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
 * The offline path has two engines (setOfflineEngine()). Direct, the
 * default, evaluates the streaming kernels, so process() equals
 * processBlock() shifted by getLatency(). TwoStage meets the same tier
 * specification at a fraction of the cost: an FFT low-pass at the input
 * rate (or at twice it) does the sharp band-limiting, after which the
 * signal is oversampled enough for a short interpolation kernel (22 to 34
 * taps at High and Ultra) to change the rate. Its arithmetic is double for
 * both instantiations. At 96 to 44.1 kHz, Ultra, it runs 8x faster than
 * Direct (see setOfflineEngine()). process() also writes into caller
 * storage, which allocates nothing once the engine is designed.
 *
 * Threading: prepare() allocates and must run on a setup thread. After
 * prepare(), processBlock() and reset() are allocation-free and belong to
 * the single stream owner; there are no cross-thread parameter setters.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, FFT.h, SimdOps.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "FFT.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
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

    /** @brief Largest target/source ratio prepare() accepts. Output per input
     *  sample follows the ratio, so past this no caller's buffer holds it. */
    static constexpr double kMaxRatio = 16777216.0;

    /** @brief Longest stage-1 kernel order of the TwoStage offline engine
     *  (see setOfflineEngine()). */
    static constexpr int kMaxTwoStageOrder = 1 << 16;

    /** @brief How process() and processRange() compute (setOfflineEngine()). */
    enum class OfflineEngine
    {
        Direct,   ///< The streaming kernels: process() is processBlock() shifted by getLatency(), bit for bit.
        TwoStage  ///< FFT band-limiting, then a short interpolator: the same tier specification, cheaper.
    };

    /**
     * @brief Chooses the offline engine (the streaming path is unaffected).
     *
     * Direct (the default) evaluates one streaming kernel per output.
     * TwoStage first low-pass filters the input by overlap-save FFT
     * convolution, at the input rate or, when the lower Nyquist lies above
     * a quarter of the input rate, at twice it (zero-stuffed): the tier's
     * passband edge and stopband edge, 12 dB more attenuation. The
     * band-limited signal is then oversampled at least 2:1 over its band, so
     * the rate change needs only a short kernel whose stopband starts at the
     * band's first image: 22 or 24 taps at High, 32 or 34 at Ultra, 40 dB
     * above the tier (the images of in-band tones fall there, and Direct's
     * longer kernel has decayed that far at those distances). When the
     * output rate is an integer fraction of that rate (48 <-> 96 kHz) the
     * output is read straight from the band-limited signal. Both stages run
     * in double, for both instantiations, so the float output is limited by
     * its own rounding only.
     *
     * Measured against Direct (MSVC x64, AVX2 dispatch, double, ns per
     * input sample once prepared, into caller storage) at 96 -> 44.1 /
     * 48 -> 44.1 / 44.1 -> 48 / 44.1 -> 96 kHz: High 3.0 / 5.0 / 5.4 / 7.9
     * instead of 16.4 / 16.6 / 20.4 / 42.1; Ultra 3.3 / 5.5 / 6.2 / 9.3
     * instead of 26.2 / 26.9 / 27.7 / 60.3. The vector-returning process()
     * adds the vector: Ultra 3.6 / 6.1 / 7.0 / 10.8. Passband edge, image
     * and alias rejection and the residual of in-band tones are those of
     * Direct or better; on in-band material the two engines agree to -152 dB
     * at High and -222 dB at Ultra.
     *
     * The transforms sit on a fixed grid, so processRange() spans agree with
     * one another and with process() bit for bit; getReach() follows the
     * engine in force. A non-finite input sample reaches every output of the
     * transforms that read it (up to the transform length, a few thousand
     * samples), not only the kernel's reach. The first offline call after
     * prepare() or after this setter designs the stages (0.1 to 0.6 ms and
     * a few hundred kB at common ratios; prepare() itself still designs the
     * streaming kernels).
     * Stage 1 is capped at kMaxTwoStageOrder: past about 190:1 downsampling
     * at Ultra its transition band widens, as Direct's does past 25:1.
     */
    void setOfflineEngine(OfflineEngine engine) noexcept { offlineEngine_ = engine; }

    /** @brief The offline engine in force (Direct by default). */
    [[nodiscard]] OfflineEngine getOfflineEngine() const noexcept { return offlineEngine_; }

    /**
     * @brief Prepares the resampler for a given rate conversion.
     *
     * Allocates internal buffers. MUST be called outside the audio thread.
     * A non-finite rate, or a ratio above kMaxRatio, is ignored and the
     * current conversion stays in force. Rates below 1 Hz are raised to 1 Hz.
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
     * zero-padded). Allocates the output vector -- offline use only; the
     * overload that writes into caller storage does not.
     *
     * @param input       Source audio samples.
     * @param inputLength Number of input samples.
     * @return Vector of resampled output samples (ceil(inputLength * ratio));
     *         empty when not prepared, for a null input or a length below 1.
     */
    [[nodiscard]] std::vector<T> process(const T* input, int inputLength)
    {
        if (table_.empty() || input == nullptr || inputLength <= 0) return {};
        std::vector<T> output(static_cast<size_t>(outputLengthFor(inputLength)));
        (void)process(input, inputLength, output.data(), static_cast<int64_t>(output.size()));
        return output;
    }

    /**
     * @brief process() into caller storage.
     *
     * Writes the first min(outputCapacity, ceil(inputLength * getRatio()))
     * samples of process()'s result, bit for bit. It allocates nothing once
     * the offline engine is designed (the first offline call after prepare()
     * or setOfflineEngine() designs it), so a converter that reuses its
     * output buffer pays no allocation per call.
     *
     * @param input          Source audio samples.
     * @param inputLength    Number of input samples.
     * @param output         Destination for the converted samples.
     * @param outputCapacity Samples the destination holds.
     * @return Samples written: 0 when not prepared, for a null pointer, or
     *         for a length or capacity below 1.
     */
    int64_t process(const T* input, int inputLength, T* output, int64_t outputCapacity)
    {
        if (table_.empty() || input == nullptr || inputLength <= 0
            || output == nullptr || outputCapacity <= 0)
            return 0;
        const int64_t outputLength = std::min<int64_t>(outputLengthFor(inputLength), outputCapacity);
        if (useTwoStage())
        {
            runTwoStage(input, inputLength, 0, outputLength, output);
            return outputLength;
        }
        int64_t outIdx = 0;
        for (; outIdx < outputLength; ++outIdx)
        {
            int64_t intPos;
            int64_t phase = 0;
            double frac = 0.0;
            if (exact_)
            {
                // Integer position arithmetic: output k reads input k*M/L.
                const int64_t num = outIdx * stepM_;
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
            output[outIdx] = interpolateOffline(input, inputLength, static_cast<int>(intPos), phase, frac);
        }
        std::fill(output + outIdx, output + outputLength, T(0));
        return outputLength;
    }

    /**
     * @brief Offline, time-aligned conversion of any span of output samples.
     *
     * Output sample k, for k in [firstOutput, firstOutput + count), is the
     * input interpolated at position k / getRatio(), with the input taken as
     * zero outside [0, inputLength). Unlike process(), which stops at the
     * input's last sample, the span may start before 0 and run past the end,
     * so it can hold the kernel's whole response to the signal: the ringing
     * before its first sample and after its last. For an impulse response,
     * whose first sample is often its loudest, that ringing is part of the
     * converted response. getReach() gives the span that holds all of it.
     * Allocates the output vector -- offline use only.
     */
    [[nodiscard]] std::vector<T> processRange(const T* input, int inputLength,
                                              int64_t firstOutput, int64_t count)
    {
        if (table_.empty() || input == nullptr || inputLength <= 0 || count <= 0) return {};
        std::vector<T> output(static_cast<size_t>(count));
        if (useTwoStage())
        {
            runTwoStage(input, inputLength, firstOutput, count, output.data());
            return output;
        }
        for (int64_t i = 0; i < count; ++i)
        {
            const int64_t outIdx = firstOutput + i;
            int64_t intPos;
            int64_t phase = 0;
            double frac = 0.0;
            if (exact_)
            {
                // Floor division: negative positions are ordinary here.
                const int64_t num = outIdx * stepM_;
                intPos = num / phasesL_;
                if (num % phasesL_ != 0 && num < 0) --intPos;
                phase = num - intPos * phasesL_;
            }
            else
            {
                const double srcPos = static_cast<double>(outIdx) / ratio_;
                const double fl = std::floor(srcPos);
                intPos = static_cast<int64_t>(fl);
                frac = srcPos - fl;
            }
            // A window that misses the input entirely is exactly zero.
            const int64_t firstTap = intPos - halfTaps_ + 1;
            if (firstTap + taps_ <= 0 || firstTap >= inputLength) continue;
            output[static_cast<size_t>(i)] =
                interpolateOffline(input, inputLength, static_cast<int>(intPos), phase, frac);
        }
        return output;
    }

    /**
     * @brief How far, in output samples, the kernel reaches before an input
     *        sample and after it: processRange() over
     *        [-getReach(), ceil(inputLength * getRatio()) + getReach()) holds
     *        the whole converted signal. It is the reach of the offline
     *        engine in force (setOfflineEngine()).
     */
    [[nodiscard]] int64_t getReach() const noexcept
    {
        if (offlineEngine_ == OfflineEngine::TwoStage && !table_.empty() && ratio_ != 1.0)
        {
            const TwoStageShape s = twoStageShape(sourceRate_, targetRate_, ratio_, quality_);
            // Input sample i drives mid samples up*i -/+ half1; a mid sample
            // reaches outputs whose interpolation window holds it.
            return static_cast<int64_t>(std::ceil(static_cast<double>(s.half1 + s.stage2.taps / 2 + 1)
                                                  * ratio_ / static_cast<double>(s.up))) + 1;
        }
        return static_cast<int64_t>(std::ceil((taps_ - halfTaps_ + 1) * ratio_)) + 1;
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
    /** @brief ceil(inputLength * ratio), clamped through double before the int
     *  cast (a huge ratio would otherwise overflow the cast itself). */
    [[nodiscard]] int outputLengthFor(int inputLength) const noexcept
    {
        const double n = std::ceil(static_cast<double>(inputLength) * ratio_);
        return static_cast<int>(std::min(n, static_cast<double>(std::numeric_limits<int>::max())));
    }

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
        // sum_k (x^2 / 4)^k / (k!)^2: each term is the previous one times
        // (x^2 / 4) / k^2, the reciprocal squares coming from a table, so the
        // series has no division in its loop.
        static constexpr std::array<double, 201> kInvSquares = [] {
            std::array<double, 201> a {};
            for (int k = 1; k <= 200; ++k) a[static_cast<size_t>(k)] = 1.0 / (static_cast<double>(k) * static_cast<double>(k));
            return a;
        }();
        const double y = 0.25 * x * x;
        double sum = 1.0, term = 1.0;
        for (int k = 1; k <= 200; ++k)
        {
            term *= y * kInvSquares[static_cast<size_t>(k)];
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
    template <typename D>
    static void buildPhase(D* dst, const Design& d, double frac)
    {
        const int half = d.taps / 2;
        if (d.identity)
        {
            for (int j = 0; j < d.taps; ++j)
                dst[j] = (j == half - 1) ? D(1) : D(0);
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
            dst[j] = static_cast<D>(sincVal * win);
            sum += sincVal * win;
        }
        if (std::abs(sum) > 1e-12)
        {
            const double inv = 1.0 / sum;
            for (int j = 0; j < d.taps; ++j)
                dst[j] = static_cast<D>(static_cast<double>(dst[j]) * inv);
        }
    }

    /**
     * @brief numPhases kernels, phase p being buildPhase() at fractional
     * position (p + offset) / L, normalised the same way.
     *
     * Every tap of every phase then sits at t = k / L for an integer k, and
     * the windowed sinc is even, so it is evaluated once per |k| on that grid
     * and shared: half the Bessel and sine evaluations of a phase-by-phase
     * build, and none repeated.
     */
    template <typename D>
    static void buildTable(D* dst, const Design& d, int64_t L, int64_t offset, int64_t numPhases)
    {
        const auto taps = static_cast<size_t>(d.taps);
        if (d.identity)
        {
            for (int64_t p = 0; p < numPhases; ++p)
                buildPhase(dst + static_cast<size_t>(p) * taps, d,
                           static_cast<double>(p + offset) / static_cast<double>(L));
            return;
        }
        constexpr double kPi = std::numbers::pi;
        const int half = d.taps / 2;
        const double i0Beta = besselI0(d.beta);
        const double scale = 1.0 / static_cast<double>(L);
        // |k| = |(j - half + 1) L - (p + offset)| stays below this bound.
        const int64_t kMax = static_cast<int64_t>(half + 1) * L + numPhases + 2;
        std::vector<double> grid(static_cast<size_t>(kMax) + 1, 0.0);
        const int64_t kWindow = static_cast<int64_t>(half) * L;   // the window is 0 from here
        for (int64_t k = 0; k < std::min(kWindow, kMax + 1); ++k)
        {
            const double t = static_cast<double>(k) * scale;
            const double x = t * d.cutoff;
            const double sincVal = (x < 1e-12) ? d.cutoff : d.cutoff * std::sin(kPi * x) / (kPi * x);
            const double wx = t / static_cast<double>(half);
            grid[static_cast<size_t>(k)] = sincVal * besselI0(d.beta * std::sqrt(1.0 - wx * wx)) / i0Beta;
        }
        for (int64_t p = 0; p < numPhases; ++p)
        {
            D* row = dst + static_cast<size_t>(p) * taps;
            double sum = 0.0;
            int64_t k = static_cast<int64_t>(1 - half) * L - (p + offset);
            for (size_t j = 0; j < taps; ++j, k += L)
            {
                const double v = grid[static_cast<size_t>(k < 0 ? -k : k)];
                row[j] = static_cast<D>(v);
                sum += v;
            }
            if (std::abs(sum) > 1e-12)
            {
                const double inv = 1.0 / sum;
                for (size_t j = 0; j < taps; ++j)
                    row[j] = static_cast<D>(static_cast<double>(row[j]) * inv);
            }
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
        if (!std::isfinite(sourceRate) || !std::isfinite(targetRate)) return;

        // Derive the complete candidate configuration without touching the
        // live object. The std::max behavior intentionally preserves the
        // established treatment of non-positive rates.
        const double stagedSourceRate = std::max(sourceRate, 1.0);
        const double stagedTargetRate = std::max(targetRate, 1.0);
        const double stagedRatio = stagedTargetRate / stagedSourceRate;
        if (stagedRatio > kMaxRatio) return;
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
            buildTable(stagedTable.data(), d, l, 0, l);
        }
        else
        {
            // kTablePhases + 3 phases: one guard phase before position 0 and
            // two after the last, so cubic interpolation across phases never
            // reads outside the table.
            stagedTable.resize(static_cast<size_t>((kTablePhases + 3) * d.taps));
            buildTable(stagedTable.data(), d, kTablePhases, -1, kTablePhases + 3);
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
        quality_ = quality;
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

    // -- Two-stage offline engine ------------------------------------------------

    /// Phases of the stage-2 table read with cubic interpolation (non-rational ratios).
    static constexpr int kTwoStagePhases = 2048;

    /** @brief The two kernels of the two-stage engine for one conversion. */
    struct TwoStageShape
    {
        int up = 1;            ///< Stage 1 rate, in multiples of the input rate (1 or 2).
        int half1 = 1;         ///< Stage 1 kernel: 2 * half1 + 1 taps at the mid rate.
        double cutoff1 = 1.0;  ///< Stage 1 sinc cutoff, 1 = mid Nyquist.
        double beta1 = 0.0;    ///< Stage 1 Kaiser beta.
        Design stage2;         ///< Stage 2 kernel, in mid samples.
        int64_t phases2 = 0;   ///< fs_out / fs_mid = phases2 / step2 when rational, else 0.
        int64_t step2 = 0;
        bool pick = false;     ///< phases2 == 1: every output is a mid sample.
    };

    /// Attenuation of each stage over the tier's. Stage 1 sets the stopband
    /// next to the band edge, so 12 dB keeps the sum of both stages inside
    /// the tier there. Stage 2's stopband starts at the first image of the
    /// band, where the images of in-band tones fall; Direct's single kernel
    /// has decayed far below its specification at those distances, so stage
    /// 2 gets 40 dB to leave in-band tones at least as clean as Direct does.
    static constexpr double kStage1MarginDb = 12.0;
    static constexpr double kStage2MarginDb = 40.0;

    [[nodiscard]] static TwoStageShape twoStageShape(double sourceRate, double targetRate,
                                                     double ratio, Quality quality) noexcept
    {
        TwoStageShape s;
        const Spec spec = qualitySpec(quality);
        const double atten1 = spec.attenuationDb + kStage1MarginDb;
        const double atten2 = spec.attenuationDb + kStage2MarginDb;
        const double band = 0.5 * std::min(1.0, ratio);   // lower Nyquist, cycles per input sample
        // The mid rate doubles the input's when the band is too close to the
        // input Nyquist to leave stage 2 a transition of half the mid rate.
        s.up = ((1.0 + spec.passband) * band <= 0.5) ? 1 : 2;
        const double b = band / static_cast<double>(s.up);   // cycles per mid sample
        // Stage 1: flat to passband * b, stopband from b (the tier's edges).
        const double order1 = std::min((atten1 - 7.95) / (14.36 * (1.0 - spec.passband) * b),
                                       static_cast<double>(kMaxTwoStageOrder));
        s.half1 = std::max(1, static_cast<int>(std::ceil(0.5 * order1)));
        s.cutoff1 = (1.0 + spec.passband) * b;
        s.beta1 = kaiserBeta(atten1);
        // Stage 2: flat to passband * b; the mid signal holds nothing above b
        // until its first image at 1 - b, where the stopband starts.
        const double order2 = (atten2 - 7.95) / (14.36 * (1.0 - (1.0 + spec.passband) * b));
        s.stage2.taps = std::max(4, 2 * static_cast<int>(std::ceil(0.5 * order2)));
        s.stage2.cutoff = spec.passband * b + 1.0 - b;
        s.stage2.beta = kaiserBeta(atten2);
        const auto [l, m] = rationalRatio(sourceRate * static_cast<double>(s.up), targetRate);
        s.phases2 = l;
        s.step2 = m;
        // An integer ratio between the mid and output rates (48 <-> 96 kHz)
        // reads the band-limited mid signal at its own samples: no stage 2.
        // Two taps, {1, 0}, keep the window arithmetic of the general case.
        if (l == 1)
        {
            s.pick = true;
            s.stage2.taps = 2;
        }
        return s;
    }

    /** @brief Kernels, transforms and scratch of the two-stage engine. */
    struct TwoStagePlan
    {
        bool ready = false;
        double sourceRate = 0.0;      ///< Key: the conversion it was designed for.
        double targetRate = 0.0;
        Quality quality = Quality::Normal;
        TwoStageShape shape;
        int fftSize = 0;              ///< Stage 1 overlap-save transform length.
        int block = 0;                ///< Mid samples each frame yields (a transform packs two).
        bool exact2 = true;           ///< Stage 2 phases exact (rational ratio).
        int64_t phases2 = 1;          ///< Exact: fs_out / fs_mid = phases2 / step2.
        int64_t step2 = 1;
        double midStep = 1.0;         ///< Mid samples per output (table mode).
        std::vector<double> gain1;       ///< Stage 1 kernel spectrum (real), fftSize bins, 1/fftSize folded in.
        std::vector<double> table2;      ///< Stage 2 kernels: in output order when exact, else by phase.
        std::vector<int64_t> advance2;   ///< Exact: mid samples between output k and k + 1 of a cycle.
        std::vector<int32_t> cycleOf2;   ///< Exact: cycle position of each phase.
        std::optional<detail::fft::SplitFFT<double>> engine;       ///< fftSize complex points.
        std::optional<detail::fft::SplitFFT<double>> engineHalf;   ///< fftSize / 2 (zero-stuffed input).
        std::vector<double> mid;
    };

    /** @brief True when the offline call runs the two-stage engine (designs it if stale). */
    [[nodiscard]] bool useTwoStage()
    {
        if (offlineEngine_ != OfflineEngine::TwoStage || ratio_ == 1.0) return false;
        if (!plan_.ready || plan_.sourceRate != sourceRate_ || plan_.targetRate != targetRate_
            || plan_.quality != quality_)
        {
            // Built aside: a failed allocation leaves the previous plan intact.
            TwoStagePlan fresh = buildTwoStagePlan(sourceRate_, targetRate_, ratio_, quality_);
            plan_ = std::move(fresh);
        }
        return true;
    }

    [[nodiscard]] static TwoStagePlan buildTwoStagePlan(double sourceRate, double targetRate,
                                                        double ratio, Quality quality)
    {
        constexpr double kPi = std::numbers::pi;
        TwoStagePlan p;
        p.sourceRate = sourceRate;
        p.targetRate = targetRate;
        p.quality = quality;
        p.shape = twoStageShape(sourceRate, targetRate, ratio, quality);
        const TwoStageShape& s = p.shape;
        const int order1 = 2 * s.half1;
        const int taps2 = s.stage2.taps;

        // At least 75% of every transform is output.
        int n = 256;
        while (n < 4 * order1 || n < 4 * taps2 + 16) n *= 2;
        p.fftSize = n;
        p.block = n - order1;

        // Stage 1: Kaiser-windowed sinc at the mid rate, DC gain `up` (the
        // zero-stuffing keeps one sample in `up`), stored centred on sample
        // 0 of the circular frame: its spectrum is then real, and the
        // filtering a multiplication of every (re, im) pair by one number.
        std::vector<double> h(static_cast<size_t>(n), 0.0);
        const double i0Beta = besselI0(s.beta1);
        double sum = 0.0;
        for (int t = -s.half1; t <= s.half1; ++t)
        {
            const double x = static_cast<double>(t) * s.cutoff1;
            const double sincVal = (std::abs(x) < 1e-12)
                ? s.cutoff1
                : s.cutoff1 * std::sin(kPi * x) / (kPi * x);
            const double wx = static_cast<double>(t) / static_cast<double>(s.half1);
            const double win = besselI0(s.beta1 * std::sqrt(std::max(0.0, 1.0 - wx * wx))) / i0Beta;
            h[static_cast<size_t>(t >= 0 ? t : n + t)] = sincVal * win;
            sum += sincVal * win;
        }
        // The inverse transform's 1/n is folded into the gain.
        const double gain = static_cast<double>(s.up) / (sum * static_cast<double>(n));
        for (auto& v : h) v *= gain;
        {
            FFTReal<double> fft(static_cast<size_t>(n));
            std::vector<double> spec(static_cast<size_t>(n) + 2, 0.0);
            fft.forward(h.data(), spec.data());
            // The imaginary parts are rounding; the spectrum is even.
            p.gain1.assign(static_cast<size_t>(n), 0.0);
            for (int k = 0; k <= n / 2; ++k)
            {
                p.gain1[static_cast<size_t>(k)] = spec[static_cast<size_t>(2 * k)];
                p.gain1[static_cast<size_t>((n - k) % n)] = spec[static_cast<size_t>(2 * k)];
            }
        }
        // Two frames per transform: one as the real part, the next as the
        // imaginary part. The kernel is real, so the filtered frames come
        // back as the real and imaginary parts of one inverse.
        p.engine.emplace(static_cast<size_t>(n));
        if (s.up == 2) p.engineHalf.emplace(static_cast<size_t>(n / 2));

        // Stage 2: exact phases when fs_out / fs_mid is a manageable fraction.
        const int64_t l = s.phases2, m = s.step2;
        p.exact2 = l > 0 && l <= kMaxExactPhases
                && l * static_cast<int64_t>(taps2) <= kMaxExactCoefficients;
        p.midStep = static_cast<double>(s.up) / ratio;
        if (s.pick)
        {
            p.phases2 = 1;
            p.step2 = m;
            p.table2 = { 1.0, 0.0 };
        }
        else if (p.exact2)
        {
            p.phases2 = l;
            p.step2 = m;
            // The rows follow the outputs: output k of a cycle of l outputs has
            // phase (k * m) mod l, so the stage-2 loop reads the table front to
            // back and steps its window by a precomputed advance (no division
            // and no data-dependent branch per output).
            p.table2.resize(static_cast<size_t>(l * taps2));
            p.advance2.resize(static_cast<size_t>(l));
            p.cycleOf2.resize(static_cast<size_t>(l));
            for (int64_t k = 0; k < l; ++k)
            {
                const int64_t ph = (k * (m % l)) % l;
                buildPhase(p.table2.data() + k * taps2, s.stage2,
                           static_cast<double>(ph) / static_cast<double>(l));
                p.advance2[static_cast<size_t>(k)] = m / l + (ph + m % l >= l ? 1 : 0);
                p.cycleOf2[static_cast<size_t>(ph)] = static_cast<int32_t>(k);
            }
        }
        else
        {
            // One guard phase before position 0 and two after the last.
            p.table2.resize(static_cast<size_t>(kTwoStagePhases + 3) * static_cast<size_t>(taps2));
            for (int ph = 0; ph < kTwoStagePhases + 3; ++ph)
                buildPhase(p.table2.data() + static_cast<size_t>(ph) * static_cast<size_t>(taps2), s.stage2,
                           static_cast<double>(ph - 1) / static_cast<double>(kTwoStagePhases));
        }
        // Room for the carried window, then the two frames of one transform
        // (see runTwoStage()).
        p.mid.assign(static_cast<size_t>(taps2 + s.up + 2 * p.block), 0.0);
        p.ready = true;
        return p;
    }

    /**
     * @brief Stage 1 for mid samples [s, s + 2 * block): two frames of the
     * zero-stuffed input convolved with the centred stage-1 kernel (m[k] is
     * the filtered signal at mid sample k), m[s + i] written to dst[i].
     *
     * One complex transform carries both frames, the first as its real part
     * and the second as its imaginary part. The kernel is real, so the
     * product with its (real, even) spectrum keeps them apart and the inverse
     * returns the two filtered frames as its real and imaginary parts. The
     * inverse runs as the forward transform of the product with re and im
     * swapped, and swaps them back on the way out. `s - half1` is a multiple
     * of `up`.
     */
    void twoStagePair(const T* input, int inputLength, int64_t s, double* dst) noexcept
    {
        TwoStagePlan& p = plan_;
        const int n = p.fftSize, up = p.shape.up, half1 = p.shape.half1, block = p.block;
        const int frameInputs = n / up;
        const int64_t q0 = (s - half1) / up;   // the first frame's first input (exact)
        const int64_t q1 = q0 + block / up;    // and the second frame's
        if (std::max<int64_t>(q0, 0) >= std::min<int64_t>(q1 + frameInputs, inputLength))
        {
            std::fill(dst, dst + 2 * block, 0.0);
            return;
        }
        detail::fft::SplitFFT<double>& first = (up == 1) ? *p.engine : *p.engineHalf;
        const auto load = [&](double* frame, int64_t q) noexcept {
            const int64_t a = std::clamp<int64_t>(-q, 0, frameInputs);
            const int64_t b = std::clamp<int64_t>(inputLength - q, a, frameInputs);
            std::fill(frame, frame + a, 0.0);
            for (int64_t t = a; t < b; ++t) frame[t] = static_cast<double>(input[q + t]);
            std::fill(frame + b, frame + frameInputs, 0.0);
        };
        load(first.inRe(), q0);
        load(first.inIm(), q1);
        const double* zr = nullptr;
        const double* zi = nullptr;
        first.run(zr, zi);

        // The zero-stuffed frames' spectrum is the half-length one repeated,
        // Z[k] = Zh[k mod n/2]. Each bin is read before it is written, so the
        // engine may hand back its own input buffers.
        const double* g = p.gain1.data();
        double* ir = p.engine->inRe();
        double* ii = p.engine->inIm();
        const int mask = n / up - 1;
        for (int k = 0; k < n; ++k)
        {
            const double re = zr[k & mask], im = zi[k & mask];
            ir[k] = im * g[k];
            ii[k] = re * g[k];
        }
        const double* wr = nullptr;
        const double* wi = nullptr;
        p.engine->run(wr, wi);
        std::copy(wi + half1, wi + half1 + block, dst);
        std::copy(wr + half1, wr + half1 + block, dst + block);
    }

    /** @brief Stage 2, table mode: cubic interpolation across four phases. */
    [[nodiscard]] double twoStageTable(const double* src, double frac) const noexcept
    {
        const int taps2 = plan_.shape.stage2.taps;
        const double exactPhase = frac * static_cast<double>(kTwoStagePhases);
        int p0 = static_cast<int>(exactPhase);
        if (p0 > kTwoStagePhases - 1) p0 = kTwoStagePhases - 1;
        const double u = exactPhase - static_cast<double>(p0);
        const double* k = plan_.table2.data() + static_cast<size_t>(p0) * static_cast<size_t>(taps2);
        const double sm = simd::dotProduct(k, src, taps2);
        const double s0 = simd::dotProduct(k + taps2, src, taps2);
        const double s1 = simd::dotProduct(k + 2 * taps2, src, taps2);
        const double s2 = simd::dotProduct(k + 3 * taps2, src, taps2);
        const double wm = -u * (u - 1.0) * (u - 2.0) / 6.0;
        const double w0 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0;
        const double w1 = -(u + 1.0) * u * (u - 2.0) / 2.0;
        const double w2 = (u + 1.0) * u * (u - 1.0) / 6.0;
        return wm * sm + w0 * s0 + w1 * s1 + w2 * s2;
    }

    /**
     * @brief Outputs firstOutput .. firstOutput + count of the two-stage
     * engine. Outputs whose combined window misses the input are exactly 0.
     * Stage 1 runs one transform (two frames) at a time into a rolling mid
     * buffer: the stage-2 window carried over from the previous transform,
     * then the new transform's samples (mid sample k at mid[k - origin]).
     */
    void runTwoStage(const T* input, int inputLength, int64_t firstOutput, int64_t count, T* out)
    {
        if (plan_.shape.pick)   runTwoStageAs<true, true>(input, inputLength, firstOutput, count, out);
        else if (plan_.exact2)  runTwoStageAs<true, false>(input, inputLength, firstOutput, count, out);
        else                    runTwoStageAs<false, false>(input, inputLength, firstOutput, count, out);
    }

    template <bool Exact, bool Pick>
    void runTwoStageAs(const T* input, int inputLength, int64_t firstOutput, int64_t count, T* out)
    {
        // Everything the loop reads lives in locals: the output stores
        // cannot then force reloads of the plan.
        TwoStagePlan& p = plan_;
        const int up = p.shape.up, half1 = p.shape.half1, block = p.block;
        const int taps2 = p.shape.stage2.taps, half2 = taps2 / 2;
        const int carry = taps2 + up;          // room before the transform
        const int64_t pairBlock = 2 * static_cast<int64_t>(block);   // mid samples per transform
        const int64_t l = p.phases2;
        const double midStep = p.midStep;
        // Mid samples that can differ from zero.
        const int64_t nzLo = -static_cast<int64_t>(half1);
        const int64_t nzHi = static_cast<int64_t>(up) * (inputLength - 1) + half1;
        double* const mid = p.mid.data();
        int64_t origin = 0, bufStart = 0, bufEnd = 0;
        bool haveBuf = false;

        // Exact: output k sits at mid position k * step2 / l; the anchor
        // (its floor) and the phase (the remainder) advance exactly.
        // Table: the position is k * midStep.
        int64_t anchor = 0, phase = 0, wholeStep = 0, remStep = 0;
        double frac = 0.0;
        if constexpr (Exact)
        {
            wholeStep = p.step2 / l;
            remStep = p.step2 % l;
            // floor(k * step2 / l) without forming k * step2.
            const int64_t r = firstOutput * remStep;
            int64_t q = r / l, rem = r % l;
            if (rem < 0) { rem += l; --q; }
            anchor = firstOutput * wholeStep + q;
            phase = rem;
        }

        int64_t i = 0;
        while (i < count)
        {
            if constexpr (!Exact)
            {
                const double pos = static_cast<double>(firstOutput + i) * midStep;
                const double fl = std::floor(pos);
                anchor = static_cast<int64_t>(fl);
                frac = pos - fl;
            }
            // Tap j weighs mid sample anchor - half2 + 1 + j (as buildPhase).
            int64_t w0 = anchor - half2 + 1;
            if (w0 + taps2 <= nzLo || w0 > nzHi)
            {
                out[i++] = T(0);
                if constexpr (Exact)
                {
                    anchor += wholeStep;
                    phase += remStep;
                    if (phase >= l) { phase -= l; ++anchor; }
                }
                continue;
            }
            if (!haveBuf || w0 >= bufEnd)
            {
                // (Re)start on the fixed grid of transforms, half1 + j * 2 *
                // block for any integer j: every call computes a mid sample
                // with the same transform, so spans agree bit for bit. The
                // grid puts an input sample on each frame's first slot (block
                // is even).
                const int64_t d = w0 - half1;
                const int64_t j = d / pairBlock - ((d % pairBlock) < 0 ? 1 : 0);
                bufStart = bufEnd = half1 + j * pairBlock;
                haveBuf = true;
            }
            while (bufEnd < w0 + taps2)
            {
                // Move the part of the window already computed (fewer than
                // taps2 samples) in front of the new transform's samples.
                const int64_t keep = std::max(w0, bufStart);
                const int r = static_cast<int>(bufEnd - keep);
                if (r > 0) std::copy(mid + (keep - origin), mid + (bufEnd - origin), mid + (carry - r));
                twoStagePair(input, inputLength, bufEnd, mid + carry);
                origin = bufEnd - carry;
                bufStart = keep;
                bufEnd += pairBlock;
            }
            // Every output whose window is now in the buffer.
            const int64_t last = std::min(bufEnd - taps2, nzHi);   // largest w0 served here
            if constexpr (Exact)
            {
                const double* src = mid + (w0 - origin);
                if constexpr (Pick)
                {
                    do
                    {
                        out[i++] = static_cast<T>(*src);
                        src += wholeStep;
                        w0 += wholeStep;
                        phase += remStep;
                        if (phase >= l) { phase -= l; ++src; ++w0; }
                    } while (i < count && w0 <= last);
                }
                else
                {
                    // The table rows follow the outputs (see buildTwoStagePlan()),
                    // so the loop walks them in order and steps the window by the
                    // cycle's precomputed advances.
                    const double* const rows = p.table2.data();
                    const int64_t* const advance = p.advance2.data();
                    int64_t k = p.cycleOf2[static_cast<size_t>(phase)];
                    const double* row = rows + k * taps2;
                    do
                    {
                        out[i++] = static_cast<T>(simd::dotProduct(row, src, taps2));
                        const int64_t a = advance[k];
                        src += a;
                        w0 += a;
                        row += taps2;
                        if (++k == l) { k = 0; row = rows; }
                    } while (i < count && w0 <= last);
                    phase = (k * remStep) % l;
                }
                anchor = w0 + half2 - 1;
            }
            else
            {
                do
                {
                    out[i++] = static_cast<T>(twoStageTable(mid + (w0 - origin), frac));
                    if (i == count) break;
                    const double pos = static_cast<double>(firstOutput + i) * midStep;
                    const double fl = std::floor(pos);
                    anchor = static_cast<int64_t>(fl);
                    frac = pos - fl;
                    w0 = anchor - half2 + 1;
                } while (w0 <= last);
                // The loop head recomputes the same position.
            }
        }
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

    Quality quality_ = Quality::Normal;
    OfflineEngine offlineEngine_ = OfflineEngine::Direct;

    std::vector<T> table_;
    std::vector<T> edgeScratch_;   ///< Offline edge window (process() allocates anyway).
    ChannelState mono_;
    std::vector<ChannelState> channelStates_;
    TwoStagePlan plan_;            ///< Two-stage engine, designed by the first offline call.
};

} // namespace dspark
