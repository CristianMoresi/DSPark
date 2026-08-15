// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralFreeze.h
 * @brief Real-time spectral hold with tonal and diffuse phase evolution.
 *
 * SpectralFreeze composes one SpectralProcessor WOLA engine. It keeps two
 * completed all-channel spectra, captures the newest one on a freeze request,
 * and advances its phases either by identity peak locking or by a deterministic
 * channel-shared diffuse sequence. Entry and exit use one N-sample polar
 * transition; the live path therefore retains the primitive's exact N-sample
 * signal latency without a parallel dry path.
 *
 * Threading: prepare(), reset(), the integer queries and processBlock() belong
 * to the stream owner. setFrozen(), isFrozen(), setPhaseMode() and
 * getPhaseMode() are lock-free and may be called from any thread. Processing,
 * setters and queries allocate no memory after prepare().
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h, SpectralProcessor.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SpectralProcessor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#if !defined(DSPARK_NO_EXCEPTIONS)
#include <stdexcept>
#endif

namespace dspark {

/**
 * @class SpectralFreeze
 * @brief Frequency-domain hold with tonal and diffuse phase modes.
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SpectralFreeze final
{
public:
    /** @brief Phase evolution used by the captured spectrum. */
    enum class PhaseMode : std::uint8_t { Tonal, Diffuse };

    /**
     * @brief Prepares the processor and allocates all audio-path storage.
     *
     * Invalid specifications are ignored and preserve the current stream.
     * FFT requests are clamped to [256,32768] and rounded upward to a power of
     * two. Only N/8, N/4 and N/2 are accepted as explicit hop sizes; every
     * other hop selects N/4.
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048, int hopSize = 0)
    {
        if (!validSpec(spec)) return;

        const int newFftSize = sanitizeFftSize(fftSize);
        const int newHopSize = sanitizeHopSize(newFftSize, hopSize);
        const int newBins = newFftSize / 2 + 1;
        const int channels = spec.numChannels;

        // Build the complete replacement off to the side. A valid prepare is
        // setup-only; if an allocation throws, the prior running instance has
        // not been partially resized.
        SpectralProcessor<T> nextProcessor;
        nextProcessor.prepare(spec, newFftSize, newHopSize);

        const std::size_t channelBins = checkedProduct(
            static_cast<std::size_t>(channels), static_cast<std::size_t>(newBins));
        const std::size_t complexChannelBins = checkedProduct(channelBins, 2u);

        std::vector<T> nextScratch(checkedProduct(
            static_cast<std::size_t>(channels),
            static_cast<std::size_t>(spec.maxBlockSize)), T(0));
        std::vector<T> nextRaw(checkedProduct(complexChannelBins, 3u), T(0));
        std::vector<T> nextMagnitude(channelBins, T(0));
        std::vector<T> nextCapturedPhase(channelBins, T(0));
        std::vector<T> nextFrozenPhase(channelBins, T(0));
        std::vector<T> nextPower(static_cast<std::size_t>(newBins), T(0));
        std::vector<T> nextOmega(static_cast<std::size_t>(newBins), T(0));
        std::vector<T> nextOmegaImag(static_cast<std::size_t>(newBins), T(0));
        std::vector<std::uint32_t> nextActive(
            static_cast<std::size_t>(newBins), 0u);
        std::vector<std::uint32_t> nextPeak(
            static_cast<std::size_t>(newBins), 0u);
        std::vector<std::uint32_t> nextRegion(
            static_cast<std::size_t>(newBins), 0u);

        processor_ = std::move(nextProcessor);
        scratch_ = std::move(nextScratch);
        rawSpectra_ = std::move(nextRaw);
        capturedMagnitude_ = std::move(nextMagnitude);
        capturedPhase_ = std::move(nextCapturedPhase);
        frozenPhase_ = std::move(nextFrozenPhase);
        binPower_ = std::move(nextPower);
        omega_ = std::move(nextOmega);
        omegaImag_ = std::move(nextOmegaImag);
        active_ = std::move(nextActive);
        peak_ = std::move(nextPeak);
        regionPeak_ = std::move(nextRegion);

        spec_ = spec;
        fftSize_ = newFftSize;
        hopSize_ = newHopSize;
        numBins_ = newBins;
        prepared_ = true;
        rebuildScratchPointers();
        clearStreamState(false);
    }

    /**
     * @brief Clears WOLA, history, capture and transition state.
     * Requested controls are intentionally preserved.
     */
    void reset() noexcept
    {
        if (!prepared_) return;
        clearStreamState(true);
    }

    /** @brief Publishes the requested freeze state with one atomic RMW. */
    void setFrozen(bool frozen) noexcept
    {
        if (frozen)
            requested_.fetch_or(kFrozenBit, std::memory_order_relaxed);
        else
            requested_.fetch_and(~kFrozenBit, std::memory_order_relaxed);
    }

    /** @brief Returns the requested freeze state, not transition completion. */
    [[nodiscard]] bool isFrozen() const noexcept
    {
        return (requested_.load(std::memory_order_relaxed) & kFrozenBit) != 0u;
    }

    /**
     * @brief Publishes the requested phase mode with one atomic RMW.
     * Any representation other than exactly Diffuse selects Tonal.
     */
    void setPhaseMode(PhaseMode mode) noexcept
    {
        if (mode == PhaseMode::Diffuse)
            requested_.fetch_or(kDiffuseBit, std::memory_order_relaxed);
        else
            requested_.fetch_and(~kDiffuseBit, std::memory_order_relaxed);
    }

    /** @brief Returns the requested phase mode. */
    [[nodiscard]] PhaseMode getPhaseMode() const noexcept
    {
        return (requested_.load(std::memory_order_relaxed) & kDiffuseBit) != 0u
                   ? PhaseMode::Diffuse
                   : PhaseMode::Tonal;
    }

    /** @brief Returns the exact input-signal latency, or zero before prepare. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return prepared_ ? fftSize_ : 0;
    }

    /** @brief Returns the prepared FFT size, or zero before prepare. */
    [[nodiscard]] int getFFTSize() const noexcept
    {
        return prepared_ ? fftSize_ : 0;
    }

    /** @brief Returns the prepared hop size, or zero before prepare. */
    [[nodiscard]] int getHopSize() const noexcept
    {
        return prepared_ ? hopSize_ : 0;
    }

    /** @brief Returns the N-sample transition duration, or zero unprepared. */
    [[nodiscard]] int getTransitionSamples() const noexcept
    {
        return prepared_ ? fftSize_ : 0;
    }

    /**
     * @brief Processes an in-place block without allocation or locking.
     *
     * Missing prepared channels advance on zero input. Runtime channels beyond
     * the prepared width remain bit-for-bit untouched. Oversize calls are
     * internally sliced without disturbing absolute hop anchors.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_ || buffer.getNumSamples() <= 0) return;

        const int runtimeChannels = buffer.getNumChannels();
        const int processedChannels = std::min(runtimeChannels, spec_.numChannels);
        const int totalSamples = buffer.getNumSamples();

        int offset = 0;
        while (offset < totalSamples)
        {
            const int block = std::min(spec_.maxBlockSize, totalSamples - offset);

            for (int ch = 0; ch < spec_.numChannels; ++ch)
            {
                T* const work = scratchChannels_[static_cast<std::size_t>(ch)];
                if (ch < processedChannels && buffer.getChannel(ch) != nullptr)
                {
                    const T* const source = buffer.getChannel(ch) + offset;
                    for (int i = 0; i < block; ++i)
                        work[i] = std::isfinite(source[i]) ? source[i] : T(0);
                }
                else
                {
                    std::fill_n(work, block, T(0));
                }
            }

            AudioBufferView<T> workView(scratchChannels_.data(),
                                        spec_.numChannels, block);
            auto callback = [this](T* spectrum, int bins) noexcept {
                processSpectrum(spectrum, bins);
            };
            processor_.processBlock(workView, callback);

            for (int ch = 0; ch < processedChannels; ++ch)
            {
                T* const destination = buffer.getChannel(ch);
                if (destination == nullptr) continue;
                const T* const work = scratchChannels_[static_cast<std::size_t>(ch)];
                for (int i = 0; i < block; ++i)
                    destination[offset + i] = std::isfinite(work[i]) ? work[i] : T(0);
            }

            offset += block;
        }
    }

private:
    static constexpr std::uint32_t kFrozenBit = 1u;
    static constexpr std::uint32_t kDiffuseBit = 2u;
    static constexpr int kMinFftSize = 256;
    static constexpr int kMaxFftSize = 32768;
    static constexpr int kMaxBlockSize = 65536;
    static constexpr int kMaxChannels = 16;
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free);

    [[nodiscard]] static bool validSpec(const AudioSpec& spec) noexcept
    {
        return std::isfinite(spec.sampleRate) && spec.sampleRate > 0.0
            && spec.maxBlockSize >= 1 && spec.maxBlockSize <= kMaxBlockSize
            && spec.numChannels >= 1 && spec.numChannels <= kMaxChannels;
    }

    [[nodiscard]] static int sanitizeFftSize(int requested) noexcept
    {
        requested = std::clamp(requested, kMinFftSize, kMaxFftSize);
        int result = kMinFftSize;
        while (result < requested) result <<= 1;
        return result;
    }

    [[nodiscard]] static int sanitizeHopSize(int fftSize, int requested) noexcept
    {
        if (requested == fftSize / 8 || requested == fftSize / 4
            || requested == fftSize / 2)
            return requested;
        return fftSize / 4;
    }

    [[nodiscard]] static std::size_t checkedProduct(std::size_t a,
                                                    std::size_t b)
    {
        // All public dimensions are tightly capped; this guard documents the
        // allocation arithmetic and remains valid if the caps later change.
        if (a != 0u && b > std::numeric_limits<std::size_t>::max() / a)
        {
#if defined(DSPARK_NO_EXCEPTIONS)
            return 0u;
#else
            throw std::length_error("SpectralFreeze allocation size overflow");
#endif
        }
        return a * b;
    }

    void rebuildScratchPointers() noexcept
    {
        scratchChannels_.fill(nullptr);
        for (int ch = 0; ch < spec_.numChannels; ++ch)
        {
            scratchChannels_[static_cast<std::size_t>(ch)] = scratch_.data()
                + static_cast<std::size_t>(ch)
                    * static_cast<std::size_t>(spec_.maxBlockSize);
        }
    }

    void clearStreamState(bool resetProcessor) noexcept
    {
        if (resetProcessor) processor_.reset();
        std::fill(scratch_.begin(), scratch_.end(), T(0));
        std::fill(rawSpectra_.begin(), rawSpectra_.end(), T(0));
        std::fill(capturedMagnitude_.begin(), capturedMagnitude_.end(), T(0));
        std::fill(capturedPhase_.begin(), capturedPhase_.end(), T(0));
        std::fill(frozenPhase_.begin(), frozenPhase_.end(), T(0));
        std::fill(binPower_.begin(), binPower_.end(), T(0));
        std::fill(omega_.begin(), omega_.end(), T(0));
        std::fill(omegaImag_.begin(), omegaImag_.end(), T(0));
        std::fill(active_.begin(), active_.end(), 0u);
        std::fill(peak_.begin(), peak_.end(), 0u);
        std::fill(regionPeak_.begin(), regionPeak_.end(), 0u);

        olderBank_ = 0;
        latestBank_ = 1;
        currentBank_ = 2;
        completedFrames_ = 0;
        callbackChannel_ = 0;
        q_ = 0;
        frameQ_ = 0;
        frameRequestWord_ = 0u;
        captured_ = false;
        capturedMode_ = PhaseMode::Tonal;
        magnitudeFloor_ = T(0);
        nextDiffuseGeneration_ = 0u;
        diffuseGeneration_ = 0u;
        diffuseFrame_ = 0u;
        tonalAdvanceCount_ = 0u;
    }

    [[nodiscard]] std::size_t rawOffset(int bank, int channel) const noexcept
    {
        return (static_cast<std::size_t>(bank)
                    * static_cast<std::size_t>(spec_.numChannels)
                + static_cast<std::size_t>(channel))
            * static_cast<std::size_t>(numBins_ * 2);
    }

    [[nodiscard]] std::size_t channelBinOffset(int channel) const noexcept
    {
        return static_cast<std::size_t>(channel)
            * static_cast<std::size_t>(numBins_);
    }

    [[nodiscard]] static T principalArgument(T phase) noexcept
    {
        constexpr T p = std::numbers::pi_v<T>;
        constexpr T tau = T(2) * p;
        T wrapped = std::fmod(phase + p, tau);
        if (wrapped < T(0)) wrapped += tau;
        return wrapped - p;
    }

    [[nodiscard]] static T phaseOf(T real, T imag) noexcept
    {
        return std::atan2(imag, real);
    }

    [[nodiscard]] static T safeMagnitude(T real, T imag) noexcept
    {
        return std::hypot(real, imag);
    }

    void processSpectrum(T* spectrum, int bins) noexcept
    {
        if (callbackChannel_ == 0) beginFrame();

        const int channel = callbackChannel_;
        T* const raw = rawSpectra_.data() + rawOffset(currentBank_, channel);
        const int usableBins = std::min(bins, numBins_);
        for (int k = 0; k < usableBins; ++k)
        {
            T real = spectrum[2 * k];
            T imag = spectrum[2 * k + 1];
            if (!std::isfinite(real) || !std::isfinite(imag))
            {
                real = T(0);
                imag = T(0);
                spectrum[2 * k] = T(0);
                spectrum[2 * k + 1] = T(0);
            }
            raw[2 * k] = real;
            raw[2 * k + 1] = imag;
        }

        if (frameQ_ > 0 && captured_)
            blendFrozenSpectrum(spectrum, channel);

        ++callbackChannel_;
        if (callbackChannel_ == spec_.numChannels)
        {
            callbackChannel_ = 0;
            completeFrame();
        }
    }

    void beginFrame() noexcept
    {
        // This is the audio owner's single requested-control load for the
        // complete all-channel frame.
        frameRequestWord_ = requested_.load(std::memory_order_relaxed);

        if (q_ == 0)
        {
            if ((frameRequestWord_ & kFrozenBit) == 0u)
            {
                captured_ = false;
            }
            else if (!captured_ && completedFrames_ >= 2)
            {
                capturedMode_ = (frameRequestWord_ & kDiffuseBit) != 0u
                                    ? PhaseMode::Diffuse
                                    : PhaseMode::Tonal;
                captureLatestFrame();
                captured_ = true;

                if (capturedMode_ == PhaseMode::Tonal)
                    advanceTonalPhase();
                else
                    diffuseFrame_ = 0u;
            }
        }

        frameQ_ = q_;
        if (frameQ_ > 0 && captured_
            && capturedMode_ == PhaseMode::Diffuse)
            prepareDiffuseFrame();
    }

    void completeFrame() noexcept
    {
        const int recycled = olderBank_;
        olderBank_ = latestBank_;
        latestBank_ = currentBank_;
        currentBank_ = recycled;
        completedFrames_ = std::min(2, completedFrames_ + 1);

        if (!captured_) return;

        if ((frameRequestWord_ & kFrozenBit) != 0u)
        {
            if (q_ < fftSize_ / hopSize_) ++q_;
        }
        else if (q_ > 0)
        {
            --q_;
        }

        if (q_ == 0 && (frameRequestWord_ & kFrozenBit) == 0u)
        {
            captured_ = false;
            return;
        }

        if (capturedMode_ == PhaseMode::Tonal)
            advanceTonalPhase();
        else if (frameQ_ > 0)
            ++diffuseFrame_;
    }

    void captureLatestFrame() noexcept
    {
        std::fill(active_.begin(), active_.end(), 0u);
        std::fill(peak_.begin(), peak_.end(), 0u);
        std::fill(regionPeak_.begin(), regionPeak_.end(), 0u);
        std::fill(binPower_.begin(), binPower_.end(), T(0));
        std::fill(omega_.begin(), omega_.end(), T(0));

        long double globalScale = 0.0L;
        for (int ch = 0; ch < spec_.numChannels; ++ch)
        {
            const T* const latest = rawSpectra_.data() + rawOffset(latestBank_, ch);
            for (int k = 0; k < numBins_; ++k)
            {
                globalScale = std::max(globalScale,
                    std::abs(static_cast<long double>(latest[2 * k])));
                globalScale = std::max(globalScale,
                    std::abs(static_cast<long double>(latest[2 * k + 1])));
            }
        }

        long double maxPower = 0.0L;
        if (globalScale > 0.0L)
        {
            for (int k = 0; k < numBins_; ++k)
            {
                long double power = 0.0L;
                for (int ch = 0; ch < spec_.numChannels; ++ch)
                {
                    const T* const latest = rawSpectra_.data()
                        + rawOffset(latestBank_, ch);
                    const long double real =
                        static_cast<long double>(latest[2 * k]) / globalScale;
                    const long double imag =
                        static_cast<long double>(latest[2 * k + 1]) / globalScale;
                    power += real * real + imag * imag;
                }
                binPower_[static_cast<std::size_t>(k)] = static_cast<T>(power);
                maxPower = std::max(maxPower, power);
            }
        }

        const long double epsilon =
            static_cast<long double>(std::numeric_limits<T>::epsilon());
        const long double minimumNormal =
            static_cast<long double>(std::numeric_limits<T>::min());
        const long double minimumMagnitude = std::sqrt(minimumNormal);
        long double normalizedPowerFloor = 0.0L;
        if (globalScale > 0.0L)
        {
            normalizedPowerFloor = 64.0L * epsilon * maxPower;
            const long double relativeMagnitudeFloor = globalScale
                * std::sqrt(normalizedPowerFloor);
            magnitudeFloor_ = static_cast<T>(std::max(
                relativeMagnitudeFloor, minimumMagnitude));
        }
        else
        {
            magnitudeFloor_ = static_cast<T>(minimumMagnitude);
        }

        const long double normalizedMagnitudeFloor =
            std::sqrt(normalizedPowerFloor);
        for (int k = 1; k < numBins_ - 1; ++k)
        {
            const long double normalizedMagnitude = std::sqrt(
                static_cast<long double>(binPower_[static_cast<std::size_t>(k)]));
            const long double rawMagnitude = globalScale * normalizedMagnitude;
            active_[static_cast<std::size_t>(k)] =
                normalizedMagnitude > normalizedMagnitudeFloor
                    && rawMagnitude > minimumMagnitude
                    ? 1u
                    : 0u;
        }

        for (int ch = 0; ch < spec_.numChannels; ++ch)
        {
            const T* const latest = rawSpectra_.data() + rawOffset(latestBank_, ch);
            const std::size_t base = channelBinOffset(ch);
            capturedMagnitude_[base] = latest[0];
            capturedPhase_[base] = T(1);
            frozenPhase_[base] = T(0);
            const int nyquist = numBins_ - 1;
            capturedMagnitude_[base + static_cast<std::size_t>(nyquist)] =
                latest[2 * nyquist];
            capturedPhase_[base + static_cast<std::size_t>(nyquist)] = T(1);
            frozenPhase_[base + static_cast<std::size_t>(nyquist)] = T(0);

            for (int k = 1; k < nyquist; ++k)
            {
                const std::size_t index = base + static_cast<std::size_t>(k);
                const T magnitude = active_[static_cast<std::size_t>(k)] != 0u
                    ? safeMagnitude(latest[2 * k], latest[2 * k + 1])
                    : T(0);
                capturedMagnitude_[index] = magnitude;
                if (magnitude > T(0))
                {
                    capturedPhase_[index] = latest[2 * k] / magnitude;
                    frozenPhase_[index] = latest[2 * k + 1] / magnitude;
                }
                else
                {
                    capturedPhase_[index] = T(1);
                    frozenPhase_[index] = T(0);
                }
            }
        }

        if (capturedMode_ == PhaseMode::Tonal)
            buildTonalRegions();

        diffuseGeneration_ = nextDiffuseGeneration_++;
        diffuseFrame_ = 0u;
        tonalAdvanceCount_ = 0u;
    }

    void buildTonalRegions() noexcept
    {
        int peakCount = 0;
        for (int k = 1; k < numBins_ - 1; ++k)
        {
            const std::size_t index = static_cast<std::size_t>(k);
            if (active_[index] != 0u
                && binPower_[index] > binPower_[index - 1u]
                && binPower_[index] >= binPower_[index + 1u])
            {
                peak_[index] = 1u;
                ++peakCount;
            }
        }

        if (peakCount == 0)
        {
            for (int k = 1; k < numBins_ - 1; ++k)
            {
                const std::size_t index = static_cast<std::size_t>(k);
                if (active_[index] != 0u)
                {
                    peak_[index] = 1u;
                    regionPeak_[index] = static_cast<std::uint32_t>(k);
                }
            }
        }
        else
        {
            int currentPeak = -1;
            int nextPeak = -1;
            for (int k = 1; k < numBins_ - 1; ++k)
            {
                if (peak_[static_cast<std::size_t>(k)] != 0u)
                {
                    currentPeak = k;
                    break;
                }
            }
            for (int k = currentPeak + 1; k < numBins_ - 1; ++k)
            {
                if (peak_[static_cast<std::size_t>(k)] != 0u)
                {
                    nextPeak = k;
                    break;
                }
            }

            for (int k = 1; k < numBins_ - 1; ++k)
            {
                if (active_[static_cast<std::size_t>(k)] == 0u) continue;
                while (nextPeak >= 0 && k > (currentPeak + nextPeak) / 2)
                {
                    currentPeak = nextPeak;
                    nextPeak = -1;
                    for (int p = currentPeak + 1; p < numBins_ - 1; ++p)
                    {
                        if (peak_[static_cast<std::size_t>(p)] != 0u)
                        {
                            nextPeak = p;
                            break;
                        }
                    }
                }
                regionPeak_[static_cast<std::size_t>(k)] =
                    static_cast<std::uint32_t>(currentPeak);
            }
        }

        for (int p = 1; p < numBins_ - 1; ++p)
        {
            if (peak_[static_cast<std::size_t>(p)] == 0u) continue;

            long double binScale = 0.0L;
            for (int ch = 0; ch < spec_.numChannels; ++ch)
            {
                const T* const latest = rawSpectra_.data()
                    + rawOffset(latestBank_, ch);
                binScale = std::max(binScale,
                    std::abs(static_cast<long double>(latest[2 * p])));
                binScale = std::max(binScale,
                    std::abs(static_cast<long double>(latest[2 * p + 1])));
            }

            long double sumWeight = 0.0L;
            long double realResidual = 0.0L;
            long double imagResidual = 0.0L;
            const long double nominal = 2.0L * std::numbers::pi_v<long double>
                * static_cast<long double>(p) * static_cast<long double>(hopSize_)
                / static_cast<long double>(fftSize_);

            if (binScale > 0.0L)
            {
                for (int ch = 0; ch < spec_.numChannels; ++ch)
                {
                    const T* const latest = rawSpectra_.data()
                        + rawOffset(latestBank_, ch);
                    const T* const older = rawSpectra_.data()
                        + rawOffset(olderBank_, ch);
                    const long double real =
                        static_cast<long double>(latest[2 * p]) / binScale;
                    const long double imag =
                        static_cast<long double>(latest[2 * p + 1]) / binScale;
                    const long double weight = real * real + imag * imag;
                    const long double latestPhase = std::atan2(
                        static_cast<long double>(latest[2 * p + 1]),
                        static_cast<long double>(latest[2 * p]));
                    const long double olderPhase = std::atan2(
                        static_cast<long double>(older[2 * p + 1]),
                        static_cast<long double>(older[2 * p]));
                    const T residual = principalArgument(static_cast<T>(
                        latestPhase - olderPhase - nominal));
                    realResidual += weight
                        * std::cos(static_cast<long double>(residual));
                    imagResidual += weight
                        * std::sin(static_cast<long double>(residual));
                    sumWeight += weight;
                }
            }

            long double rho = 0.0L;
            const long double residualMagnitude =
                std::hypot(realResidual, imagResidual);
            if (residualMagnitude > 64.0L
                    * static_cast<long double>(std::numeric_limits<T>::epsilon())
                    * sumWeight)
                rho = std::atan2(imagResidual, realResidual);

            const T advance = principalArgument(static_cast<T>(nominal + rho));
            omega_[static_cast<std::size_t>(p)] = std::cos(advance);
            omegaImag_[static_cast<std::size_t>(p)] = std::sin(advance);
        }

        // The two phase banks are a compact unit-complex representation. Peak
        // entries retain their captured absolute phase; non-peaks retain the
        // captured phase relative to their region peak. This lets the held
        // audio path use multiplies instead of transcendental functions.
        for (int ch = 0; ch < spec_.numChannels; ++ch)
        {
            const std::size_t base = channelBinOffset(ch);
            for (int k = 1; k < numBins_ - 1; ++k)
            {
                if (active_[static_cast<std::size_t>(k)] == 0u
                    || peak_[static_cast<std::size_t>(k)] != 0u)
                    continue;
                const std::size_t index = base + static_cast<std::size_t>(k);
                const std::size_t peakIndex = base + regionPeak_[
                    static_cast<std::size_t>(k)];
                const T real = capturedPhase_[index];
                const T imag = frozenPhase_[index];
                const T peakReal = capturedPhase_[peakIndex];
                const T peakImag = frozenPhase_[peakIndex];
                capturedPhase_[index] = real * peakReal + imag * peakImag;
                frozenPhase_[index] = imag * peakReal - real * peakImag;
            }
        }
    }

    void advanceTonalPhase() noexcept
    {
        ++tonalAdvanceCount_;
        const bool renormalize = (tonalAdvanceCount_ & 255u) == 0u;
        for (int ch = 0; ch < spec_.numChannels; ++ch)
        {
            const std::size_t base = channelBinOffset(ch);
            for (int p = 1; p < numBins_ - 1; ++p)
            {
                if (peak_[static_cast<std::size_t>(p)] == 0u) continue;
                const std::size_t index = base + static_cast<std::size_t>(p);
                const T real = capturedPhase_[index];
                const T imag = frozenPhase_[index];
                const T advanceReal = omega_[static_cast<std::size_t>(p)];
                const T advanceImag = omegaImag_[static_cast<std::size_t>(p)];
                T nextReal = real * advanceReal - imag * advanceImag;
                T nextImag = real * advanceImag + imag * advanceReal;
                if (renormalize)
                {
                    const T length = std::hypot(nextReal, nextImag);
                    if (length > T(0) && std::isfinite(length))
                    {
                        nextReal /= length;
                        nextImag /= length;
                    }
                    else
                    {
                        nextReal = T(1);
                        nextImag = T(0);
                    }
                }
                capturedPhase_[index] = nextReal;
                frozenPhase_[index] = nextImag;
            }
        }
    }

    [[nodiscard]] static T diffuseAngle(std::uint64_t generation,
                                        std::uint64_t frame,
                                        std::uint64_t bin) noexcept
    {
        std::uint64_t x = UINT64_C(0xD1B54A32D192ED03)
            ^ (generation * UINT64_C(0x9E3779B97F4A7C15))
            ^ (frame * UINT64_C(0xBF58476D1CE4E5B9))
            ^ (bin * UINT64_C(0x94D049BB133111EB));
        x += UINT64_C(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30u)) * UINT64_C(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27u)) * UINT64_C(0x94D049BB133111EB);
        x ^= x >> 31u;
        const double uniform = static_cast<double>(x >> 11u) * 0x1.0p-53;
        return static_cast<T>(2.0 * std::numbers::pi_v<double> * uniform);
    }

    void prepareDiffuseFrame() noexcept
    {
        for (int k = 1; k < numBins_ - 1; ++k)
        {
            const T angle = diffuseAngle(diffuseGeneration_, diffuseFrame_,
                                         static_cast<std::uint64_t>(k));
            omega_[static_cast<std::size_t>(k)] = std::cos(angle);
            omegaImag_[static_cast<std::size_t>(k)] = std::sin(angle);
        }
    }

    void targetPhasor(int channel, int bin, T& real, T& imag) const noexcept
    {
        const std::size_t base = channelBinOffset(channel);
        const std::size_t index = base + static_cast<std::size_t>(bin);
        if (capturedMode_ == PhaseMode::Diffuse)
        {
            const T capturedReal = capturedPhase_[index];
            const T capturedImag = frozenPhase_[index];
            const T randomReal = omega_[static_cast<std::size_t>(bin)];
            const T randomImag = omegaImag_[static_cast<std::size_t>(bin)];
            real = capturedReal * randomReal - capturedImag * randomImag;
            imag = capturedReal * randomImag + capturedImag * randomReal;
            return;
        }

        if (peak_[static_cast<std::size_t>(bin)] != 0u)
        {
            real = capturedPhase_[index];
            imag = frozenPhase_[index];
            return;
        }

        const std::size_t peakIndex = base
            + regionPeak_[static_cast<std::size_t>(bin)];
        const T peakReal = capturedPhase_[peakIndex];
        const T peakImag = frozenPhase_[peakIndex];
        const T relativeReal = capturedPhase_[index];
        const T relativeImag = frozenPhase_[index];
        real = peakReal * relativeReal - peakImag * relativeImag;
        imag = peakReal * relativeImag + peakImag * relativeReal;
    }

    void blendFrozenSpectrum(T* spectrum, int channel) noexcept
    {
        const int transitionSteps = fftSize_ / hopSize_;
        const std::size_t base = channelBinOffset(channel);
        const int nyquist = numBins_ - 1;

        if (frameQ_ == transitionSteps)
        {
            spectrum[0] = capturedMagnitude_[base];
            spectrum[1] = T(0);
            spectrum[2 * nyquist] =
                capturedMagnitude_[base + static_cast<std::size_t>(nyquist)];
            spectrum[2 * nyquist + 1] = T(0);
            for (int k = 1; k < nyquist; ++k)
            {
                const std::size_t index = base + static_cast<std::size_t>(k);
                const T liveMagnitude = safeMagnitude(spectrum[2 * k],
                                                       spectrum[2 * k + 1]);
                const T frozenMagnitude = capturedMagnitude_[index];
                const bool liveActive = liveMagnitude >= magnitudeFloor_;
                const bool frozenActive = frozenMagnitude >= magnitudeFloor_;
                if (!liveActive && !frozenActive)
                {
                    spectrum[2 * k] = T(0);
                    spectrum[2 * k + 1] = T(0);
                    continue;
                }
                if (!frozenActive)
                {
                    const T scale = frozenMagnitude / liveMagnitude;
                    spectrum[2 * k] *= scale;
                    spectrum[2 * k + 1] *= scale;
                    continue;
                }
                T targetReal = T(0);
                T targetImag = T(0);
                targetPhasor(channel, k, targetReal, targetImag);
                spectrum[2 * k] = frozenMagnitude * targetReal;
                spectrum[2 * k + 1] = frozenMagnitude * targetImag;
            }
            return;
        }

        const T phase = std::numbers::pi_v<T> * static_cast<T>(frameQ_)
            / static_cast<T>(transitionSteps);
        const T alpha = T(0.5) - T(0.5) * std::cos(phase);
        const T oneMinusAlpha = T(1) - alpha;
        spectrum[0] = oneMinusAlpha * spectrum[0]
            + alpha * capturedMagnitude_[base];
        spectrum[1] = T(0);
        spectrum[2 * nyquist] = oneMinusAlpha * spectrum[2 * nyquist]
            + alpha * capturedMagnitude_[base + static_cast<std::size_t>(nyquist)];
        spectrum[2 * nyquist + 1] = T(0);

        for (int k = 1; k < nyquist; ++k)
        {
            const std::size_t index = base + static_cast<std::size_t>(k);
            const T liveMagnitude = safeMagnitude(spectrum[2 * k],
                                                  spectrum[2 * k + 1]);
            const T frozenMagnitude = capturedMagnitude_[index];
            const bool liveActive = liveMagnitude >= magnitudeFloor_;
            const bool frozenActive = frozenMagnitude >= magnitudeFloor_;
            if (!liveActive && !frozenActive)
            {
                spectrum[2 * k] = T(0);
                spectrum[2 * k + 1] = T(0);
                continue;
            }

            T targetReal = T(0);
            T targetImag = T(0);
            targetPhasor(channel, k, targetReal, targetImag);
            T livePhase = phaseOf(spectrum[2 * k], spectrum[2 * k + 1]);
            T targetPhase = phaseOf(targetReal, targetImag);
            if (!liveActive) livePhase = targetPhase;
            if (!frozenActive) targetPhase = livePhase;

            const T magnitude = oneMinusAlpha * liveMagnitude
                + alpha * frozenMagnitude;
            const T outputPhase = principalArgument(livePhase
                + alpha * principalArgument(targetPhase - livePhase));
            spectrum[2 * k] = magnitude * std::cos(outputPhase);
            spectrum[2 * k + 1] = magnitude * std::sin(outputPhase);
            if (!std::isfinite(spectrum[2 * k])
                || !std::isfinite(spectrum[2 * k + 1]))
            {
                spectrum[2 * k] = T(0);
                spectrum[2 * k + 1] = T(0);
            }
        }
    }

    AudioSpec spec_ {};
    bool prepared_ = false;
    int fftSize_ = 0;
    int hopSize_ = 0;
    int numBins_ = 0;

    SpectralProcessor<T> processor_;
    std::vector<T> scratch_;
    std::array<T*, kMaxChannels> scratchChannels_ {};

    // Three raw banks are current, latest completed and the completed frame
    // before it. Integer bank rotation avoids frame-sized copies.
    std::vector<T> rawSpectra_;
    int olderBank_ = 0;
    int latestBank_ = 1;
    int currentBank_ = 2;
    int completedFrames_ = 0;
    int callbackChannel_ = 0;

    std::vector<T> capturedMagnitude_;
    std::vector<T> capturedPhase_;
    std::vector<T> frozenPhase_;
    std::vector<T> binPower_;
    std::vector<T> omega_;
    std::vector<T> omegaImag_;
    std::vector<std::uint32_t> active_;
    std::vector<std::uint32_t> peak_;
    std::vector<std::uint32_t> regionPeak_;

    int q_ = 0;
    int frameQ_ = 0;
    std::uint32_t frameRequestWord_ = 0u;
    bool captured_ = false;
    PhaseMode capturedMode_ = PhaseMode::Tonal;
    T magnitudeFloor_ = T(0);
    std::uint64_t nextDiffuseGeneration_ = 0u;
    std::uint64_t diffuseGeneration_ = 0u;
    std::uint64_t diffuseFrame_ = 0u;
    std::uint32_t tonalAdvanceCount_ = 0u;

    std::atomic<std::uint32_t> requested_ { 0u };
};

} // namespace dspark
