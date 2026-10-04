// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflineTransientAnalyzer.h
 * @brief Complete-source, stereo-linked attack and pulse analysis.
 *
 * OFFLINE ONLY. Reuses the energy analyzer, FFT and onset feature/picker
 * kernels. Two complete bounded source scans produce separate SuperFlux attack
 * and spectral pulse maps. Spectra pool channel power, not L+R. A 1 ms power
 * map refines timing, recovers isolated quiet restarts, and follows each event
 * into its local body. Upper-band evidence recovers attacks masked by decaying
 * bass without interpreting the high-pass filter's bass leakage as new onsets.
 * Zero padding
 * covers both source boundaries; returned event positions are source-relative
 * int64 frames.
 *
 * Storage is O(duration / 1 ms), never an owned copy of the source PCM. Only
 * sparse events and the 100 ms energy map remain by default. Optional retained
 * spectral features can feed a subsequent tempo analysis without recomputing
 * FFTs.
 *
 * Threading: analyze() is a synchronous worker call, never an audio callback.
 * Analysis owns immutable, move-only results; borrowed spans require its
 * lifetime. Dependencies: OfflineEnergyAnalyzer.h, detail/OnsetFeatures.h and
 * Core/Biquad.h.
 */

#include "OfflineEnergyAnalyzer.h"

#if DSPARK_HAS_OFFLINE

#include "detail/OnsetFeatures.h"
#include "../Core/Biquad.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace dspark
{

/** @brief Source-bound attack and pulse maps for offline Dynamics. */
template <FloatType T> class OfflineTransientAnalyzer final
{
  public:
    static constexpr std::uint32_t algorithmRevision = 4;
    struct Options
    {
        bool attacks = true;
        bool pulses = true;
        bool retainFeatures = false;
    };

    /** @brief A detected event, including its locally measured attack/body
     * interval. */
    struct Event
    {
        std::int64_t begin = 0;
        std::int64_t peak = 0;
        std::int64_t end = 0;     ///< Exclusive; begin <= peak < end <= source frames.
        double strength = 0;      ///< Normalized novelty, or normalized RMS for energy restarts.
        double contrast = 0;      ///< Local power-rise contrast, not a probability.
        double peakAmplitude = 0; ///< Linked pooled-channel peak in the event.
        double sustainRms = 0;    ///< Local body/noise reference after the event.
        bool endLimited = false;  ///< Body transition not resolved before the search boundary.
        bool overlapsNext = false;
        bool energyRestart = false; ///< Source-energy restart instead of a spectral candidate.
        bool bandRefined = false; ///< Attack/body localized above 800 Hz under a lower-frequency bed.
    };

    /** @brief Distinct novelty functions and SuperFlux registers from one
     * spectrum. */
    struct FeatureFrame
    {
        T attack = T(0);
        T pulse = T(0);
        std::array<T, 4> registers{};
        double windowPower = 0;    ///< Normalized, Hann-weighted mean power by Parseval's identity.
        double lowEnergyShare = 0; ///< Fraction of that power below 800 Hz.
    };

    /**
     * @brief Read-only analysis with source provenance; no owned audio samples.
     * Threading: energy() is an owner-thread reference view, not an atomic
     * publication. Const readers may share a completed analysis while it remains
     * alive and unmoved; no reader may overlap a move or other mutable access.
     */
    class Analysis final
    {
      public:
        Analysis() = default;
        Analysis(Analysis &&) noexcept = default;
        Analysis &operator=(Analysis &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept
        {
            return energy_.isValid();
        }
        /**
         * @brief Borrows the energy map as an owner-thread reference view.
         * Valid while this analysis remains alive and unmoved. No concurrent
         * mutable access is permitted; returning a reference does not publish it.
         */
        [[nodiscard]] const typename OfflineEnergyAnalyzer<T>::Analysis &energy() const noexcept
        {
            return energy_;
        }
        [[nodiscard]] std::span<const Event> attacks() const noexcept
        {
            return isValid() ? std::span<const Event>(attacks_.get(), attackCount_)
                             : std::span<const Event>();
        }
        [[nodiscard]] std::span<const Event> pulses() const noexcept
        {
            return isValid() ? std::span<const Event>(pulses_.get(), pulseCount_)
                             : std::span<const Event>();
        }
        [[nodiscard]] std::span<const FeatureFrame> features() const noexcept
        {
            return isValid() && features_
                       ? std::span<const FeatureFrame>(features_.get(), featureCount_)
                       : std::span<const FeatureFrame>();
        }
        [[nodiscard]] int frameSize() const noexcept
        {
            return frameSize_;
        }
        [[nodiscard]] int hopFrames() const noexcept
        {
            return hop_;
        }
        /** @brief Spectral window center, before event localization/refinement. */
        [[nodiscard]] std::int64_t featureCenter(std::size_t index) const noexcept
        {
            return index < featureCount_
                       ? static_cast<std::int64_t>(index + 1) * hop_ - frameSize_ / 2
                       : 0;
        }
        [[nodiscard]] Options getOptions() const noexcept
        {
            return options_;
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid()
                       ? energy_.retainedBytes() + (attackCount_ + pulseCount_) * sizeof(Event) +
                             (features_ ? featureCount_ * sizeof(FeatureFrame) : 0)
                       : 0;
        }

      private:
        friend class OfflineTransientAnalyzer;
        typename OfflineEnergyAnalyzer<T>::Analysis energy_;
        Options options_;
        int frameSize_ = 0, hop_ = 0;
        std::size_t attackCount_ = 0, pulseCount_ = 0, featureCount_ = 0;
        std::unique_ptr<Event[]> attacks_, pulses_;
        std::unique_ptr<FeatureFrame[]> features_;
    };

    struct Result
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Analysis analysis;
        std::size_t memoryBytes =
            0; ///< Charged payload, including conservative FFT setup allowance.
        [[nodiscard]] bool succeeded() const noexcept
        {
            return offlineSucceeded(status);
        }
    };

    /**
     * @brief Analyzes immutable mono/stereo audio at 8-384 kHz on the calling
     * worker.
     *
     * A full-source peak normalizes analysis PCM before the FFT. This avoids
     * overflow for extreme finite input and makes source gain a neutral operation
     * apart from floating-point rounding. Processing audio is never normalized.
     * Failure publishes no partial/stale analysis. The source must remain
     * unchanged across both scans; fingerprints check content as well as source
     * metadata.
     */
    [[nodiscard]] Result analyze(OfflineAudioSource<T> &source, Options options = {},
                                 const OfflineJobOptions &jobOptions = {}) const
    {
        Result result;
        try
        {
            detail::OfflineSession job(jobOptions);
            Analysis candidate;
            candidate.options_ = options;
            const auto spec = source.getSpec();
            detail::offlineValidateSpec(spec, 2);
            if ((spec.sampleRate < 8000) || (spec.sampleRate > 384000))
                detail::offlineFail(OfflineStatus::InvalidInput);
            job.checkpoint(OfflinePhase::Analyze, 0, spec.frames);
            const bool spectrumNeeded = options.attacks || options.pulses || options.retainFeatures;
            candidate.frameSize_ = detail::OnsetSpectrum<T>::defaultFrameSize(spec.sampleRate);
            candidate.hop_ = std::max(1, static_cast<int>(std::lround(spec.sampleRate / 200)));
            const int microHop = std::max(1, static_cast<int>(std::lround(spec.sampleRate / 1000)));
            const auto microCount = static_cast<std::uint64_t>((spec.frames - 1) / microHop + 1);
            if (spec.frames > std::numeric_limits<std::int64_t>::max() - candidate.frameSize_)
                detail::offlineFail(OfflineStatus::MemoryLimit);
            const auto featureCount =
                static_cast<std::uint64_t>((spec.frames + candidate.frameSize_) / candidate.hop_);
            if (spectrumNeeded)
            {
                // Charge all length-dependent feature work before scanning any PCM.
                job.charge(detail::offlineBytes(microCount, sizeof(MicroBin)));
                job.charge(detail::offlineBytes(featureCount, sizeof(FeatureFrame)));
                if (options.attacks)
                {
                    job.charge(detail::offlineBytes(microCount, sizeof(double)));
                    job.charge(detail::offlineBytes(featureCount, sizeof(bool)));
                    job.charge(detail::offlineBytes(
                        static_cast<std::uint64_t>(candidate.frameSize_ / 2 + 1), sizeof(double)));
                }
                job.charge(detail::offlineBytes(static_cast<std::uint64_t>(candidate.frameSize_) *
                                                    2 * spec.channels,
                                                sizeof(T)));
                job.charge(detail::OnsetSpectrum<T>::preparationBytes(candidate.frameSize_));
                job.charge(detail::OnsetNovelty<T>::preparationBytes(candidate.frameSize_));
            }
            auto energy = OfflineEnergyAnalyzer<T>().analyze(
                source, detail::offlineRemaining(jobOptions, job.bytes()));
            if (!energy.succeeded())
                detail::offlineFail(energy.status);
            job.charge(energy.memoryBytes);
            if (energy.analysis.getSpec() != spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            candidate.energy_ = std::move(energy.analysis);
            if (spectrumNeeded && candidate.energy_.samplePeak() > 0)
            {
                auto micro = std::make_unique<MicroBin[]>(static_cast<std::size_t>(microCount));
                auto features =
                    std::make_unique<FeatureFrame[]>(static_cast<std::size_t>(featureCount));
                auto highPower = options.attacks
                                     ? std::make_unique<double[]>(static_cast<std::size_t>(microCount))
                                     : nullptr;
                auto highEvidence = options.attacks
                                        ? std::make_unique<bool[]>(static_cast<std::size_t>(featureCount))
                                        : nullptr;
                candidate.featureCount_ = static_cast<std::size_t>(featureCount);
                scanFeatures(source, spec, candidate, microHop, micro.get(), highPower.get(),
                             highEvidence.get(), features.get(), job, jobOptions.blockFrames);
                const auto microView =
                    std::span<const MicroBin>(micro.get(), static_cast<std::size_t>(microCount));
                const auto featureView =
                    std::span<const FeatureFrame>(features.get(), candidate.featureCount_);
                const auto highView = std::span<const double>(
                    highPower.get(), highPower ? static_cast<std::size_t>(microCount) : 0);
                const auto evidenceView = std::span<const bool>(
                    highEvidence.get(), highEvidence ? candidate.featureCount_ : 0);
                if (options.attacks)
                    buildEvents(candidate, featureView, microView, highView, evidenceView,
                                microHop, false, job);
                if (options.pulses)
                    buildEvents(candidate, featureView, microView, {}, {}, microHop, true, job);
                if (options.retainFeatures)
                    candidate.features_ = std::move(features);
            }
            if (source.getSpec() != spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            job.checkpoint(OfflinePhase::Analyze, spec.frames, spec.frames);
            result.memoryBytes = job.bytes();
            result.analysis = std::move(candidate);
            result.status = OfflineStatus::Success;
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

  private:
    struct MicroBin
    {
        double power = 0;
        double peakPower = 0;
        std::int64_t peakFrame = 0;
    };

    // The same refinement and body tracker serve full-band and high-band power.
    // Peak samples always come from the original linked source, not the filter.
    struct PowerView
    {
        std::span<const MicroBin> source;
        std::span<const double> high;
        PowerView(std::span<const MicroBin> full, std::span<const double> band = {}) noexcept
            : source(full), high(band)
        {
        }
        [[nodiscard]] std::size_t size() const noexcept { return source.size(); }
        [[nodiscard]] double operator[](std::size_t i) const noexcept
        {
            return high.empty() ? source[i].power : high[i];
        }
    };

    static void scanFeatures(OfflineAudioSource<T> &source, const OfflineAudioSpec &spec,
                             const Analysis &analysis, int microHop, MicroBin *micro,
                             double *highPower, bool *highEvidence,
                             FeatureFrame *features, detail::OfflineSession &job, int blockFrames)
    {
        const int size = analysis.frameSize_, hop = analysis.hop_;
        auto ring = std::make_unique<T[]>(static_cast<std::size_t>(2 * size * spec.channels));
        detail::OnsetSpectrum<T> spectrum;
        detail::OnsetNovelty<T> novelty;
        spectrum.prepare(size);
        novelty.prepare(spec.sampleRate, size, true);
        detail::OfflineBlock<T> scratch(job, spec, blockFrames);
        Biquad<double> highPass;
        std::unique_ptr<double[]> highGains;
        if (highPower)
        {
            const auto coefficients = BiquadCoeffs::makeHighPass(spec.sampleRate, 800);
            highPass.setCoeffsNow(coefficients);
            highGains = std::make_unique<double[]>(static_cast<std::size_t>(size / 2 + 1));
            for (int k = 0; k <= size / 2; ++k)
            {
                const double gain = coefficients.getMagnitude(
                    static_cast<double>(k) * spec.sampleRate / size, spec.sampleRate);
                highGains[static_cast<std::size_t>(k)] = gain * gain;
            }
        }
        OfflineFingerprint fingerprint;
        int write = 0, inHop = 0;
        std::size_t frame = 0;
        auto push = [&](const std::array<T, 2> &values) {
            std::array<const T *, 2> pointers{};
            for (int c = 0; c < spec.channels; ++c)
            {
                T *channel = ring.get() + static_cast<std::size_t>(c * 2 * size);
                channel[write] = channel[write + size] = values[static_cast<std::size_t>(c)];
            }
            if (++write == size)
                write = 0;
            if (++inHop < hop)
                return;
            inHop = 0;
            for (int c = 0; c < spec.channels; ++c)
                pointers[static_cast<std::size_t>(c)] = ring.get() + c * 2 * size + write;
            if (!spectrum.computePooled({pointers.data(), spec.channels, size}))
                detail::offlineFail(OfflineStatus::NumericalFailure);
            const auto magnitudes = spectrum.magnitudes();
            double windowPower = 0;
            double lowPower = 0;
            double filteredLow = 0, filteredHigh = 0;
            for (std::size_t k = 0; k < magnitudes.size(); ++k)
            {
                const double magnitude = static_cast<double>(magnitudes[k]) / size;
                const double power =
                    magnitude * magnitude * (k == 0 || k + 1 == magnitudes.size() ? 1 : 2);
                windowPower += power;
                if (static_cast<double>(k) * spec.sampleRate / size < 800)
                {
                    lowPower += power;
                    if (highGains)
                        filteredLow += power * highGains[k];
                }
                else if (highGains)
                    filteredHigh += power * highGains[k];
            }
            const auto value = novelty.process(spectrum.magnitudes(), {},
                                               detail::OnsetNovelty<T>::Method::BothFlux);
            if (!value.valid || !std::isfinite(value.value) || !std::isfinite(value.spectralFlux))
                detail::offlineFail(OfflineStatus::NumericalFailure);
            // A finite-order high-pass still passes some bass. Require a majority
            // of its predicted spectral power above the split before using the
            // band for timing; residual bass cycles alone are not new attacks.
            if (highEvidence)
                highEvidence[frame] = filteredHigh > 0 && filteredHigh >= filteredLow;
            features[frame++] = {value.value, value.spectralFlux, value.registers, windowPower,
                                 windowPower > 0 ? lowPower / windowPower : 0};
        };
        const double normalization = analysis.energy_.samplePeak();
        for (std::int64_t first = 0; first < spec.frames;)
        {
            const int count =
                static_cast<int>(std::min<std::int64_t>(scratch.frames(), spec.frames - first));
            auto block = scratch.view(count);
            detail::offlineRead(source, spec, first, block);
            for (int f = 0; f < count; ++f)
            {
                std::array<T, 2> values{};
                double power = 0;
                double high = 0;
                for (int c = 0; c < spec.channels; ++c)
                {
                    const T raw = block.getChannel(c)[f];
                    if (!std::isfinite(raw))
                        detail::offlineFail(OfflineStatus::NonFiniteInput);
                    detail::offlineHash(fingerprint, raw);
                    const double value = static_cast<double>(raw) / normalization;
                    if (!std::isfinite(value) || std::abs(value) > 1)
                        detail::offlineFail(OfflineStatus::SourceMismatch);
                    values[static_cast<std::size_t>(c)] = static_cast<T>(value);
                    power += value * value;
                    if (highPower)
                    {
                        const double filtered = highPass.processSample(value, c);
                        high += filtered * filtered;
                    }
                }
                power /= spec.channels;
                const auto position = first + f;
                auto &bin = micro[static_cast<std::size_t>(position / microHop)];
                bin.power += power / microHop;
                if (highPower)
                    highPower[static_cast<std::size_t>(position / microHop)] +=
                        high / spec.channels / microHop;
                if (power > bin.peakPower)
                {
                    bin.peakPower = power;
                    bin.peakFrame = position;
                }
                push(values);
            }
            first += count;
            job.checkpoint(OfflinePhase::Analyze, first, spec.frames);
        }
        if (fingerprint != analysis.energy_.fingerprint())
            detail::offlineFail(OfflineStatus::SourceMismatch);
        for (int f = 0; f < size; ++f)
            push({});
        if (frame != analysis.featureCount_)
            detail::offlineFail(OfflineStatus::NumericalFailure);
    }

    [[nodiscard]] static double meanPower(PowerView micro, std::int64_t begin,
                                          std::int64_t end) noexcept
    {
        double sum = 0;
        for (auto i = std::max<std::int64_t>(0, begin);
             i < std::min<std::int64_t>(static_cast<std::int64_t>(micro.size()), end); ++i)
            sum += micro[static_cast<std::size_t>(i)];
        const double mean = end > begin ? sum / static_cast<double>(end - begin) : 0;
        return micro.high.empty() ? std::min(1.0, mean) : mean;
    }

    [[nodiscard]] static bool refine(Event &event, std::int64_t reference,
                                     PowerView micro, int microHop,
                                     const OfflineAudioSpec &spec, bool bass) noexcept
    {
        const auto radius =
            static_cast<std::int64_t>(std::lround((bass ? 0.045 : 0.025) * spec.sampleRate));
        if (reference < -radius || reference >= spec.frames + radius)
            return false;
        const auto from = std::max<std::int64_t>(0, reference - radius) / microHop;
        const auto to = std::min<std::int64_t>(spec.frames - 1, reference + radius) / microHop;
        double best = 0, before = 0, after = 0;
        std::int64_t selected = from;
        for (auto i = from; i <= to; ++i)
        {
            const double left = meanPower(micro, i - 3, i);
            const double right = meanPower(micro, i, i + 3);
            if (right - left > best)
            {
                best = right - left;
                before = left;
                after = right;
                selected = i;
            }
        }
        if (!(best > 1e-12) || best < 0.12 * after)
            return false;
        if (bass)
        {
            // A bass waveform's second half-cycle can have a steeper slope than
            // its actual attack. Use the low-register evidence already computed
            // from the shared spectrum to locate the first sustained rise above the
            // preceding body, rather than following that later half-cycle. Longer
            // context averages the bass cycle; broadband attacks retain fine
            // localization.
            const double baseline = maximumPower(micro, from - 40, from);
            double maximum = 0;
            for (auto i = from; i <= to; ++i)
                maximum = std::max(maximum, meanPower(micro, i, i + 3));
            const double crossing = baseline + 0.08 * std::max(0.0, maximum - baseline);
            bool found = false;
            for (auto i = from; i <= to; ++i)
            {
                const double broadLeft = maximumPower(micro, i - 42, i - 2);
                const double broadRight = maximumPower(micro, i, i + 40);
                const double left = meanPower(micro, i - 3, i);
                const double right = meanPower(micro, i, i + 3);
                if (right > crossing && right > left && broadRight - broadLeft >= 0.4 * broadRight)
                {
                    before = left;
                    after = right;
                    best = right - left;
                    selected = i;
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        if (!bass)
        {
            // A rising event can have its largest slope near its peak. Recover
            // its actual restart when a preceding quiet interval is observable;
            // do not move through sustained energy or a preceding event.
            const double quiet = after * 1e-6;
            const auto earliest = std::max<std::int64_t>(2, selected - 40);
            for (auto i = selected - 1; i >= earliest; --i)
                if (micro[static_cast<std::size_t>(i)] <= quiet &&
                    micro[static_cast<std::size_t>(i - 1)] <= quiet &&
                    micro[static_cast<std::size_t>(i - 2)] <= quiet)
                {
                    auto onset = i + 1;
                    while (onset < selected && micro[static_cast<std::size_t>(onset)] <= quiet)
                        ++onset;
                    if (onset < selected)
                    {
                        selected = onset;
                        before = meanPower(micro, onset - 3, onset);
                        after = meanPower(micro, onset, onset + 3);
                        best = after - before;
                    }
                    break;
                }
        }
        event.contrast = best / std::max(1e-30, before + after);
        const double floor = before + 0.05 * best;
        auto begin = selected;
        while (begin > 0 && begin > selected - 5 &&
               micro[static_cast<std::size_t>(begin - 1)] > floor)
            --begin;
        while (begin < to && micro[static_cast<std::size_t>(begin)] <= floor)
            ++begin;
        // A waveform half-cycle can rise sharply without a new attack. Check
        // the enclosing energy step as well as the 3 ms localization gradient.
        // This also rejects the spectral spread caused by cutting a sustained
        // tone at the end of a file; zero padding must not invent an attack.
        const double contextBefore = meanPower(micro, begin - 12, begin - 2);
        const auto availableEnd = static_cast<std::int64_t>(micro.size());
        const double contextAfter = meanPower(micro, begin, std::min(begin + 10, availableEnd));
        if (contextAfter - contextBefore < 0.08 * contextAfter)
            return false;
        const double broadBefore = meanPower(micro, begin - 22, begin - 2);
        const double broadAfter = meanPower(micro, begin, std::min(begin + 20, availableEnd));
        if (broadAfter - broadBefore < 0.08 * broadAfter)
            return false;
        const auto &bin = micro.source[static_cast<std::size_t>(begin)];
        event.begin = begin * microHop;
        // Preserve the exact position of an isolated single-frame impulse.
        if (micro.high.empty() && bin.peakPower > 0.95 * bin.power * microHop)
            event.begin = bin.peakFrame;
        event.peak = event.begin;
        event.end = event.begin + 1;
        return true;
    }

    [[nodiscard]] static double maximumPower(PowerView micro, std::int64_t begin,
                                             std::int64_t end) noexcept
    {
        double maximum = 0;
        for (auto i = begin; i < end; ++i)
            maximum = std::max(maximum, meanPower(micro, i, i + 3));
        return maximum;
    }

    static void followBody(Event &event, const Event *next, std::span<const MicroBin> micro,
                           std::span<const double> highPower,
                           int microHop, const OfflineAudioSpec &spec, double scale) noexcept
    {
        const auto begin = event.begin / microHop;
        const auto maximum =
            std::max<std::int64_t>(1, static_cast<std::int64_t>(0.25 * spec.sampleRate));
        const auto limitFrame =
            std::min(spec.frames, event.begin + std::min(maximum, spec.frames - event.begin));
        const auto horizon = (next ? std::min(limitFrame, next->begin) : limitFrame);
        const auto stop = std::max(begin + 1, (horizon - 1) / microHop + 1);
        const auto tailBegin = std::max(begin, stop - 25);
        const PowerView body(micro, event.bandRefined ? highPower : std::span<const double>{});
        const double sustain = meanPower(body, tailBegin, stop);
        double crest = 0;
        auto crestBin = begin;
        for (auto i = begin; i < std::min(stop, begin + 40); ++i)
        {
            const double power = meanPower(body, i - 1, i + 2);
            if (power > crest)
            {
                crest = power;
                crestBin = i;
            }
        }
        auto end = stop;
        if (crest > 1.25 * sustain)
        {
            const double floor = sustain + 0.1 * (crest - sustain);
            for (auto i = crestBin + 1; i < stop; ++i)
                if (meanPower(body, i - 1, i + 2) <= floor && meanPower(body, i, i + 3) <= floor)
                {
                    end = i;
                    break;
                }
        }
        else
            end = std::min(stop, crestBin + 2);
        event.endLimited = end == stop && horizon < spec.frames;
        event.overlapsNext = event.endLimited && next && horizon == next->begin;
        event.end = std::min(horizon, std::max(event.begin + 1, end * microHop));
        event.sustainRms = std::sqrt(meanPower(micro, tailBegin, stop)) * scale;
        double peak = 0;
        for (auto i = begin; i < stop && i * microHop < event.end; ++i)
        {
            const auto &bin = micro[static_cast<std::size_t>(i)];
            if (bin.peakPower > peak && bin.peakFrame >= event.begin && bin.peakFrame < event.end)
            {
                peak = bin.peakPower;
                event.peak = bin.peakFrame;
            }
        }
        event.peakAmplitude = std::sqrt(peak) * scale;
    }

    static void buildEvents(Analysis &analysis, std::span<const FeatureFrame> features,
                            std::span<const MicroBin> micro, std::span<const double> highPower,
                            std::span<const bool> highEvidence, int microHop, bool pulse,
                            detail::OfflineSession &job)
    {
        const auto spec = analysis.energy_.getSpec();
        T maximum = T(0);
        for (const auto &f : features)
            maximum = std::max(maximum, pulse ? f.pulse : f.attack);
        if (!(maximum > T(0)))
            return;
        const auto value = [&](std::size_t i) {
            return (pulse ? features[i].pulse : features[i].attack) / maximum;
        };
        const auto average =
            static_cast<std::size_t>(std::lround(0.1 * spec.sampleRate / analysis.hop_));
        const auto gap = static_cast<std::size_t>(
            std::lround((pulse ? 0.05 : 0.03) * spec.sampleRate / analysis.hop_));
        const auto gapFrames =
            static_cast<std::int64_t>(std::llround((pulse ? 0.05 : 0.03) * spec.sampleRate));
        const typename detail::OnsetPeakPicker<T>::Windows windows{
            pulse ? 1u : 6u, pulse ? 1u : 6u, average, average, gap > 0 ? gap - 1 : 0};
        const auto offset = static_cast<std::int64_t>(std::lround(0.34 * analysis.frameSize_));
        const auto visitSpectral = [&](auto emit) {
            detail::OnsetPeakPicker<T> picker;
            std::int64_t last = -1;
            for (std::size_t i = 0; i < features.size(); ++i)
            {
                if ((i & 1023u) == 0)
                    job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i),
                                   static_cast<std::int64_t>(features.size()));
                // Preserve the separate pulse policy. An attack may add upper
                // frequencies while a louder bass body loses total energy;
                // source-domain refinement below decides whether it is real.
                if (pulse && i > 0 && features[i].windowPower <= features[i - 1].windowPower)
                    continue;
                // A padded spectrum can turn the file's ending into novelty.
                // Events there require source-domain evidence, supplied below.
                if (!pulse && analysis.featureCenter(i) + offset >= spec.frames)
                    continue;
                if (!picker.accept(i, features.size(), value, windows, T(0.06)))
                    continue;
                Event event;
                // Source energy, not novelty energy: stopping a bass note spreads
                // its novelty into high bands without changing the source register.
                const bool bass = features[i].lowEnergyShare > 0.8;
                if (!refine(event, analysis.featureCenter(i) + offset, micro, microHop, spec, bass))
                {
                    if (pulse || highPower.empty() ||
                        !refine(event, analysis.featureCenter(i) + offset,
                                PowerView(micro, highPower), microHop, spec, false))
                        continue;
                    // The novelty window precedes the actual onset. Check the
                    // nearest window to the refined source position, not that
                    // earlier window dominated by the preceding bass body.
                    const auto evidence = static_cast<std::size_t>(std::clamp<std::int64_t>(
                        (event.begin + analysis.frameSize_ / 2 + analysis.hop_ / 2) /
                                analysis.hop_ - 1,
                        0, static_cast<std::int64_t>(features.size() - 1)));
                    if (!highEvidence[evidence])
                        continue;
                    event.bandRefined = true;
                }
                if (last >= 0 && event.begin - last < gapFrames)
                    continue;
                event.strength =
                    static_cast<double>((pulse ? features[i].pulse : features[i].attack) / maximum);
                emit(event);
                last = event.begin;
            }
        };
        const auto visit = [&](auto emit) {
            if (pulse)
            {
                visitSpectral(emit);
                return;
            }
            // A global novelty maximum can mask an isolated quiet attack, and
            // the spectral peak window can merge closely spaced separate hits.
            // Add source-energy restarts after at least 8 ms of local quiet.
            // The 30 dB contrast requirement rejects ordinary waveform valleys;
            // slow bass onsets remain the spectral/refinement path's job.
            std::size_t nextBin = 0;
            Event restart;
            bool hasRestart = false;
            const auto seekRestart = [&] {
                hasRestart = false;
                while (nextBin < micro.size())
                {
                    const auto at = static_cast<std::int64_t>(nextBin++);
                    if (micro[static_cast<std::size_t>(at)].power <= 1e-12)
                        continue;
                    const auto &bin = micro[static_cast<std::size_t>(at)];
                    const bool impulse = bin.peakPower > .95 * bin.power * microHop;
                    // Missing left context is not eight milliseconds of silence.
                    if (at < 8 && !impulse)
                        continue;
                    double before = 0;
                    for (auto j = std::max<std::int64_t>(0, at - 8); j < at; ++j)
                        before = std::max(before, micro[static_cast<std::size_t>(j)].power);
                    const double after = meanPower(micro, at, at + 3);
                    if (before > after * .001)
                        continue;
                    restart = {};
                    restart.begin = bin.peakFrame;
                    restart.peak = restart.begin;
                    restart.end = restart.begin + 1;
                    restart.strength = std::sqrt(after);
                    restart.contrast = (after - before) / (after + before);
                    restart.energyRestart = true;
                    hasRestart = true;
                    return;
                }
            };
            std::int64_t last = -1;
            const auto sameAttack = static_cast<std::int64_t>(std::llround(.006 * spec.sampleRate));
            const auto append = [&](Event event) {
                if (last < 0 || event.begin - last >= gapFrames)
                {
                    emit(event);
                    last = event.begin;
                }
            };
            seekRestart();
            visitSpectral([&](Event event) {
                while (hasRestart && restart.begin < event.begin - sameAttack)
                {
                    append(restart);
                    seekRestart();
                }
                if (hasRestart && restart.begin <= event.begin + sameAttack)
                {
                    append(restart);
                    seekRestart();
                }
                else
                    append(event);
            });
            while (hasRestart)
            {
                append(restart);
                seekRestart();
            }
        };
        std::size_t count = 0;
        visit([&](Event) { ++count; });
        if (count == 0)
            return;
        auto events = job.allocate<Event>(count);
        std::size_t at = 0;
        visit([&](Event event) { events[at++] = event; });
        for (std::size_t i = 0; i < count; ++i)
        {
            followBody(events[i], i + 1 < count ? &events[i + 1] : nullptr, micro, highPower,
                       microHop, spec, analysis.energy_.samplePeak());
            job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i + 1),
                           static_cast<std::int64_t>(count));
        }
        if (pulse)
        {
            analysis.pulses_ = std::move(events);
            analysis.pulseCount_ = count;
        }
        else
        {
            analysis.attacks_ = std::move(events);
            analysis.attackCount_ = count;
        }
    }
};

} // namespace dspark

#endif // DSPARK_HAS_OFFLINE
