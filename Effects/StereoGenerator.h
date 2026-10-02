// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file StereoGenerator.h
 * @brief Twenty-band stereo generation through a processed-copy delta.
 */
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Convolver.h"
#include "../Core/DryWetMixer.h"
#include "../Core/FIRFilter.h"
#include "../Core/SmoothedValue.h"
#include "../Core/StateBlob.h"
#include "../Core/detail/StereoColor.h"
#include "../Core/detail/TptSvf.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace dspark
{
/** @brief A stereo generator that preserves the delayed original mid signal.
 *
 * A parallel copy passes through twenty moving bands and asymmetric/symmetric
 * rational color. Its stereo delta is added to the original: L += delta,
 * R -= delta. Width does not scale the original side. There is no limiter or
 * automatic gain trim. Only the generated delta receives the optional low cut.
 *
 * The color branch uses an explicit 0.45 Fs to 0.50 Fs FIR transition. The
 * default factor is 1: continuous polynomial reconstruction, shared nonlinear
 * integration and FIR moment projection all run at the source sample rate.
 * Factors 2, 4, 8 and 16 select the oversampled color implementation at prepare
 * time; factor 2 has lower measured antialias rejection than the 1x path.
 * The source-clock DC response is retained at every factor. At 1x the alignment
 * delay is 256 samples before an optional low cut. This is a mathematical
 * source-topology model, not a claim of physical-device equivalence.
 *
 * Width defaults to zero (exact delayed identity). Automation ramps for 5 ms
 * using Core SmoothedValue and keeps latency constant. Input must be stereo;
 * to generate stereo from mono, the caller duplicates mono into two channels.
 * Feed zero input to flush a tail; getLatency() is alignment delay, not tail
 * length. Reset clears history and snaps width to its current target.
 *
 * Threading: prepare(), getOptions(), getState() and setState() require exclusive
 * setup access. processBlock(), reset(), resetAtFrame() and getSourceFrame()
 * belong to one audio stream. Only setWidth(), getWidth(), getStatus() and
 * getLatency() may run concurrently with that stream. No callback allocations
 * or locks. Destruction and prepare must not race any other access.
 * @tparam T float or double sample storage; the DSP path computes in double.
 */
template <typename T> class StereoGenerator final
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    static_assert(std::atomic<float>::is_always_lock_free && std::atomic<int>::is_always_lock_free);

  public:
    /** @brief Configuration applied by prepare(), never by the audio callback. */
    struct Options
    {
        int oversampling = 1; ///< Local color-branch factor: 1, 2, 4, 8 or 16.
        float lowCutHz = 0;   ///< Generated delta only: off, or 20 to 5000 Hz.
        bool operator==(const Options &) const noexcept = default;
    };

    /** @brief Result of the last processing call (also readable by a UI). */
    enum class Status
    {
        Ok,
        NotPrepared,
        InvalidInput,
        ClockOverflow,
        NumericalFailure
    };

    StereoGenerator() = default;
    StereoGenerator(const StereoGenerator &) = delete;
    StereoGenerator &operator=(const StereoGenerator &) = delete;

    /** @brief Conservative cumulative setup allocation bound, or zero for invalid settings.
     * Includes state, temporary filter designs and supported vector/FFT overhead;
     * excludes caller buffers and allocator bookkeeping. This query allocates nothing.
     */
    [[nodiscard]] static std::size_t getPrepareMemoryBound(const AudioSpec &spec,
                                                           Options options) noexcept
    {
        if (!validConfiguration(spec, options))
            return 0;
        using Color = detail::StereoColor<detail::ClipperCurve::AsymmetricKnee>;
        const int source = State::sourceFactor(spec.sampleRate);
        const int colorLatency = Color::latencyFor(options.oversampling, source);
        const auto hp = highPassLayout(spec.sampleRate, options.lowCutHz);
        const int totalLatency = colorLatency + hp.order / 2 + hp.block;
        const auto buffer = [](std::size_t channels, std::size_t frames, std::size_t bytes) {
            return channels * ((frames * bytes + 31) / 32 * 32) + 64;
        };
        auto bytes = sizeof(State) + 64 + 2 * Color::allocationBound(options.oversampling, source) +
                     buffer(2, 256, sizeof(double)) + buffer(1, 256, sizeof(double)) +
                     buffer(1, 256, sizeof(double)) + buffer(1, colorLatency, sizeof(double)) +
                     buffer(2, 256, sizeof(T)) + buffer(2, totalLatency, sizeof(T));
        if (hp.block)
            bytes += 3 * ((static_cast<std::size_t>(hp.order) + 1) * sizeof(double) + 64) +
                     detail::ConvolutionRate::convolutionAllocationBound(hp.block, hp.order + 1);
        return bytes;
    }

    /** @brief Prepares pending options for stereo at 8 to 384 kHz.
     * Invalid settings return false, leaving the previous prepared state intact.
     * Allocation failure propagates with the same strong state guarantee.
     */
    bool prepare(const AudioSpec &spec)
    {
        return prepare(spec, options_);
    }

    /** @brief Prepares explicit options; on success clears all stream history. */
    bool prepare(const AudioSpec &spec, Options options)
    {
        if (!validConfiguration(spec, options))
            return false;
        auto next = std::make_unique<State>(spec, options, getWidth());
        state_ = std::move(next);
        options_ = options;
        latency_.store(state_->latency, std::memory_order_relaxed);
        publish(Status::Ok);
        return true;
    }

    /** @brief Publishes width in [0,1]. Invalid values return false unchanged. */
    bool setWidth(float width) noexcept
    {
        if (!std::isfinite(width) || width < 0 || width > 1)
            return false;
        width_.store(width, std::memory_order_relaxed);
        return true;
    }
    /** @brief Returns the published target, not the instantaneous ramp value. */
    [[nodiscard]] float getWidth() const noexcept
    {
        return width_.load(std::memory_order_relaxed);
    }
    /** @brief Returns options for the next prepare (including restored presets). */
    [[nodiscard]] Options getOptions() const noexcept
    {
        return options_;
    }
    /** @brief Exact active delay in source frames, including the optional FIR. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return latency_.load(std::memory_order_relaxed);
    }
    /** @brief Returns the last processing status. Numerical failure latches until reset. */
    [[nodiscard]] Status getStatus() const noexcept
    {
        return static_cast<Status>(status_.load(std::memory_order_relaxed));
    }
    /** @brief Returns the next input frame on this stream's source clock. */
    [[nodiscard]] std::uint64_t getSourceFrame() const noexcept
    {
        return state_ ? state_->frame : 0;
    }
    /** @brief Clears history and restarts modulation at source frame zero. */
    void reset() noexcept
    {
        resetAtFrame(0);
    }

    /** @brief Clears history and aligns modulation to an absolute source frame.
     * This is not a seek restore: replay the preceding audio to recover filter
     * history. The 64-frame modulation grid stays anchored to source frame zero.
     * The clock is 64-bit; LFO time/phase is evaluated in double precision.
     */
    void resetAtFrame(std::uint64_t frame) noexcept
    {
        if (!state_)
            return;
        state_->reset(frame, getWidth());
        publish(Status::Ok);
    }

    /** @brief Processes writable, nonoverlapping stereo channels in place.
     * Returns false without changing input or advancing state for invalid layout,
     * nonfinite input, oversized blocks or clock overflow. A numerical DSP failure
     * returns false and replaces the affected internal chunk and subsequent audio
     * with aligned dry input until reset; earlier chunks remain processed. Inspect
     * getStatus() and handle the failure in the host. No finite-value clamp occurs.
     */
    bool processBlock(AudioBufferView<T> buffer) noexcept
    {
        return processBlock(buffer, {});
    }

    /** @brief Also captures the generated mono delta before rounded L/R add-back.
     * An empty span disables capture. Otherwise its length must match the block
     * and its storage must not overlap either audio channel. The delta includes
     * width and low cut, with the same alignment delay as output. A numerical
     * failure writes zero delta for each affected chunk. No subtraction of large
     * original samples is needed to recover a small generated component.
     */
    bool processBlock(AudioBufferView<T> buffer, std::span<double> generatedDelta) noexcept
    {
        if (!state_)
            return publish(Status::NotPrepared);
        const int count = buffer.getNumSamples();
        if (buffer.getNumChannels() != 2 || count > state_->spec.maxBlockSize ||
            (!generatedDelta.empty() && generatedDelta.size() != static_cast<std::size_t>(count)))
            return publish(Status::InvalidInput);
        if (static_cast<std::uint64_t>(count) >
            std::numeric_limits<std::uint64_t>::max() - state_->frame)
            return publish(Status::ClockOverflow);
        if (count == 0)
            return publish(state_->fault ? Status::NumericalFailure : Status::Ok);
        auto *left = buffer.getChannel(0);
        auto *right = buffer.getChannel(1);
        if (!left || !right)
            return publish(Status::InvalidInput);
        if (!generatedDelta.empty() &&
            (!generatedDelta.data() ||
             overlaps(generatedDelta.data(),
                      static_cast<std::uint64_t>(generatedDelta.size()) * sizeof(double), left,
                      static_cast<std::uint64_t>(count) * sizeof(T)) ||
             overlaps(generatedDelta.data(),
                      static_cast<std::uint64_t>(generatedDelta.size()) * sizeof(double), right,
                      static_cast<std::uint64_t>(count) * sizeof(T))))
            return publish(Status::InvalidInput);
        const auto a = reinterpret_cast<std::uintptr_t>(left);
        const auto b = reinterpret_cast<std::uintptr_t>(right);
        if (static_cast<std::uint64_t>(a > b ? a - b : b - a) <
            static_cast<std::uint64_t>(count) * sizeof(T))
            return publish(Status::InvalidInput);
        for (int i = 0; i < count; ++i)
            if (!std::isfinite(left[i]) || !std::isfinite(right[i]))
                return publish(Status::InvalidInput);
        state_->width.setTargetValue(getWidth());
        for (int offset = 0; offset < count;)
        {
            const int size = std::min(256, count - offset);
            state_->process(buffer.getSubView(offset, size),
                            generatedDelta.empty()
                                ? std::span<double>{}
                                : generatedDelta.subspan(static_cast<std::size_t>(offset),
                                                         static_cast<std::size_t>(size)));
            offset += size;
        }
        return publish(state_->fault ? Status::NumericalFailure : Status::Ok);
    }

    /** @brief Serializes parameters only; filter history and source clock are excluded. */
    [[nodiscard]] std::vector<std::uint8_t> getState() const
    {
        StateWriter writer(stateId("STGN"), 1);
        writer.write("width", getWidth());
        writer.write("oversampling", static_cast<std::int32_t>(options_.oversampling));
        writer.write("lowCutHz", options_.lowCutHz);
        return writer.blob();
    }

    /** @brief Restores a validated preset. Factor/low cut apply at the next prepare.
     * Missing keys retain their settings; malformed/duplicate known keys reject
     * the entire preset unchanged. Unknown keys are ignored for forward compatibility.
     */
    bool setState(const std::uint8_t *data, std::size_t size)
    {
        StateReader reader(data, size);
        if (!reader.isValid() || reader.processorId() != stateId("STGN") ||
            reader.processorVersion() != 1)
            return false;
        std::array<bool, 3> seen{};
        for (const auto &entry : reader.entries())
        {
            const int index = entry.key == "width"          ? 0
                              : entry.key == "oversampling" ? 1
                              : entry.key == "lowCutHz"     ? 2
                                                            : -1;
            if (index < 0)
                continue;
            if (seen[index] || entry.type != (index == 1 ? 1 : 0))
                return false;
            seen[index] = true;
        }
        const float width = reader.read("width", getWidth());
        const Options options{reader.read("oversampling", options_.oversampling),
                              reader.read("lowCutHz", options_.lowCutHz)};
        if (!std::isfinite(width) || width < 0 || width > 1 || !validOptions(options))
            return false;
        options_ = options;
        width_.store(width, std::memory_order_relaxed);
        return true;
    }

  private:
    struct HighPassLayout
    {
        int order = 0, block = 0;
        bool allReject = false;
    };
    static HighPassLayout highPassLayout(double rate, double lowCut) noexcept
    {
        HighPassLayout result;
        if (lowCut <= 0)
            return result;
        const double transition =
            std::min(std::max(50., lowCut * .25), std::min(lowCut * 1.5, rate - 2 * lowCut));
        if (lowCut >= rate * .5 || transition < 30)
            result.allReject = true;
        else
        {
            result.order =
                static_cast<int>(std::ceil(94 / (2.285 * twoPi<double> * transition / rate)));
            if (result.order & 1)
                ++result.order;
            result.block = 1;
            while (result.block * 2 <= std::max(64, static_cast<int>(std::round(rate / 93.75))))
                result.block *= 2;
        }
        return result;
    }
    static bool overlaps(const void *left, std::uint64_t leftBytes, const void *right,
                         std::uint64_t rightBytes) noexcept
    {
        const auto a = reinterpret_cast<std::uintptr_t>(left);
        const auto b = reinterpret_cast<std::uintptr_t>(right);
        return a <= b ? b - a < leftBytes : a - b < rightBytes;
    }
    struct Band
    {
        detail::TptSvfState<double> state{};
        int index;
        double rate, base, corridor, g = 0, gain = 0, dg = 0, dGain = 0;
        int untilUpdate = 0;

        void prepare(int i, double fs) noexcept
        {
            index = i;
            rate = fs;
            const double first = i < 16 ? 210 : 45;
            const double last = i < 16 ? std::min(14000., fs * .35) : 165;
            corridor = std::log(last / first) / (i < 16 ? 15 : 3);
            base = first * std::exp((i < 16 ? i : i - 16) * corridor);
        }
        double coefficient(double t) const noexcept
        {
            const double f =
                base *
                std::exp(.22 * corridor *
                         std::sin(twoPi<double> * (.029 + index * .0017) * t + index * 2.399963));
            return std::tan(pi<double> * f / rate);
        }
        double amplitude(double t) const noexcept
        {
            const double db = (((index / 2 + index % 2) & 1) == 0 ? 1 : -1) *
                              (2.6 + .4 * std::sin(twoPi<double> * (.037 + index * .0013) * t +
                                                   index * 1.618034));
            return std::expm1(db * std::log(10.) / 20);
        }
        void position(std::uint64_t frame) noexcept
        {
            const auto offset = static_cast<int>(frame % 64);
            const double t = static_cast<double>(frame - offset) / rate;
            const double future = t + 64. / rate;
            g = coefficient(t);
            gain = amplitude(t);
            dg = (coefficient(future) - g) / 64;
            dGain = (amplitude(future) - gain) / 64;
            for (int i = 0; i < offset; ++i)
            {
                g += dg;
                gain += dGain;
            }
            untilUpdate = 64 - offset;
        }
        double process(double input, std::uint64_t frame) noexcept
        {
            if (untilUpdate == 0)
                position(frame);
            const double a1 = 1 / (1 + g * (g + .5));
            const double a2 = g * a1;
            const double a3 = g * a2;
            const double y = gain * .5 * detail::tptSvfStep(input, state, a1, a2, a3).bandpass;
            g += dg;
            gain += dGain;
            --untilUpdate;
            return y;
        }
    };

    struct State
    {
        AudioSpec spec;
        std::array<Band, 20> bands{};
        AudioBuffer<double> copy, delta;
        DryWetMixer<T, 2> dry;
        DryWetMixer<double, 1> linear;
        Convolver<double> highPass;
        detail::StereoColor<detail::ClipperCurve::AsymmetricKnee> tube;
        detail::StereoColor<detail::ClipperCurve::SymmetricKnee> tape;
        SmoothedValue<double> width;
        std::uint64_t frame = 0;
        int latency;
        bool useHighPass = false, allReject = false, fault = false;

        State(AudioSpec environment, Options options, float initialWidth)
            : spec(environment), tube(options.oversampling, sourceFactor(spec.sampleRate)),
              tape(options.oversampling, sourceFactor(spec.sampleRate)), latency(tube.latency())
        {
            for (int i = 0; i < 20; ++i)
                bands[i].prepare(i, spec.sampleRate);
            copy.resize(2, 256);
            delta.resize(1, 256);
            linear.prepare({spec.sampleRate, 256, 1});
            linear.setLatencyCompensation(tube.latency());
            const auto hp = highPassLayout(spec.sampleRate, options.lowCutHz);
            allReject = hp.allReject;
            if (hp.block)
            {
                const auto taps = FIRDesign<double>::highPass(spec.sampleRate, options.lowCutHz,
                                                              hp.order + 1, 10.2);
                highPass.prepare(hp.block, taps.data(), static_cast<int>(taps.size()));
                latency += hp.order / 2 + hp.block;
                useHighPass = true;
            }
            dry.prepare({spec.sampleRate, 256, 2});
            dry.setLatencyCompensation(latency);
            width.prepare(spec.sampleRate, 5);
            width.setSmoothingType(SmoothedValue<double>::SmoothingType::Linear);
            reset(0, initialWidth);
        }
        static int sourceFactor(double rate) noexcept
        {
            return rate <= 48000 ? 8 : rate <= 96000 ? 4 : rate <= 192000 ? 2 : 1;
        }
        void reset(std::uint64_t position, float targetWidth) noexcept
        {
            frame = position;
            for (auto &band : bands)
            {
                band.state = {};
                band.position(position);
            }
            tube.reset();
            tape.reset();
            highPass.reset();
            dry.reset();
            linear.reset();
            width.reset(targetWidth);
            fault = false;
        }
        void process(AudioBufferView<T> input, std::span<double> generatedDelta) noexcept
        {
            const int count = input.getNumSamples();
            dry.pushDry(input);
            double *l = copy.getChannel(0), *r = copy.getChannel(1), *d = delta.getChannel(0);
            if (!fault)
            {
                for (int i = 0; i < count; ++i)
                {
                    l[i] = r[i] = 0;
                    for (int j = 0; j < 20; ++j)
                        (j & 1 ? r[i] : l[i]) +=
                            bands[j].process(input.getChannel(j & 1)[i], frame + i);
                    d[i] = .5 * (l[i] - r[i]);
                    if (!std::isfinite(l[i]) || !std::isfinite(r[i]) || !std::isfinite(d[i]))
                        fault = true;
                }
                if (!fault)
                {
                    linear.pushDry(delta.toView().getSubView(0, count));
                    double *lc[]{l}, *rc[]{r};
                    tube.process({lc, 1, count});
                    tape.process({rc, 1, count});
                    const auto *aligned = linear.getDryChannel(0);
                    for (int i = 0; i < count; ++i)
                        d[i] = aligned[i] + .1 * (.5 * (l[i] - r[i]) - aligned[i]);
                    if (useHighPass)
                        highPass.processInPlace(d, count);
                    for (int i = 0; i < count; ++i)
                        if (!std::isfinite(d[i]))
                            fault = true;
                    if (allReject)
                        std::fill_n(d, count, 0.);
                }
            }
            const auto *dl = dry.getDryChannel(0), *dr = dry.getDryChannel(1);
            std::array<double, 256> changes{};
            for (int i = 0; i < count; ++i)
            {
                const double w = width.getNextValue();
                const double change = fault || w == 0 ? 0 : 8 * w * d[i];
                const double outL = static_cast<double>(dl[i]) + change;
                const double outR = static_cast<double>(dr[i]) - change;
                if (!std::isfinite(outL) || !std::isfinite(outR) ||
                    std::abs(outL) > std::numeric_limits<T>::max() ||
                    std::abs(outR) > std::numeric_limits<T>::max())
                    fault = true;
                changes[i] = change;
            }
            for (int i = 0; i < count; ++i)
            {
                if (!generatedDelta.empty())
                    generatedDelta[i] = fault ? 0 : changes[i];
                // Assignment preserves signed zero and exact dry PCM at zero width.
                input.getChannel(0)[i] =
                    fault || changes[i] == 0 ? dl[i] : static_cast<T>(dl[i] + changes[i]);
                input.getChannel(1)[i] =
                    fault || changes[i] == 0 ? dr[i] : static_cast<T>(dr[i] - changes[i]);
            }
            frame += static_cast<std::uint64_t>(count);
        }
    };

    static bool validConfiguration(const AudioSpec &spec, Options options) noexcept
    {
        return spec.isValid() && spec.numChannels == 2 && (spec.sampleRate >= 8000) &&
               (spec.sampleRate <= 384000) && validOptions(options);
    }
    static bool validOptions(Options options) noexcept
    {
        return options.oversampling >= 1 && options.oversampling <= 16 &&
               (options.oversampling & (options.oversampling - 1)) == 0 &&
               std::isfinite(options.lowCutHz) &&
               (options.lowCutHz == 0 || (options.lowCutHz >= 20 && options.lowCutHz <= 5000));
    }
    bool publish(Status status) noexcept
    {
        status_.store(static_cast<int>(status), std::memory_order_relaxed);
        return status == Status::Ok;
    }
    std::unique_ptr<State> state_;
    Options options_;
    std::atomic<float> width_{0};
    std::atomic<int> status_{static_cast<int>(Status::NotPrepared)}, latency_{0};
};
} // namespace dspark
