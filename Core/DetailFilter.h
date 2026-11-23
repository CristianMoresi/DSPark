// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DetailFilter.h
 * @brief A time-varying minimum-phase filter drawn on a log-frequency grid,
 *        with detail in the low end too (a 24th of an octave from 200 Hz).
 *
 * A gain curve a twelfth of an octave fine at 100 Hz needs an impulse
 * response of a third of a second: one transform long enough for it, every
 * hop and over the whole band, costs several times what the top needs. The
 * curve is split in decibels by a crossover an octave wide (900 to 1800 Hz
 * at 48 kHz, 830 to 1650 at 44.1) into two minimum-phase filters in series,
 * whose product is the curve. Under the crossover they share it by
 * resolution as well: what is broad (over 60 Hz across) is the high part's,
 * and the low part gets what is left, the fine detail, whose mean is zero.
 * That is what keeps the low part's correction inside its band: a
 * minimum-phase filter that moves the mean of the log gain turns the phase
 * up to the top of the spectrum, which no decimated filter can do, and cut
 * off there it rings ahead of every attack.
 *
 * - The low part runs at an eighth of the rate (a sixteenth from 88.2 kHz),
 *   where a response of a third of a second is two thousand samples. It
 *   works as a correction: the programme, delayed, plus the interpolated
 *   difference the low filter makes. A flat curve therefore returns the
 *   programme, whatever the decimator does to the top. The decimator and the
 *   interpolator are one linear-phase low-pass, stopped from the slow
 *   Nyquist up (not at it, as a half-band would be): the low filter turns the
 *   phase well above its crossover, and what a half-band lets through there
 *   comes back as an image of the programme.
 * - The high part runs at the rate, in windows of 1024 samples with a
 *   response of up to three windows: a cut a 24th of an octave wide at 2
 *   kHz takes that room to be as deep as it is drawn. It takes the curve as
 *   it is drawn, so it leads the programme by what the latency allows.
 *
 * The curve has SpectralReadout::kPoints values in decibels, evenly in log
 * frequency from 20 Hz to 20 kHz (the end values hold beyond), so what is
 * designed, what is applied and what is drawn share one grid. The owner's
 * callback writes it every getHopSize() samples: one curve per channel, or
 * one for all with setLinked(true). An owner that designs from the signal
 * reads it through the feed callback, which keeps its analysis on the same
 * samples whatever the blocks.
 *
 * A point of the grid can be narrower than a bin of the part that plays it
 * (2.9 Hz in the low part, 12 over it, at 48 kHz), which takes the curve at
 * its bins and plays between them as they allow. The played response is
 * therefore no estimate from the curve: it is the response of the kernels
 * applied, for a display that shows what is heard and not what was asked.
 * The audio thread only hands the kernels over (publishPlayed(), a copy);
 * the thread that reads the display (playedValue(), playedExtreme()) makes
 * the response when they are new.
 *
 * Latency (getLatency()): with DetailLatency::Frame, the default, 2048
 * samples, doubling by octaves with the rate. A curve is drawn for each
 * kernel, 512 apart, and they are
 * played as Hann frames of 2048 at a quarter's overlap would play them, so
 * the gain moves as smoothly as through such frames and the high part
 * leads the programme as their middle would. Before the kernels are
 * designed each point holds its moves (setHold()) over the three curves
 * that play one in full (32 ms at 48 kHz), so that smoothing leaves a
 * brief move as deep as it was drawn.
 * With DetailLatency::Block the
 * latency is 774 samples at 44.1 and 48 kHz, half that at 22.05 and 32, a
 * quarter at 16 kHz and under. What is broad and what is high, the high
 * part's, then leads the programme by 10 ms at 48 kHz (the kernel of a curve
 * is half in 486 samples ahead of the instant it was drawn at); the fine
 * detail of the low end, whose own readings are windows of up to a third of
 * a second, follows it.
 * The output does not depend on how the stream is cut into blocks.
 *
 * Threading: prepare() on the setup thread (allocates); processBlock(),
 * reset(), publishPlayed() and the setters on the audio thread
 * (owner-managed); playedValue() and playedExtreme() on any thread (the
 * readers wait for one another and, for the moment a copy takes, for the
 * audio thread; the audio thread never waits for them).
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DenormalGuard.h, DspMath.h,
 * FFT.h, FIRFilter.h, SimdOps.h, SpectralFilter.h, SpinLock.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "DenormalGuard.h"
#include "DspMath.h"
#include "FFT.h"
#include "FIRFilter.h"
#include "SimdOps.h"
#include "SpectralFilter.h"
#include "SpinLock.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numbers>
#include <thread>
#include <type_traits>
#include <vector>

namespace dspark {

/**
 * @brief When a part of a DetailFilter hands its samples back.
 */
enum class DetailLatency
{
    Frame, ///< After a window of 2048: the curve leads the programme and moves as smoothly as overlapped frames of it.
    Block  ///< After a quarter of a frame, by partitioned convolution: the same kernels, at 774 samples.
};

/**
 * @brief How a DetailFilter by the frame holds the moves of each point of
 *        its curve, over the three curves that play a move in full.
 */
enum class DetailHold
{
    Largest, ///< Its largest move of the three: the steadiest; a move ends two curves late.
    Reach,   ///< A new move for the three, then the curve as drawn: a move ends when it is drawn.
    None     ///< The curve as drawn: a brief move plays at part of its depth.
};

namespace detail {

/**
 * @class KernelConvolver
 * @brief A part of a DetailFilter: minimum-phase kernels designed as a
 *        SpectralFilter designs its own (detail::MinimumPhase), at the
 *        window and padding given, convolved in partitions of a block (a
 *        quarter of that window, up to the whole of it).
 *
 * A kernel is designed every `steps` blocks. With DetailLatency::Frame the
 * output runs through the last four as Hann frames four spacings long and
 * one apart would play them. Their weights pair up (the first and the third
 * add up to a half, as do the other two), so three convolutions play four
 * kernels: their mean and the two differences. With DetailLatency::Block it
 * runs through the last three by a quadratic B-spline. A fade between two images
 * a moving curve at the rate of the kernels: by its first power near that
 * rate with a raised cosine, by its square with a straight line, which then
 * falls slowly away from it. The spline images by the cube and has no corner.
 * Kernels are designed in double, the signal runs in T. A flat curve returns
 * the input to the bit, late by a block.
 *
 * Audio thread, except prepare() (setup thread; allocates).
 *
 * @tparam T Sample type (float or double).
 * @tparam L How the output runs through the kernels.
 */
template <FloatType T, DetailLatency L>
class KernelConvolver
{
public:
    /**
     * @brief Allocates.
     * @param sampleRate The rate this part runs at.
     * @param channels   The channels it filters.
     * @param window     The window the curve is designed for (a power of two, 4 at least).
     * @param padding    The transform of the design over that window (a power of two).
     * @param block      Samples of a partition: a quarter of the window to all of it.
     * @param steps      Blocks from one kernel to the next.
     */
    void prepare(double sampleRate, int channels, int window, int padding, int block, int steps)
    {
        prepared_ = false;
        fs_ = sampleRate;
        channels_ = std::max(channels, 0);
        size_ = window * padding;
        hop_ = window / 2;
        block_ = std::clamp(block, std::max(1, window / 4), std::max(1, window));
        steps_ = std::max(steps, 1);
        bins_ = size_ / 2 + 1;
        length_ = size_ - window + 1;
        partitions_ = (length_ + block_ - 1) / block_;
        stride_ = 2 * (block_ + 1);

        design_ = std::make_unique<FFTReal<double>>(static_cast<std::size_t>(size_));
        transform_ = std::make_unique<FFTReal<T>>(static_cast<std::size_t>(2 * block_));
        minimumPhase_.prepare(bins_);
        logGain_.assign(static_cast<std::size_t>(bins_), T(0));
        spectrum_.assign(static_cast<std::size_t>(size_ + 2), 0.0);
        kernel_.assign(static_cast<std::size_t>(size_ + 2), 0.0);
        taper_.assign(static_cast<std::size_t>(length_), 1.0);
        const int tail = std::max(1, length_ / 4);
        for (int i = 0; i < tail; ++i)
            taper_[static_cast<std::size_t>(length_ - tail + i)] = 0.5 + 0.5 * std::cos(std::numbers::pi * (i + 1) / (tail + 1));
        // The short kernel: a window at most, tapered over its whole length.
        shortLength_ = std::min(length_, window);
        shortTaper_.resize(static_cast<std::size_t>(shortLength_));
        for (int i = 0; i < shortLength_; ++i)
            shortTaper_[static_cast<std::size_t>(i)] = 0.5 + 0.5 * std::cos(std::numbers::pi * i / shortLength_);
        const int fade = steps_ * block_;
        for (auto* v : { &older_, &newer_ }) v->resize(static_cast<std::size_t>(fade));
        for (int i = 0; i < fade; ++i)
        {
            const double u = static_cast<double>(i + 1) / fade, turn = 0.5 * std::numbers::pi * u;
            older_[static_cast<std::size_t>(i)] = static_cast<T>(kByFour ? -0.25 * std::cos(turn) : 0.5 * (1.0 - u) * (1.0 - u));
            newer_[static_cast<std::size_t>(i)] = static_cast<T>(kByFour ? 0.25 * std::sin(turn) : 0.5 * u * u);
        }

        time_.assign(static_cast<std::size_t>(2 * block_ + 2), T(0));
        sum_.assign(static_cast<std::size_t>(stride_), T(0));
        for (auto* v : { &oldest_, &newest_ }) v->assign(static_cast<std::size_t>(block_), T(0));
        const auto spectra = static_cast<std::size_t>(partitions_) * static_cast<std::size_t>(stride_);
        channel_.resize(static_cast<std::size_t>(channels_));
        slot_.resize(static_cast<std::size_t>(channels_));
        for (auto& c : channel_)
        {
            for (auto* v : { &c.in, &c.before, &c.out }) v->assign(static_cast<std::size_t>(block_), T(0));
            c.past.assign(spectra, T(0));
        }
        for (auto& s : slot_)
        {
            for (auto& k : s.partition) k.assign(spectra, T(0));
            if constexpr (kByFour)
                for (auto* v : { &s.mean, &s.even, &s.odd }) v->assign(spectra, T(0));
            s.kept.assign(static_cast<std::size_t>(length_), 0.0);
        }
        prepared_ = true;
        reset();
    }

    /** @brief Clears the signal and the kernels (every one is then flat). */
    void reset() noexcept
    {
        if (!prepared_) return;
        for (auto& c : channel_)
            for (auto* v : { &c.in, &c.before, &c.out, &c.past }) std::fill(v->begin(), v->end(), T(0));
        for (auto& s : slot_)
        {
            s.flat.fill(true);
            s.newest = 0;
        }
        at_ = head_ = count_ = 0;
        fading_ = steps_;
    }

    /** @brief One kernel for every channel. */
    void setLinked(bool on) noexcept { linked_ = on; }
    /** @brief Kernels a window long, tapered over their whole length, from the next one designed. */
    void setShortKernel(bool on) noexcept { short_ = on; }

    /** @return The latency: a block. */
    [[nodiscard]] int getLatency() const noexcept { return block_; }
    /** @return Half a window: the owner's hop, read from the low part. */
    [[nodiscard]] int getHopSize() const noexcept { return hop_; }
    /** @return The bins of a curve. */
    [[nodiscard]] int getNumBins() const noexcept { return bins_; }
    /** @return The transform of the design. */
    [[nodiscard]] int getTransformSize() const noexcept { return size_; }
    /** @return Samples in a kernel. */
    [[nodiscard]] int getKernelLength() const noexcept { return length_; }
    /** @return Hertz a bin of the curve. */
    [[nodiscard]] double getBinHz() const noexcept { return fs_ / static_cast<double>(size_); }
    /** @return The newest kernel of a channel (0 when linked), or nullptr where it is flat. */
    [[nodiscard]] const double* getKeptKernel(int channel) const noexcept
    {
        if (channel < 0 || channel >= static_cast<int>(slot_.size())) return nullptr;
        const Slot& s = slot_[static_cast<std::size_t>(channel)];
        return s.flat[static_cast<std::size_t>(s.newest)] ? nullptr : s.kept.data();
    }

    /**
     * @brief Filters a block in place.
     * @tparam Design `void(int channel, const T* power, T* logGain, int bins)`:
     *         writes natural logarithms of gains (preset to 0). The power is
     *         nullptr and the channel -1 when linked. Must not throw.
     * @param buffer The audio.
     * @param design The callback, run as each kernel starts.
     */
    template <typename Design>
    void processBlock(AudioBufferView<T> buffer, Design&& design) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        const int channels = std::min(buffer.getNumChannels(), channels_);
        for (int done = 0; done < buffer.getNumSamples();)
        {
            const int count = std::min(buffer.getNumSamples() - done, block_ - at_);
            for (int ch = 0; ch < channels; ++ch)
            {
                Channel& c = channel_[static_cast<std::size_t>(ch)];
                T* x = buffer.getChannel(ch) + done;
                std::copy_n(x, count, c.in.data() + at_);
                std::copy_n(c.out.data() + at_, count, x);
            }
            done += count;
            at_ += count;
            if (at_ < block_) continue;
            at_ = 0;
            step(channels, design);
        }
    }

private:
    static constexpr bool kByFour = L == DetailLatency::Frame;
    static constexpr int kKernels = kByFour ? 4 : 3; ///< Read at once.

    struct Channel
    {
        std::vector<T> in, before, out;
        std::vector<T> past;                ///< The spectra of the last blocks, a partition each.
    };

    struct Slot
    {
        std::array<std::vector<T>, kKernels> partition;
        std::vector<double> kept;
        std::array<bool, kKernels> flat {};  ///< All set by reset().
        std::vector<T> mean, even, odd;      ///< By four: their mean and the differences of the two pairs.
        int newest = 0;
    };

    template <typename Design>
    void step(int channels, Design& design) noexcept
    {
        if (++count_ == steps_)
        {
            count_ = 0;
            fading_ = 0;
            const bool one = linked_ && channels > 1;
            for (int s = 0; s < (one ? 1 : channels); ++s)
            {
                std::fill(logGain_.begin(), logGain_.end(), T(0));
                design(one ? -1 : s, static_cast<const T*>(nullptr), logGain_.data(), bins_);
                build(slot_[static_cast<std::size_t>(s)]);
            }
        }
        head_ = head_ + 1 == partitions_ ? 0 : head_ + 1;
        for (int ch = 0; ch < channels; ++ch)
            convolve(channel_[static_cast<std::size_t>(ch)], slot_[static_cast<std::size_t>(linked_ ? 0 : ch)]);
        if (fading_ < steps_) ++fading_;
    }

    void build(Slot& s) noexcept
    {
        s.newest = (s.newest + 1) % kKernels;
        const auto newest = static_cast<std::size_t>(s.newest);
        bool flat = true;
        for (int k = 0; k < bins_ && flat; ++k) flat = logGain_[static_cast<std::size_t>(k)] == T(0);
        s.flat[newest] = flat;
        if (flat)
        {
            if constexpr (kByFour)
            {
                // The sums read a flat kernel as a spectrum: ones in the first partition.
                std::fill(s.partition[newest].begin(), s.partition[newest].end(), T(0));
                for (int k = 0; k <= block_; ++k) s.partition[newest][static_cast<std::size_t>(2 * k)] = T(1);
                combine(s);
            }
            return;
        }
        double* h = spectrum_.data();
        double* c = kernel_.data();
        for (int k = 0; k < bins_; ++k)
        {
            h[2 * k] = std::max(static_cast<double>(logGain_[static_cast<std::size_t>(k)]), MinimumPhase::kLeastLogGain);
            h[2 * k + 1] = 0.0;
        }
        minimumPhase_.build(*design_, h, c, size_, short_ ? shortTaper_.data() : taper_.data(), short_ ? shortLength_ : length_);
        std::copy_n(c, length_, s.kept.data());
        T* to = s.partition[newest].data();
        for (int p = 0; p < partitions_; ++p)
        {
            std::fill(time_.begin(), time_.end(), T(0));
            const int from = p * block_, count = std::min(block_, length_ - from);
            for (int i = 0; i < count; ++i) time_[static_cast<std::size_t>(i)] = static_cast<T>(c[from + i]);
            transform_->forward(time_.data(), to + p * stride_);
        }
        if constexpr (kByFour) combine(s);
    }

    /// By four: the mean of the kernels and the difference of each pair.
    void combine(Slot& s) noexcept
    {
        if (allFlat(s)) return;
        const auto at = [&s](int age) { return s.partition[static_cast<std::size_t>((s.newest + kKernels - age) % kKernels)].data(); };
        const T* h0 = at(0);
        const T* h1 = at(1);
        const T* h2 = at(2);
        const T* h3 = at(3);
        T* mean = s.mean.data();
        T* even = s.even.data();
        T* odd = s.odd.data();
        const int count = partitions_ * stride_;
        for (int i = 0; i < count; ++i) mean[i] = T(0.25) * ((h0[i] + h2[i]) + (h1[i] + h3[i]));
        for (int i = 0; i < count; ++i) even[i] = h0[i] - h2[i];
        for (int i = 0; i < count; ++i) odd[i] = h1[i] - h3[i];
    }

    [[nodiscard]] static bool allFlat(const Slot& s) noexcept
    {
        for (const bool flat : s.flat)
            if (!flat) return false;
        return true;
    }

    /// The newest block through one of a slot's kernels.
    void play(const Channel& c, const Slot& s, int which, T* to) noexcept
    {
        const auto w = static_cast<std::size_t>(which);
        if (s.flat[w])
        {
            std::copy_n(c.in.data(), block_, to);
            return;
        }
        playSpectra(c, s.partition[w].data(), to);
    }

    /// The newest block through a kernel given as the spectra of its partitions.
    void playSpectra(const Channel& c, const T* kernel, T* to) noexcept
    {
        std::fill(sum_.begin(), sum_.end(), T(0));
        // The ring of spectra in its two runs, newest first.
        for (int p = 0; p <= head_; ++p)
            simd::complexMulAccum(sum_.data(), c.past.data() + (head_ - p) * stride_, kernel + p * stride_, block_ + 1);
        for (int p = head_ + 1; p < partitions_; ++p)
            simd::complexMulAccum(sum_.data(), c.past.data() + (head_ - p + partitions_) * stride_, kernel + p * stride_, block_ + 1);
        transform_->inverse(sum_.data(), time_.data());
        std::copy_n(time_.data() + block_, block_, to);
    }

    void convolve(Channel& c, const Slot& s) noexcept
    {
        std::copy_n(c.before.data(), block_, time_.data());
        std::copy_n(c.in.data(), block_, time_.data() + block_);
        transform_->forward(time_.data(), c.past.data() + head_ * stride_);
        if constexpr (kByFour) playFour(c, s);
        else playThree(c, s);
        std::copy_n(c.in.data(), block_, c.before.data());
    }

    void playFour(Channel& c, const Slot& s) noexcept
    {
        T* out = c.out.data();
        if (fading_ >= steps_ || allFlat(s))
        {
            std::copy_n(c.in.data(), block_, out); // nothing designed yet, or nothing but flat
            return;
        }
        playSpectra(c, s.mean.data(), out);
        playSpectra(c, s.even.data(), oldest_.data());
        playSpectra(c, s.odd.data(), newest_.data());
        const T* evenShare = older_.data() + fading_ * block_;
        const T* oddShare = newer_.data() + fading_ * block_;
        const T* even = oldest_.data();
        const T* odd = newest_.data();
        for (int i = 0; i < block_; ++i) out[i] += evenShare[i] * even[i] + oddShare[i] * odd[i];
    }

    void playThree(Channel& c, const Slot& s) noexcept
    {
        // The middle one, and what the two others differ from it by: flat or
        // equal kernels play exactly as one.
        play(c, s, (s.newest + kKernels - 1) % kKernels, c.out.data());
        if (fading_ < steps_ && !allFlat(s))
        {
            play(c, s, (s.newest + 1) % kKernels, oldest_.data());
            play(c, s, s.newest, newest_.data());
            const T* older = older_.data() + fading_ * block_;
            const T* newer = newer_.data() + fading_ * block_;
            const T* first = oldest_.data();
            const T* last = newest_.data();
            T* out = c.out.data();
            for (int i = 0; i < block_; ++i)
            {
                const T middle = out[i];
                out[i] = middle + older[i] * (first[i] - middle) + newer[i] * (last[i] - middle);
            }
        }
    }

    double fs_ = 48000.0;
    int channels_ = 0, size_ = 0, hop_ = 0, block_ = 1, steps_ = 2, bins_ = 0, length_ = 0, partitions_ = 1, stride_ = 4;
    int shortLength_ = 0;
    bool short_ = false;
    int at_ = 0, head_ = 0, count_ = 0, fading_ = 2;
    bool prepared_ = false, linked_ = false;
    std::unique_ptr<FFTReal<double>> design_;
    std::unique_ptr<FFTReal<T>> transform_;
    MinimumPhase minimumPhase_;
    std::vector<T> logGain_, time_, sum_, oldest_, newest_, older_, newer_;
    std::vector<double> spectrum_, kernel_, taper_, shortTaper_;
    std::vector<Channel> channel_;
    std::vector<Slot> slot_;
};

} // namespace detail

/**
 * @class DetailFilter
 * @brief Minimum-phase filtering by a curve on a log-frequency grid, fine in
 *        the low end.
 *
 * @tparam T Sample type (float or double).
 * @tparam L When it hands the samples back (DetailLatency; Frame unless given).
 */
template <FloatType T, DetailLatency L = DetailLatency::Frame>
class DetailFilter
{
    static constexpr bool kByBlock = L == DetailLatency::Block;
    using Part = detail::KernelConvolver<T, L>;

public:
    static constexpr int kPoints = SpectralReadout<T>::kPoints; ///< Points of a curve.
    static constexpr int kMaxChannels = 16;                     ///< Channels filtered at most.

    /** @return The frequency of a point of the curve, in hertz. */
    [[nodiscard]] static double frequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }

    /**
     * @brief Allocates for a layout (an invalid spec is ignored; channels
     *        over kMaxChannels pass through).
     * @param spec Rate and channels.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        fs_ = spec.sampleRate;
        channels_ = std::min(spec.numChannels, kMaxChannels);
        const int octaves = std::clamp(static_cast<int>(std::lround(std::log2(std::max(fs_, 44100.0) / 44100.0))), 0, 3);
        const int scale = 1 << octaves;

        // Everything grows with the rate; the factor stops at 16, and the
        // slow windows grow from there. By the block it shrinks with the rate
        // as well, to two octaves down, at the same hertz a bin.
        const int down = kByBlock ? std::clamp(-static_cast<int>(std::lround(std::log2(fs_ / 48000.0))), 0, 2) : 0;
        const int factor = kFactor >> down;
        factor_ = std::min(16, factor * scale);
        const int slowScale = scale * factor / factor_;
        quantum_ = kSlowQuantum * factor_;
        const double slowRate = fs_ / static_cast<double>(factor_);

        rate_.prepare(channels_, factor_);
        const int slowWindow = kSlowWindow * slowScale, window = (kWindow >> down) * scale;
        low_.prepare(slowRate, channels_, slowWindow, kSlowPadding, slowWindow / (kByBlock ? 4 : 1), kSlowSteps);
        high_.prepare(fs_, channels_, window, kPadding, window / 4, kSteps);

        const int lowSpan = low_.getLatency() * factor_;
        correctionDelay_ = 2 * rate_.half_ + lowSpan;
        const int needed = quantum_ + correctionDelay_ + high_.getLatency();
        // By the block, what is over the engine's own delay is lead for the high part.
        latency_ = kByBlock ? std::max(needed, (kBlockLatency * scale) >> down) : std::max(kLatency * scale, needed);

        align_.prepare(channels_, latency_ - needed);
        programme_.prepare(channels_, correctionDelay_);
        slowDry_.prepare(channels_, low_.getLatency());
        removed_.prepare(channels_, latency_ - quantum_);   // the output waits a quantum more in out_
        slow_.resize(channels_, kSlowQuantum);
        before_.resize(channels_, kSlowQuantum);
        original_.resize(channels_, quantum_);
        change_.resize(channels_, quantum_);
        in_.resize(channels_, quantum_);
        out_.resize(channels_, quantum_);

        // By the frame a curve is drawn once a kernel: between them nothing reads it.
        hopQuanta_ = std::max(1, low_.getHopSize() * factor_ / quantum_ * (kByBlock ? 1 : 2));

        // The high part takes the curve as it is drawn: held back to the low
        // part's, every move of the broad curve would come later than the
        // latency allows it.
        curvesKept_ = 1;
        const auto kept = static_cast<std::size_t>(curvesKept_) * static_cast<std::size_t>(kPoints);
        curve_.assign(static_cast<std::size_t>(channels_), std::vector<T>(static_cast<std::size_t>(kPoints), T(0)));
        past_.assign(static_cast<std::size_t>(channels_), std::vector<T>(kept, T(0)));
        recent_.assign(kByBlock ? 0u : static_cast<std::size_t>(channels_), std::vector<T>(static_cast<std::size_t>(kHoldCurves * kPoints), T(0)));
        held_.assign(kByBlock ? 0u : static_cast<std::size_t>(channels_), std::vector<T>(static_cast<std::size_t>(kPoints), T(0)));
        heldAge_.assign(kByBlock ? 0u : static_cast<std::size_t>(channels_), std::vector<std::uint8_t>(static_cast<std::size_t>(kPoints), kHoldCurves));

        // The crossover: well inside what the decimator keeps at any rate.
        const double upper = std::min(1800.0, 0.3 * slowRate);
        lowMap_.prepare(low_.getBinHz(), low_.getNumBins(), upper, true);
        highMap_.prepare(high_.getBinHz(), high_.getNumBins(), upper, false);
        split_.prepare(upper / (Split::kStep * high_.getBinHz()), low_.getNumBins(), (high_.getNumBins() - 1) / Split::kStep + 1);
        fine_.assign(static_cast<std::size_t>(channels_), std::vector<T>(static_cast<std::size_t>(split_.fineBins_), T(0)));
        pastBroad_.assign(static_cast<std::size_t>(channels_),
                          std::vector<T>(static_cast<std::size_t>(curvesKept_ * split_.broadBins_), T(0)));
        for (std::size_t ch = 0; ch < 2; ++ch)
        {
            playedLow_[ch] = std::vector<std::atomic<float>>(static_cast<std::size_t>(low_.getKernelLength()));
            playedHigh_[ch] = std::vector<std::atomic<float>>(static_cast<std::size_t>(high_.getKernelLength()));
            readLow_[ch].assign(static_cast<std::size_t>(low_.getKernelLength()), 0.0);
            readHigh_[ch].assign(static_cast<std::size_t>(high_.getKernelLength()), 0.0);
        }
        readCurve_.assign(static_cast<std::size_t>(kPoints), T(0));
        lowResponse_.prepare(low_.getTransformSize(), low_.getKernelLength(), low_.getBinHz(),
                             static_cast<double>(split_.fineBins_ - 1) * low_.getBinHz());
        highResponse_.prepare(high_.getTransformSize(), high_.getKernelLength(), high_.getBinHz(), 0.5 * fs_);
        prepared_ = true;
        reset();
    }

    /** @brief Clears the signal and the curves. */
    void reset() noexcept
    {
        if (!prepared_) return;
        rate_.reset();
        low_.reset();
        high_.reset();
        align_.reset();
        programme_.reset();
        slowDry_.reset();
        removed_.reset();
        in_.clear();
        out_.clear();
        for (auto& c : curve_) std::fill(c.begin(), c.end(), T(0));
        for (auto& p : past_) std::fill(p.begin(), p.end(), T(0));
        for (auto& p : recent_) std::fill(p.begin(), p.end(), T(0));
        recentAt_ = 0;
        for (auto& p : held_) std::fill(p.begin(), p.end(), T(0));
        for (auto& a : heldAge_) std::fill(a.begin(), a.end(), static_cast<std::uint8_t>(kHoldCurves));
        for (auto& f : fine_) std::fill(f.begin(), f.end(), T(0));
        for (auto& b : pastBroad_) std::fill(b.begin(), b.end(), T(0));
        filled_ = 0;
        quanta_ = 0;
        newest_ = 0;
        curveCount_ = 0;
        publishPlayed();
    }

    /** @brief One curve and one filter for every channel. */
    void setLinked(bool on) noexcept
    {
        linked_ = on;
        low_.setLinked(on);
        high_.setLinked(on);
    }
    /** @return Whether one curve serves every channel. */
    [[nodiscard]] bool isLinked() const noexcept { return linked_; }

    /**
     * @brief Cuts the kernels of the high part (all but the fine detail of
     *        the low end) to its window and tapers them over their whole
     *        length, from the next one designed (off by default: a kernel keeps
     *        the room the padding leaves, with only its tail tapered). A cut
     *        there then rings for a shorter time after the event, at the same
     *        depth; the fine detail of the low end keeps the length it takes.
     */
    void setShortKernel(bool on) noexcept
    {
        shortKernel_ = on;
        high_.setShortKernel(on);
    }
    /** @return Whether the kernels are short. */
    [[nodiscard]] bool isShortKernel() const noexcept { return shortKernel_; }

    /**
     * @brief By the frame, how each point holds its moves before the
     *        kernels are designed (DetailHold::Largest by default). By the block
     *        there is no hold.
     */
    void setHold(DetailHold hold) noexcept { hold_ = hold; }
    /** @return How the points hold their moves. */
    [[nodiscard]] DetailHold getHold() const noexcept { return hold_; }

    /** @brief Outputs what is taken out: the programme, delayed, less the filtered one. */
    void setRemovalListen(bool on) noexcept { removalListen_ = on; }
    /** @return Whether the output is what is taken out. */
    [[nodiscard]] bool isRemovalListen() const noexcept { return removalListen_; }

    /** @return The latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }
    /** @return Samples from one curve to the next. */
    [[nodiscard]] int getHopSize() const noexcept { return hopQuanta_ * quantum_; }

    /** @return Curves designed since reset(), modulo 2^30: a display follows this. */
    [[nodiscard]] int getCurveCount() const noexcept { return curveCount_; }
    /**
     * @param shownAt The curve count a display was last drawn at.
     * @return Whether the next drawing is due: 21 ms later (four curves
     *         by the block, two by the frame), some 47 a second, as often
     *         as a screen shows it, whatever the blocks.
     */
    [[nodiscard]] bool isDisplayDue(int shownAt) const noexcept { return ((curveCount_ - shownAt) & 0x3FFFFFFF) >= (kByBlock ? kDisplayCurves : kDisplayCurves / 2); }

    /**
     * @brief Hands the last kernels of both parts to a display (audio
     *        thread): a copy of them, for playedValue() and playedExtreme().
     * @param channels The channels processed (one shows on both).
     */
    void publishPlayed(int channels = 2) noexcept
    {
        if (!prepared_) return;
        const unsigned version = playedVersion_.load(std::memory_order_relaxed);
        playedVersion_.store(version + 1u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        const bool one = linked_ || std::min(channels, channels_) < 2;
        playedOne_.store(one, std::memory_order_relaxed);
        handOverChannel<0>();
        if (!one) handOverChannel<1>();
        playedVersion_.store(version + 2u, std::memory_order_release);
    }

    /**
     * @return A point of the response of the kernels last handed over, in
     *         decibels (any thread; made by the first read after they
     *         change). Channel 0 or 1 (the same when linked); others read 0.
     */
    [[nodiscard]] T playedValue(int channel, int point) const noexcept
    {
        refreshPlayed();
        return played_.value(channel, point);
    }

    /** @return The value of the largest magnitude of that response, over both channels (any thread). */
    [[nodiscard]] T playedExtreme() const noexcept
    {
        refreshPlayed();
        return played_.extreme();
    }

    /**
     * @brief The response being applied now: publishPlayed() and the points
     *        of a channel (for the owner's thread, not while a display reads).
     * @param channel 0 or 1 (0 when linked); others give zeros.
     * @param played  kPoints values, in decibels.
     */
    void getPlayedCurve(int channel, T* played) noexcept
    {
        std::fill_n(played, kPoints, T(0));
        if (!prepared_ || channel < 0 || channel >= 2) return;
        publishPlayed();
        for (int i = 0; i < kPoints; ++i) played[i] = playedValue(channel, i);
    }

    /**
     * @brief Filters a block in place by curves that do not read the signal.
     *
     * @tparam Design `void(int channel, T* curveDb)`, as below.
     * @param buffer The audio.
     * @param design The callback, run once a hop.
     */
    template <typename Design>
    void processBlock(AudioBufferView<T> buffer, Design&& design) noexcept
    {
        processBlock(buffer, [](int, int) noexcept {}, design);
    }

    /**
     * @brief Filters a block in place.
     *
     * @tparam Feed   `void(int offset, int count)`: the samples [offset,
     *         offset + count) of the block, still untouched, are about to be
     *         taken; all of a hop's have been given when its curve is asked
     *         for. Must not throw.
     * @tparam Design `void(int channel, T* curveDb)`: writes the kPoints
     *         gains wanted now, in decibels (held from one hop to the next).
     *         The channel is 0 when linked. Must not throw.
     * @param buffer The audio.
     * @param feed   The callback that reads the block (or a key beside it).
     * @param design The callback, run once at the end of each hop.
     */
    template <typename Feed, typename Design>
    void processBlock(AudioBufferView<T> buffer, Feed&& feed, Design&& design) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        const int channels = std::min(buffer.getNumChannels(), channels_);
        const int samples = buffer.getNumSamples();
        if (channels <= 0) return;
        int done = 0;
        while (done < samples)
        {
            const int count = std::min(samples - done, quantum_ - filled_);
            feed(done, count);
            for (int ch = 0; ch < channels; ++ch)
            {
                T* x = buffer.getChannel(ch) + done;
                std::copy_n(x, count, in_.getChannel(ch) + filled_);
                std::copy_n(out_.getChannel(ch) + filled_, count, x);
            }
            filled_ += count;
            done += count;
            if (filled_ == quantum_)
            {
                filled_ = 0;
                runQuantum(channels, design);
            }
        }
    }

private:
    /// A whole-sample delay of every channel.
    struct Line
    {
        void prepare(int channels, int length)
        {
            length_ = std::max(0, length);
            data_.assign(static_cast<std::size_t>(channels), std::vector<T>(static_cast<std::size_t>(length_), T(0)));
            at_.assign(static_cast<std::size_t>(channels), 0);
        }

        void reset() noexcept
        {
            for (auto& d : data_) std::fill(d.begin(), d.end(), T(0));
            std::fill(at_.begin(), at_.end(), 0);
        }

        void process(int channel, T* x, int count) noexcept
        {
            if (length_ == 0) return;
            const auto c = static_cast<std::size_t>(channel);
            T* d = data_[c].data();
            int at = at_[c];
            // By spans of the ring: each sample out is the one in `length_` ago.
            while (count > 0)
            {
                const int run = std::min(count, length_ - at);
                std::swap_ranges(x, x + run, d + at);
                x += run;
                count -= run;
                at += run;
                if (at == length_) at = 0;
            }
            at_[c] = at;
        }

        std::vector<std::vector<T>> data_;
        std::vector<int> at_;
        int length_ = 0;
    };

    /// Down and up by the factor through one linear-phase low-pass, 100 dB
    /// down from the slow Nyquist. Together they delay by exactly twice half_.
    struct RateChange
    {
        void prepare(int channels, int factor)
        {
            constexpr double depth = 100.0, passEdge = 0.35; // of the slow rate
            factor_ = factor;
            const double transition = (0.5 - passEdge) / factor;
            half_ = static_cast<int>(std::ceil(0.5 * (depth - 7.95) / (14.36 * transition)));
            length_ = 2 * half_ + 1;
            perPhase_ = (length_ + factor_ - 1) / factor_;
            const auto h = FIRDesign<double>::lowPass(1.0, 0.5 * (0.5 + passEdge) / factor, length_,
                                                      FIRDesign<double>::estimateKaiserBeta(depth));
            taps_.resize(static_cast<std::size_t>(length_));
            for (int i = 0; i < length_; ++i)
                taps_[static_cast<std::size_t>(i)] = static_cast<T>(h[static_cast<std::size_t>(i)]);
            // Each phase of the interpolator, oldest sample first.
            phases_.assign(static_cast<std::size_t>(perPhase_ * factor_), T(0));
            for (int r = 0; r < factor_; ++r)
                for (int i = 0; i < perPhase_; ++i)
                {
                    const int n = (perPhase_ - 1 - i) * factor_ + r;
                    if (n < length_)
                        phases_[static_cast<std::size_t>(r * perPhase_ + i)] = static_cast<T>(factor_ * h[static_cast<std::size_t>(n)]);
                }
            const auto count = static_cast<std::size_t>(channels);
            down_.assign(count, std::vector<T>(static_cast<std::size_t>(length_ - 1 + factor_ * kSlowQuantum), T(0)));
            up_.assign(count, std::vector<T>(static_cast<std::size_t>(perPhase_ - 1 + kSlowQuantum), T(0)));
        }

        void reset() noexcept
        {
            for (auto& d : down_) std::fill(d.begin(), d.end(), T(0));
            for (auto& u : up_) std::fill(u.begin(), u.end(), T(0));
        }

        void decimate(int channel, const T* fast, T* slow) noexcept
        {
            T* past = down_[static_cast<std::size_t>(channel)].data();
            const int count = factor_ * kSlowQuantum;
            std::copy_n(fast, count, past + length_ - 1);
            // The kernel is symmetric, so it correlates as it is.
            for (int j = 0; j < kSlowQuantum; ++j)
                slow[j] = simd::dotProductT(past + j * factor_, taps_.data(), length_);
            std::copy(past + count, past + count + length_ - 1, past);
        }

        void interpolate(int channel, const T* slow, T* fast) noexcept
        {
            T* past = up_[static_cast<std::size_t>(channel)].data();
            std::copy_n(slow, kSlowQuantum, past + perPhase_ - 1);
            for (int j = 0; j < kSlowQuantum; ++j)
                for (int r = 0; r < factor_; ++r)
                    fast[j * factor_ + r] = simd::dotProductT(past + j, phases_.data() + r * perPhase_, perPhase_);
            std::copy(past + kSlowQuantum, past + kSlowQuantum + perPhase_ - 1, past);
        }

        std::vector<T> taps_, phases_;
        std::vector<std::vector<T>> down_, up_;
        int factor_ = kFactor, half_ = 0, length_ = 1, perPhase_ = 1;
    };

    /// The magnitude of a kernel at the points. Centred on the time origin,
    /// the kernel fills under half of twice its transform: its complex
    /// spectrum is then smooth enough to be read between those bins by a
    /// short windowed sinc (its power would not be: that spans twice the time).
    struct Response
    {
        void prepare(int transform, int length, double binHz, double limitHz)
        {
            length_ = length;
            size_ = 2 * transform;
            fft_ = std::make_unique<FFTReal<T>>(static_cast<std::size_t>(size_));
            time_.assign(static_cast<std::size_t>(size_), T(0));
            spectrum_.assign(static_cast<std::size_t>(size_ + 2), T(0));
            places_.clear();
            // Kaiser: flat to 0.001 over what the centred kernel holds (a quarter of the band).
            const double half = 0.5 * kTaps;
            const auto bessel = [](double x) {
                double sum = 1.0, term = 1.0;
                for (int k = 1; k < 20; ++k)
                {
                    term *= 0.25 * x * x / static_cast<double>(k * k);
                    sum += term;
                }
                return sum;
            };
            for (int i = 0; i < kPoints; ++i)
            {
                const double hz = frequency(i);
                if (hz > limitHz) break;
                const double at = std::min(2.0 * hz / binHz, static_cast<double>(size_ / 2 - kTaps / 2 - 1));
                Place p;
                const int under = static_cast<int>(at);
                p.first = under - kTaps / 2 + 1;
                double weight[kTaps] {}, sum = 0.0;
                for (int j = 0; j < kTaps; ++j)
                {
                    const double away = static_cast<double>(p.first + j) - at, x = std::numbers::pi * away;
                    const double window = bessel(5.5 * std::sqrt(std::max(0.0, 1.0 - away * away / (half * half))));
                    weight[j] = window * (std::abs(x) < 1e-9 ? 1.0 : std::sin(x) / x);
                    sum += weight[j];
                }
                for (int j = 0; j < kTaps; ++j) p.weight[j] = static_cast<T>(weight[j] / sum);
                places_.push_back(p);
            }
        }

        /// Adds the decibels of a kernel (none: one everywhere) at the points.
        void add(const double* kernel, T* db) noexcept
        {
            if (kernel == nullptr) return;
            const int middle = length_ / 2;
            T* time = time_.data();
            for (int i = middle; i < length_; ++i) time[i - middle] = static_cast<T>(kernel[i]);
            std::fill(time + (length_ - middle), time + (size_ - middle), T(0));
            for (int i = 0; i < middle; ++i) time[size_ - middle + i] = static_cast<T>(kernel[i]);
            fft_->forward(time, spectrum_.data());
            const T* s = spectrum_.data();
            const std::size_t count = places_.size();
            std::size_t i = 0;
            // By DC the taps reach below bin 0 and read its mirror (the conjugate).
            for (; i < count && places_[i].first < 0; ++i)
            {
                const Place& p = places_[i];
                T re = T(0), im = T(0);
                for (int j = 0; j < kTaps; ++j)
                {
                    const int bin = p.first + j;
                    const int k = bin < 0 ? -bin : bin;
                    re += p.weight[j] * s[2 * k];
                    im += (bin < 0 ? -p.weight[j] : p.weight[j]) * s[2 * k + 1];
                }
                db[i] += T(4.3429448190325183) * std::log(std::max(re * re + im * im, T(1e-12)));
            }
            for (; i < count; ++i)
            {
                const Place& p = places_[i];
                const T* bins = s + 2 * p.first;
                T re = T(0), im = T(0);
                for (int j = 0; j < kTaps; ++j)
                {
                    re += p.weight[j] * bins[2 * j];
                    im += p.weight[j] * bins[2 * j + 1];
                }
                db[i] += T(4.3429448190325183) * std::log(std::max(re * re + im * im, T(1e-12)));
            }
        }

        static constexpr int kTaps = 8;

        struct Place
        {
            T weight[kTaps] {};
            int first = 0;
        };

        std::unique_ptr<FFTReal<T>> fft_;
        std::vector<T> time_, spectrum_;
        std::vector<Place> places_;
        int length_ = 0, size_ = 0;
    };

    /// The share of the curve that one filter plays, on that filter's bins.
    struct BinMap
    {
        void prepare(double binHz, int bins, double upper, bool isLow)
        {
            const auto size = static_cast<std::size_t>(bins);
            point_.assign(size, 0);
            part_.assign(size, T(0));
            share_.assign(size, T(0));
            const double lastPoint = static_cast<double>(kPoints - 1), span = std::log(1000.0);
            for (std::size_t k = 0; k < size; ++k)
            {
                const double hz = std::max(static_cast<double>(k) * binHz, 1.0);
                const double at = std::clamp(lastPoint * std::log(hz / 20.0) / span, 0.0, lastPoint);
                const int under = std::min(static_cast<int>(at), kPoints - 2);
                point_[k] = under;
                part_[k] = static_cast<T>(at - static_cast<double>(under));
                // All low under half the upper edge, none from the edge on.
                const double over = std::clamp(1.0 + std::log2(hz / upper), 0.0, 1.0);
                const double low = 0.5 + 0.5 * std::cos(std::numbers::pi * over);
                share_[k] = static_cast<T>(isLow ? low : 1.0 - low);
            }
        }

        /// The natural logarithm of the gain of each bin.
        void write(const T* curveDb, T* logGain, int bins) const noexcept
        {
            constexpr T perDb = T(0.11512925464970229); // ln(10) / 20
            const int count = std::min(bins, static_cast<int>(share_.size()));
            for (int k = 0; k < count; ++k)
            {
                const auto u = static_cast<std::size_t>(k);
                const auto p = static_cast<std::size_t>(point_[u]);
                logGain[k] = perDb * share_[u] * (curveDb[p] + part_[u] * (curveDb[p + 1] - curveDb[p]));
            }
        }

        std::vector<int> point_;
        std::vector<T> part_, share_;
    };

    /// The low share of a curve in two: its broad shape on the high part's
    /// bins, and the fine detail left on the low part's.
    struct Split
    {
        void prepare(double upperBin, int lowBins, int highBins)
        {
            broadBins_ = std::min(highBins, static_cast<int>(std::ceil(upperBin)) + kReach + 1);
            fineBins_ = std::min(lowBins, kRatio * (broadBins_ + 1) + 1);
            binned_.assign(static_cast<std::size_t>(broadBins_ + 2 * kReach), 0.0);
            around_.assign(static_cast<std::size_t>(broadBins_ + 4), 0.0);
            // Three boxes of five bins, over their whole weight.
            double box[kWidth] { 1.0 };
            for (int pass = 0, length = 1; pass < 3; ++pass, length += 4)
                for (int i = length + 3; i >= 0; --i)
                {
                    double sum = 0.0;
                    for (int j = std::max(0, i - 4); j <= std::min(i, length - 1); ++j) sum += box[j];
                    box[i] = sum;
                }
            for (int i = 0; i < kWidth; ++i) kernel_[i] = box[i] / 125.0;
            // Catmull-Rom between two bins of the high part.
            for (int i = 0; i < kRatio; ++i)
            {
                const double t = static_cast<double>(i) / kRatio;
                between_[i][0] = 0.5 * t * ((2.0 - t) * t - 1.0);
                between_[i][1] = 0.5 * (t * t * (3.0 * t - 5.0) + 2.0);
                between_[i][2] = 0.5 * t * ((4.0 - 3.0 * t) * t + 1.0);
                between_[i][3] = 0.5 * t * t * (t - 1.0);
            }
        }

        /// Adds a broad shape to the high part's log gains, read between its
        /// bins as run() takes it from the low part's: the two make the curve.
        void addBroad(const T* broad, T* gain, int bins) const noexcept
        {
            const auto at = [&](int j) noexcept {
                const int k = j < 0 ? -j : j;
                return k < broadBins_ ? static_cast<double>(broad[k]) : 0.0;
            };
            const int count = std::min(bins, kStep * (broadBins_ + 1));
            for (int k = 0; k < count; ++k)
            {
                const int j = k / kStep;
                const double* w = between_[(k % kStep) * (kRatio / kStep)];
                gain[k] += static_cast<T>(w[0] * at(j - 1) + w[1] * at(j) + w[2] * at(j + 1) + w[3] * at(j + 2));
            }
        }

        /// fine: the low share's log gains on the low bins on entry, the detail left on return.
        void run(T* fine, T* broad) noexcept
        {
            // Onto the high part's bins by areas, with the spectrum's mirror at zero.
            double* binned = binned_.data() + kReach;
            for (int k = 0; k < broadBins_ + kReach; ++k)
            {
                double sum = 0.0;
                for (int j = -kRatio / 2; j <= kRatio / 2; ++j)
                {
                    const int from = kRatio * k + j, at = from < 0 ? -from : from;
                    if (at < fineBins_) sum += (j == -kRatio / 2 || j == kRatio / 2 ? 0.5 : 1.0) * static_cast<double>(fine[at]);
                }
                binned[k] = sum / kRatio;
            }
            for (int k = 1; k <= kReach; ++k) binned[-k] = binned[k];

            double* smooth = around_.data() + 1;
            for (int k = 0; k < broadBins_; ++k)
            {
                double sum = 0.0;
                for (int i = 0; i < kWidth; ++i) sum += kernel_[i] * binned[k + i - kReach];
                smooth[k] = sum;
                broad[k] = static_cast<T>(sum);
            }
            smooth[-1] = smooth[1];
            for (int k = broadBins_; k < broadBins_ + 3; ++k) smooth[k] = 0.0;

            for (int j = 0; j < fineBins_; ++j)
            {
                const double* s = smooth + std::min(j / kRatio, broadBins_);
                const double* w = between_[j % kRatio];
                fine[j] -= static_cast<T>(w[0] * s[-1] + w[1] * s[0] + w[2] * s[1] + w[3] * s[2]);
            }
        }

        static constexpr int kRatio = 8;   ///< Bins of the low part in one of the broad shape's, at every rate.
        static constexpr int kStep = 2;    ///< And of the high part.
        static constexpr int kReach = 6;   ///< The smoothing's, each side, in bins of the broad shape.
        static constexpr int kWidth = 2 * kReach + 1;

        int broadBins_ = 0, fineBins_ = 0;
        double kernel_[kWidth] {};
        double between_[kRatio][4] {};
        std::vector<double> binned_, around_;
    };

    /// The newest curve of a channel as setHold() asks. Moves are compared
    /// by their size, whatever their sign. DetailHold::Largest: each point at its
    /// largest of the last kHoldCurves curves. DetailHold::Reach: a point that
    /// moves further than it was (by more than kHoldStep) keeps that move
    /// for kHoldCurves curves and then follows the curve.
    void holdMoves(std::size_t c) noexcept
    {
        T* now = curve_[c].data();
        T* past = recent_[c].data();
        T* held = held_[c].data();
        std::uint8_t* age = heldAge_[c].data();
        std::copy_n(now, kPoints, past + recentAt_ * kPoints);
        if (hold_ != DetailHold::Reach)
        {
            if (hold_ == DetailHold::Largest)
                for (int h = 0; h < kHoldCurves; ++h)
                    for (int i = 0; i < kPoints; ++i)
                        if (std::abs(past[h * kPoints + i]) > std::abs(now[i])) now[i] = past[h * kPoints + i];
            std::copy_n(now, kPoints, held);
            std::fill_n(age, kPoints, kHoldCurves);
            return;
        }
        for (int i = 0; i < kPoints; ++i)
        {
            const T size = std::abs(now[i]), kept = std::abs(held[i]);
            if (size > kept)
            {
                if (size > kept + T(kHoldStep)) age[i] = 0;
                held[i] = now[i];
            }
            if (age[i] < kHoldCurves)
            {
                now[i] = held[i];
                ++age[i];
            }
            else
                held[i] = now[i];
        }
    }

    template <typename Design>
    void runQuantum(int channels, Design& design) noexcept
    {
        // The last quantum of a hop: the low filter takes its frame in it.
        if (quanta_ == hopQuanta_ - 1)
        {
            if constexpr (!kByBlock) recentAt_ = (recentAt_ + 1) % kHoldCurves;
            newest_ = (newest_ + 1) % curvesKept_;
            curveCount_ = (curveCount_ + 1) & 0x3FFFFFFF;
            const int curves = linked_ ? 1 : channels;
            for (int ch = 0; ch < curves; ++ch)
            {
                const auto c = static_cast<std::size_t>(ch);
                design(ch, curve_[c].data());
                if constexpr (!kByBlock) holdMoves(c);
                std::copy_n(curve_[c].data(), kPoints, past_[c].data() + newest_ * kPoints);
                lowMap_.write(curve_[c].data(), fine_[c].data(), split_.fineBins_);
                split_.run(fine_[c].data(), pastBroad_[c].data() + newest_ * split_.broadBins_);
            }
        }
        quanta_ = (quanta_ + 1) % hopQuanta_;

        T* slow[kMaxChannels] {};
        T* out[kMaxChannels] {};
        for (int ch = 0; ch < channels; ++ch)
        {
            T* in = in_.getChannel(ch);
            slow[ch] = slow_.getChannel(ch);
            out[ch] = out_.getChannel(ch);
            std::copy_n(in, quantum_, original_.getChannel(ch));
            removed_.process(ch, original_.getChannel(ch), quantum_);
            align_.process(ch, in, quantum_);
            rate_.decimate(ch, in, slow[ch]);
            std::copy_n(slow[ch], kSlowQuantum, before_.getChannel(ch));
            slowDry_.process(ch, before_.getChannel(ch), kSlowQuantum);
        }

        // The low filter and what it changed.
        low_.processBlock(AudioBufferView<T>(slow, channels, kSlowQuantum), [this](int ch, const T*, T* gain, int bins) noexcept {
            std::copy_n(fine_[curveIndex(ch)].data(), std::min(bins, split_.fineBins_), gain);
        });

        // That change onto the programme, then the high filter.
        for (int ch = 0; ch < channels; ++ch)
        {
            T* in = in_.getChannel(ch);
            const T* before = before_.getChannel(ch);
            for (int i = 0; i < kSlowQuantum; ++i) slow[ch][i] -= before[i];
            T* added = change_.getChannel(ch);
            rate_.interpolate(ch, slow[ch], added);
            programme_.process(ch, in, quantum_);
            for (int i = 0; i < quantum_; ++i) out[ch][i] = in[i] + added[i];
        }
        high_.processBlock(AudioBufferView<T>(out, channels, quantum_), [this](int ch, const T*, T* gain, int bins) noexcept {
            const int oldest = (newest_ + 1) % curvesKept_;
            const auto c = curveIndex(ch);
            highMap_.write(past_[c].data() + oldest * kPoints, gain, bins);
            split_.addBroad(pastBroad_[c].data() + oldest * split_.broadBins_, gain, bins);
        });

        if (removalListen_)
            for (int ch = 0; ch < channels; ++ch)
            {
                const T* original = original_.getChannel(ch);
                for (int i = 0; i < quantum_; ++i) out[ch][i] = original[i] - out[ch][i];
            }
    }

    /// A kernel (none where flat) into what a display reads.
    static void handOver(const double* kernel, std::vector<std::atomic<float>>& to, std::atomic<bool>& flat) noexcept
    {
        flat.store(kernel == nullptr, std::memory_order_relaxed);
        if (kernel == nullptr) return;
        for (std::size_t i = 0; i < to.size(); ++i) to[i].store(static_cast<float>(kernel[i]), std::memory_order_relaxed);
    }

    /// A channel's kernels of both parts into what a display reads.
    template <std::size_t Channel>
    void handOverChannel() noexcept
    {
        static_assert(Channel < 2);
        handOver(low_.getKeptKernel(static_cast<int>(Channel)), playedLow_[Channel], playedLowFlat_[Channel]);
        handOver(high_.getKeptKernel(static_cast<int>(Channel)), playedHigh_[Channel], playedHighFlat_[Channel]);
    }

    /// The response of the kernels last handed over, into played_, if they are new.
    void refreshPlayed() const noexcept
    {
        const unsigned now = playedVersion_.load(std::memory_order_acquire);
        if ((now & 1u) == 0 && now == shownVersion_.load(std::memory_order_acquire)) return;
        std::lock_guard<SpinLock> lock(readerLock_);
        for (;;)
        {
            const unsigned version = playedVersion_.load(std::memory_order_acquire);
            if ((version & 1u) != 0)
            {
                std::this_thread::yield();
                continue;
            }
            if (version == shownVersion_.load(std::memory_order_relaxed)) return;
            const bool one = playedOne_.load(std::memory_order_relaxed);
            bool lowFlat[2] {}, highFlat[2] {};
            for (std::size_t ch = 0; ch < (one ? 1u : 2u); ++ch)
            {
                lowFlat[ch] = playedLowFlat_[ch].load(std::memory_order_relaxed);
                highFlat[ch] = playedHighFlat_[ch].load(std::memory_order_relaxed);
                if (!lowFlat[ch])
                    for (std::size_t i = 0; i < readLow_[ch].size(); ++i) readLow_[ch][i] = static_cast<double>(playedLow_[ch][i].load(std::memory_order_relaxed));
                if (!highFlat[ch])
                    for (std::size_t i = 0; i < readHigh_[ch].size(); ++i) readHigh_[ch][i] = static_cast<double>(playedHigh_[ch][i].load(std::memory_order_relaxed));
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (playedVersion_.load(std::memory_order_relaxed) != version) continue;
            for (std::size_t ch = 0; ch < 2; ++ch)
            {
                const std::size_t from = one ? 0u : ch;
                T* curve = readCurve_.data();
                std::fill_n(curve, kPoints, T(0));
                lowResponse_.add(lowFlat[from] ? nullptr : readLow_[from].data(), curve);
                highResponse_.add(highFlat[from] ? nullptr : readHigh_[from].data(), curve);
                played_.publishPoints(static_cast<int>(ch), curve);
            }
            shownVersion_.store(version, std::memory_order_release);
            return;
        }
    }

    [[nodiscard]] std::size_t curveIndex(int channel) const noexcept
    {
        return static_cast<std::size_t>(linked_ ? 0 : std::max(channel, 0));
    }

    static constexpr int kFactor = 8;        ///< Decimation of the low part at 44.1 and 48 kHz.
    static constexpr int kSlowQuantum = 4;   ///< Slow samples worked on at once.
    static constexpr int kSlowWindow = 64;   ///< Window of the low filter, in slow samples.
    static constexpr int kSlowPadding = 32;  ///< Its transform over its window: 2.9 Hz a bin, a third of a second of response.
    static constexpr int kWindow = 1024;     ///< Window of the high filter (Split::kRatio and kStep hold the two parts' bins together).
    static constexpr int kSlowSteps = kByBlock ? 4 : 1; ///< Blocks of the low part from a kernel to the next (11 ms).
    static constexpr int kSteps = kByBlock ? 1 : 2; ///< And of the high part (5 ms by the block, 11 by the frame).
    static constexpr int kBlockLatency = 774; ///< The latency kept by the block at 44.1 and 48 kHz.
    static constexpr std::uint8_t kHoldCurves = 3; ///< By the frame: curves a new reach is held for (32 ms at 48 kHz).
    static constexpr double kHoldStep = 0.5;  ///< By the frame: what a reach passes the held move by, in dB, to be held anew.
    static constexpr int kPadding = 4;       ///< Its transform over its window: 11.7 Hz a bin, 64 ms of response.
    static constexpr int kLatency = 2048;    ///< The latency kept at 44.1 and 48 kHz.
    static constexpr int kDisplayCurves = 4; ///< Curves from one drawing of a display to the next.

    double fs_ = 48000.0;
    int channels_ = 0, factor_ = kFactor, quantum_ = kFactor * kSlowQuantum, latency_ = 0;
    int correctionDelay_ = 0, hopQuanta_ = 1, quanta_ = 0, filled_ = 0;
    int curvesKept_ = 1, newest_ = 0, curveCount_ = 0;
    bool prepared_ = false, linked_ = false, removalListen_ = false, shortKernel_ = false;
    RateChange rate_;
    Part low_, high_;
    std::vector<std::vector<T>> recent_; ///< By the frame: the last kHoldCurves curves of each channel, as drawn.
    int recentAt_ = 0;
    std::vector<std::vector<T>> held_;   ///< By the frame: the move each point holds, per channel.
    std::vector<std::vector<std::uint8_t>> heldAge_; ///< And the curves it has been held for.
    DetailHold hold_ = DetailHold::Largest;
    BinMap lowMap_, highMap_;
    // The display: kernels handed over by the audio thread, and what its reader makes of them.
    std::atomic<unsigned> playedVersion_ { 0 };          ///< Odd while a copy is being made.
    std::atomic<bool> playedOne_ { true };               ///< One response for both channels.
    std::array<std::vector<std::atomic<float>>, 2> playedLow_, playedHigh_;
    std::array<std::atomic<bool>, 2> playedLowFlat_ { true, true }, playedHighFlat_ { true, true };
    mutable SpinLock readerLock_;
    mutable std::atomic<unsigned> shownVersion_ { ~0u };  ///< The copy played_ was made from.
    mutable std::array<std::vector<double>, 2> readLow_, readHigh_;
    mutable std::vector<T> readCurve_;
    mutable Response lowResponse_, highResponse_;
    mutable SpectralReadout<T> played_;
    Split split_;
    std::vector<std::vector<T>> fine_, pastBroad_; ///< The detail for the low part; the broad shapes kept with past_.
    Line align_, programme_, slowDry_, removed_;
    AudioBuffer<T, kMaxChannels> slow_, before_, original_, change_, in_, out_;
    std::vector<std::vector<T>> curve_; ///< The newest curve of each channel.
    std::vector<std::vector<T>> past_;  ///< Its last curvesKept_ curves, the oldest for the high filter.
};

} // namespace dspark
