// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TimeStretch.h
 * @brief Phase-vocoder time stretching: change tempo without changing pitch.
 *
 * A phase vocoder analyses the signal at one hop and re-synthesises it at
 * another. Reading the input every `Ra` samples and writing the output every
 * `Rs` samples stretches the signal by `Rs / Ra` while every partial keeps
 * its frequency, because each frame's phases are advanced by the
 * instantaneous frequency measured between the analysis frames rather than
 * simply repeated. This class is that engine with nothing after it. (Its
 * sibling `Effects/PitchShifter.h` is the same engine plus a resampler that
 * squeezes the stretched stream back to the original duration, which is why
 * that one moves pitch and this one does not.)
 *
 * How it keeps its quality:
 *
 * - **Identity phase locking** (Laroche & Dolson, "Improved phase vocoder
 *   time-scale modification of audio", IEEE Trans. Speech and Audio
 *   Processing 7(3), 1999): the spectral peaks are found each frame and
 *   every bin in a peak's region of influence is rotated by the SAME phase
 *   increment as its peak, so the partial stays vertically coherent instead
 *   of dissolving into the hollow, chorused sound a plain vocoder makes.
 * - **Onset detection and a transient-locked hop.** Attacks are found with
 *   half-wave-rectified spectral flux over a log-frequency filterbank, which
 *   sees a strike over a sustained bed where a broadband energy test cannot:
 *   on a drum strike over a 110 Hz harmonic bed the weakest onset frame
 *   carries 14.46 dB more flux than the median frame and only 0.13 dB more
 *   energy. Across a detected attack the synthesis phases are reset to the
 *   analysis phases AND the analysis hop is held at the synthesis hop for
 *   one whole window, so every frame that sees the strike places it at the
 *   same offset. Without that hold, a stretch spreads one strike over
 *   `fftSize * |ratio - 1|` samples and the attack audibly doubles; with it,
 *   the strike measures as concentrated as the input's own, at every frame
 *   size from 256 to 4096 and every ratio. The input the hold does not
 *   consume is repaid over the following hops, so the timeline stays exact:
 *   measured over 30 s, no onset lands more than 1.71 ms from where the
 *   stretched timeline puts it, at 120, 480 and 960 strikes per minute
 *   alike.
 *
 * Peaks are grouped by the plain magnitude valley between them; no
 * perceptual band table is involved anywhere in this file.
 *
 * Everything is implemented here from those papers. No third-party code.
 *
 * Hops: the synthesis hop is fixed at `fftSize / 4` (75% overlap, which the
 * sqrt-Hann analysis and synthesis windows overlap-add to a constant), and
 * the analysis hop is `round(Rs / ratio)` carried through a fractional
 * accumulator, so the realised stretch is exactly the requested ratio for
 * arbitrary ratios and never drifts, however long the stream runs. The
 * transient hold above is the one exception, and it borrows rather than
 * skips: what it does not consume it repays.
 *
 * Latency and the two processing paths:
 *
 * - `processBlock()` is the streaming path: real-time safe, no allocation,
 *   no lock, no throw, latency `fftSize` samples as reported by
 *   `getLatency()`.
 * - `process()` is the offline path for a whole signal. It returns a buffer
 *   of `round(inputLength * ratio)` samples aligned with the input - the
 *   algorithmic delay is removed inside, so there is nothing to compensate
 *   and the effective latency of that path is zero.
 *
 * **What streaming can and cannot do.** Stretching changes duration: over
 * `n` input samples the stretched signal is `ratio * n` samples long. A block
 * call that is handed `n` samples and must return `n` samples cannot make
 * that difference disappear, so this class states plainly where it goes.
 * `processBlock()` always accepts the whole block into a bounded input queue
 * sized in prepare(), and always emits `n` samples of the stretched stream,
 * in order, never repeating or dropping one:
 *
 * - `ratio == 1`: exact, indefinitely. The queue level does not move and the
 *   output is the input delayed by `getLatency()`.
 * - `ratio > 1` (slower): the stretched stream is longer than the input, so
 *   the caller supplies more input than the stretch consumes and the surplus
 *   accumulates. The queue absorbs it for a bounded time - about 4.5 s at
 *   ratio 1.081, 48 kHz and the default 2048 frame (queue capacity divided
 *   by the surplus rate `1 - 1/ratio`) - after which the newest input is
 *   refused and the stream splices.
 * - `ratio < 1` (faster): the stretch needs more input than the block brings.
 *   The class never invents material: it emits the stretched stream as far as
 *   the input it has allows and waits, in silence, for exactly as many
 *   samples as are missing before going on. Everything the caller feeds still
 *   comes out, in order; what it cannot do is come out sooner than it
 *   arrived.
 *
 * Input is taken in step with output, one sample in for one sample out, so
 * every one of those decisions falls at a position in the stream rather than
 * at a block boundary: the output is bit-identical however the host chops the
 * stream, at every ratio, including while the queue is saturated.
 *
 * If you have the whole signal, use `process()`: it is exact at every ratio
 * with none of the above. If you are stretching a live stream by a lot, the
 * input has to arrive at `1 / ratio` the output rate, which means a
 * variable-rate source feeding the block - a file player reading faster or
 * slower - not a fixed host block of the same size in and out.
 *
 * Frame size and sample rate: `fftSize` is a quality/latency trade-off, not
 * a tuning constant. Its frequency resolution is `sampleRate / fftSize` Hz
 * and its time resolution `fftSize / sampleRate` s, so the same fftSize is a
 * different trade at a different rate: 2048 spans 42.7 ms at 48 kHz and
 * 21.3 ms at 96 kHz. Low material wants the longer window (partials must be
 * resolved into separate bins), percussive material the shorter one -
 * though the transient hold above is what actually protects a strike, and it
 * measures the same at every frame size from 256 to 4096. One consequence of
 * the frame size is worth stating: the hold needs an unlocked hop between
 * strikes to repay in, so the strike density it can serve scales as
 * `sampleRate / fftSize`. At 2048 and 48 kHz the 1.71 ms figure above holds
 * measured from 100 to 1040 strikes per minute and breaks down past about
 * 1060, where per-onset timing degrades to roughly 3.9 ms.
 *
 * Threading (single control thread + single audio thread):
 * - `processBlock()`, `reset()`: audio thread (the stream owner).
 * - `setTimeRatio()`, `setTempoChangePercent()`, `setTransientPreserve()`,
 *   `setPhaseLock()`: control thread. Each one
 *   publishes the WHOLE parameter set as a unit, so a gesture that moves two
 *   of them at once can never be adopted half-way; the audio thread picks the
 *   set up at the next frame boundary through a bounded read that gives up
 *   and keeps the set already in use rather than ever waiting on the control
 *   thread. Non-finite values are ignored.
 * - `prepare()`, `setState()`, `process()`: setup thread only; they allocate
 *   and must not run concurrently with processing.
 * - `getState()` and the getters read the control-side values and are safe
 *   from any thread.
 *
 * Dependencies: Effects/detail/PhaseVocoderEngine.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h, Core/StateBlob.h.
 *
 * @code
 * dspark::TimeStretch<float> ts;
 * ts.prepare(spec);                       // 2048-sample frame by default
 * ts.setTempoChangePercent(-8.3f);        // 120 BPM played at 110 BPM
 * dspark::AudioBuffer<float> stretched;
 * ts.process(source.toView(), stretched); // exact, whole-signal
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/StateBlob.h"
#include "detail/PhaseVocoderEngine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class TimeStretch
 * @brief Real-time and offline time stretching, 0.5x to 2x, pitch unchanged.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TimeStretch final
{
public:
    // The published parameter words must never make the audio thread take a
    // lock to read them.
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "audio-thread stores must not lock");

    /** @brief Smallest and largest stretch this class accepts. */
    static constexpr T kMinRatio = T(0.5);
    static constexpr T kMaxRatio = T(2);

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Allocates every buffer this class will ever use.
     *
     * Invalid specs (non-positive or non-finite rate, block size or channel
     * count) and fftSize values that are not a power of two in [256, 1 << 20]
     * are ignored: the previous state is kept and an unprepared instance
     * stays pass-through.
     *
     * @param spec    Audio environment specification.
     * @param fftSize STFT frame size, power of two (default 2048). Larger
     *                favours low-pitched and sustained material, smaller
     *                favours transients and lowers latency.
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048)
    {
        if (!spec.isValid() || (fftSize & (fftSize - 1)) != 0
            || fftSize < 256 || fftSize > (1 << 20))
            return;

        prepared_.store(false, std::memory_order_relaxed);

        numChannels_ = std::max(1, spec.numChannels);
        fftSize_  = fftSize;
        synthHop_ = fftSize / 4;

        // The reader trails the completed frontier of the synthesis stream by
        // one synthesis hop. The analysis-synthesis chain itself accounts for
        // fftSize - synthHop samples of delay, so that trailing distance puts
        // the reported latency at exactly one frame.
        latency_ = fftSize_;

        // The input queue holds what the stretch has not consumed yet. It
        // needs room for one whole analysis hop (which can be as long as a
        // frame), for one host block, and for the surplus a ratio above 1
        // accumulates until the caller stops or the queue refuses more.
        int capacity = 4 * fftSize_ + std::max(1, spec.maxBlockSize);
        queueSize_ = 1;
        while (queueSize_ < capacity) queueSize_ <<= 1;
        queueMask_ = queueSize_ - 1;

        // No resample-back stage and no harmonic/percussive split; the
        // locked analysis hop and the spectral-flux onset detector are both
        // asked for, because this owner reads the synthesis stream directly
        // and its hop schedule has to act on strikes the frame-energy test
        // cannot see over sustained material.
        engine_.prepare(spec.sampleRate, numChannels_, fftSize_, false, false, true, true);
        accumMask_ = engine_.olaMask();

        queue_.assign(static_cast<size_t>(numChannels_), {});
        for (int ch = 0; ch < numChannels_; ++ch)
            queue_[static_cast<size_t>(ch)].assign(static_cast<size_t>(queueSize_), T(0));
        feed_.assign(static_cast<size_t>(fftSize_), T(0));

        publishEngineParams();
        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears all signal state and empties the input queue (keeps
     *  parameters). Belongs to the owner of the stream. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        engine_.reset();
        for (auto& q : queue_) std::fill(q.begin(), q.end(), T(0));

        queueWrite_ = 0;
        queueRead_  = 0;
        queued_     = 0;
        // Start one synthesis hop behind the write head: those cells are
        // still silent, which is the latency the class reports, and from
        // there the reader stays exactly that far behind.
        readPos_ = engine_.writeHead() - static_cast<int64_t>(synthHop_);
    }

    // -- Parameters ---------------------------------------------------------------

    /**
     * @brief Sets the stretch as an output/input length ratio.
     *
     * Above 1 the signal gets longer (slower); below 1 shorter (faster).
     * Clamped to [0.5, 2]. Non-finite values are ignored. The change glides
     * in over a few frames rather than jumping, so it is click-free on a
     * running stream; reset() adopts it immediately.
     */
    void setTimeRatio(T ratio) noexcept
    {
        if (!std::isfinite(ratio)) return;
        timeRatio_.store(std::clamp(ratio, kMinRatio, kMaxRatio),
                         std::memory_order_relaxed);
        publishEngineParams();
    }

    /**
     * @brief Sets the stretch as a tempo change in percent.
     *
     * Playing a passage `pct` percent faster means fitting it into
     * `1 / (1 + pct/100)` of the time, so that is the ratio this sets: +10
     * gives 0.909, -10 gives 1.111. Values outside the ratio range clamp to
     * it; non-finite values are ignored.
     */
    void setTempoChangePercent(T pct) noexcept
    {
        if (!std::isfinite(pct)) return;
        const T denom = T(1) + pct / T(100);
        if (!(denom > T(0))) { setTimeRatio(kMaxRatio); return; }
        setTimeRatio(T(1) / denom);
    }

    /** @brief Enables phase reset on detected transients (default on).
     *  Without it, attacks are stretched along with everything else and
     *  soften audibly. */
    void setTransientPreserve(bool on) noexcept
    {
        transientPreserve_.store(on, std::memory_order_relaxed);
        publishEngineParams();
    }

    /**
     * @brief Enables identity phase locking (default on).
     *
     * Off leaves the plain phase vocoder, where every bin advances on its own
     * instantaneous frequency and the bins of one partial lose their relative
     * phase - the classic phasiness. It is exposed so the difference can be
     * heard and measured, not because it is ever the better setting.
     */
    void setPhaseLock(bool on) noexcept
    {
        phaseLock_.store(on, std::memory_order_relaxed);
        publishEngineParams();
    }

    /** @return Current stretch ratio (output length / input length). */
    [[nodiscard]] T getTimeRatio() const noexcept
    {
        return timeRatio_.load(std::memory_order_relaxed);
    }

    /** @return Whether transient phase reset is enabled. */
    [[nodiscard]] bool getTransientPreserve() const noexcept
    {
        return transientPreserve_.load(std::memory_order_relaxed);
    }

    /** @return Whether identity phase locking is enabled. */
    [[nodiscard]] bool getPhaseLock() const noexcept
    {
        return phaseLock_.load(std::memory_order_relaxed);
    }

    /** @brief Latency of the streaming path in samples: one frame (42.7 ms at
     *  the default 2048 frame and 48 kHz), 0 before prepare() succeeds. The
     *  offline process() removes its own delay and needs no compensation. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return prepared_.load(std::memory_order_relaxed) ? latency_ : 0;
    }

    /** @brief Input samples accepted but not yet consumed by the stretch.
     *  Rises while the ratio is above 1 and the caller feeds a fixed block
     *  size; at the queue capacity further input is refused. */
    [[nodiscard]] int getQueuedInputSamples() const noexcept { return queued_; }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TSTR"), 1);
        w.write("ratio", static_cast<float>(timeRatio_.load(std::memory_order_relaxed)));
        w.write("transient", transientPreserve_.load(std::memory_order_relaxed));
        w.write("phaselock", phaseLock_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TSTR")) return false;
        setTimeRatio(static_cast<T>(r.read("ratio", 1.0f)));
        setTransientPreserve(r.read("transient", true));
        setPhaseLock(r.read("phaselock", true));
        return true;
    }

    // -- Processing ----------------------------------------------------------------

    /**
     * @brief Streaming, in-place, real-time safe.
     *
     * Accepts the block's samples into the input queue and writes the same
     * number of samples of the stretched stream back into the block. See the
     * file header for what happens to the length difference at ratios away
     * from 1. Pass-through until prepare() succeeds; channels beyond the
     * prepared count are left untouched.
     *
     * @param buffer Audio block; all prepared channels are processed.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS  = buffer.getNumSamples();
        if (nS <= 0) return;

        // Input is taken into the queue in step with the output leaving it,
        // never a whole block ahead. That is what the device physically is -
        // one sample in for every sample out - and it is also what makes the
        // result independent of the host's block size: every decision below
        // is taken at a position in the stream where the queue holds exactly
        // the input the stream has delivered by then, whether the host sends
        // one sample at a time or four thousand.
        int taken = 0;    // samples of this block already queued
        int i = 0;        // output samples already written
        while (i < nS)
        {
            takeInput(buffer, nCh, taken, i);

            const int64_t avail = engine_.writeHead() - readPos_;
            if (avail > 0)
            {
                const int take = static_cast<int>(
                    std::min<int64_t>(avail, static_cast<int64_t>(nS - i)));
                takeInput(buffer, nCh, taken, i + take);
                readSynthesis(buffer, nCh, i, take);
                readPos_ += take;
                i += take;
                continue;
            }

            // More synthesis is needed: run one analysis frame. Below a ratio
            // of 1 the stretch wants more input than the stream has brought,
            // and no amount of buffering invents it: the output waits, in
            // silence, for exactly as many samples as are missing, and picks
            // up where it left off.
            const int need = engine_.samplesToNextHop();
            if (need > queued_)
            {
                const int silence = std::min(need - queued_, nS - i);
                // Take the matching input FIRST: the block is about to be
                // overwritten with silence, and a sample the stream delivered
                // must be queued whether or not anything came out against it.
                takeInput(buffer, nCh, taken, i + silence);
                for (int ch = 0; ch < nCh; ++ch)
                    std::fill(buffer.getChannel(ch) + i, buffer.getChannel(ch) + i + silence, T(0));
                i += silence;
                continue;
            }

            for (int ch = 0; ch < nCh; ++ch)
            {
                dequeueInto(ch, need);
                engine_.pushInput(ch, feed_.data(), need);
            }
            engine_.commitInput(need, nCh);
            queueRead_ = (queueRead_ + need) & queueMask_;
            queued_ -= need;
        }
    }

    /**
     * @brief Offline whole-signal stretch (setup thread; allocates `out`).
     *
     * `out` is resized to `round(inputLength * ratio)` samples and holds the
     * stretched signal aligned with the input: the algorithmic delay is
     * removed here, so no compensation is needed on this path. The ratio in
     * force is adopted immediately rather than glided, and the streaming
     * state is reset. An unprepared instance copies the input through.
     *
     * @param in  Source signal.
     * @param out Destination; resized by this call.
     */
    void process(AudioBufferView<const T> in, AudioBuffer<T>& out)
    {
        const int inLen = in.getNumSamples();
        const int inCh  = in.getNumChannels();

        if (!prepared_.load(std::memory_order_relaxed) || inLen <= 0 || inCh <= 0)
        {
            out.resize(std::max(0, inCh), std::max(0, inLen));
            for (int ch = 0; ch < inCh; ++ch)
                std::copy(in.getChannel(ch), in.getChannel(ch) + inLen, out.getChannel(ch));
            return;
        }

        reset();

        const int nCh = std::min(inCh, numChannels_);
        const double ratio = engine_.activeRatio();
        const auto outLen = static_cast<int>(
            std::lround(static_cast<double>(inLen) * ratio));
        out.resize(inCh, std::max(0, outLen));
        if (outLen <= 0) return;

        // A frame carries its content at its centre, so the first output
        // sample that lines up with input sample 0 sits half an input frame
        // plus half a stretched frame into the synthesis stream.
        const int64_t skip = std::lround(0.5 * static_cast<double>(fftSize_) * (1.0 + ratio));
        const int64_t stop = skip + outLen;

        int64_t streamPos = 0;    // position in the synthesis stream
        int64_t inPos = 0;        // input samples handed to the engine

        while (streamPos < stop)
        {
            const int64_t avail = engine_.writeHead() - readPos_;
            if (avail <= 0)
            {
                // Offline has the whole signal, so nothing rations the input;
                // past its end the engine is flushed with silence.
                const int need = engine_.samplesToNextHop();
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* src = in.getChannel(ch);
                    for (int k = 0; k < need; ++k)
                    {
                        const int64_t p = inPos + k;
                        feed_[static_cast<size_t>(k)] =
                            (p < inLen) ? src[static_cast<size_t>(p)] : T(0);
                    }
                    engine_.pushInput(ch, feed_.data(), need);
                }
                engine_.commitInput(need, nCh);
                inPos += need;
                continue;
            }

            const int64_t take = std::min(avail, stop - streamPos);
            const int64_t from = std::max(streamPos, skip);
            const int64_t count = streamPos + take - from;
            if (count > 0)
            {
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* acc = engine_.olaData(ch);
                    T* dst = out.getChannel(ch) + (from - skip);
                    int64_t rp = readPos_ + (from - streamPos);
                    for (int64_t k = 0; k < count; ++k)
                        dst[k] = acc[static_cast<size_t>((rp + k) & accumMask_)];
                }
            }
            readPos_ += take;
            streamPos += take;
        }

        // Channels the stretch does not cover would otherwise be left at the
        // zero resize() gives them; carry them through instead, truncated or
        // silence-padded to the stretched length.
        for (int ch = nCh; ch < inCh; ++ch)
        {
            const T* src = in.getChannel(ch);
            T* dst = out.getChannel(ch);
            const int n = std::min(outLen, inLen);
            std::copy(src, src + n, dst);
            std::fill(dst + n, dst + outLen, T(0));
        }
    }

private:
    /** @brief Publishes the whole engine parameter set (control thread). */
    void publishEngineParams() noexcept
    {
        typename detail::PhaseVocoderEngine<T>::Params p;
        // The engine carries its stretch target logarithmically, because the
        // same core drives a pitch shifter where semitones are the natural
        // unit. Ratio and semitones are the same number in two spellings.
        p.targetSemitones = 12.0 * std::log2(static_cast<double>(
                                timeRatio_.load(std::memory_order_relaxed)));
        p.transientPreserve = transientPreserve_.load(std::memory_order_relaxed);
        p.formantPreserve = false;   // nothing resamples the output here
        p.phaseLock = phaseLock_.load(std::memory_order_relaxed);
        p.percussiveSplit = false;   // no public switch selects the split here
        engine_.publishParams(p);
    }

    /**
     * @brief Queues this block's input up to sample `upTo`, refusing what
     * does not fit.
     *
     * `taken` is how far the block has been read so far and moves with the
     * output, so the queue level at any decision point is a property of the
     * stream position and not of the block boundaries. The buffer is read
     * before the output overwrites it, which the caller of this function
     * guarantees by queueing a run before emitting it.
     */
    void takeInput(AudioBufferView<T> buffer, int nCh, int& taken, int upTo) noexcept
    {
        if (upTo <= taken) return;
        const int room = queueSize_ - queued_;
        const int count = std::min(upTo - taken, room);
        if (count <= 0) { taken = upTo; return; }   // full: this input is refused
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = buffer.getChannel(ch) + taken;
            auto& q = queue_[static_cast<size_t>(ch)];
            int wp = queueWrite_;
            for (int k = 0; k < count; ++k)
            {
                q[static_cast<size_t>(wp)] = src[k];
                wp = (wp + 1) & queueMask_;
            }
        }
        queueWrite_ = (queueWrite_ + count) & queueMask_;
        queued_ += count;
        taken = upTo;
    }

    /** @brief Lays `count` queued samples of one channel out contiguously.
     *  The engine's ring writer takes one run per hop, so the queue's own
     *  wrap has to be flattened first. */
    void dequeueInto(int ch, int count) noexcept
    {
        const auto& q = queue_[static_cast<size_t>(ch)];
        int rp = queueRead_;
        for (int k = 0; k < count; ++k)
        {
            feed_[static_cast<size_t>(k)] = q[static_cast<size_t>(rp)];
            rp = (rp + 1) & queueMask_;
        }
    }

    /** @brief Copies `count` synthesis samples into the block at `offset`. */
    void readSynthesis(AudioBufferView<T> buffer, int nCh, int offset, int count) noexcept
    {
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* acc = engine_.olaData(ch);
            T* dst = buffer.getChannel(ch) + offset;
            const int64_t rp = readPos_;
            for (int k = 0; k < count; ++k)
                dst[k] = acc[static_cast<size_t>((rp + k) & accumMask_)];
        }
    }

    // -- Members -------------------------------------------------------------------
    int numChannels_ = 0;
    std::atomic<bool> prepared_ { false };

    int fftSize_ = 2048;
    int synthHop_ = 512;
    int latency_ = 2048;
    int64_t accumMask_ = 8191;     ///< Cached engine OLA ring mask.

    detail::PhaseVocoderEngine<T> engine_;   ///< Shared analysis/synthesis core.

    std::vector<std::vector<T>> queue_;   ///< Per-channel input queue.
    std::vector<T> feed_;                 ///< One analysis hop, laid out flat.
    int queueSize_ = 0;
    int queueMask_ = 0;
    int queueWrite_ = 0;
    int queueRead_ = 0;
    int queued_ = 0;

    int64_t readPos_ = 0;       ///< Reader position in the synthesis stream.

    std::atomic<T> timeRatio_ { T(1) };
    std::atomic<bool> transientPreserve_ { true };
    std::atomic<bool> phaseLock_ { true };
};

} // namespace dspark
