// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PitchShifter.h
 * @brief Real-time pitch shifter: a phase vocoder stretches, a sinc reader
 *        resamples the stretched stream back to the original duration.
 *
 * Two engines sit behind setQuality():
 *
 * - **Quality::Studio (default)**, Effects/detail/StudioVocoder.h:
 *   phase-gradient heap integration (Prusa & Holighaus 2017) on a reference
 *   summed over every channel, instantaneous frequency from a second
 *   transform 16 samples on, and a time map anchored on strikes found by a
 *   look-ahead spectral-flux detector. The stretched stream is read back at
 *   the pitch ratio with the 32-tap windowed sinc of Core/Interpolation.h.
 * - **Quality::Standard / Quality::High**, the Standard/High engine
 *   (Effects/detail/PhaseVocoderEngine.h): identity phase locking (Laroche &
 *   Dolson 1999) with phase reset on energy onsets, decided on the FIRST
 *   channel. Standard reads with a 4-point Catmull-Rom interpolator and is
 *   the earlier Standard rendering, pinned by stored references; High reads with the
 *   32-tap sinc at the same latency.
 *
 * Measured against the ideal (each source re-synthesised at the shifted
 * pitch) at -12, -5, -2, +3, +7 and +12 semitones, stereo, 48 kHz: spectral
 * scores over a chord, a bass line and a sung vowel, strike scores over a
 * drum pattern and a mix:
 *
 *   engine                    LSD dB  conv dB  attack dB  pre-echo dB  onset ms
 *   Studio, 2048 (default)     1.96   -14.22     -2.84      -94.1       0.68
 *   Studio, 4096               1.47   -18.66     -2.84      -79.5       0.89
 *   Standard engine            2.50   -12.16     -4.74      -78.4      13.18
 *
 * (LSD: multi-resolution log-spectral distance; conv: spectral convergence;
 * attack: energy of the first 5 ms against the ideal's, closer to 0 is
 * better; pre-echo: energy in the 20 ms before each strike above the
 * ideal's, relative to the strike; onset: median error of the unaligned
 * onset time. Lower is better elsewhere.)
 *
 * The Standard/High engine's reader trails a write head that advances at the pitch
 * ratio, so its real delay drifts with the pitch: a tone burst's energy
 * centroid lands 64 ms late at -12 semitones and 32 ms early at +12 against
 * one reported latency, and a source absent from the first channel (panned
 * hard right) comes out 48 dB down. Studio steers its timeline so the reader
 * stands a fixed latency behind the input at every pitch (a mean 1.6 ms off
 * at -12 semitones, within 0.2 ms from -1 to +12) and shifts a source panned
 * to either side exactly.
 *
 * Studio's latency, 1.5 frames plus the onset look-ahead, a quarter frame of
 * anchor lead and the sinc's reach, is fixed for every pitch: 5184 samples
 * (108 ms) at the default 2048 frame and 48 kHz. The Standard/High engine's is 2 *
 * fftSize (4096). The dry path of the mix control is delay-compensated to
 * the reported value, so partial mixes stay comb-free. Crossing between
 * Studio and the Standard/High engine restarts the stream on the next block (their
 * latencies differ); Standard and High cross-fade into each other live.
 * Channels beyond the prepared count pass through untouched.
 *
 * Both engines glide the active shift toward the target at up to 0.5
 * semitones per analysis hop, keep the average stretch exactly equal to the
 * ratio (a fractional-accumulator hop, no cumulative drift), taper the bins
 * an upward shift would fold above Nyquist, and share one rotation across
 * channels, which keeps the stereo image from wandering.
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored); prepare() is setup-thread
 * only (allocates; invalid specs are ignored); reset() belongs to the owner
 * of the stream; getState()/setState() are setup/UI threads.
 *
 * Dependencies: Effects/detail/PhaseVocoderEngine.h,
 * Effects/detail/StudioVocoder.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h,
 * Core/Interpolation.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Interpolation.h"
#include "../Core/StateBlob.h"
#include "detail/PhaseVocoderEngine.h"
#include "detail/StudioVocoder.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class PitchShifter
 * @brief Real-time phase-vocoder pitch shifter (+-12 semitones, stereo-linked).
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PitchShifter
{
public:
    // -- Lifecycle -------------------------------------------------------------

    /**
     * @brief Allocates all rings and spectral state.
     *
     * Invalid specs (non-positive/non-finite rate, block size or channel
     * count) and fftSize values that are not a power of two in [256, 1 << 20]
     * are ignored: the previous state is kept and an unprepared instance
     * stays pass-through.
     *
     * @param spec    Audio environment specification.
     * @param fftSize STFT frame size, a power of two, used by whichever
     *                engine is selected. 0 (the default) selects 2048 for
     *                Standard and High, and the power of two nearest 43 ms
     *                for Studio (2048 at 44.1 and 48 kHz, 4096 at 88.2 and
     *                96 kHz). Smaller sizes lower latency and favour
     *                transients; larger sizes favour low-pitched material
     *                (Studio at 4096: lower spectral error, see the file
     *                overview, at 1.5 times the latency).
     */
    void prepare(const AudioSpec& spec, int fftSize = 0)
    {
        if (!spec.isValid()) return;
        if (fftSize != 0 && ((fftSize & (fftSize - 1)) != 0
                             || fftSize < 256 || fftSize > (1 << 20)))
            return;

        prepared_.store(false, std::memory_order_relaxed);

        numChannels_ = std::max(1, spec.numChannels);
        // 0 selects each engine's default frame: 2048 for Standard/High (the
        // published rendering), about 43 ms for Studio (2048 at 44.1 and
        // 48 kHz, 4096 at 88.2 and 96 kHz).
        const int fftStudio = fftSize != 0 ? fftSize : studioDefaultFrame(spec.sampleRate);
        if (fftSize == 0) fftSize = 2048;

        // The reader trails the write head by readOffset_; the window/OLA
        // chain adds another fftSize - synthHop, so the measured wet latency
        // is readOffset_ + fftSize - synthHop = 2 * fftSize (exact at unity
        // ratio). The dry path must delay by the SAME value or partial mixes
        // comb-filter.
        const int synthHop = fftSize / 4;
        readOffset_ = fftSize + synthHop;
        latency_    = readOffset_ + fftSize - synthHop;
        // Studio: the resample reader runs at the pitch ratio behind the
        // completed stream. At ratio a an input sample reaches the completed
        // stream after half a frame, a frame of analysis, the onset
        // lookahead and one analysis hop, and the reader needs its sinc
        // reach beyond that; the worst case is the lowest ratio, 0.5:
        // N/(2a) + N/2 + lookahead + taps/a = 1.5 N + lookahead + 32. A
        // strike anchor may move an analysis frame ahead of the nominal
        // timeline by up to a quarter frame at that ratio, which the last
        // term covers. One fixed latency for every ratio, so the dry path
        // and host compensation hold at any pitch.
        studio_.prepare(spec.sampleRate, numChannels_, fftStudio, true);
        studio_.setAnchorLeadLimit(static_cast<double>(fftStudio / 4));
        latencyStudio_ = fftStudio + fftStudio / 2 + studio_.lookahead() + fftStudio / 4 + 64;
        latencyLegacy_ = latency_;
        drySize_ = 1;
        while (drySize_ < std::max(latencyLegacy_, latencyStudio_) + 1) drySize_ <<= 1;
        dryMask_ = drySize_ - 1;

        // The engine owns the analysis rings, spectral state and OLA ring;
        // resample-back compensation stages (anti-alias taper, formant
        // pre-warp target) are enabled because this owner resamples.
        //
        // The three remaining capabilities are refused, and the last two are
        // written out rather than defaulted because a future default that
        // flips would change what this effect sounds like. The locked
        // analysis hop breaks the resample-back reader's Ra = Rs/ratio
        // assumption and buys nothing here. The spectral-flux onset detector
        // fires earlier and more often than the frame-energy test, and every
        // firing resets phase, so adopting it smears energy ahead of a strike
        // - audible pre-echo on percussive material - and re-renders every
        // release already in a user's hands. A mix finished last month is not
        // restored by a release note. How large the smear is depends on the
        // material and on the estimator it is measured with, so no figure is
        // quoted here; that it is audible on percussive material is the
        // durable part, and it is why this effect keeps the detector it
        // shipped with.
        engine_.prepare(spec.sampleRate, numChannels_, fftSize, true, false, false, false);
        mixMaxStep_ = static_cast<T>(1.0 / std::max(1.0, spec.sampleRate * 0.02));
        accumMask_ = engine_.olaMask();

        dryRing_.assign(static_cast<size_t>(numChannels_), {});
        for (int ch = 0; ch < numChannels_; ++ch)
            dryRing_[static_cast<size_t>(ch)].assign(static_cast<size_t>(drySize_), T(0));

        publishEngineParams();
        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears all signal state (keeps parameters). Safe on the audio thread. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        studioActive_ = quality_.load(std::memory_order_relaxed) == Quality::Studio;
        latency_ = studioActive_ ? latencyStudio_ : latencyLegacy_;
        engine_.reset();
        studio_.reset();
        for (auto& r : dryRing_) std::fill(r.begin(), r.end(), T(0));
        studioOut_ = 0;
        {
            // The reader starts where output 0 shows input -latency, and
            // advances at the pitch ratio from there.
            const double start = studio_.streamPositionOf(-static_cast<double>(latencyStudio_));
            studioReadInt_ = static_cast<int64_t>(std::floor(start));
            studioReadFrac_ = start - static_cast<double>(studioReadInt_);
        }

        dryPos_ = 0;
        readPosInt_ = engine_.writeHead() - readOffset_;
        readPosFrac_ = 0.0;
        currentMix_ = mix_.load(std::memory_order_relaxed);
        highReader_ = quality_.load(std::memory_order_relaxed) != Quality::Standard;
        readerFadeLeft_ = 0;   // start settled on the selected reader
    }

    // -- Parameters (thread-safe) -----------------------------------------------

    /** @brief Engine and resample-back reader (see the file overview). */
    enum class Quality
    {
        Standard,   ///< Standard/High engine, 4-point Catmull-Rom reader: the earlier default rendering.
        High,       ///< Standard/High engine, 32-tap windowed-sinc reader: transparent HF.
        Studio      ///< Studio engine (default): see the file overview.
    };

    /**
     * @brief Selects the engine and reader. Thread-safe. Standard and High
     *        crossfade into each other over 64 samples; crossing to or from
     *        Studio restarts the stream at the next block (the latencies
     *        differ), so a host should re-read getLatency() after it.
     * @param quality Engine and reader. Out-of-range values clamp to Studio.
     */
    void setQuality(Quality quality) noexcept
    {
        quality = static_cast<Quality>(std::clamp(static_cast<int>(quality), 0,
                                                  static_cast<int>(Quality::Studio)));
        quality_.store(quality, std::memory_order_relaxed);
    }

    /** @return The resample-back reader quality. */
    [[nodiscard]] Quality getQuality() const noexcept
    {
        return quality_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Sets the pitch shift in semitones, clamped to +-12.
     *
     * The active shift glides toward the target at up to 0.5 semitones per
     * analysis hop (a few ms at the default frame size), so live changes are
     * click-free. Non-finite values are ignored.
     */
    void setSemitones(T st) noexcept
    {
        if (!std::isfinite(st)) return;
        semitones_.store(std::clamp(st, T(-12), T(12)), std::memory_order_relaxed);
        publishEngineParams();
    }

    /** @brief Sets the pitch shift as a frequency ratio, clamped to [0.5, 2].
     *  Non-finite values are ignored. */
    void setPitchRatio(T ratio) noexcept
    {
        if (!std::isfinite(ratio)) return;
        ratio = std::clamp(ratio, T(0.5), T(2));
        semitones_.store(static_cast<T>(12.0 * std::log2(static_cast<double>(ratio))),
                         std::memory_order_relaxed);
        publishEngineParams();
    }

    /** @brief Dry/wet mix, [0, 1]. The dry path is latency-compensated and the
     *  mix is ramped over at least 20 ms (the wet stream is decorrelated
     *  from the dry, so an unsmoothed step would click). Non-finite values are
     *  ignored. */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Enables phase reset on detected transients (default on). */
    void setTransientPreserve(bool enabled) noexcept
    {
        transientPreserve_.store(enabled, std::memory_order_relaxed);
        publishEngineParams();
    }

    /**
     * @brief Keeps formants (vocal timbre) in place while pitch moves.
     *
     * A cepstral lift extracts the smooth spectral envelope of each frame
     * (quefrencies below ~1 ms) and the synthesis magnitudes are pre-warped
     * by env(k*ratio)/env(k), so after the output resampler the envelope
     * lands back where it started - the classic anti-chipmunk correction.
     * Costs two extra FFTs per frame. Default off.
     */
    void setFormantPreserve(bool enabled) noexcept
    {
        formantPreserve_.store(enabled, std::memory_order_relaxed);
        publishEngineParams();
    }

    /** @return Current shift in semitones. */
    [[nodiscard]] T getSemitones() const noexcept
    {
        return semitones_.load(std::memory_order_relaxed);
    }

    /** @return Current dry/wet mix. */
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @return Whether transient phase reset is enabled. */
    [[nodiscard]] bool getTransientPreserve() const noexcept
    {
        return transientPreserve_.load(std::memory_order_relaxed);
    }

    /** @return Whether formant preservation is enabled. */
    [[nodiscard]] bool getFormantPreserve() const noexcept
    {
        return formantPreserve_.load(std::memory_order_relaxed);
    }

    /** @brief Reports total latency in samples. Studio: 1.5 * fftSize +
     *  lookahead + fftSize / 4 + 64 (5184, 108 ms, at the default frame and
     *  48 kHz), held at every pitch. Standard and High: 2 * fftSize, exact at
     *  unity only (their real delay drifts with the pitch; file overview). */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("PSHF"), 1);
        w.write("semitones", static_cast<float>(semitones_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("transient", transientPreserve_.load(std::memory_order_relaxed));
        w.write("formant", formantPreserve_.load(std::memory_order_relaxed));
        w.write("quality", static_cast<int32_t>(quality_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("PSHF")) return false;
        setSemitones(static_cast<T>(r.read("semitones", 0.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setTransientPreserve(r.read("transient", true));
        setFormantPreserve(r.read("formant", false));
        setQuality(static_cast<Quality>(r.read("quality", 0)));   // clamped inside
        return true;
    }

    // -- Processing --------------------------------------------------------------

    /**
     * @brief Processes audio in-place.
     *
     * Pass-through until prepare() succeeds. Channels beyond the prepared
     * count are left untouched.
     *
     * @param buffer Audio block; all prepared channels are processed.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS  = buffer.getNumSamples();
        // Crossing between Studio and the Standard/High engine restarts the stream:
        // the two run different latencies, so no crossfade can join them.
        if ((quality_.load(std::memory_order_relaxed) == Quality::Studio) != studioActive_)
            reset();
        if (studioActive_)
        {
            processStudio(buffer, nCh, nS);
            return;
        }

        // Rate-limited mix ramp (moveTowards, exact landing; settled it
        // reduces to the constant, bit-identically). A per-block ramp landed
        // in 0.7 ms with 32-sample blocks.
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;

        // Reader selection: a change crossfades from the old reader to the
        // new one over kReaderFade samples (the readers differ at HF).
        const bool highQuality = quality_.load(std::memory_order_relaxed) == Quality::High;
        if (highQuality != highReader_)
        {
            highReader_ = highQuality;
            readerFadeLeft_ = kReaderFade;
        }

        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, engine_.samplesToNextHop());

            // 1. Push input into the analysis ring and the dry-compensation ring.
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* in = buffer.getChannel(ch) + i;
                engine_.pushInput(ch, in, chunk);
                auto& dry = dryRing_[static_cast<size_t>(ch)];
                int dp = dryPos_;
                for (int k = 0; k < chunk; ++k)
                {
                    dry[static_cast<size_t>(dp)] = in[k];
                    dp = (dp + 1) & dryMask_;
                }
            }

            // 2. Produce output: fractional read of the synthesis stream + mix.
            const double ratio = engine_.activeRatio();
            int64_t rpEnd = readPosInt_;
            double  rfEnd = readPosFrac_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = buffer.getChannel(ch) + i;
                const T* acc = engine_.olaData(ch);
                const auto& dry = dryRing_[static_cast<size_t>(ch)];

                int64_t rp = readPosInt_;
                double  rf = readPosFrac_;
                int     dp = dryPos_;
                int     fadeLeft = readerFadeLeft_;

                for (int k = 0; k < chunk; ++k)
                {
                    T wet = readWet(highReader_, acc, rp, rf);
                    if (fadeLeft > 0)
                    {
                        const T w = static_cast<T>(fadeLeft) * (T(1) / T(kReaderFade));
                        const T old = readWet(!highReader_, acc, rp, rf);
                        wet += (old - wet) * w;
                        --fadeLeft;
                    }
                    const int dryIdx = (dp - latency_) & dryMask_;
                    const T drySample = dry[static_cast<size_t>(dryIdx)];
                    const T mixVal = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(i + k + 1));
                    // Two-product blend: exact at both ends (mix 1 emits the
                    // wet stream bit-exactly, mix 0 the delayed dry).
                    out[k] = drySample * (T(1) - mixVal) + wet * mixVal;

                    rf += ratio;
                    const auto adv = static_cast<int64_t>(rf);
                    rp += adv;
                    rf -= static_cast<double>(adv);
                    dp = (dp + 1) & dryMask_;
                }
                if (ch == 0) { rpEnd = rp; rfEnd = rf; }   // recurrence result
            }

            // Commit shared positions once per chunk, using the SAME
            // per-sample recurrence result the output loops computed (taken
            // from channel 0, whose loop ran it already): a single
            // frac + ratio * chunk product rounds differently for different
            // chunk sizes, and chunk boundaries follow the host block size,
            // so committing the product form made the output depend on how
            // the host chopped the stream (~1 ulp per flip, but a bit-exact
            // contract is a bit-exact contract).
            {
                if (nCh == 0)   // channel-less call: advance the stream anyway
                {
                    for (int k = 0; k < chunk; ++k)
                    {
                        rfEnd += ratio;
                        const auto adv = static_cast<int64_t>(rfEnd);
                        rpEnd += adv;
                        rfEnd -= static_cast<double>(adv);
                    }
                }
                readPosInt_  = rpEnd;
                readPosFrac_ = rfEnd;
                dryPos_ = (dryPos_ + chunk) & dryMask_;
                readerFadeLeft_ = std::max(0, readerFadeLeft_ - chunk);
            }

            // 3. Advance the engine (runs an STFT hop at the analysis boundary).
            engine_.commitInput(chunk, nCh);

            i += chunk;
        }

        currentMix_ = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(nS));
    }

private:
    /**
     * @brief The Studio path: stream the input into the engine, read its
     *        stretched stream back at the pitch ratio with the 32-tap sinc.
     *
     * The reader's rate is the engine's glided ratio, so the pitch is exactly
     * that ratio; its position is steered, not free: before each frame the
     * engine is told where on the input timeline the reader will stand when
     * it reaches that frame, so the latency stays latencyStudio_ through
     * pitch changes instead of drifting with them.
     */
    void processStudio(AudioBufferView<T> buffer, int nCh, int nS) noexcept
    {
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;
        const int64_t mask = studio_.olaMask();
        int i = 0;
        while (i < nS)
        {
            const int need = studio_.samplesToNextHop();
            if (need == 0)
            {
                studio_.commitInput(0, nCh);
                continue;
            }
            const int chunk = std::min(nS - i, need);
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* in = buffer.getChannel(ch) + i;
                studio_.pushInput(ch, in, chunk);
                auto& dry = dryRing_[static_cast<size_t>(ch)];
                int dp = dryPos_;
                for (int k = 0; k < chunk; ++k)
                {
                    dry[static_cast<size_t>(dp)] = in[k];
                    dp = (dp + 1) & dryMask_;
                }
            }

            const double ratio = studio_.activeRatio();
            int64_t rpEnd = studioReadInt_;
            double rfEnd = studioReadFrac_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = buffer.getChannel(ch) + i;
                const T* acc = studio_.olaData(ch);
                const auto& dry = dryRing_[static_cast<size_t>(ch)];
                int64_t rp = studioReadInt_;
                double rf = studioReadFrac_;
                int dp = dryPos_;
                for (int k = 0; k < chunk; ++k)
                {
                    const T wet = reader_.readRing(acc, mask, rp, rf);
                    const T drySample = dry[static_cast<size_t>((dp - latency_) & dryMask_)];
                    const T mixVal = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(i + k + 1));
                    out[k] = drySample * (T(1) - mixVal) + wet * mixVal;
                    rf += ratio;
                    const auto adv = static_cast<int64_t>(rf);
                    rp += adv;
                    rf -= static_cast<double>(adv);
                    dp = (dp + 1) & dryMask_;
                }
                if (ch == 0) { rpEnd = rp; rfEnd = rf; }
            }
            if (nCh == 0)
            {
                for (int k = 0; k < chunk; ++k)
                {
                    rfEnd += ratio;
                    const auto adv = static_cast<int64_t>(rfEnd);
                    rpEnd += adv;
                    rfEnd -= static_cast<double>(adv);
                }
            }
            studioReadInt_ = rpEnd;
            studioReadFrac_ = rfEnd;
            dryPos_ = (dryPos_ + chunk) & dryMask_;
            studioOut_ += chunk;

            // Steer the frame after the next one: where the reader will stand
            // on the input timeline when it reaches that frame's centre.
            const double c = studio_.nextStreamCentre() + static_cast<double>(studio_.synthHop());
            const double here = static_cast<double>(studioReadInt_) + studioReadFrac_;
            studio_.steerTimeline(static_cast<double>(studioOut_) + (c - here) / ratio
                                  - static_cast<double>(latencyStudio_));
            studio_.commitInput(chunk, nCh);
            i += chunk;
        }
        currentMix_ = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(nS));
    }

    /** @brief Studio's default frame: the power of two nearest 43 ms. */
    [[nodiscard]] static int studioDefaultFrame(double sampleRate) noexcept
    {
        int n = 256;
        while (n < (1 << 16) && static_cast<double>(n) * 1.5 < 0.0427 * sampleRate) n <<= 1;
        return n;
    }

    /** @brief Reads the synthesis stream with the selected reader. */
    [[nodiscard]] T readWet(bool high, const T* acc, int64_t ip, double frac) const noexcept
    {
        return high ? reader_.readRing(acc, accumMask_, ip, frac)
                    : readCatmullRom(acc, ip, frac);
    }

    /** @brief 4-point Catmull-Rom read of the synthesis accumulator. */
    [[nodiscard]] T readCatmullRom(const T* acc, int64_t ip, double frac) const noexcept
    {
        const int64_t m = accumMask_;
        const T x0 = acc[static_cast<size_t>((ip - 1) & m)];
        const T x1 = acc[static_cast<size_t>(ip & m)];
        const T x2 = acc[static_cast<size_t>((ip + 1) & m)];
        const T x3 = acc[static_cast<size_t>((ip + 2) & m)];
        const T f  = static_cast<T>(frac);
        return x1 + T(0.5) * f * (x2 - x0
                 + f * (T(2) * x0 - T(5) * x1 + T(4) * x2 - x3
                 + f * (T(3) * (x1 - x2) + x3 - x0)));
    }

    /** @brief Publishes the engine's parameter set from the atomic parameters
     *  (control thread; one seqlock publish per setter call). */
    void publishEngineParams() noexcept
    {
        typename detail::PhaseVocoderEngine<T>::Params p;
        p.targetSemitones = static_cast<double>(semitones_.load(std::memory_order_relaxed));
        p.transientPreserve = transientPreserve_.load(std::memory_order_relaxed);
        p.formantPreserve = formantPreserve_.load(std::memory_order_relaxed);
        engine_.publishParams(p);
        typename detail::StudioVocoder<T>::Params q;
        q.targetSemitones = p.targetSemitones;
        q.transientPreserve = p.transientPreserve;
        q.formantPreserve = p.formantPreserve;
        studio_.publishParams(q);
    }

    // -- Members -----------------------------------------------------------------
    int numChannels_ = 0;
    std::atomic<bool> prepared_ { false };

    int readOffset_ = 2560;   ///< Reader distance behind the write head.
    int latency_ = 4096;      ///< Measured wet latency (= readOffset_ + N - Rs).
    int drySize_ = 8192;
    int dryMask_ = 8191;
    int64_t accumMask_ = 8191;   ///< Cached engine OLA ring mask.

    detail::PhaseVocoderEngine<T> engine_;   ///< Standard/High engine (Standard / High).
    detail::StudioVocoder<T> studio_;        ///< Studio engine.
    bool studioActive_ = true;               ///< Engine in force since reset().
    int latencyStudio_ = 5184;               ///< Studio's fixed latency.
    int latencyLegacy_ = 4096;               ///< Standard/High engine's latency (2 * fftSize).
    int64_t studioReadInt_ = 0;              ///< Studio reader position (integer part).
    double studioReadFrac_ = 0.0;            ///< Studio reader position (fraction).
    int64_t studioOut_ = 0;                  ///< Output samples since reset().
    SincInterpolator<T> reader_;             ///< High-quality reader (it trails the
                                             ///< OLA write head by far more than
                                             ///< its 16-sample reach).
    static constexpr int kReaderFade = 64;   ///< Reader-switch crossfade length.
    bool highReader_ = false;                ///< Reader in use (audio thread).
    int readerFadeLeft_ = 0;                 ///< Samples left in a reader crossfade.

    std::vector<std::vector<T>> dryRing_;    ///< Per-channel latency-matched dry.

    int dryPos_ = 0;
    int64_t readPosInt_ = 0;
    double readPosFrac_ = 0.0;
    T currentMix_ = T(1);     ///< Audio-thread mix ramp state (exact landing).
    T mixMaxStep_ = T(1.0 / 960.0); ///< Mix ramp rate: full scale per 20 ms.

    std::atomic<T> semitones_ { T(0) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<bool> transientPreserve_ { true };
    std::atomic<bool> formantPreserve_ { false };
    std::atomic<Quality> quality_ { Quality::Studio };
};

} // namespace dspark
