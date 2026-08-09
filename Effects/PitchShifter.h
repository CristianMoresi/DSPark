// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PitchShifter.h
 * @brief Phase-vocoder pitch shifter with identity phase locking.
 *
 * Time-stretches the signal with a phase vocoder and resamples the stretched
 * stream back to the original duration, shifting pitch by the same ratio.
 * Quality rests on three techniques:
 *
 * - **Identity phase locking** (Laroche & Dolson 1999): spectral peaks are
 *   detected each frame and every bin in a peak's region of influence is
 *   rotated by the *same* phase increment as its peak, preserving the vertical
 *   phase coherence whose loss causes the classic phase-vocoder "phasiness".
 * - **Transient phase reset**: onsets (energy rising > 6 dB over the tracked
 *   envelope) re-initialise synthesis phases to the analysis phases, keeping
 *   attacks sharp instead of smeared. Peaks with no history (new partials)
 *   are reset individually even without a global onset.
 * - **Spectral anti-alias cut**: shifting up reads the synthesis stream
 *   faster, so content above Nyquist/ratio would alias; those bins are
 *   tapered to zero in the frequency domain before synthesis.
 *
 * Architecture (per block, streaming, zero allocation):
 *
 *   input ring -> analysis hop Ra (variable, fractional-accumulator exact)
 *     -> FFT -> peak picking & phase propagation (reference channel)
 *     -> per-channel rigid phase rotation per region -> IFFT
 *     -> overlap-add at fixed synthesis hop Rs = N/4 (exact COLA)
 *     -> Catmull-Rom fractional reader at rate `ratio` -> output
 *
 * The analysis/synthesis stages up to the OLA ring are the shared vocoder
 * engine (Effects/detail/PhaseVocoderEngine.h); this class owns the
 * resample-back stage (the Catmull-Rom fractional reader) and the
 * latency-compensated dry path.
 *
 * The analysis hop carries a fractional accumulator so the average stretch is
 * exactly Rs/(Rs/ratio) = ratio: tuning is exact for arbitrary ratios, with
 * no cumulative drift. Channels share the reference channel's peak/phase
 * decisions (rigid per-region rotation), which preserves inter-channel phase
 * differences exactly - the stereo image does not wander.
 *
 * Latency: 2 * fftSize samples (reported by getLatency(), measured exact at
 * unity ratio: reader offset fftSize + fftSize/4 behind the write head plus
 * the fftSize - fftSize/4 window/OLA delay of the analysis-synthesis chain).
 * The dry path of the mix control is delay-compensated to the same value, so
 * partial mixes stay comb-free. Channels beyond the prepared count pass
 * through untouched (and therefore uncompensated).
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored); prepare() is setup-thread
 * only (allocates; invalid specs are ignored); reset() belongs to the owner
 * of the stream; getState()/setState() are setup/UI threads.
 *
 * Dependencies: Effects/detail/PhaseVocoderEngine.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h, Core/StateBlob.h.
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
     * @param fftSize STFT frame size, power of two (default 2048). Smaller
     *                sizes lower latency and favour transients; larger sizes
     *                favour low-pitched material.
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048)
    {
        if (!spec.isValid() || (fftSize & (fftSize - 1)) != 0
            || fftSize < 256 || fftSize > (1 << 20))
            return;

        prepared_.store(false, std::memory_order_relaxed);

        numChannels_ = std::max(1, spec.numChannels);

        // The reader trails the write head by readOffset_; the window/OLA
        // chain adds another fftSize - synthHop, so the measured wet latency
        // is readOffset_ + fftSize - synthHop = 2 * fftSize (exact at unity
        // ratio). The dry path must delay by the SAME value or partial mixes
        // comb-filter.
        const int synthHop = fftSize / 4;
        readOffset_ = fftSize + synthHop;
        latency_    = readOffset_ + fftSize - synthHop;
        drySize_ = 1;
        while (drySize_ < latency_ + 1) drySize_ <<= 1;
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
        // fires more often than the frame-energy test, and every firing
        // resets phase, so adopting it moves the rendering of every release
        // already in a user's hands: measured on strikes over a sustained
        // bed, strike concentration falls by up to 91.6% and pre-echo rises
        // by up to 16.5 dB. This effect keeps the detector it shipped with.
        engine_.prepare(spec.sampleRate, numChannels_, fftSize, true, false, false, false);
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
        engine_.reset();
        for (auto& r : dryRing_) std::fill(r.begin(), r.end(), T(0));

        dryPos_ = 0;
        readPosInt_ = engine_.writeHead() - readOffset_;
        readPosFrac_ = 0.0;
        currentMix_ = mix_.load(std::memory_order_relaxed);
    }

    // -- Parameters (thread-safe) -----------------------------------------------

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
     *  mix is smoothed linearly over one block (the wet stream is decorrelated
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

    /** @brief Reports total latency in samples (2 * fftSize, measured exact at
     *  unity ratio; ~85 ms at the default 2048 frame and 48 kHz). */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("PSHF"), 1);
        w.write("semitones", static_cast<float>(semitones_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("transient", transientPreserve_.load(std::memory_order_relaxed));
        w.write("formant", formantPreserve_.load(std::memory_order_relaxed));
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

        // Linear per-block mix ramp with exact landing (settled: step == 0 and
        // the per-sample value reduces to the constant, bit-identically).
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;
        const T mixStep   = (nS > 0) ? (mixTarget - mixStart) / static_cast<T>(nS) : T(0);

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

                for (int k = 0; k < chunk; ++k)
                {
                    const T wet = readCatmullRom(acc, rp, rf);
                    const int dryIdx = (dp - latency_) & dryMask_;
                    const T drySample = dry[static_cast<size_t>(dryIdx)];
                    const T mixVal = mixStart + mixStep * static_cast<T>(i + k);
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
            }

            // 3. Advance the engine (runs an STFT hop at the analysis boundary).
            engine_.commitInput(chunk, nCh);

            i += chunk;
        }

        currentMix_ = mixTarget;   // exact landing
    }

private:
    /** @brief Publishes the engine's parameter set from the atomic parameters
     *  (control thread; one seqlock publish per setter call). */
    void publishEngineParams() noexcept
    {
        typename detail::PhaseVocoderEngine<T>::Params p;
        p.targetSemitones = static_cast<double>(semitones_.load(std::memory_order_relaxed));
        p.transientPreserve = transientPreserve_.load(std::memory_order_relaxed);
        p.formantPreserve = formantPreserve_.load(std::memory_order_relaxed);
        engine_.publishParams(p);
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

    // -- Members -----------------------------------------------------------------
    int numChannels_ = 0;
    std::atomic<bool> prepared_ { false };

    int readOffset_ = 2560;   ///< Reader distance behind the write head.
    int latency_ = 4096;      ///< Measured wet latency (= readOffset_ + N - Rs).
    int drySize_ = 8192;
    int dryMask_ = 8191;
    int64_t accumMask_ = 8191;   ///< Cached engine OLA ring mask.

    detail::PhaseVocoderEngine<T> engine_;   ///< Shared analysis/synthesis core.

    std::vector<std::vector<T>> dryRing_;    ///< Per-channel latency-matched dry.

    int dryPos_ = 0;
    int64_t readPosInt_ = 0;
    double readPosFrac_ = 0.0;
    T currentMix_ = T(1);     ///< Audio-thread mix ramp state (exact landing).

    std::atomic<T> semitones_ { T(0) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<bool> transientPreserve_ { true };
    std::atomic<bool> formantPreserve_ { false };
};

} // namespace dspark
