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
 * Dependencies: Core/FFT.h, Core/WindowFunctions.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/StateBlob.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
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

        sampleRate_  = spec.sampleRate;
        numChannels_ = std::max(1, spec.numChannels);
        fftSize_     = fftSize;
        numBins_     = fftSize_ / 2 + 1;
        synthHop_    = fftSize_ / 4;
        ringMask_    = fftSize_ - 1;

        accumSize_ = fftSize_ * 4;          // power of two: frame + read margin
        accumMask_ = accumSize_ - 1;

        // The reader trails the write head by readOffset_; the window/OLA
        // chain adds another fftSize - synthHop_, so the measured wet latency
        // is readOffset_ + fftSize_ - synthHop_ = 2 * fftSize (exact at unity
        // ratio). The dry path must delay by the SAME value or partial mixes
        // comb-filter.
        readOffset_ = fftSize_ + synthHop_;
        latency_    = readOffset_ + fftSize_ - synthHop_;
        drySize_ = 1;
        while (drySize_ < latency_ + 1) drySize_ <<= 1;
        dryMask_ = drySize_ - 1;

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize_));

        window_.resize(static_cast<size_t>(fftSize_));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        for (auto& w : window_) w = std::sqrt(w);   // sqrt-Hann analysis+synthesis

        const int nCh = numChannels_;
        inputRing_.assign(static_cast<size_t>(nCh), {});
        dryRing_.assign(static_cast<size_t>(nCh), {});
        accum_.assign(static_cast<size_t>(nCh), {});
        for (int ch = 0; ch < nCh; ++ch)
        {
            inputRing_[static_cast<size_t>(ch)].assign(static_cast<size_t>(fftSize_), T(0));
            dryRing_[static_cast<size_t>(ch)].assign(static_cast<size_t>(drySize_), T(0));
            accum_[static_cast<size_t>(ch)].assign(static_cast<size_t>(accumSize_), T(0));
        }

        fftIn_.resize(static_cast<size_t>(fftSize_));
        spec_.resize(static_cast<size_t>(fftSize_ + 2));
        fftResult_.resize(static_cast<size_t>(fftSize_));
        prevAnalysis_.resize(static_cast<size_t>(fftSize_ + 2));
        prevSynth_.resize(static_cast<size_t>(fftSize_ + 2));

        // Formant preservation scratch (cepstral envelope).
        cepsTime_.resize(static_cast<size_t>(fftSize_));
        cepsSpec_.resize(static_cast<size_t>(fftSize_ + 2));
        envLog_.resize(static_cast<size_t>(numBins_));
        formantGain_.resize(static_cast<size_t>(numBins_));
        mag_.resize(static_cast<size_t>(numBins_));
        prevMag_.resize(static_cast<size_t>(numBins_));
        rotRe_.resize(static_cast<size_t>(numBins_));
        rotIm_.resize(static_cast<size_t>(numBins_));
        peakBin_.resize(static_cast<size_t>(numBins_ / 2 + 2));

        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears all signal state (keeps parameters). Safe on the audio thread. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        for (auto& r : inputRing_) std::fill(r.begin(), r.end(), T(0));
        for (auto& r : dryRing_)   std::fill(r.begin(), r.end(), T(0));
        for (auto& a : accum_)     std::fill(a.begin(), a.end(), T(0));
        std::fill(prevAnalysis_.begin(), prevAnalysis_.end(), T(0));
        std::fill(prevSynth_.begin(), prevSynth_.end(), T(0));
        std::fill(prevMag_.begin(), prevMag_.end(), T(0));

        inputPos_ = 0;
        dryPos_ = 0;
        writeHead_ = static_cast<int64_t>(accumSize_);       // keep indices positive
        readPosInt_ = writeHead_ - readOffset_;
        readPosFrac_ = 0.0;

        stActive_ = std::clamp(static_cast<double>(semitones_.load(std::memory_order_relaxed)),
                               -12.0, 12.0);
        ratioActive_ = std::exp2(stActive_ / 12.0);
        currentMix_ = mix_.load(std::memory_order_relaxed);
        hopCarry_ = 0.0;
        inputSinceHop_ = 0;
        analysisHop_ = computeAnalysisHop();
        onsetEnv_ = 0.0;
        firstFrame_ = true;
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
    }

    /** @brief Sets the pitch shift as a frequency ratio, clamped to [0.5, 2].
     *  Non-finite values are ignored. */
    void setPitchRatio(T ratio) noexcept
    {
        if (!std::isfinite(ratio)) return;
        ratio = std::clamp(ratio, T(0.5), T(2));
        semitones_.store(static_cast<T>(12.0 * std::log2(static_cast<double>(ratio))),
                         std::memory_order_relaxed);
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
            const int chunk = std::min(nS - i, analysisHop_ - inputSinceHop_);

            // 1. Push input into the analysis ring and the dry-compensation ring.
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* in = buffer.getChannel(ch) + i;
                auto& ring = inputRing_[static_cast<size_t>(ch)];
                auto& dry  = dryRing_[static_cast<size_t>(ch)];
                int wp = inputPos_;
                int dp = dryPos_;
                for (int k = 0; k < chunk; ++k)
                {
                    ring[static_cast<size_t>(wp)] = in[k];
                    dry[static_cast<size_t>(dp)]  = in[k];
                    wp = (wp + 1) & ringMask_;
                    dp = (dp + 1) & dryMask_;
                }
            }

            // 2. Produce output: fractional read of the synthesis stream + mix.
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = buffer.getChannel(ch) + i;
                const auto& acc = accum_[static_cast<size_t>(ch)];
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

                    rf += ratioActive_;
                    const auto adv = static_cast<int64_t>(rf);
                    rp += adv;
                    rf -= static_cast<double>(adv);
                    dp = (dp + 1) & dryMask_;
                }
            }

            // Commit shared positions once per chunk.
            {
                double rf = readPosFrac_ + ratioActive_ * chunk;
                const auto adv = static_cast<int64_t>(rf);
                readPosInt_ += adv;
                readPosFrac_ = rf - static_cast<double>(adv);
                inputPos_ = (inputPos_ + chunk) & ringMask_;
                dryPos_   = (dryPos_ + chunk) & dryMask_;
            }

            // 3. Run an STFT hop at the analysis boundary.
            inputSinceHop_ += chunk;
            if (inputSinceHop_ >= analysisHop_)
            {
                inputSinceHop_ = 0;
                processHop(nCh);
            }

            i += chunk;
        }

        currentMix_ = mixTarget;   // exact landing
    }

private:
    static constexpr double kTwoPi = 2.0 * std::numbers::pi;

    /** @brief Wraps a phase difference into (-pi, pi]. */
    [[nodiscard]] static double princArg(double x) noexcept
    {
        return x - kTwoPi * std::round(x / kTwoPi);
    }

    /** @brief Next analysis hop from the active ratio, with exact fractional carry. */
    [[nodiscard]] int computeAnalysisHop() noexcept
    {
        const double ideal = static_cast<double>(synthHop_) / ratioActive_ + hopCarry_;
        int hop = static_cast<int>(ideal);
        hop = std::clamp(hop, 1, fftSize_);
        hopCarry_ = ideal - static_cast<double>(hop);
        return hop;
    }

    /** @brief 4-point Catmull-Rom read of the synthesis accumulator. */
    [[nodiscard]] T readCatmullRom(const std::vector<T>& acc, int64_t ip, double frac) const noexcept
    {
        const auto m = static_cast<int64_t>(accumMask_);
        const T x0 = acc[static_cast<size_t>((ip - 1) & m)];
        const T x1 = acc[static_cast<size_t>(ip & m)];
        const T x2 = acc[static_cast<size_t>((ip + 1) & m)];
        const T x3 = acc[static_cast<size_t>((ip + 2) & m)];
        const T f  = static_cast<T>(frac);
        return x1 + T(0.5) * f * (x2 - x0
                 + f * (T(2) * x0 - T(5) * x1 + T(4) * x2 - x3
                 + f * (T(3) * (x1 - x2) + x3 - x0)));
    }

    /** @brief One analysis->synthesis frame for all channels. */
    void processHop(int nCh) noexcept
    {
        // --- glide the active ratio toward the target (max 0.5 st per hop) ----
        const double stTarget = std::clamp(
            static_cast<double>(semitones_.load(std::memory_order_relaxed)), -12.0, 12.0);
        stActive_ += std::clamp(stTarget - stActive_, -0.5, 0.5);
        ratioActive_ = std::exp2(stActive_ / 12.0);

        // --- reference-channel analysis ---------------------------------------
        analyzeChannel(0);

        const bool transient = detectTransient();
        buildRotations(transient);

        // Snapshot the reference analysis spectrum for the next frame's
        // heterodyned phase increments (after decisions, before modification).
        std::copy(spec_.begin(), spec_.end(), prevAnalysis_.begin());
        std::copy(mag_.begin(), mag_.end(), prevMag_.begin());

        // --- synthesis: reference channel first, then the rest -----------------
        synthesizeChannel(0, true);
        for (int ch = 1; ch < nCh; ++ch)
        {
            analyzeChannel(ch);
            synthesizeChannel(ch, false);
        }

        analysisHop_ = computeAnalysisHop();
        writeHead_ += synthHop_;
        firstFrame_ = false;
    }

    /** @brief Windows the input ring into fftIn_ and fills spec_ for `ch`. */
    void analyzeChannel(int ch) noexcept
    {
        const auto& ring = inputRing_[static_cast<size_t>(ch)];
        const int readPos = inputPos_;   // oldest sample (ring size == fftSize)
        for (int k = 0; k < fftSize_; ++k)
        {
            const int idx = (readPos + k) & ringMask_;
            fftIn_[static_cast<size_t>(k)] = ring[static_cast<size_t>(idx)]
                                           * window_[static_cast<size_t>(k)];
        }
        fft_->forward(fftIn_.data(), spec_.data());

        if (ch == 0)
        {
            for (int k = 0; k < numBins_; ++k)
            {
                const T re = spec_[static_cast<size_t>(2 * k)];
                const T im = spec_[static_cast<size_t>(2 * k + 1)];
                mag_[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
            }
        }
    }

    /** @brief Frame-energy onset detector (6 dB rise over tracked envelope). */
    [[nodiscard]] bool detectTransient() noexcept
    {
        double energy = 0.0;
        for (int k = 0; k < numBins_; ++k)
        {
            const double m = static_cast<double>(mag_[static_cast<size_t>(k)]);
            energy += m * m;
        }
        const double prevEnv = onsetEnv_;
        onsetEnv_ = std::max(energy, onsetEnv_ * 0.7);
        if (!transientPreserve_.load(std::memory_order_relaxed))
            return firstFrame_;
        return firstFrame_ || (energy > 4.0 * prevEnv && energy > 1e-12);
    }

    /**
     * @brief Peak picking + phase propagation on the reference channel.
     *
     * Fills rotRe_/rotIm_ with one rigid rotation phasor per bin (constant
     * across each peak's region of influence). On transients the rotation is
     * identity, which resets synthesis phases to analysis phases.
     */
    void buildRotations(bool transient) noexcept
    {
        if (transient)
        {
            std::fill(rotRe_.begin(), rotRe_.end(), T(1));
            std::fill(rotIm_.begin(), rotIm_.end(), T(0));
            return;
        }

        // --- find spectral peaks (local maxima over +-2 bins, above floor) -----
        T maxMag = T(0);
        for (int k = 0; k < numBins_; ++k)
            maxMag = std::max(maxMag, mag_[static_cast<size_t>(k)]);

        numPeaks_ = 0;
        if (maxMag > T(1e-9))
        {
            const T floorMag = maxMag * T(1e-4);   // -80 dB relative floor
            const int last = numBins_ - 2;
            for (int k = 2; k <= last; ++k)
            {
                const T m = mag_[static_cast<size_t>(k)];
                if (m < floorMag) continue;
                if (m > mag_[static_cast<size_t>(k - 1)] && m >= mag_[static_cast<size_t>(k + 1)]
                    && m > mag_[static_cast<size_t>(k - 2)] && m >= mag_[static_cast<size_t>(k + 2)])
                {
                    peakBin_[static_cast<size_t>(numPeaks_++)] = k;
                    k += 2;   // a neighbour cannot also be a peak
                }
            }
        }

        if (numPeaks_ == 0)
        {
            std::fill(rotRe_.begin(), rotRe_.end(), T(1));
            std::fill(rotIm_.begin(), rotIm_.end(), T(0));
            return;
        }

        // --- per-peak heterodyned phase propagation ----------------------------
        // analysisHop_ still holds the hop just consumed (it is recomputed
        // after this call), which is exactly the Ra of this frame interval.
        const double Ra = static_cast<double>(analysisHop_);
        const double Rs = static_cast<double>(synthHop_);
        const double binW = kTwoPi / static_cast<double>(fftSize_);

        int regionStart = 0;
        for (int p = 0; p < numPeaks_; ++p)
        {
            const int bin = peakBin_[static_cast<size_t>(p)];
            const double re  = static_cast<double>(spec_[static_cast<size_t>(2 * bin)]);
            const double im  = static_cast<double>(spec_[static_cast<size_t>(2 * bin + 1)]);
            const double pre = static_cast<double>(prevAnalysis_[static_cast<size_t>(2 * bin)]);
            const double pim = static_cast<double>(prevAnalysis_[static_cast<size_t>(2 * bin + 1)]);

            double rotR = 1.0, rotI = 0.0;
            // A peak with no spectral history is a fresh partial: keep its
            // analysis phase instead of propagating from noise.
            if (prevMag_[static_cast<size_t>(bin)] > T(0.1) * mag_[static_cast<size_t>(bin)])
            {
                // Measured inter-frame phase advance, inherently wrapped.
                const double deltaPhi = std::atan2(im * pre - re * pim, re * pre + im * pim);
                const double omegaK = binW * static_cast<double>(bin);
                const double omegaInst = omegaK + princArg(deltaPhi - omegaK * Ra) / Ra;

                const double psiPrev = std::atan2(
                    static_cast<double>(prevSynth_[static_cast<size_t>(2 * bin + 1)]),
                    static_cast<double>(prevSynth_[static_cast<size_t>(2 * bin)]));
                const double phi = std::atan2(im, re);
                const double theta = psiPrev + Rs * omegaInst - phi;
                rotR = std::cos(theta);
                rotI = std::sin(theta);
            }

            // Region of influence: up to the magnitude valley before the next peak.
            int regionEnd = numBins_;   // exclusive
            if (p + 1 < numPeaks_)
            {
                const int nextBin = peakBin_[static_cast<size_t>(p + 1)];
                int valley = bin + 1;
                T valleyMag = mag_[static_cast<size_t>(valley)];
                for (int k = bin + 2; k < nextBin; ++k)
                {
                    if (mag_[static_cast<size_t>(k)] < valleyMag)
                    {
                        valleyMag = mag_[static_cast<size_t>(k)];
                        valley = k;
                    }
                }
                regionEnd = valley + 1;
            }

            for (int k = regionStart; k < regionEnd; ++k)
            {
                rotRe_[static_cast<size_t>(k)] = static_cast<T>(rotR);
                rotIm_[static_cast<size_t>(k)] = static_cast<T>(rotI);
            }
            regionStart = regionEnd;
        }

        // DC and Nyquist are real-valued in the packed spectrum: never rotate.
        rotRe_[0] = T(1);                                     rotIm_[0] = T(0);
        rotRe_[static_cast<size_t>(numBins_ - 1)] = T(1);     rotIm_[static_cast<size_t>(numBins_ - 1)] = T(0);
    }

    /** @brief Applies rotations + anti-alias cut to spec_, IFFTs and overlap-adds. */
    void synthesizeChannel(int ch, bool isReference) noexcept
    {
        // Rigid per-region rotation preserves intra-region (and inter-channel)
        // relative phases exactly.
        for (int k = 1; k < numBins_ - 1; ++k)
        {
            const T re = spec_[static_cast<size_t>(2 * k)];
            const T im = spec_[static_cast<size_t>(2 * k + 1)];
            const T rr = rotRe_[static_cast<size_t>(k)];
            const T ri = rotIm_[static_cast<size_t>(k)];
            spec_[static_cast<size_t>(2 * k)]     = re * rr - im * ri;
            spec_[static_cast<size_t>(2 * k + 1)] = re * ri + im * rr;
        }

        // Formant preservation: pre-warp synthesis magnitudes by
        // env(k*ratio)/env(k) so the output resampler puts the spectral
        // envelope back where the input had it. The gain table is computed
        // once on the reference channel and shared, keeping the stereo
        // image coherent.
        if (formantPreserve_.load(std::memory_order_relaxed)
            && std::abs(ratioActive_ - 1.0) > 1e-6)
        {
            if (isReference)
                computeFormantGains();
            for (int k = 1; k < numBins_ - 1; ++k)
            {
                const T g = formantGain_[static_cast<size_t>(k)];
                spec_[static_cast<size_t>(2 * k)]     *= g;
                spec_[static_cast<size_t>(2 * k + 1)] *= g;
            }
        }

        // Anti-alias for upward shifts: the resampler multiplies frequencies
        // by `ratio`, so taper bins that would land above Nyquist.
        if (ratioActive_ > 1.0)
        {
            const int cut = static_cast<int>(static_cast<double>(fftSize_ / 2) / ratioActive_);
            const int taperStart = std::max(1, cut - 4);
            for (int k = taperStart; k < numBins_; ++k)
            {
                T g = T(0);
                if (k <= cut)
                    g = static_cast<T>(cut - k + 1) / static_cast<T>(cut - taperStart + 1);
                spec_[static_cast<size_t>(2 * k)]     *= g;
                spec_[static_cast<size_t>(2 * k + 1)] *= g;
            }
        }

        if (isReference)
            std::copy(spec_.begin(), spec_.end(), prevSynth_.begin());

        fft_->inverse(spec_.data(), fftResult_.data());

        synthesizeTail(ch);
    }

    /**
     * @brief Cepstral envelope of the current synthesis spectrum and the
     * env(k*ratio)/env(k) pre-warp gains.
     *
     * The log magnitude is mirrored to an even sequence, transformed, low-
     * quefrency liftered (~1 ms keeps formant structure, drops the harmonic
     * comb) and transformed back into a smooth log envelope.
     */
    void computeFormantGains() noexcept
    {
        // Even-symmetric log magnitude.
        for (int k = 0; k < numBins_; ++k)
        {
            const T re = spec_[static_cast<size_t>(2 * k)];
            const T im = spec_[static_cast<size_t>(2 * k + 1)];
            cepsTime_[static_cast<size_t>(k)] =
                std::log(std::sqrt(re * re + im * im) + T(1e-9));
        }
        for (int k = numBins_; k < fftSize_; ++k)
            cepsTime_[static_cast<size_t>(k)] =
                cepsTime_[static_cast<size_t>(fftSize_ - k)];

        fft_->forward(cepsTime_.data(), cepsSpec_.data());

        // Lifter: keep quefrencies below ~1 ms (the smooth envelope), zero
        // the rest (the harmonic comb). Cepstral index is quefrency in
        // samples: 1 ms = 0.001 * fs.
        const int qCut = std::max(8, static_cast<int>(0.001 * sampleRate_));
        const int keep = std::min(qCut, numBins_ - 1);
        for (int k = keep + 1; k < numBins_; ++k)
        {
            cepsSpec_[static_cast<size_t>(2 * k)]     = T(0);
            cepsSpec_[static_cast<size_t>(2 * k + 1)] = T(0);
        }

        fft_->inverse(cepsSpec_.data(), cepsTime_.data());
        for (int k = 0; k < numBins_; ++k)
            envLog_[static_cast<size_t>(k)] = cepsTime_[static_cast<size_t>(k)];

        // Gains with linear interpolation at k*ratio, clamped to +-40 dB.
        for (int k = 0; k < numBins_; ++k)
        {
            const double pos = std::min(static_cast<double>(k) * ratioActive_,
                                        static_cast<double>(numBins_ - 1));
            const auto i0 = static_cast<int>(pos);
            const auto frac = static_cast<T>(pos - i0);
            const T target = envLog_[static_cast<size_t>(i0)]
                + (envLog_[static_cast<size_t>(std::min(i0 + 1, numBins_ - 1))]
                   - envLog_[static_cast<size_t>(i0)]) * frac;
            const T delta = std::clamp(target - envLog_[static_cast<size_t>(k)],
                                       T(-4.6), T(4.6));
            formantGain_[static_cast<size_t>(k)] = std::exp(delta);
        }
    }

    /** @brief Overlap-add of the already-inverted frame (split for clarity). */
    void synthesizeTail(int ch) noexcept
    {

        // Overlap-add at the fixed synthesis hop. sqrt-Hann analysis+synthesis
        // with Rs = N/4 overlap-adds to a constant 2.0, hence the 0.5 norm.
        auto& acc = accum_[static_cast<size_t>(ch)];
        const auto m = static_cast<int64_t>(accumMask_);

        // The tail region [w + N - Rs, w + N) is entered for the first time by
        // this frame: clear the stale ring content before accumulating.
        for (int k = fftSize_ - synthHop_; k < fftSize_; ++k)
            acc[static_cast<size_t>((writeHead_ + k) & m)] = T(0);

        constexpr T kNorm = T(0.5);
        for (int k = 0; k < fftSize_; ++k)
        {
            const auto idx = static_cast<size_t>((writeHead_ + k) & m);
            acc[idx] += fftResult_[static_cast<size_t>(k)]
                      * window_[static_cast<size_t>(k)] * kNorm;
        }
    }

    // -- Members -----------------------------------------------------------------
    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    std::atomic<bool> prepared_ { false };

    int fftSize_ = 2048;
    int numBins_ = 1025;
    int synthHop_ = 512;
    int ringMask_ = 2047;
    int accumSize_ = 8192;
    int accumMask_ = 8191;
    int readOffset_ = 2560;   ///< Reader distance behind the write head.
    int latency_ = 4096;      ///< Measured wet latency (= readOffset_ + N - Rs).
    int drySize_ = 8192;
    int dryMask_ = 8191;

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_;

    std::vector<std::vector<T>> inputRing_;   ///< Per-channel analysis ring (size N).
    std::vector<std::vector<T>> dryRing_;     ///< Per-channel latency-matched dry.
    std::vector<std::vector<T>> accum_;       ///< Per-channel synthesis OLA ring.

    std::vector<T> fftIn_, spec_, fftResult_;
    std::vector<T> prevAnalysis_, prevSynth_; ///< Reference-channel spectra (re,im).
    std::vector<T> mag_, prevMag_;            ///< Reference-channel magnitudes.
    std::vector<T> cepsTime_, cepsSpec_;      ///< Formant cepstrum scratch.
    std::vector<T> envLog_, formantGain_;     ///< Envelope + pre-warp gains.
    std::vector<T> rotRe_, rotIm_;            ///< Per-bin rigid rotation phasor.
    std::vector<int> peakBin_;
    int numPeaks_ = 0;

    int inputPos_ = 0;
    int dryPos_ = 0;
    int64_t writeHead_ = 0;
    int64_t readPosInt_ = 0;
    double readPosFrac_ = 0.0;

    double stActive_ = 0.0;
    double ratioActive_ = 1.0;
    double hopCarry_ = 0.0;
    int analysisHop_ = 512;
    int inputSinceHop_ = 0;
    double onsetEnv_ = 0.0;
    bool firstFrame_ = true;
    T currentMix_ = T(1);     ///< Audio-thread mix ramp state (exact landing).

    std::atomic<T> semitones_ { T(0) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<bool> transientPreserve_ { true };
    std::atomic<bool> formantPreserve_ { false };
};

} // namespace dspark
