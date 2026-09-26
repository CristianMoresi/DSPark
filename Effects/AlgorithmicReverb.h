// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file AlgorithmicReverb.h
 * @brief True-stereo 32-line FDN reverb with exact per-band decay and a
 *        dispersive spring-tank model.
 *
 * Architecture:
 * ```
 * Input L                         Input R
 *   |                               |
 * [Pre-delay]                     [Pre-delay]
 *   |                               |
 *   +--> [Early reflections: velvet-noise taps per side, density rising
 *   |     with time; the first few discrete (raw input), the rest from the
 *   |     diffused feed; ipsilateral early, diffuse later; grouped air/wall
 *   |     absorption]
 *   |
 * [Input diffusion: 4 allpass]    [Input diffusion: 4 allpass, other delays]
 *   |                               |
 *   +--> [Late injection: one tap per line, L/R interleaved, after the
 *   |     ER-to-late gap]
 *   |       |
 *   |       v
 *   |    [FDN core: 32 lines]
 *   |      +- allpass-interpolated read at a smoothly modulated length
 *   |      +- Jot absorption shelf (mid/high T60) per line
 *   |      +- bass shelf (bass/mid T60) per line   --> output taps
 *   |      +- Hadamard 32x32 mix + line rotation (lossless, dense)
 *   |      +- short in-loop allpass (echo density), scaled with the size
 *   |      +- safety soft limit, write back + injection
 *   |       |
 *   |       v
 *   |    [Output: orthogonal L/R sign sums of the taps -> 2 allpass
 *   |     diffusers per side -> coherent low band -> width]
 *   v
 * [Early + late] -> [DC block] -> [Tone EQ] -> DryWetMixer -> Output
 * ```
 * Type::Spring replaces the FDN with twelve spring tanks (see runSprings()).
 *
 * Design notes:
 * - **True stereo.** Each input channel has its own pre-delay, diffusion and
 *   early reflections, so a left source stays on the left in the early field
 *   while the late tail becomes diffuse (early taps start lateralised and
 *   blend toward equal L/R as they get later, like a real room).
 * - **Realistic early field.** The early reflections are a velvet-noise
 *   sequence (Valimaki et al.): sparse +-1 taps whose density rises with
 *   time, like a room's echo density, spectrally flat and free of the
 *   periodic ring of chained allpasses. setDiffusion() moves the split
 *   between the first discrete reflections (crisp, localisable) and the
 *   smooth diffuse ones.
 * - **Exact decay per band.** Each line carries two first-order shelves
 *   designed by the bilinear transform: the Jot shelf pins the loop gain at
 *   DC to the mid T60 and at Nyquist to the HF T60 (midpoint at the high
 *   crossover), the bass shelf pins DC to the bass T60 and leaves the top at
 *   unity (midpoint at the bass crossover). The gains use each line's full
 *   loop length (line plus the in-loop allpass's mean group delay). There is
 *   no DC blocker or smoothing filter in the loop, so nothing else bends the
 *   decay: measured T60s land within a few percent of the targets in every
 *   octave, at any sample rate.
 * - **Room-like statistics.** Benchmarked against measured halls, churches
 *   and theatres (the Detmold SRIR database and CT-AudioLink) and against
 *   an ideal diffuse field (Gaussian noise with the same decay per band),
 *   the tail matches both: its fine spectrum, its sub-Hz modal peaks
 *   (within about 0.8 dB of the ideal at the 99.9th percentile, rooms
 *   0.4 dB), its envelope in every octave, its echo density (0.96 of
 *   Gaussian over 20-150 ms, rooms 0.98) and the shape of its decay (the
 *   early decay time is 0.93-1.05 of the T30, rooms 0.84-1.03). This rests
 *   on five choices:
 *   - 32 lines with about 3.7 s of delay at full size: the modal density
 *     (modes per Hz equals the total delay in seconds) keeps the modes
 *     overlapping, and the Hadamard matrix plus a line rotation feeds
 *     every line from every other with equal weight;
 *   - short input diffusers (0.4-1.8 ms): an allpass holds back some
 *     frequencies longer than others, and in a short decay those would
 *     be heard ringing on;
 *   - in-loop allpasses scaled with the room size, so their group-delay
 *     ripple stays a small share of every loop (no frequency lingers in a
 *     small room);
 *   - two output allpasses per side, so every echo of the tank leaves as
 *     a short dense burst and the late field is smooth from its onset;
 *   - the output taps follow the absorption, so even the first pass
 *     through a line leaves attenuated by its own decay: the tail decays
 *     exponentially from its first echoes instead of holding a plateau
 *     for one round trip.
 * - **Natural stereo bass.** A diffuse field is coherent between the ears
 *   at low frequencies. The late tail's side signal is high-passed (2nd
 *   order at 220 Hz), so its L/R correlation is 0.98 below 100 Hz, about
 *   0.35 at 250-500 Hz and 0 above 1 kHz: no phasey low end, full width on
 *   top.
 * - **Transparent modulation.** Each line's length wanders with its own
 *   smooth random LFO (Lexicon style), updated at control rate with a
 *   per-sample linear ramp and read through a first-order allpass
 *   interpolator kept in its low-dispersion range ([0.5, 1.5) samples of
 *   fractional delay), so there is no cumulative HF loss. Depth is set in
 *   time and scales with the room size: the same chorus at every sample
 *   rate, and small rooms do not warble. The presets smear the tail's
 *   remaining resonances while keeping steady tones clean: over tones
 *   from 400 Hz to 3 kHz, the energy more than 3 Hz from the tone is
 *   34-39 dB down (Plate, lusher by design, about 22 dB).
 * - **Smooth size changes.** setSize() glides the line and in-loop allpass
 *   lengths (a short tape-style Doppler) instead of jumping, so it can be
 *   automated.
 * - **Spring tanks.** Type::Spring models six springs per side as
 *   dispersive loops (Valimaki, Parker & Abel 2010): stretched-allpass
 *   cascades make every round trip a chirp whose band edge arrives last,
 *   re-dispersed on each pass, band-limited like a real tank. Six springs
 *   of different lengths with a little round-trip jitter keep long decays
 *   from ringing metallic (the tail spectrum's ripple is half that of a
 *   two-spring tank). The springs run in SIMD lanes side by side.
 *   setDiffusion() sets the chirp strength for this type.
 * - **Efficient.** The sparse early-reflection FIR runs tap by tap over
 *   64-sample chunks as SIMD multiply-adds on contiguous memory; the
 *   per-line arithmetic runs in SIMD around the scalar delay-line reads,
 *   and the delay lines are padded so their streams do not collide in the
 *   cache. A stereo hall costs about 2% of one core at 48 kHz.
 * - **Eco quality** (setQuality): 16 lines, 40 early taps per side and one
 *   spring per side, with the same decay calibration and loudness (within
 *   0.5 dB), at about 60% of the CPU.
 *
 * Threading: prepare() belongs to the setup thread (allocates). processBlock(),
 * processSample() and reset() belong to the audio thread. All setters are
 * lock-free atomic publications, safe from any thread; topology changes
 * (type, quality) and coefficient refreshes are drained at the start of the
 * next processBlock()/processSample() on the audio thread. Non-finite setter
 * arguments are ignored.
 *
 * Four levels of API complexity:
 *
 * - **Level 1:** `reverb.setType(Hall);`
 * - **Level 2:** `reverb.setSize(0.7f); reverb.setDamping(0.3f);`
 * - **Level 3:** `reverb.setHighDecayMultiplier(0.4f);`
 * - **Level 4:** Inherit and override protected members.
 *
 * References:
 * - Jot & Chaigne (1991): FDN with frequency-dependent decay (shelving absorption)
 * - Schlecht & Habets (2017, 2019): FDN mixing matrices, echo density, modulation
 * - Valimaki, Parker & Abel (2010, JAES): parametric spring reverberation
 * - Dattorro (1997, JAES): plate topology, allpass-interpolated modulation
 * - Griesinger / Lexicon 480L: random modulation
 * - Valimaki et al. (2012, IEEE): "50 Years of Artificial Reverberation"
 * - Abel & Huang (2006): normalized echo density
 *
 * Dependencies: DryWetMixer.h, DspMath.h, Biquad.h, AudioSpec.h,
 *               AudioBuffer.h, DenormalGuard.h, StateBlob.h, SimdOps.h.
 *
 * @code
 *   dspark::AlgorithmicReverb<float> reverb;
 *   reverb.prepare(spec);
 *   reverb.setType(dspark::AlgorithmicReverb<float>::Type::Hall);
 *   reverb.setDecay(2.0f);
 *   reverb.setMix(0.3f);
 *   reverb.processBlock(buffer);
 *
 *   // Advanced: tune frequency-dependent decay
 *   reverb.setHighDecayMultiplier(0.4f);  // HF decays 2.5x faster
 *   reverb.setBassDecayMultiplier(1.3f);  // bass lingers 1.3x longer
 * @endcode
 */

#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Biquad.h"
#include "../Core/StateBlob.h"
#include "../Core/SimdOps.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class AlgorithmicReverb
 * @brief True-stereo 32-line FDN reverb with exact per-band decay and 6 presets.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class AlgorithmicReverb
{
public:
    /** @brief Reverb type presets. */
    enum class Type
    {
        Room,       ///< Small room, short decay, dense close reflections.
        Hall,       ///< Concert hall, spacious, long smooth tail.
        Chamber,    ///< Recording studio chamber, warm, balanced.
        Plate,      ///< Metal plate, dense shimmer, no early reflections.
        Spring,     ///< Spring reverb, bouncy vintage character.
        Cathedral   ///< Large cathedral, immense decay, vast space.
    };

    /**
     * @brief Engine quality / CPU cost trade-off (see setQuality()).
     */
    enum class Quality
    {
        Full,   ///< Complete 32-line engine. Default.
        Eco     ///< Reduced 8-line engine, about 3x cheaper. For constrained targets.
    };

    ~AlgorithmicReverb() = default; // non-virtual: leaf class (no virtual dispatch)

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the reverberation engine and allocates required memory.
     *
     * Must be called before processing. An invalid spec (non-positive or
     * non-finite fields) is a no-op that keeps the previous state.
     *
     * @param spec Audio specification detailing sample rate and maximum block size.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        spec_ = spec;
        mixer_.prepare(spec);
        const double sr = spec.sampleRate;

        // Re-derive the sample counts stored at set-time: after a re-prepare
        // at a different rate they would keep the OLD rate's sample count.
        preDelaySamples_.store(msToSamples(preDelayMs_.load(std::memory_order_relaxed)),
                               std::memory_order_relaxed);
        erToLateSamples_.store(msToSamples(erToLateMs_.load(std::memory_order_relaxed)),
                               std::memory_order_relaxed);

        const auto samples = [sr](double ms) { return static_cast<int>(ms * sr / 1000.0) + 8; };
        for (int c = 0; c < 2; ++c)
        {
            // The early taps read the raw input straight from here.
            preDelay_[c].prepare(samples(kMaxPreDelayMs + kMaxErMs));
            // The diffused ring feeds the early taps (up to 170 ms, their
            // contralateral copies up to 1.3x that) and the late injection
            // (ER-to-late gap plus the injection spread).
            ring_[c].prepare(samples(kMaxErToLateMs + kMaxInjectMs));
            for (auto& ap : inAP_[c]) ap.prepare(samples(kMaxInDiffMs));
            for (int k = 0; k < kOutStages; ++k)
            {
                outAP_[c][k].prepare(samples(kOutDiffMs_[c][k]));
                outAPLen_[c][k] = nearestPrime(std::max(1, static_cast<int>(kOutDiffMs_[c][k] * sr / 1000.0)));
            }
        }
        const int maxLine = samples(kBaseDelaysMs_[kMaxLines - 1] + kModMaxMs) + 4;
        lines_.prepare(kMaxLines, maxLine);
        loopAP_.prepare(kMaxLines, samples(kMaxLoopApMs));

        // Spring dispersion: stretching factor K puts the chirp band edge at
        // fs / (2K) ~ kSpringCutHz; a 4th-order Butterworth just below it
        // removes the stretched allpasses' mirrored bands.
        springK_ = std::max(1, static_cast<int>(std::lround(sr / (2.0 * kSpringCutHz))));
        springP_ = 1;
        while (springP_ < springK_ + 1) springP_ <<= 1;
        springHist_.assign(static_cast<std::size_t>(kSprings * (kSpringStages + 1) * springP_), T(0));
        {
            const double fc = 0.9 * sr / (2.0 * springK_);
            const double qs[2] = { 0.5411961001461970, 1.3065629648763764 };
            for (int k = 0; k < 2; ++k)
            {
                const auto c = BiquadCoeffs::makeLowPass(sr, fc, qs[k]);
                springLP_[k] = { static_cast<T>(c.b0), static_cast<T>(c.b1), static_cast<T>(c.b2),
                                 static_cast<T>(c.a1), static_cast<T>(c.a2) };
            }
        }

        const T srT = static_cast<T>(sr);
        dcR_ = T(1) - T(6.283185307179586) * T(kDcCutHz) / srT;
        cohCoeff_ = static_cast<T>(1.0 - std::exp(-6.283185307179586 * kCoherenceHz / sr));
        maxReadPos_ = static_cast<T>(maxLine - 2);

        eco_ = quality_.load(std::memory_order_relaxed) == Quality::Eco;
        spring_ = type_.load(std::memory_order_relaxed) == Type::Spring;
        refreshTopology();
        prepared_ = true;

        applyPreset(type_.load(std::memory_order_relaxed));
        // prepare() applied whatever the caller configured before it, so no
        // drain is needed on the first processBlock.
        presetDirty_.store(false, std::memory_order_relaxed);
        paramsDirty_.store(false, std::memory_order_relaxed);
        qualityDirty_.store(false, std::memory_order_relaxed);
        toneDirty_.store(true, std::memory_order_relaxed);
        drainTone();
        reset();
    }

    /**
     * @brief Processes an audio block in place with zero allocations.
     *
     * Stereo buffers are reverberated in true stereo; a mono buffer feeds
     * both inputs and receives the average of the two outputs. Extra
     * channels beyond two are left untouched.
     *
     * @param buffer View of the audio buffers (mono or stereo).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), 2);
        const int nS  = buffer.getNumSamples();
        if (nCh == 0 || nS == 0 || !prepared_) return;

        // Front-door non-finite guard: the FDN is fully recursive, so a single
        // NaN/Inf input would poison the tail forever. No-op on finite input.
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                if (!std::isfinite(d[i])) d[i] = T(0);
        }

        drainPendingChanges();

        mixer_.pushDry(buffer);
        refreshCachedParams();

        T* chL = buffer.getChannel(0);
        T* chR = nCh >= 2 ? buffer.getChannel(1) : nullptr;
        alignas(64) T outL[kChunk];
        alignas(64) T outR[kChunk];
        for (int start = 0; start < nS; start += kChunk)
        {
            const int n = std::min(kChunk, nS - start);
            processChunk(chL + start, (chR ? chR : chL) + start, outL, outR, n);
            if (chR)
            {
                std::copy(outL, outL + n, chL + start);
                std::copy(outR, outR + n, chR + start);
            }
            else
            {
                for (int i = 0; i < n; ++i)
                    chL[start + i] = (outL[i] + outR[i]) * T(0.5);
            }
        }

        mixer_.mixWet(buffer, mix_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Processes a single mono sample and returns the wet stereo pair.
     *
     * The input feeds both reverb inputs; the result is the wet signal only
     * (the mix is not applied). Pending parameter, preset and quality changes
     * are drained here too.
     *
     * @param input The mono input sample to reverberate.
     * @return The {Left, Right} reverberated output.
     */
    [[nodiscard]] std::pair<T, T> processSample(T input) noexcept
    {
        if (!prepared_) return { T(0), T(0) };
        DenormalGuard guard;
        drainPendingChanges();
        refreshCachedParams();
        // Front-door non-finite guard (see processBlock).
        if (!std::isfinite(input)) input = T(0);
        T outL = T(0), outR = T(0);
        processChunk(&input, &input, &outL, &outR, 1);
        return { outL, outR };
    }

    /**
     * @brief Clears all delay lines and filter states and restarts the
     *        modulation, snapping any length glide to its target.
     */
    void reset() noexcept
    {
        for (int c = 0; c < 2; ++c)
        {
            preDelay_[c].clear();
            ring_[c].clear();
            for (auto& ap : inAP_[c]) ap.clear();
            for (auto& ap : outAP_[c]) ap.clear();
            erLP_[c].fill(T(0));
            dcX1_[c] = dcY1_[c] = T(0);
        }
        lines_.clear();
        loopAP_.clear();
        cohLP_.fill(T(0));
        std::fill(springHist_.begin(), springHist_.end(), T(0));
        springW_ = 0;
        for (auto& st : springLPState_) st.fill(T(0));
        apY1_.fill(T(0));
        jotX1_.fill(T(0));
        jotY1_.fill(T(0));
        bassX1_.fill(T(0));
        bassY1_.fill(T(0));

        const double sr = spec_.sampleRate > 0 ? spec_.sampleRate : 48000.0;
        const T rate = modRate_.load(std::memory_order_relaxed);
        for (int i = 0; i < kMaxLines; ++i)
        {
            lfo_[i].prepare(sr, rate * lfoRateFactor(i), lfoSeed(i));
            lenCur_[i] = static_cast<T>(lenTarget_[i]);
            loopAPCur_[i] = static_cast<T>(loopAPLen_[i]);
            pos_[i] = std::max(lenCur_[i], T(kMinReadPos));
            posInc_[i] = T(0);
        }
        ctrlPhase_ = 0;
        loopAPGliding_ = false;

        toneLPBiquad_.reset();
        toneHPBiquad_.reset();
        mixer_.reset();
    }

    // =========================================================================
    // Level 1: Simple API
    // =========================================================================

    /**
     * @brief Loads the selected preset baseline.
     *
     * Call setDecay(), setSize() and other parameter setters after setType()
     * when those values should override the selected preset. Applied at the
     * start of the next processBlock(); the running tail is cleared.
     */
    void setType(Type type) noexcept
    {
        // Wild enum values (a corrupted blob, a stray cast) clamp into range.
        const int t = std::clamp(static_cast<int>(type), 0,
                                 static_cast<int>(Type::Cathedral));
        type_.store(static_cast<Type>(t), std::memory_order_relaxed);
        // A preset load re-establishes the baseline: forget prior overrides so
        // the preset's own values apply, unless a setter is called AFTER this.
        userParamMask_.store(0u, std::memory_order_relaxed);
        presetDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Selects the engine quality / CPU cost trade-off.
     *
     * Quality::Full (default) runs the complete engine. Quality::Eco runs
     * at about 60% of its CPU for embedded and other constrained targets,
     * with the same controls, decay calibration and loudness:
     *
     * - 16 FDN lines instead of 32 (every other base delay, so the lines
     *   still span the full range; per-line decay gains keep every T60
     *   exact)
     * - 40 early-reflection taps per side instead of 96
     * - one spring per side instead of six, with a 40-stage dispersion
     *   cascade instead of 72 (a stronger coefficient keeps the chirp)
     *
     * Half the modal density is the audible difference: long, exposed
     * tails (a slow decay on sparse material) ring a little more, and the
     * build-up is slightly less dense; short and mid rooms sound almost
     * the same.
     *
     * Like setType(), this is an engine-mode switch, not an automation
     * target: it is applied at the start of the next processBlock() and
     * clears the reverb state (the running tail is dropped).
     *
     * @param q Quality::Full or Quality::Eco.
     */
    void setQuality(Quality q) noexcept
    {
        quality_.store(q == Quality::Eco ? Quality::Eco : Quality::Full,
                       std::memory_order_relaxed);
        qualityDirty_.store(true, std::memory_order_release);
    }

    /** @brief Mid-band decay time T60 in seconds (0.1 - 30). */
    void setDecay(T seconds) noexcept
    {
        if (!std::isfinite(seconds)) return;
        decayTime_.store(std::clamp(seconds, T(0.1), T(30)),
                         std::memory_order_relaxed);
        markUserParam(kUserDecay);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Dry/wet mix (0 = dry, 1 = wet). */
    void setMix(T dryWet) noexcept
    {
        if (!std::isfinite(dryWet)) return;
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed);
    }

    // =========================================================================
    // Level 2: Intermediate API
    // =========================================================================

    /**
     * @brief Room size (0.01 - 1). Scales the delay lines from 36% to 100%
     *        of their base lengths. Changes glide smoothly (with a brief
     *        Doppler, like a tape delay), so the size can be automated.
     */
    void setSize(T size) noexcept
    {
        if (!std::isfinite(size)) return;
        size_.store(std::clamp(size, T(0.01), T(1)), std::memory_order_relaxed);
        markUserParam(kUserSize);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets high-frequency damping (0 = bright, 1 = dark).
     *
     * Maps internally to highDecayMultiplier: 0->1.0 (HF=mid), 1->0.1 (HF 10x faster).
     * For finer control, use setHighDecayMultiplier() directly.
     */
    void setDamping(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        T clamped = std::clamp(amount, T(0), T(1));
        damping_.store(clamped, std::memory_order_relaxed);
        highDecayMult_.store(T(1) - clamped * T(0.9), std::memory_order_relaxed);
        markUserParam(kUserDamping);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Pre-delay before the early reflections, in ms (0 - 200). */
    void setPreDelay(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        T clamped = std::clamp(ms, T(0), T(kMaxPreDelayMs));
        preDelayMs_.store(clamped, std::memory_order_relaxed);
        if (spec_.sampleRate > 0)
            preDelaySamples_.store(msToSamples(clamped), std::memory_order_relaxed);
    }

    /**
     * @brief Diffusion (0 - 1): the strength of the input, in-loop and
     *        output allpass diffusers, and how many of the first early
     *        reflections stay discrete (7 at 0, 1 at 1). High values give a
     *        smooth, dense onset; low values keep crisp, localisable early
     *        echoes. For Type::Spring it sets the strength of the chirp.
     */
    void setDiffusion(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        diffusion_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        markUserParam(kUserDiffusion);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Modulation depth (0 - 1) of the delay-line wander. It smears
     *        the tail's resonances; 0 is fully static, 1 is a strong chorus.
     *        The depth is defined in time and scales with the room size.
     */
    void setModulation(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        modDepth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        markUserParam(kUserModDepth);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets stereo width of the late reverb tail.
     *
     * Uses M/S processing: 0 = mono, 1 = natural stereo, 2 = extra wide.
     * Applied to the late tail (its low band stays coherent), before the
     * early reflections are added.
     *
     * @param width Stereo width (0.0 - 2.0). Default: 1.0.
     */
    void setWidth(T width) noexcept
    {
        if (!std::isfinite(width)) return;
        width_.store(std::clamp(width, T(0), T(2)), std::memory_order_relaxed);
    }

    /** @brief Extra gap between the early reflections and the late tail, in ms (0 - 200). */
    void setErToLateDelay(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        T clamped = std::clamp(ms, T(0), T(kMaxErToLateMs));
        erToLateMs_.store(clamped, std::memory_order_relaxed);
        markUserParam(kUserErToLate);
        if (spec_.sampleRate > 0)
            erToLateSamples_.store(msToSamples(clamped), std::memory_order_relaxed);
    }

    // =========================================================================
    // Level 3: Expert API - Frequency-Dependent Decay
    // =========================================================================

    /**
     * @brief Sets HF decay as a multiplier of mid decay time.
     *
     * 0.1 = HF decays 10x faster (very dark).
     * 0.5 = HF decays 2x faster (natural room).
     * 1.0 = HF same as mid (bright/metallic).
     *
     * @param mult Multiplier (0.05 - 1.0).
     */
    void setHighDecayMultiplier(T mult) noexcept
    {
        if (!std::isfinite(mult)) return;
        T clamped = std::clamp(mult, T(0.05), T(1));
        highDecayMult_.store(clamped, std::memory_order_relaxed);
        damping_.store(std::clamp((T(1) - clamped) / T(0.9), T(0), T(1)),
                       std::memory_order_relaxed);
        markUserParam(kUserDamping);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets bass decay as a multiplier of mid decay time.
     *
     * 0.5 = bass decays 2x faster (tight).
     * 1.0 = bass same as mid (neutral).
     * 1.5 = bass lingers 1.5x longer (natural large room).
     * 2.0 = bass lingers 2x longer (boomy).
     *
     * @param mult Multiplier (0.3 - 3.0).
     */
    void setBassDecayMultiplier(T mult) noexcept
    {
        if (!std::isfinite(mult)) return;
        bassDecayMult_.store(std::clamp(mult, T(0.3), T(3)),
                             std::memory_order_relaxed);
        markUserParam(kUserBassDecay);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Frequency where the HF decay transition is centered.
     *
     * The per-line absorption is a first-order pole-zero shelf pinned to the
     * mid decay at DC and the HF decay at Nyquist (the T60 anchors stay exact
     * for any crossover); this sets where the transition happens: the decay
     * midpoint (geometric mean of the mid and HF loop gains) lands at this
     * frequency. Lower values darken the tail earlier, higher values keep
     * the damping in the top octaves only.
     *
     * @param hz Crossover in Hz (1000 - 16000). Default: 5000.
     *           Non-finite values are ignored.
     */
    void setHighCrossover(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        highCrossover_.store(std::clamp(hz, T(1000), T(16000)),
                             std::memory_order_relaxed);
        markUserParam(kUserHighXover);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Frequency where the bass decay transition is centered.
     *
     * A first-order shelf per line gives the bass T60 exactly at DC and
     * leaves the mid band untouched; its geometric midpoint lands here.
     *
     * @param hz Crossover in Hz (50 - 500). Default: 200.
     */
    void setBassCrossover(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        bassCrossover_.store(std::clamp(hz, T(50), T(500)),
                             std::memory_order_relaxed);
        markUserParam(kUserBassXover);
        paramsDirty_.store(true, std::memory_order_release);
    }

    // =========================================================================
    // Level 3: Expert API - Tone & Levels
    // =========================================================================

    /** @brief Early reflections level in dB (-60 to +6). */
    void setEarlyLevel(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        earlyLevel_.store(decibelsToGain(std::clamp(dB, T(-60), T(6))), std::memory_order_relaxed);
        markUserParam(kUserEarly);
    }

    /** @brief Late tail level in dB (-60 to +6). */
    void setLateLevel(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        lateLevel_.store(decibelsToGain(std::clamp(dB, T(-60), T(6))), std::memory_order_relaxed);
        markUserParam(kUserLate);
    }

    /** @brief Modulation rate in Hz (0.1 - 5); each line wanders at its own multiple. */
    void setModRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        modRate_.store(std::clamp(hz, T(0.1), T(5)), std::memory_order_relaxed);
        markUserParam(kUserModRate);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets a post-reverb low-cut filter on the wet signal (12 dB/oct).
     * @param hz Cutoff in Hz (0 = off, 20-500 typical).
     */
    void setToneLowCut(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        toneLowCutHz_.store(hz, std::memory_order_relaxed);
        toneDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets a post-reverb high-cut filter on the wet signal (12 dB/oct).
     * @param hz Cutoff in Hz (0 = off, 2000-16000 typical).
     */
    void setToneHighCut(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        toneHighCutHz_.store(hz, std::memory_order_relaxed);
        toneDirty_.store(true, std::memory_order_release);
    }

    // =========================================================================
    // Getters
    // =========================================================================

    [[nodiscard]] Type getType() const noexcept { return type_.load(std::memory_order_relaxed); }
    [[nodiscard]] Quality getQuality() const noexcept { return quality_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDecay() const noexcept { return decayTime_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSize() const noexcept { return size_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDamping() const noexcept { return damping_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDiffusion() const noexcept { return diffusion_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getModulation() const noexcept { return modDepth_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getModRate() const noexcept { return modRate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getPreDelay() const noexcept { return preDelayMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getErToLateDelay() const noexcept { return erToLateMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHighDecayMultiplier() const noexcept { return highDecayMult_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBassDecayMultiplier() const noexcept { return bassDecayMult_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWidth() const noexcept { return width_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHighCrossover() const noexcept { return highCrossover_.load(std::memory_order_relaxed); }
    /** @brief Early reflections level in dB (as set, or the preset's). */
    [[nodiscard]] T getEarlyLevel() const noexcept { return levelDb(earlyLevel_.load(std::memory_order_relaxed)); }
    /** @brief Late tail level in dB (as set, or the preset's). */
    [[nodiscard]] T getLateLevel() const noexcept { return levelDb(lateLevel_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getBassCrossover() const noexcept { return bassCrossover_.load(std::memory_order_relaxed); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("ARVB"), 1);
        w.write("type", static_cast<int32_t>(type_.load(std::memory_order_relaxed)));
        w.write("quality", static_cast<int32_t>(quality_.load(std::memory_order_relaxed)));
        w.write("decay", static_cast<float>(decayTime_.load(std::memory_order_relaxed)));
        w.write("size", static_cast<float>(size_.load(std::memory_order_relaxed)));
        w.write("damping", static_cast<float>(damping_.load(std::memory_order_relaxed)));
        w.write("diffusion", static_cast<float>(diffusion_.load(std::memory_order_relaxed)));
        w.write("modDepth", static_cast<float>(modDepth_.load(std::memory_order_relaxed)));
        w.write("modRate", static_cast<float>(modRate_.load(std::memory_order_relaxed)));
        w.write("preDelay", static_cast<float>(preDelayMs_.load(std::memory_order_relaxed)));
        w.write("erToLate", static_cast<float>(erToLateMs_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("width", static_cast<float>(width_.load(std::memory_order_relaxed)));
        w.write("highDecay", static_cast<float>(highDecayMult_.load(std::memory_order_relaxed)));
        w.write("bassDecay", static_cast<float>(bassDecayMult_.load(std::memory_order_relaxed)));
        w.write("highXover", static_cast<float>(highCrossover_.load(std::memory_order_relaxed)));
        w.write("bassXover", static_cast<float>(bassCrossover_.load(std::memory_order_relaxed)));
        w.write("earlyDb", static_cast<float>(getEarlyLevel()));
        w.write("lateDb", static_cast<float>(getLateLevel()));
        w.write("toneLowCut", static_cast<float>(toneLowCutHz_.load(std::memory_order_relaxed)));
        w.write("toneHighCut", static_cast<float>(toneHighCutHz_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("ARVB")) return false;
        setType(static_cast<Type>(r.read("type", 0)));
        // Older blobs have no "quality" key: default 0 = Full.
        setQuality(r.read("quality", 0) == 1 ? Quality::Eco : Quality::Full);
        setDecay(static_cast<T>(r.read("decay", 1.0f)));
        setSize(static_cast<T>(r.read("size", 0.5f)));
        setDamping(static_cast<T>(r.read("damping", 0.5f)));
        setDiffusion(static_cast<T>(r.read("diffusion", 0.7f)));
        setModulation(static_cast<T>(r.read("modDepth", 0.1f)));
        setModRate(static_cast<T>(r.read("modRate", 1.0f)));
        setPreDelay(static_cast<T>(r.read("preDelay", 0.0f)));
        setErToLateDelay(static_cast<T>(r.read("erToLate", 0.0f)));
        setMix(static_cast<T>(r.read("mix", 0.3f)));
        setWidth(static_cast<T>(r.read("width", 1.0f)));
        setHighDecayMultiplier(static_cast<T>(r.read("highDecay", 0.5f)));
        setBassDecayMultiplier(static_cast<T>(r.read("bassDecay", 1.2f)));
        setHighCrossover(static_cast<T>(r.read("highXover", 5000.0f)));
        setBassCrossover(static_cast<T>(r.read("bassXover", 200.0f)));
        setEarlyLevel(static_cast<T>(r.read("earlyDb", 0.0f)));
        setLateLevel(static_cast<T>(r.read("lateDb", 0.0f)));
        const float lo = r.read("toneLowCut", -1.0f);
        const float hi = r.read("toneHighCut", -1.0f);
        if (lo > 0.0f) setToneLowCut(static_cast<T>(lo));
        if (hi > 0.0f) setToneHighCut(static_cast<T>(hi));
        return true;
    }

protected:
    // --- Constants -----------------------------------------------------------

    static constexpr int kMaxLines   = 32;
    static constexpr int kEcoLines   = 16;
    static constexpr int kInStages   = 4;   ///< input diffusers per channel feeding the late field (Full)
    static constexpr int kMaxERTaps  = 96;  ///< velvet early-reflection taps per side (Full)
    static constexpr int kEcoERTaps  = 40;
    static constexpr int kMaxDiscreteER = 7; ///< raw (discrete) early taps at diffusion 0
    static constexpr int kERGroups   = 4;   ///< absorption groups (early -> late)
    static constexpr int kCtrl       = 16;  ///< modulation control period (samples)
    static constexpr int kChunk      = 64;  ///< sparse-FIR processing chunk (samples)

    static constexpr double kMaxPreDelayMs = 200.0;
    static constexpr double kMaxErToLateMs = 200.0;
    static constexpr double kMaxErMs       = 170.0;
    static constexpr double kMaxInjectMs   = 8.0;
    static constexpr double kMaxInDiffMs   = 2.0;
    static constexpr double kMaxLoopApMs   = 5.5;
    static constexpr double kCoherenceHz   = 220.0; ///< late field coherent below this
    static constexpr double kModMaxMs      = 2.0;   ///< peak line wander at modulation 1, size 1
    static constexpr double kGlideMs       = 60.0;  ///< size-change glide time constant
    static constexpr double kMaxGlideSpeed = 0.04;  ///< max length change per sample (4% Doppler)
    static constexpr double kDcCutHz       = 5.0;   ///< wet-output DC blocker
    static constexpr T      kMinReadPos    = T(2);
    static constexpr T      kSoftLimit     = T(2);  ///< in-loop safety limiter threshold

    /// Wet output scale: calibrated so the steady-state level matches the
    /// previous engine's (and Full matches Eco within a fraction of a dB).
    static constexpr T kOutGain = T(0.25);
    /// Early-reflection energy per side before earlyLevel.
    static constexpr T kErEnergy = T(0.0625);

    // FDN base delay times in ms (at size = 1); lengths are rounded to primes.
    // About 3.7 s of delay in all: the modal density (modes per Hz equals the
    // total delay in seconds) keeps the modes overlapping, so they do not
    // ring out on their own.
    static constexpr double kBaseDelaysMs_[kMaxLines] = {
         44.5,  46.5,  49.7,  52.3,  56.1,  58.9,  61.9,  65.2,
         68.6,  73.7,  76.2,  79.8,  86.4,  89.8,  95.7, 101.2,
        105.4, 112.5, 116.7, 126.4, 132.1, 140.6, 146.9, 154.1,
        162.9, 175.8, 184.1, 192.8, 204.0, 218.5, 224.2, 240.1
    };

    // In-loop allpass delays (ms): short and fixed in time, shuffled against
    // the line lengths so no line's total is a near multiple of another's.
    static constexpr double kLoopApMs_[kMaxLines] = {
        3.55, 1.47, 3.80, 1.96, 1.84, 2.33, 4.41, 4.90,
        1.10, 1.35, 4.29, 3.43, 4.16, 4.53, 2.08, 3.06,
        1.22, 2.94, 3.92, 3.18, 2.57, 4.04, 4.65, 2.45,
        2.69, 4.78, 3.67, 1.71, 2.20, 1.59, 2.82, 3.31
    };

    // Input diffusion allpass delays (ms) of the late-field feed,
    // decorrelated between channels. Short, so the frequencies an allpass
    // holds back longest (its group-delay peaks) are not heard ringing on
    // through a short decay.
    static constexpr double kInDiffMs_[2][kInStages] = {
        { 0.36, 0.69, 1.10, 1.65 },
        { 0.43, 0.80, 1.20, 1.79 }
    };
    static constexpr double kInDiffCoeffs_[kInStages] = { 0.75, 0.72, 0.68, 0.64 };

    // Output diffusion allpass delays (ms) on the late sum, per channel: each
    // echo of the tank leaves as a short dense burst, so the late field is
    // smooth from its first milliseconds.
    static constexpr int kOutStages = 2;
    static constexpr double kOutDiffMs_[2][kOutStages] = {
        { 1.13, 2.41 },
        { 1.31, 2.67 }
    };

    // Late injection taps (ms after the ER-to-late gap), one per line.
    static constexpr double kInjectMs_[kMaxLines] = {
        7.14, 3.31, 0.76, 0.25, 5.86, 2.04, 4.84, 5.35,
        5.61, 6.88, 0.00, 7.65, 2.80, 1.02, 7.90, 1.53,
        1.27, 4.33, 6.63, 2.55, 3.06, 6.12, 5.10, 6.37,
        3.82, 1.78, 3.57, 7.39, 2.29, 0.51, 4.08, 4.59
    };
    static constexpr int kInjectSign_[kMaxLines] = {
         1,  1, -1,  1, -1,  1,  1, -1,
        -1,  1,  1, -1,  1, -1,  1,  1,
         1,  1,  1, -1,  1,  1,  1,  1,
         1, -1,  1,  1, -1,  1, -1, -1
    };

    // Orthogonal stereo output sign vectors (inner product 0; the first 8
    // entries are orthogonal too, so Eco keeps the decorrelation).
    static constexpr int kOutSignL_[kMaxLines] = {
        -1, -1,  1, -1, -1,  1,  1,  1,
        -1, -1,  1,  1, -1, -1, -1,  1,
        -1, -1,  1, -1, -1,  1,  1, -1,
         1,  1, -1, -1, -1, -1, -1,  1
    };
    static constexpr int kOutSignR_[kMaxLines] = {
        -1,  1,  1,  1, -1, -1,  1, -1,
        -1,  1,  1, -1, -1,  1, -1, -1,
        -1,  1,  1,  1, -1, -1,  1,  1,
         1, -1, -1,  1, -1,  1, -1, -1
    };

    // Spring tanks (Type::Spring): six springs per side (even = left), all
    // of different lengths; Eco keeps one per side.
    static constexpr int    kSprings          = 12;
    static constexpr int    kEcoSprings       = 2;
    static constexpr int    kSpringStages     = 72;   ///< dispersion allpasses per spring (Full)
    static constexpr int    kEcoSpringStages  = 40;
    static constexpr double kSpringJitterMs   = 3.0;     ///< delay jitter at modulation 1
    /// Round-trip time of spring s at the reference size: spread over
    /// 33-60 ms with a little irregularity, interleaved between the sides.
    static double springBaseMs(int s) noexcept
    {
        return 33.0 + 27.0 * s / (kSprings - 1) + 1.3 * (hash01(s, 90) - 0.5);
    }
    static constexpr double kSpringCutHz      = 4400.0;  ///< dispersion band edge (transition frequency)
    static constexpr T      kSpringOutGain    = T(2);

    // =========================================================================
    // Building blocks
    // =========================================================================

    /** @brief Power-of-two circular delay line: at(k) is the sample pushed k samples ago. */
    struct Line
    {
        std::vector<T> buf;
        int mask = 0;
        int w = 0;

        void prepare(int maxDelay)
        {
            int size = 1;
            while (size < maxDelay + 2) size <<= 1;
            buf.assign(static_cast<std::size_t>(size), T(0));
            mask = size - 1;
            w = 0;
        }
        void clear() noexcept { std::fill(buf.begin(), buf.end(), T(0)); w = 0; }
        [[nodiscard]] T at(int k) const noexcept { return buf[static_cast<std::size_t>((w - k) & mask)]; }
        void push(T x) noexcept { buf[static_cast<std::size_t>(w)] = x; w = (w + 1) & mask; }
    };

    /**
     * @brief Equal-size delay lines in one contiguous buffer sharing a write
     *        index (every line is written once per sample): at(i, k) is the
     *        sample line i received k samples ago.
     */
    struct DelayBank
    {
        std::vector<T> buf;
        int size = 0;
        int stride = 0;
        int mask = 0;
        int w = 0;

        void prepare(int numLines, int maxDelay)
        {
            size = 1;
            while (size < maxDelay + 2) size <<= 1;
            mask = size - 1;
            // One cache line of padding per line: with a power-of-two stride
            // every line would start in the same L1 set, and the 32 streams
            // read each sample would keep evicting each other.
            stride = size + 64 / static_cast<int>(sizeof(T));
            buf.assign(static_cast<std::size_t>(stride) * static_cast<std::size_t>(numLines), T(0));
            w = 0;
        }
        void clear() noexcept { std::fill(buf.begin(), buf.end(), T(0)); w = 0; }
        [[nodiscard]] T at(int i, int k) const noexcept
        { return buf[static_cast<std::size_t>(i * stride + ((w - k) & mask))]; }
        void write(int i, T x) noexcept { buf[static_cast<std::size_t>(i * stride + w)] = x; }
        void advance() noexcept { w = (w + 1) & mask; }
    };

    /**
     * @brief Hermite-interpolated random noise generator for organic modulation.
     *
     * Band-limited random values via cubic Hermite (Catmull-Rom)
     * interpolation between xorshift32 random targets: smooth, non-periodic
     * modulation.
     */
    struct SmoothRandomLFO
    {
        T h0_ = T(0), h1_ = T(0), h2_ = T(0), h3_ = T(0);
        T phase_ = T(0);
        T phaseInc_ = T(0);
        uint32_t state_ = 1;

        void prepare(double sr, T rate, uint32_t seed) noexcept
        {
            phaseInc_ = rate / static_cast<T>(sr);
            state_ = seed ? seed : 1;
            h0_ = nextRandom(); h1_ = nextRandom();
            h2_ = nextRandom(); h3_ = nextRandom();
            phase_ = T(0);
        }

        void setRate(T rate, double sr) noexcept { phaseInc_ = rate / static_cast<T>(sr); }

        /// Advances by `stride` samples (rates stay far below sr / stride, so
        /// at most one new target is drawn per call).
        T nextStride(int stride) noexcept
        {
            phase_ += phaseInc_ * static_cast<T>(stride);
            if (phase_ >= T(1))
            {
                phase_ -= T(1);
                h0_ = h1_; h1_ = h2_; h2_ = h3_;
                h3_ = nextRandom();
            }
            const T d = phase_;
            const T c1 = T(0.5) * (h2_ - h0_);
            const T c2 = h0_ - T(2.5) * h1_ + T(2) * h2_ - T(0.5) * h3_;
            const T c3 = T(0.5) * (h3_ - h0_) + T(1.5) * (h1_ - h2_);
            return ((c3 * d + c2) * d + c1) * d + h1_;
        }

    private:
        T nextRandom() noexcept
        {
            state_ ^= state_ << 13;
            state_ ^= state_ >> 17;
            state_ ^= state_ << 5;
            return static_cast<T>(state_) / static_cast<T>(0xFFFFFFFFu) * T(2) - T(1);
        }
    };

    static constexpr T lfoRateFactor(int i) noexcept
    {
        return T(0.71) + T(0.063) * static_cast<T>(i);
    }
    static constexpr uint32_t lfoSeed(int i) noexcept
    {
        return static_cast<uint32_t>(i) * 7919u + 1u;
    }

    /// Deterministic hash in [0, 1) for the early-reflection layout.
    static double hash01(int k, int s) noexcept
    {
        const double v = std::sin(static_cast<double>(k) * 12.9898
                                  + static_cast<double>(s) * 78.233) * 43758.5453;
        return v - std::floor(v);
    }

    /// Linear gain to dB (a zero gain, a preset's muted early field, reads -120 dB).
    static T levelDb(T gain) noexcept
    {
        return static_cast<T>(20.0 * std::log10(std::max(static_cast<double>(gain), 1e-6)));
    }

    int msToSamples(T ms) const noexcept
    {
        return static_cast<int>(static_cast<T>(spec_.sampleRate) * ms / T(1000));
    }

    // --- State ---------------------------------------------------------------

    AudioSpec spec_ {};
    bool prepared_ = false;
    bool eco_ = false;
    int nLines_ = kMaxLines;
    int ctrlPhase_ = 0;

    // Input stage (per channel)
    std::array<Line, 2> preDelay_;
    std::array<std::array<Line, kInStages>, 2> inAP_;
    std::array<std::array<int, kInStages>, 2> inAPLen_ {};
    std::array<T, kInStages> inAPCoeff_ {};
    std::array<Line, 2> ring_;             ///< allpass-diffused input, feeds the late field
    std::array<std::array<Line, kOutStages>, 2> outAP_;   ///< late-field output diffusers
    std::array<std::array<int, kOutStages>, 2> outAPLen_ {};
    T outAPCoeff_ = T(0.5);
    T lateComp_ = T(1);   ///< late level compensation for the output after absorption

    // Early reflections (per side): velvet taps, each reading its own side
    // (erChan_ 0) or the other (1)
    int numERTaps_ = 0;
    Type erType_ = Type::Room;
    std::array<std::array<int, kMaxERTaps>, 2> erTap_ {}, erChan_ {};
    std::array<std::array<T, kMaxERTaps>, 2> erGain_ {};
    std::array<int, kERGroups + 1> erGroupStart_ {};
    std::array<T, kERGroups> erLPCoeff_ {};
    std::array<std::array<T, kERGroups>, 2> erLP_ {};

    // FDN
    DelayBank lines_;
    std::array<int, kMaxLines> lenTarget_ {};    ///< prime line lengths (samples)
    std::array<T, kMaxLines> lenCur_ {};         ///< gliding length
    std::array<T, kMaxLines> pos_ {};            ///< current read position
    std::array<T, kMaxLines> posInc_ {};         ///< per-sample ramp
    std::array<T, kMaxLines> apY1_ {};           ///< allpass interpolator state
    std::array<int, kMaxLines> injTap_ {};
    std::array<T, kMaxLines> injGain_ {};        ///< sign / sqrt(lines)
    std::array<T, kMaxLines> outSignL_ {}, outSignR_ {};
    std::array<SmoothRandomLFO, kMaxLines> lfo_;
    T modDepthSamples_ = T(0);
    T glideCoeff_ = T(0);
    T maxGlideStep_ = T(0);
    T maxReadPos_ = T(8);

    // Spring tanks: stretched-allpass histories [spring][stage + 1][P]
    // (stage m reads history m and writes history m + 1), shared write
    // index; 4th-order Butterworth band limit per spring.
    bool spring_ = false;
    std::vector<T> springHist_;
    int springP_ = 8;
    int springW_ = 0;
    int springK_ = 5;
    int springStages_ = kSpringStages;
    T springA_ = T(0.5);
    std::array<std::array<T, 5>, 2> springLP_ {};              ///< b0 b1 b2 a1 a2, two sections
    std::array<std::array<T, kSprings>, 4> springLPState_ {};   ///< [section * 2 + state][spring]

    // In-loop allpasses
    DelayBank loopAP_;
    std::array<int, kMaxLines> loopAPLen_ {};   ///< in-loop allpass target lengths
    std::array<T, kMaxLines> loopAPCur_ {};     ///< gliding lengths after a size change
    bool loopAPGliding_ = false;
    T loopAPCoeff_ = T(0.4);

    // Absorption: Jot shelf (mid/high) and bass shelf, per line
    std::array<T, kMaxLines> jotB0_ {}, jotB1_ {}, jotA1_ {}, jotX1_ {}, jotY1_ {};
    std::array<T, kMaxLines> bassB0_ {}, bassB1_ {}, bassA1_ {}, bassX1_ {}, bassY1_ {};

    // Output stage
    std::array<T, 2> cohLP_ {};            ///< states of the two side high-passes (coherent low band)
    T cohCoeff_ = T(0);
    std::array<T, 2> dcX1_ {}, dcY1_ {};
    T dcR_ = T(0.999);

    // Tone correction EQ (Biquad 12 dB/oct)
    Biquad<T, 2> toneLPBiquad_;
    Biquad<T, 2> toneHPBiquad_;
    bool toneLPActive_ = false;
    bool toneHPActive_ = false;

    DryWetMixer<T> mixer_;

    // --- Parameters ----------------------------------------------------------
    //
    // Thread-safety model: every setter mutates only the atomic shadows below
    // and raises a dirty flag; the audio thread drains the flags at the top of
    // processBlock()/processSample() and rebuilds the non-atomic coefficient
    // arrays there, so nothing on the audio path is written from other threads.

    std::atomic<Type> type_      { Type::Room };
    std::atomic<T> decayTime_    { T(1) };
    std::atomic<T> size_         { T(0.5) };
    std::atomic<T> damping_      { T(0.5) };
    std::atomic<T> diffusion_    { T(0.7) };
    std::atomic<T> modDepth_     { T(0.1) };
    std::atomic<T> modRate_      { T(1) };
    std::atomic<T> preDelayMs_   { T(0) };
    std::atomic<T> erToLateMs_   { T(0) };
    std::atomic<T> mix_          { T(0.3) };
    std::atomic<T> earlyLevel_   { T(1) };
    std::atomic<T> lateLevel_    { T(1) };
    std::atomic<T> width_        { T(1) };     // 0 = mono, 1 = natural, 2 = wide

    std::atomic<T> highDecayMult_ { T(0.5) };   // HF T60 multiplier (0.05-1.0)
    std::atomic<T> bassDecayMult_ { T(1.2) };   // bass T60 multiplier (0.3-3.0)
    std::atomic<T> highCrossover_ { T(5000) };  // Hz
    std::atomic<T> bassCrossover_ { T(200) };   // Hz

    std::atomic<Quality> quality_ { Quality::Full };

    std::atomic<int> preDelaySamples_ { 0 };
    std::atomic<int> erToLateSamples_ { 0 };

    std::atomic<bool> presetDirty_  { false };  // setType() -> rebuild + reset
    std::atomic<bool> paramsDirty_  { false };  // other setters -> refresh coefficients
    std::atomic<bool> qualityDirty_ { false };  // setQuality() -> resize engine + reset
    std::atomic<bool> toneDirty_    { false };  // tone EQ cutoff changes
    std::atomic<T> toneLowCutHz_  { T(-1) };    // <= 0 = off
    std::atomic<T> toneHighCutHz_ { T(-1) };

    // Per-param user-override mask. setType() clears it so the preset
    // re-establishes the baseline; each param setter marks its bit;
    // commitPreset() skips any atomic the user overrode, so a setter issued
    // after setType() always wins regardless of drain timing (the documented
    // quick-start is `setType(Hall); setDecay(2.0f);`).
    enum : uint32_t {
        kUserDecay = 1u, kUserSize = 2u, kUserDamping = 4u, kUserDiffusion = 8u,
        kUserBassDecay = 16u, kUserHighXover = 32u, kUserBassXover = 64u,
        kUserModDepth = 128u, kUserModRate = 256u, kUserEarly = 512u,
        kUserLate = 1024u, kUserErToLate = 2048u
    };
    std::atomic<uint32_t> userParamMask_ { 0u };
    void markUserParam(uint32_t bit) noexcept
    { userParamMask_.fetch_or(bit, std::memory_order_release); }

    /// Block-cached copies of the atomics read in the sample loop.
    struct CachedParams
    {
        int preDelaySamples = 0;
        int erToLateSamples = 0;
        T earlyLevel = T(1);
        T lateLevel  = T(1);
        T width      = T(1);
    };
    CachedParams cachedParams_ {};

    // =========================================================================
    // Audio-thread control
    // =========================================================================

    /**
     * @brief Drains deferred parameter changes on the audio thread.
     *
     * The acquire-ordered exchanges synchronize with the release stores in
     * the setters. Each flag is pre-checked with a plain load so the
     * per-sample path pays no read-modify-write when nothing is pending.
     */
    void drainPendingChanges() noexcept
    {
        if (qualityDirty_.load(std::memory_order_acquire)
            && qualityDirty_.exchange(false, std::memory_order_acquire))
        {
            const bool eco = quality_.load(std::memory_order_relaxed) == Quality::Eco;
            if (eco != eco_)
            {
                eco_ = eco;
                refreshTopology();
                updateAll();
                reset();   // engine topology changed: old state is stale
            }
        }

        if (presetDirty_.load(std::memory_order_acquire)
            && presetDirty_.exchange(false, std::memory_order_acquire))
        {
            applyPreset(type_.load(std::memory_order_relaxed));
            reset();       // new room: drop the old tail, snap the lengths
            paramsDirty_.store(false, std::memory_order_relaxed);
        }
        else if (paramsDirty_.load(std::memory_order_acquire)
                 && paramsDirty_.exchange(false, std::memory_order_acquire))
        {
            // Knob change: refresh coefficients without touching the audio
            // state; line lengths glide to their new targets.
            updateAll();
        }

        drainTone();
    }

    void drainTone() noexcept
    {
        if (!(toneDirty_.load(std::memory_order_acquire)
              && toneDirty_.exchange(false, std::memory_order_acquire)))
            return;
        const T hpHz = toneLowCutHz_.load(std::memory_order_relaxed);
        const T lpHz = toneHighCutHz_.load(std::memory_order_relaxed);
        toneHPActive_ = hpHz > T(0) && spec_.sampleRate > 0;
        if (toneHPActive_)
            toneHPBiquad_.setCoeffsNow(BiquadCoeffs::makeHighPass(
                spec_.sampleRate, static_cast<double>(std::clamp(hpHz, T(20), T(500)))));
        toneLPActive_ = lpHz > T(0) && spec_.sampleRate > 0;
        if (toneLPActive_)
            toneLPBiquad_.setCoeffsNow(BiquadCoeffs::makeLowPass(
                spec_.sampleRate, static_cast<double>(std::clamp(lpHz, T(2000), T(16000)))));
    }

    /// Pulls the atomic parameters into the block-local cache (audio thread).
    void refreshCachedParams() noexcept
    {
        cachedParams_.preDelaySamples = preDelaySamples_.load(std::memory_order_relaxed);
        cachedParams_.erToLateSamples = erToLateSamples_.load(std::memory_order_relaxed);
        cachedParams_.earlyLevel = earlyLevel_.load(std::memory_order_relaxed);
        cachedParams_.lateLevel  = lateLevel_.load(std::memory_order_relaxed);
        cachedParams_.width      = width_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Control-rate update: glides the line lengths toward their
     *        targets, draws the next modulation values and sets the
     *        per-sample read-position ramps for the next kCtrl samples.
     */
    void controlTick() noexcept
    {
        const int n = nLines_;
        for (int i = 0; i < n; ++i)
        {
            const T diff = static_cast<T>(lenTarget_[i]) - lenCur_[i];
            lenCur_[i] += std::clamp(diff * glideCoeff_, -maxGlideStep_, maxGlideStep_);
            const T m = lfo_[i].nextStride(kCtrl) * modDepthSamples_;
            const T target = std::clamp(lenCur_[i] + m, kMinReadPos, maxReadPos_);
            posInc_[i] = (target - pos_[i]) * (T(1) / T(kCtrl));
        }
        if (loopAPGliding_)
        {
            // The in-loop allpasses follow a size change with the lines.
            bool moving = false;
            for (int i = 0; i < n; ++i)
            {
                const T diff = static_cast<T>(loopAPLen_[i]) - loopAPCur_[i];
                if (std::abs(diff) < T(0.01))
                    loopAPCur_[i] = static_cast<T>(loopAPLen_[i]);
                else
                {
                    loopAPCur_[i] += std::clamp(diff * glideCoeff_, -maxGlideStep_, maxGlideStep_);
                    moving = true;
                }
            }
            loopAPGliding_ = moving;
        }
    }

    /// Unnormalised M-point Sylvester Hadamard transform in place, fully
    /// unrolled at compile time (the halves, then one butterfly stage).
    template <int M>
    static void hadamardSmall(T* x) noexcept
    {
        if constexpr (M > 1)
        {
            hadamardSmall<M / 2>(x);
            hadamardSmall<M / 2>(x + M / 2);
            [x]<std::size_t... I>(std::index_sequence<I...>) {
                ((void) [x] {
                    const T a = x[I], b = x[I + M / 2];
                    x[I] = a + b;
                    x[I + M / 2] = a - b;
                }(), ...);
            }(std::make_index_sequence<M / 2> {});
        }
    }

    /// In-place unnormalised fast Walsh-Hadamard transform of N values
    /// (in-vector strides unrolled scalar, the rest in SIMD).
    template <int N, int W, typename O>
    static void fwht(T* x) noexcept
    {
        using V = typename O::V;
        for (int b = 0; b < N; b += W)
            hadamardSmall<W>(x + b);
        for (int h = W; h < N; h <<= 1)
            for (int i = 0; i < N; i += h << 1)
                for (int j = i; j < i + h; j += W)
                {
                    const V a = O::load(x + j), b = O::load(x + j + h);
                    O::store(x + j, O::add(a, b));
                    O::store(x + j + h, O::sub(a, b));
                }
    }

    /**
     * @brief One sample of the N-line FDN (compile-time N so the per-line
     *        stages vectorize): allpass-interpolated modulated reads, the
     *        L/R output sums, Hadamard mix plus rotation, absorption
     *        shelves, in-loop allpass, safety limiter, injection, write.
     */
    template <int N>
    void runFdn(T& lateL, T& lateR, int injLag) noexcept
    {
        constexpr int W = std::min(simd::kVecWidth<T>, N);
        using O = simd::Vec<T, W>;
        using V = typename O::V;
        static_assert(N % W == 0, "line count must be a multiple of the SIMD width");

        alignas(64) std::array<T, N + 1> r;   // +1: wrap slot for the rotation
        alignas(64) std::array<T, N> v;
        alignas(64) std::array<T, N> tmp;
        alignas(64) std::array<int, N> idx;

        // Allpass-interpolated modulated reads. eta = (1 - d) / (1 + d) for
        // the fractional part d in [0.5, 1.5), as the series -h / (1 + h) in
        // h = (d - 1) / 2, |h| <= 1/4: the error (< 1e-3) only nudges the
        // fractional delay, the interpolator stays exactly allpass. The
        // arithmetic runs in separate (vectorizable) passes around the
        // scalar gather.
        for (int i = 0; i < N; ++i)
        {
            const T p = pos_[i] += posInc_[i];
            const int M = static_cast<int>(p - T(0.5));
            const T h = (p - static_cast<T>(M) - T(1)) * T(0.5);
            v[i] = -h * (T(1) - h * (T(1) - h * (T(1) - h)));
            idx[i] = M;
        }
        for (int i = 0; i < N; ++i)
        {
            r[i] = lines_.at(i, idx[i]);
            tmp[i] = lines_.at(i, idx[i] + 1);
        }
        for (int i = 0; i < N; i += W)
        {
            const V y = O::madd(O::load(v.data() + i),
                                O::sub(O::load(r.data() + i), O::load(apY1_.data() + i)),
                                O::load(tmp.data() + i));
            O::store(apY1_.data() + i, y);
            O::store(r.data() + i, y);
        }

        // Absorption of what each line delivers (its own round trip: the
        // in-loop allpass that fed it plus the line): Jot mid/high shelf,
        // then bass shelf, y = b0 x + b1 x1 - a1 y1 each.
        for (int i = 0; i < N; i += W)
        {
            const V x = O::load(r.data() + i);
            const V j = O::sub(O::madd(O::load(jotB0_.data() + i), x,
                                       O::mul(O::load(jotB1_.data() + i), O::load(jotX1_.data() + i))),
                               O::mul(O::load(jotA1_.data() + i), O::load(jotY1_.data() + i)));
            O::store(jotX1_.data() + i, x);
            O::store(jotY1_.data() + i, j);
            const V b = O::sub(O::madd(O::load(bassB0_.data() + i), j,
                                       O::mul(O::load(bassB1_.data() + i), O::load(bassX1_.data() + i))),
                               O::mul(O::load(bassA1_.data() + i), O::load(bassY1_.data() + i)));
            O::store(bassX1_.data() + i, j);
            O::store(bassY1_.data() + i, b);
            O::store(r.data() + i, b);
        }

        // Orthogonal L/R output sums, taken after the absorption: even the
        // first pass through a line leaves attenuated by that line's own
        // decay, so the tail decays exponentially from its first echoes
        // instead of holding a plateau for one round trip.
        {
            V aL = O::set1(T(0)), aR = aL;
            for (int i = 0; i < N; i += W)
            {
                const V x = O::load(r.data() + i);
                aL = O::madd(O::load(outSignL_.data() + i), x, aL);
                aR = O::madd(O::load(outSignR_.data() + i), x, aR);
            }
            alignas(64) std::array<T, W> sumL, sumR;
            O::store(sumL.data(), aL);
            O::store(sumR.data(), aR);
            for (int k = 0; k < W; ++k) { lateL += sumL[k]; lateR += sumR[k]; }
        }

        // Hadamard mix, rotation of the lines by one, normalisation.
        const V norm = O::set1(T(1) / std::sqrt(static_cast<T>(N)));
        fwht<N, W, O>(r.data());
        r[N] = r[0];
        for (int i = 0; i < N; i += W)
            O::store(v.data() + i, O::mul(norm, O::load(r.data() + i + 1)));

        {
            if (loopAPGliding_) [[unlikely]]
            {
                for (int i = 0; i < N; ++i)
                {
                    const int k = static_cast<int>(loopAPCur_[i]);
                    const T f = loopAPCur_[i] - static_cast<T>(k);
                    const T a = loopAP_.at(i, k);
                    tmp[i] = a + f * (loopAP_.at(i, k + 1) - a);
                }
            }
            else
            {
                for (int i = 0; i < N; ++i)
                    tmp[i] = loopAP_.at(i, loopAPLen_[i]);
            }
            const V g = O::set1(loopAPCoeff_);
            for (int i = 0; i < N; i += W)
            {
                const V d = O::load(tmp.data() + i);
                const V w = O::madd(g, d, O::load(v.data() + i));
                O::store(v.data() + i, O::sub(d, O::mul(g, w)));
                O::store(tmp.data() + i, w);
            }
            for (int i = 0; i < N; ++i)
                loopAP_.write(i, tmp[i]);
            loopAP_.advance();
        }

        for (int i = 0; i < N; ++i)
        {
            T x = v[i];
            // Safety limiter (never reached at sane levels): identity below
            // kSoftLimit, a rational soft knee above.
            if (std::abs(x) > kSoftLimit) [[unlikely]]
            {
                const T over = std::abs(x) - kSoftLimit;
                x = std::copysign(kSoftLimit + over / (T(1) + over), x);
            }
            lines_.write(i, x + injGain_[i] * ring_[(i >> 1) & 1].at(injLag + injTap_[i]));
        }
        lines_.advance();
    }

    /**
     * @brief One sample of the spring tanks (Type::Spring): six springs
     *        per side (one in Eco), all of different lengths.
     *
     * Each spring is a feedback loop after Valimaki, Parker & Abel (2010):
     * the round-trip delay line (modulated, allpass-interpolated like the
     * FDN lines), a cascade of stretched first-order allpasses
     * (a + z^-K) / (1 + a z^-K) whose group delay rises from DC to the
     * transition frequency fs / (2K) (the characteristic chirp, re-dispersed
     * on every round trip), a 4th-order band limit just below fs / (2K), and
     * the same exact-T60 absorption shelves as the FDN. Even springs feed
     * the left output and hear mostly the left input, odd ones the right;
     * six springs of different lengths per side, with a little round-trip
     * jitter, keep the sparse modes of a single loop (the metallic ring of
     * long decays) from dominating. Each spring's band-limited chirp train
     * is its output. setDiffusion() sets the chirp strength here.
     */
    template <int Springs, int Stages>
    void runSprings(T& lateL, T& lateR, const T (&raw)[2]) noexcept
    {
        // The springs are independent, so the stage cascade runs across
        // them: one SIMD lane per spring, the history laid out as
        // [stage][slot][spring].
        constexpr int NW = simd::kVecNarrowWidth<T>;
        constexpr int W = (Springs % NW == 0) ? NW : 1;
        using O = simd::Vec<T, W>;
        using V = typename O::V;

        const int P = springP_;
        const int mask = P - 1;
        const int iw = springW_;
        const int ik = (springW_ - springK_) & mask;
        // Each spring hears mostly its own side (a mono input drives both);
        // the raw transducer signal, since any diffusion would blur the chirps.
        const T drive[2] = { T(0.75) * raw[0] + T(0.25) * raw[1],
                             T(0.25) * raw[0] + T(0.75) * raw[1] };

        alignas(64) std::array<T, Springs> x;
        for (int s = 0; s < Springs; ++s)
        {
            const T p = pos_[s] += posInc_[s];
            const int M = static_cast<int>(p - T(0.5));
            const T h = (p - static_cast<T>(M) - T(1)) * T(0.5);
            const T eta = -h * (T(1) - h * (T(1) - h * (T(1) - h)));
            const T y = eta * (lines_.at(s, M) - apY1_[s]) + lines_.at(s, M + 1);
            apY1_[s] = y;
            x[s] = y;
        }

        // Dispersion: y = a (x - y[n-K]) + x[n-K], stage by stage.
        const V a = O::set1(springA_);
        T* hist = springHist_.data();
        const std::size_t stageStride = static_cast<std::size_t>(P) * kSprings;
        const std::size_t wOff = static_cast<std::size_t>(iw) * kSprings;
        const std::size_t kOff = static_cast<std::size_t>(ik) * kSprings;
        for (int g = 0; g < Springs; g += W)
        {
            V xv = O::load(x.data() + g);
            T* hin = hist + g;
            for (int m = 0; m < Stages; ++m, hin += stageStride)
            {
                O::store(hin + wOff, xv);
                xv = O::madd(a, O::sub(xv, O::load(hin + stageStride + kOff)), O::load(hin + kOff));
            }
            O::store(hin + wOff, xv);
            O::store(x.data() + g, xv);
        }

        // Band limit: two TDF-II Butterworth sections.
        for (int k = 0; k < 2; ++k)
        {
            const auto& c = springLP_[static_cast<std::size_t>(k)];
            const V b0 = O::set1(c[0]), b1 = O::set1(c[1]), b2 = O::set1(c[2]);
            const V a1 = O::set1(c[3]), a2 = O::set1(c[4]);
            T* s1 = springLPState_[static_cast<std::size_t>(2 * k)].data();
            T* s2 = springLPState_[static_cast<std::size_t>(2 * k + 1)].data();
            for (int g = 0; g < Springs; g += W)
            {
                const V xv = O::load(x.data() + g);
                const V y = O::madd(b0, xv, O::load(s1 + g));
                O::store(s1 + g, O::sub(O::madd(b1, xv, O::load(s2 + g)), O::mul(a1, y)));
                O::store(s2 + g, O::sub(O::mul(b2, xv), O::mul(a2, y)));
                O::store(x.data() + g, y);
            }
        }

        T out[2] = { T(0), T(0) };
        for (int s = 0; s < Springs; ++s) out[s & 1] += x[s];

        // Absorption shelves (exact per-band T60), limiter, injection.
        for (int g = 0; g < Springs; g += W)
        {
            const V xv = O::load(x.data() + g);
            const V j = O::sub(O::madd(O::load(jotB0_.data() + g), xv,
                                       O::mul(O::load(jotB1_.data() + g), O::load(jotX1_.data() + g))),
                               O::mul(O::load(jotA1_.data() + g), O::load(jotY1_.data() + g)));
            O::store(jotX1_.data() + g, xv);
            O::store(jotY1_.data() + g, j);
            const V b = O::sub(O::madd(O::load(bassB0_.data() + g), j,
                                       O::mul(O::load(bassB1_.data() + g), O::load(bassX1_.data() + g))),
                               O::mul(O::load(bassA1_.data() + g), O::load(bassY1_.data() + g)));
            O::store(bassX1_.data() + g, j);
            O::store(bassY1_.data() + g, b);
            O::store(x.data() + g, b);
        }
        for (int s = 0; s < Springs; ++s)
        {
            T v = x[s];
            if (std::abs(v) > kSoftLimit) [[unlikely]]
            {
                const T over = std::abs(v) - kSoftLimit;
                v = std::copysign(kSoftLimit + over / (T(1) + over), v);
            }
            lines_.write(s, v + drive[s & 1]);
        }
        lines_.advance();
        springW_ = (springW_ + 1) & mask;
        const T og = kSpringOutGain / std::sqrt(static_cast<T>(Springs / 2));
        lateL += out[0] * og;
        lateR += out[1] * og;
    }

    /// Line count of the active engine: the springs, or the FDN size.
    void refreshTopology() noexcept
    {
        nLines_ = spring_ ? (eco_ ? kEcoSprings : kSprings) : eco_ ? kEcoLines : kMaxLines;
    }

    /// Sum of gain * line.at(lag0 - n) over n in [0, count): a sparse-FIR
    /// tap applied to a whole chunk, reading contiguous ring memory.
    static void addTap(T* acc, const Line& l, int lag0, T gain, int count) noexcept
    {
        // The span is contiguous in the ring except across its wrap: split it
        // there so both parts run as SIMD multiply-adds.
        const T* b = l.buf.data();
        const int start = (l.w - lag0) & l.mask;
        const int first = std::min(count, l.mask + 1 - start);
        simd::addWithGain(acc, b + start, gain, first);
        if (first < count)
            simd::addWithGain(acc + first, b, gain, count - first);
    }

    /**
     * @brief Core processing of up to kChunk samples: stereo in, wet stereo out.
     *
     * The sparse FIRs (velvet pre-diffuser and early field) run tap by tap
     * over the whole chunk, on contiguous ring memory; the late field and the
     * output stage run per sample.
     */
    void processChunk(const T* inL, const T* inR, T* outL, T* outR, int count) noexcept
    {
        const int pd = cachedParams_.preDelaySamples;
        const int gap = cachedParams_.erToLateSamples;
        const int last = count - 1;

        // --- Pre-delay, then the allpass diffusers of the late-field feed ---
        alignas(64) T raw[2][kChunk];
        for (int c = 0; c < 2; ++c)
        {
            Line& pre = preDelay_[static_cast<std::size_t>(c)];
            const T* in = c == 0 ? inL : inR;
            for (int n = 0; n < count; ++n) pre.push(in[n]);
            std::fill(raw[c], raw[c] + count, T(0));
            addTap(raw[c], pre, pd + 1 + last, T(1), count);
            Line& ring = ring_[static_cast<std::size_t>(c)];
            auto& aps = inAP_[static_cast<std::size_t>(c)];
            for (int n = 0; n < count; ++n)
            {
                T x = raw[c][n];
                for (int st = 0; st < kInStages; ++st)
                {
                    Line& ap = aps[static_cast<std::size_t>(st)];
                    const T g = inAPCoeff_[static_cast<std::size_t>(st)];
                    const T d = ap.at(inAPLen_[static_cast<std::size_t>(c)][static_cast<std::size_t>(st)]);
                    const T w = x + g * d;
                    ap.push(w);
                    x = d - g * w;
                }
                ring.push(x);
            }
        }

        // --- Early reflections: velvet taps with rising density, each on its
        //     own side or the other; the first few read the raw input (crisp
        //     discrete reflections), the rest the diffused feed (each a short
        //     dense burst); grouped absorption, later groups darker ---
        alignas(64) T early[2][kChunk];
        for (int s = 0; s < 2; ++s)
        {
            const Line* src[4] = { &ring_[static_cast<std::size_t>(s)], &ring_[static_cast<std::size_t>(1 - s)],
                                   &preDelay_[static_cast<std::size_t>(s)], &preDelay_[static_cast<std::size_t>(1 - s)] };
            const int lagOff[4] = { last, last, pd + last, pd + last };
            std::fill(early[s], early[s] + count, T(0));
            auto& lp = erLP_[static_cast<std::size_t>(s)];
            for (int g = 0; g < kERGroups; ++g)
            {
                alignas(64) T acc[kChunk] = {};
                for (int k = erGroupStart_[g]; k < erGroupStart_[g + 1]; ++k)
                    addTap(acc, *src[erChan_[s][k]], erTap_[s][k] + lagOff[erChan_[s][k]], erGain_[s][k], count);
                const T c = erLPCoeff_[g];
                T y = lp[g];
                for (int n = 0; n < count; ++n)
                {
                    y += c * (acc[n] - y);
                    early[s][n] += y;
                }
                lp[g] = y;
            }
        }

        for (int n = 0; n < count; ++n)
        {
            if (ctrlPhase_ == 0) controlTick();
            ctrlPhase_ = (ctrlPhase_ + 1) & (kCtrl - 1);

            // --- Late field ---
            T lateL = T(0), lateR = T(0);
            if (spring_)
            {
                const T rawN[2] = { raw[0][n], raw[1][n] };
                if (eco_) runSprings<kEcoSprings, kEcoSpringStages>(lateL, lateR, rawN);
                else      runSprings<kSprings, kSpringStages>(lateL, lateR, rawN);
            }
            else if (eco_) runFdn<kEcoLines>(lateL, lateR, gap + last - n);
            else           runFdn<kMaxLines>(lateL, lateR, gap + last - n);

            const auto [yl, yr] = outputStage(lateL, lateR, early[0][n], early[1][n]);
            outL[n] = yl;
            outR[n] = yr;
        }
    }

    /// Per-sample output stage: coherent low band, width, early + late, DC
    /// block, tone EQ.
    std::pair<T, T> outputStage(T lateL, T lateR, T earlyL, T earlyR) noexcept
    {
        // --- Output: coherent low band, width, early + late, DC block, tone ---
        const T lateLvl = cachedParams_.lateLevel * kOutGain * lateComp_;
        lateL *= lateLvl;
        lateR *= lateLvl;
        if (!spring_)
        {
            // Output diffusion (the springs' chirps stay crisp).
            T* lr[2] = { &lateL, &lateR };
            const T g = outAPCoeff_;
            for (int c = 0; c < 2; ++c)
            {
                T x = *lr[c];
                for (int k = 0; k < kOutStages; ++k)
                {
                    Line& ap = outAP_[static_cast<std::size_t>(c)][static_cast<std::size_t>(k)];
                    const T d = ap.at(outAPLen_[static_cast<std::size_t>(c)][static_cast<std::size_t>(k)]);
                    const T w = x + g * d;
                    ap.push(w);
                    x = d - g * w;
                }
                *lr[c] = x;
            }
        }
        {
            // A diffuse field is coherent between the ears at low frequencies
            // (the wavelength dwarfs the head): the side signal is high-passed
            // (2 poles at kCoherenceHz), so the bass is not phasey.
            const T mid  = (lateL + lateR) * T(0.5);
            T side = (lateL - lateR) * T(0.5);
            cohLP_[0] += cohCoeff_ * (side - cohLP_[0]);
            const T hp1 = side - cohLP_[0];                 // first-order high-pass
            cohLP_[1] += cohCoeff_ * (hp1 - cohLP_[1]);
            side = (hp1 - cohLP_[1]) * cachedParams_.width;  // and a second one
            lateL = mid + side;
            lateR = mid - side;
        }

        T out[2] = { earlyL * cachedParams_.earlyLevel + lateL,
                     earlyR * cachedParams_.earlyLevel + lateR };
        for (int c = 0; c < 2; ++c)
        {
            const T y = out[c] - dcX1_[c] + dcR_ * dcY1_[c];
            dcX1_[c] = out[c];
            dcY1_[c] = y;
            out[c] = y;
        }
        if (toneHPActive_)
        {
            out[0] = toneHPBiquad_.processSample(out[0], 0);
            out[1] = toneHPBiquad_.processSample(out[1], 1);
        }
        if (toneLPActive_)
        {
            out[0] = toneLPBiquad_.processSample(out[0], 0);
            out[1] = toneLPBiquad_.processSample(out[1], 1);
        }
        return { out[0], out[1] };
    }

    // =========================================================================
    // Coefficient update helpers (audio thread or prepare)
    // =========================================================================

    void updateAll() noexcept
    {
        if (spec_.sampleRate <= 0) return;
        updateDelayLengths();
        updateDiffCoeffs();
        updateModulation();
        updateDecayParams();
        const T hd = highDecayMult_.load(std::memory_order_relaxed);
        damping_.store(std::clamp((T(1) - hd) / T(0.9), T(0), T(1)), std::memory_order_relaxed);
        erToLateSamples_.store(msToSamples(erToLateMs_.load(std::memory_order_relaxed)),
                               std::memory_order_relaxed);
        generateERTapsForType(erType_);   // the discrete/diffuse split follows the diffusion
    }

    /// Size factor: size 0 -> 0.35, size 1 -> 1.
    T sizeFactor() const noexcept
    {
        return T(0.35) + T(0.65) * size_.load(std::memory_order_relaxed);
    }

    void updateDelayLengths() noexcept
    {
        const double sr = spec_.sampleRate;
        const double sz = static_cast<double>(sizeFactor());
        const auto ms = [sr](double v) { return std::max(1, static_cast<int>(v * sr / 1000.0)); };

        for (int i = 0; i < kMaxLines; ++i)
        {
            // Eco picks every other base delay, so its 16 lines still span
            // the full range (even mode density).
            const int src = eco_ ? std::min(i * (kMaxLines / kEcoLines) + 1, kMaxLines - 1) : i;
            lenTarget_[i] = nearestPrime(ms(kBaseDelaysMs_[src] * sz));
            // The in-loop allpasses scale with the room like the lines, so
            // their group-delay ripple stays a small, size-independent share
            // of each loop: no frequency lingers longer in a small room.
            loopAPLen_[i] = nearestPrime(ms(kLoopApMs_[i] * sz));
            loopAPGliding_ = true;
            injTap_[i] = std::max(1, ms(kInjectMs_[i]));
            injGain_[i] = static_cast<T>(kInjectSign_[i]) / std::sqrt(static_cast<T>(nLines_));
            outSignL_[i] = static_cast<T>(kOutSignL_[i]);
            outSignR_[i] = static_cast<T>(kOutSignR_[i]);
        }
        if (spring_)
        {
            // Round trips of 27-49 ms at the preset size (0.11), up to
            // 156 ms at size 1 (within the line capacity).
            const double springScale = 0.6 + 2.0 * static_cast<double>(size_.load(std::memory_order_relaxed));
            for (int s = 0; s < nLines_; ++s)
                lenTarget_[s] = nearestPrime(ms(springBaseMs(s) * springScale));
        }
        for (int c = 0; c < 2; ++c)
            for (int st = 0; st < kInStages; ++st)
                inAPLen_[static_cast<std::size_t>(c)][static_cast<std::size_t>(st)]
                    = nearestPrime(ms(kInDiffMs_[c][st]));
    }

    void updateDiffCoeffs() noexcept
    {
        const double diff = static_cast<double>(diffusion_.load(std::memory_order_relaxed));
        for (int st = 0; st < kInStages; ++st)
            inAPCoeff_[static_cast<std::size_t>(st)] = static_cast<T>(kInDiffCoeffs_[st] * diff);
        loopAPCoeff_ = static_cast<T>(0.075 + 0.175 * diff);
        outAPCoeff_ = static_cast<T>(0.6 * diff);

        // Spring chirp strength: the group-delay rise per stage is
        // (1 + a) / (1 - a) - (1 - a) / (1 + a). The total dispersion is
        // that of a 72-stage cascade at a = 0.3 + 0.5 x diffusion; shorter
        // cascades get a larger coefficient to match it.
        {
            const double aRef = 0.3 + 0.5 * diff;
            springStages_ = eco_ ? kEcoSpringStages : kSpringStages;
            const double f = 72.0 / springStages_ * (1.0 + aRef) / (1.0 - aRef);
            springA_ = static_cast<T>((f - 1.0) / (f + 1.0));
        }
    }

    void updateModulation() noexcept
    {
        const double sr = spec_.sampleRate;
        const T rate = modRate_.load(std::memory_order_relaxed);
        for (int i = 0; i < kMaxLines; ++i)
            lfo_[i].setRate(rate * lfoRateFactor(i), sr);
        // Springs jitter by a fixed time (the random round-trip variation of
        // spring models, which smears their sparse modes); the FDN wanders
        // in proportion to the room size.
        modDepthSamples_ = modDepth_.load(std::memory_order_relaxed)
                           * (spring_ ? static_cast<T>(kSpringJitterMs * sr / 1000.0)
                                      : static_cast<T>(kModMaxMs * sr / 1000.0) * sizeFactor());
        glideCoeff_ = static_cast<T>(1.0 - std::exp(-kCtrl / (kGlideMs * sr / 1000.0)));
        maxGlideStep_ = static_cast<T>(kMaxGlideSpeed * kCtrl);
    }

    /**
     * @brief Per-line absorption: Jot mid/high shelf and bass shelf.
     *
     * Both are first-order sections from the bilinear transform of
     * H(s) = (s + a) / (s + b) (prewarped at the crossover):
     *   H(z) = ((1 + a) + (a - 1) z^-1) / ((1 + b) + (b - 1) z^-1).
     * Jot: gHigh * H with a = K sqrt(gMid/gHigh), b = K sqrt(gHigh/gMid),
     * so |H(1)| = gMid and |H(-1)| = gHigh exactly. Bass: a = K sqrt(r),
     * b = K / sqrt(r) with r = gBass / gMid, so the DC gain is r and the
     * Nyquist gain 1. Each midpoint (geometric mean) lands on its crossover.
     * The loop length includes the in-loop allpass (its mean group delay
     * equals its length).
     */
    void updateDecayParams() noexcept
    {
        const double sr = spec_.sampleRate;
        const double decay = static_cast<double>(decayTime_.load(std::memory_order_relaxed));
        const double t60H = std::max(0.05, decay * static_cast<double>(
            highDecayMult_.load(std::memory_order_relaxed)));
        const double t60B = std::max(0.05, decay * static_cast<double>(
            bassDecayMult_.load(std::memory_order_relaxed)));
        const double kPi = 3.14159265358979323846;
        const double fh = std::clamp(static_cast<double>(
            highCrossover_.load(std::memory_order_relaxed)), 100.0, 0.45 * sr);
        const double fb = std::clamp(static_cast<double>(
            bassCrossover_.load(std::memory_order_relaxed)), 10.0, 0.45 * sr);
        const double Kh = std::tan(kPi * fh / sr);
        const double Kb = std::tan(kPi * fb / sr);
        double passEnergy = 0.0;
        for (int i = 0; i < kMaxLines; ++i)
        {
            // Loop length: line + group delay of the in-loop allpasses. The
            // FDN's short allpasses count their mean (their length). A
            // spring's dispersion makes the loop time frequency dependent,
            // from K (1 - a) / (1 + a) per stage at DC up to the chirp at the
            // band edge; the low and mid band, where the energy is, set it.
            const double M = spring_
                ? static_cast<double>(lenTarget_[i]) + springStages_ * springK_
                      * (1.0 - static_cast<double>(springA_)) / (1.0 + static_cast<double>(springA_))
                : static_cast<double>(lenTarget_[i] + loopAPLen_[i]);
            const double gM = std::pow(0.001, M / (decay * sr));
            if (i < nLines_) passEnergy += gM * gM;
            const double gH = std::min(std::pow(0.001, M / (t60H * sr)), gM);
            // Keep the bass loop gain below 1 so extreme settings ring out
            // instead of self-sustaining.
            const double gB = std::min(std::pow(0.001, M / (t60B * sr)), 0.9995);

            {
                const double ratio = std::sqrt(gM / gH);
                const double a = Kh * ratio, b = Kh / ratio, inv = 1.0 / (1.0 + b);
                jotB0_[i] = static_cast<T>(gH * (1.0 + a) * inv);
                jotB1_[i] = static_cast<T>(gH * (a - 1.0) * inv);
                jotA1_[i] = static_cast<T>((b - 1.0) * inv);
            }
            {
                const double sq = std::sqrt(gB / gM);
                const double a = Kb * sq, b = Kb / sq, inv = 1.0 / (1.0 + b);
                bassB0_[i] = static_cast<T>((1.0 + a) * inv);
                bassB1_[i] = static_cast<T>((a - 1.0) * inv);
                bassA1_[i] = static_cast<T>((b - 1.0) * inv);
            }
        }
        // The FDN output is taken after the absorption, one round trip
        // quieter than the lines' content; this restores the late level (so
        // the early/late balance does not depend on the decay time).
        lateComp_ = spring_ ? T(1)
                            : static_cast<T>(1.0 / std::sqrt(passEnergy / std::max(1, nLines_)));
    }

    // --- Early reflection generation -----------------------------------------

    /// Regenerates the early-reflection taps for a reverb type (Eco caps the count).
    void generateERTapsForType(Type type) noexcept
    {
        const int cap = eco_ ? kEcoERTaps : kMaxERTaps;
        switch (type)
        {
            case Type::Room:      generateERTaps(1.5, 35.0, cap);                break;
            case Type::Hall:      generateERTaps(5.0, 110.0, cap);               break;
            case Type::Chamber:   generateERTaps(3.0, 60.0, cap);                break;
            case Type::Plate:     generateERTaps(0.0, 0.0, 0);                   break;
            case Type::Spring:    generateERTaps(0.0, 0.0, 0);                   break;
            case Type::Cathedral: generateERTaps(10.0, 160.0, cap);              break;
        }
    }

    /**
     * @brief Velvet-noise early field (Valimaki et al.): numTaps sparse +-1
     *        impulses per side between minMs and maxMs, one per slot at a
     *        random position, denser as time goes on (density ~ t^0.6, like
     *        the rising echo density of a room). The first four are positive
     *        discrete reflections. Each tap reads its own side (probability
     *        falling from 0.85 to 0.5 over the window, so the field starts
     *        lateralised and turns diffuse) or the other. Amplitudes follow a
     *        decaying envelope normalised by the local density, so the energy
     *        flows smoothly into the late field; later tap groups pass darker
     *        absorption low-passes. The taps read the pre-diffused signal, so
     *        each reflection is itself a short dense burst.
     */
    void generateERTaps(double minMs, double maxMs, int numTaps) noexcept
    {
        numERTaps_ = std::clamp(numTaps, 0, kMaxERTaps);
        for (int g = 0; g <= kERGroups; ++g)
            erGroupStart_[g] = numERTaps_ * g / kERGroups;
        if (numERTaps_ == 0) return;

        const double sr = spec_.sampleRate;
        constexpr double p = 1.6;
        // Diffusion sets how many of the first reflections stay discrete
        // (crisp, room-like) instead of reading the diffused feed (smooth).
        const int discrete = 1 + static_cast<int>(std::lround(
            (kMaxDiscreteER - 1) * (1.0 - static_cast<double>(diffusion_.load(std::memory_order_relaxed)))));
        for (int s = 0; s < 2; ++s)
        {
            double energy = 0.0;
            std::array<double, kMaxERTaps> gv {};
            for (int k = 0; k < numERTaps_; ++k)
            {
                const double u = (k + 0.15 + 0.7 * hash01(k, s + 20)) / numERTaps_;
                // Warped time in [0, 1): a floor of uniform density plus a
                // share rising like t^(p - 1).
                const double w = 0.3 * u + 0.7 * std::pow(u, 1.0 / p);
                const double ms = minMs + (maxMs - minMs) * w;
                erTap_[s][k] = std::max(1, static_cast<int>(ms * sr / 1000.0));
                erChan_[s][k] = (hash01(k, s + 30) < 0.85 - 0.35 * w ? 0 : 1) + (k < discrete ? 2 : 0);
                const double density = 1.0 / (0.3 + 0.7 / p * std::pow(std::max(u, 1e-3), 1.0 / p - 1.0));   // dk/dw
                const double sign = k < 4 || hash01(k, s + 40) < 0.5 ? 1.0 : -1.0;
                // Energy per unit time follows the envelope: sparse taps are
                // individually louder.
                gv[k] = sign * std::exp(-1.6 * w) / std::sqrt(density);
                energy += gv[k] * gv[k];
            }
            const double norm = std::sqrt(static_cast<double>(kErEnergy) / energy);
            for (int k = 0; k < numERTaps_; ++k)
                erGain_[s][k] = static_cast<T>(gv[k] * norm);
        }

        static constexpr double kGroupCutHz[kERGroups] = { 14000.0, 9000.0, 5500.0, 3200.0 };
        for (int g = 0; g < kERGroups; ++g)
        {
            const double fc = std::min(kGroupCutHz[g], 0.45 * sr);
            erLPCoeff_[g] = static_cast<T>(1.0 - std::exp(-6.283185307179586 * fc / sr));
        }
    }

    // Sieve of Eratosthenes up to kPrimeTableMax, computed once on first use.
    static constexpr int kPrimeTableMax = 131072;

    static const std::vector<uint8_t>& getPrimeSieve() noexcept
    {
        static const std::vector<uint8_t> sieve = []
        {
            std::vector<uint8_t> s(static_cast<size_t>(kPrimeTableMax), 1);
            s[0] = 0;
            s[1] = 0;
            for (int i = 2; i * i < kPrimeTableMax; ++i)
                if (s[static_cast<size_t>(i)])
                    for (int j = i * i; j < kPrimeTableMax; j += i)
                        s[static_cast<size_t>(j)] = 0;
            return s;
        }();
        return sieve;
    }

    /// Smallest prime >= n (lengths only grow by a few samples).
    static int nearestPrime(int n) noexcept
    {
        if (n <= 2) return 2;
        if (n < kPrimeTableMax)
        {
            const auto& sieve = getPrimeSieve();
            while (n < kPrimeTableMax && !sieve[static_cast<size_t>(n)])
                ++n;
            if (n < kPrimeTableMax) return n;
        }
        if (n % 2 == 0) ++n;
        while (true)
        {
            bool isPrime = true;
            for (int d = 3; d * d <= n; d += 2)
                if (n % d == 0) { isPrime = false; break; }
            if (isPrime) return n;
            n += 2;
        }
    }

    // --- Preset application --------------------------------------------------

    struct PresetValues {
        T size, decay, hdMult, bdMult, hxover, bxover;
        T diff, modDepth, modRate, earlyLvl, lateLvl, erToLate;
    };

    void commitPreset(const PresetValues& p) noexcept
    {
        // Only write the atomics the user did NOT override since the last
        // setType(). The acquire-read pairs with the release-mark in each setter.
        const uint32_t m = userParamMask_.load(std::memory_order_acquire);
        if (!(m & kUserSize))      size_.store(p.size, std::memory_order_relaxed);
        if (!(m & kUserDecay))     decayTime_.store(p.decay, std::memory_order_relaxed);
        if (!(m & kUserDamping))   highDecayMult_.store(p.hdMult, std::memory_order_relaxed);
        if (!(m & kUserBassDecay)) bassDecayMult_.store(p.bdMult, std::memory_order_relaxed);
        if (!(m & kUserHighXover)) highCrossover_.store(p.hxover, std::memory_order_relaxed);
        if (!(m & kUserBassXover)) bassCrossover_.store(p.bxover, std::memory_order_relaxed);
        if (!(m & kUserDiffusion)) diffusion_.store(p.diff, std::memory_order_relaxed);
        if (!(m & kUserModDepth))  modDepth_.store(p.modDepth, std::memory_order_relaxed);
        if (!(m & kUserModRate))   modRate_.store(p.modRate, std::memory_order_relaxed);
        if (!(m & kUserEarly))     earlyLevel_.store(p.earlyLvl, std::memory_order_relaxed);
        if (!(m & kUserLate))      lateLevel_.store(p.lateLvl, std::memory_order_relaxed);
        if (!(m & kUserErToLate))  erToLateMs_.store(p.erToLate, std::memory_order_relaxed);
    }

    void applyPreset(Type type) noexcept
    {
        //                          size     decay    hdMult   bdMult   hxover   bxover
        //                          diff     modDep   modRate  early    late     erToLate
        switch (type)
        {
            case Type::Room:
                commitPreset({T(0.22), T(0.5),  T(0.40), T(1.1),  T(5000), T(250),
                              T(0.72), T(0.14), T(1.2),  T(1),    T(0.8),  T(0)});
                break;
            case Type::Hall:
                commitPreset({T(0.68), T(2.2),  T(0.32), T(1.3),  T(4500), T(200),
                              T(0.84), T(0.18), T(0.55), T(0.7),  T(1),    T(15)});
                break;
            case Type::Chamber:
                commitPreset({T(0.38), T(1.2),  T(0.38), T(1.1),  T(5000), T(250),
                              T(0.78), T(0.14), T(0.8),  T(0.9),  T(0.9),  T(8)});
                break;
            case Type::Plate:
                commitPreset({T(0.14), T(1.5),  T(0.55), T(0.8),  T(7000), T(150),
                              T(0.94), T(0.16), T(1.4),  T(0),    T(1),    T(0)});
                break;
            case Type::Spring:
                commitPreset({T(0.11), T(0.9),  T(0.28), T(1.0),  T(4000), T(200),
                              T(0.6),  T(0.08), T(0.35), T(0.5),  T(1),    T(0)});
                break;
            case Type::Cathedral:
                commitPreset({T(0.98), T(5.0),  T(0.24), T(1.5),  T(3500), T(150),
                              T(0.91), T(0.22), T(0.35), T(0.5),  T(1),    T(25)});
                break;
        }

        spring_ = type == Type::Spring;
        erType_ = type;
        refreshTopology();
        if (prepared_)
            updateAll();
    }
};

} // namespace dspark
