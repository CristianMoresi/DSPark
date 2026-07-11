// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AlgorithmicReverb.h
 * @brief World-class 16-line FDN reverb with Jot absorption and Hadamard mixing.
 *
 * Architecture:
 * ```
 * Input (mono sum)
 *   │
 *   ▼
 * [Pre-delay: 0-200ms]
 *   │
 *   ▼
 * [Input Diffusion: 8 cascaded allpass, 1.0-9.5ms]
 *   │
 *   ├──▶ [Early Reflections: 40 taps with progressive HF absorption, L/R decorrelated]
 *   │
 *   ├──▶ [ER-to-Late gap]
 *   │       │
 *   │       ▼
 *   │    [Parallel Allpass Diffuser: 16 parallel AP + Hadamard → 16 delay + Hadamard]
 *   │    (2-step, 16² = 256 echo paths, each FDN line gets unique dense input)
 *   │       │
 *   │       ▼
 *   │    [FDN Core: 16 delay lines]
 *   │      ├─ Read with dual smooth-random-LFO modulated delay
 *   │      ├─ Hadamard 16×16 butterfly (O(N log N))
 *   │      ├─ Jot absorption filter (1st-order shelving) per line
 *   │      ├─ Bass shelf (1-pole) per line
 *   │      ├─ 2 feedback allpass per line
 *   │      ├─ DC blocker + soft limiter
 *   │      └─ Write back + input injection
 *   │       │
 *   │       ▼
 *   │    [Output: sign-weighted + Dattorro multi-tap]
 *   │       │
 *   │       ▼
 *   │    [Output Diffusion: 2 allpass/channel, L/R decorrelated]
 *   │
 *   ▼
 * [Combine early + late] → [Tone EQ: Biquad LP + HP (12 dB/oct)] → DryWetMixer → Output
 * ```
 *
 * Key features:
 * - **Jot absorption filter** (Jot 1991): 1st-order shelving IIR per delay line
 *   for smooth frequency-dependent decay — the #1 factor for natural sound.
 *   Separate bass shelf for independent LF control.
 * - **Hadamard 16×16**: all eigenvalues ±1, zero coloring
 * - **Parallel allpass diffuser** (Signalsmith-inspired): 16 parallel allpass
 *   + 2-step Hadamard mixing → 256 unique echo paths per input sample.
 *   Each FDN line receives a different, densely-mixed version of the input.
 * - **Feedback allpass**: 2 regular allpass per delay line for in-loop density
 * - **Output diffusion**: 2 allpass per channel with decorrelated delays,
 *   smears residual temporal patterns
 * - **Smooth random modulation** (Lexicon-style): Hermite-interpolated
 *   noise replaces periodic sine LFOs for organic character
 * - **Multi-tap output** (Dattorro-style): 7 taps/channel from different
 *   delay line positions for true temporal decorrelation
 * - **Tone correction EQ**: Biquad high/low cut (12 dB/oct) for tonal shaping
 * - **Early reflections**: 40-tap with progressive frequency absorption
 *   simulating wall absorption — late taps are naturally darker
 * - **Soft saturation**: Fast branchless rational soft-clip (transparent below ±1)
 *   — more musical than hard clamp, prevents pipeline stalls.
 * - **Allpass interpolation**: in modulated FDN reads, preserves HF over
 *   hundreds of feedback iterations (linear/cubic causes cumulative dulling)
 * - **Stereo width**: M/S width control on late reverb tail
 * - **Eco quality mode** (setQuality): opt-in reduced engine (8 FDN lines,
 *   control-rate modulation, linear allpass interpolation, no extra output
 *   taps, single-stage input scatter, 12 early taps) at a fraction of the
 *   CPU cost, for embedded and other constrained targets. The default Full
 *   quality path is bit-identical to previous releases.
 *
 * Four levels of API complexity:
 *
 * - **Level 1:** `reverb.setType(Hall);`
 * - **Level 2:** `reverb.setSize(0.7f); reverb.setDamping(0.3f);`
 * - **Level 3:** `reverb.setHighDecayMultiplier(0.4f);`
 * - **Level 4:** Inherit and override protected members.
 *
 * References:
 * - Jot & Chaigne (1991) — FDN with frequency-dependent decay (shelving absorption)
 * - Dattorro (1997, JAES) — Multi-tap output, plate topology
 * - Griesinger / Lexicon 480L — Random modulation, Spin/Wander
 * - Sean Costello / Valhalla DSP — Practical FDN design, absorbent allpass
 * - Valimaki et al. (2012, IEEE) — "50 Years of Artificial Reverberation"
 * - Smith (CCRMA) — Hadamard matrices, prime power delay selection
 *
 * Dependencies: RingBuffer.h, DryWetMixer.h, DspMath.h, Biquad.h,
 *               AudioSpec.h, AudioBuffer.h.
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

#include "../Core/RingBuffer.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Biquad.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class AlgorithmicReverb
 * @brief 16-line FDN reverb with Jot absorption and 6 presets.
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
        Full,   ///< Complete 16-line engine. Default; bit-identical to previous releases.
        Eco     ///< Reduced engine, roughly 3-4x cheaper. For constrained targets.
    };

    ~AlgorithmicReverb() = default; // non-virtual: leaf class (no virtual dispatch)

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the reverberation engine and allocates required memory.
     *
     * Initializes delay lines, filters, and LFOs based on the specified sample rate.
     * This method avoids memory allocations on the audio thread and must be called
     * prior to any processing.
     *
     * @param spec Audio specification detailing sample rate and maximum block size.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);
        double sr = spec.sampleRate;

        preDelayBuf_.prepare(static_cast<int>(sr * 0.2) + 1);
        erBuf_.prepare(static_cast<int>(sr * 0.2) + 1);
        erToLateBuf_.prepare(static_cast<int>(sr * 0.2) + 1);

        int maxDiff = static_cast<int>(sr * 0.012) + 1;
        for (auto& buf : diffBufs_)
            buf.prepare(maxDiff);

        int maxFDN = static_cast<int>(sr * 0.5) + 1;
        for (auto& dl : fdnDelays_)
            dl.prepare(maxFDN);

        // Parallel allpass diffuser — step 1 (~20ms max)
        int maxParAP = static_cast<int>(sr * 0.021) + 1;
        for (auto& buf : parAPBufs_) buf.prepare(maxParAP);

        // Multi-channel diffuser — step 2 (~47ms max)
        int maxDiffS2 = static_cast<int>(sr * 0.047) + 1;
        for (auto& buf : diffuserStep2_) buf.prepare(maxDiffS2);

        // Feedback allpass buffers (~60ms max, proportional to FDN delays)
        int maxFbAP = static_cast<int>(sr * 0.06) + 1;
        for (auto& buf : fbAPBufsA_) buf.prepare(maxFbAP);
        for (auto& buf : fbAPBufsB_) buf.prepare(maxFbAP);

        // Internal serial allpass buffers (~35ms max)
        int maxIntAP = static_cast<int>(sr * 0.035) + 1;
        for (auto& buf : intAPBufsA_) buf.prepare(maxIntAP);
        for (auto& buf : intAPBufsB_) buf.prepare(maxIntAP);

        // Output diffusion buffers (~3ms max)
        int maxOutDiff = static_cast<int>(sr * 0.003) + 1;
        for (auto& buf : outDiffBufsL_)
            buf.prepare(maxOutDiff);
        for (auto& buf : outDiffBufsR_)
            buf.prepare(maxOutDiff);

        // Initialize smooth random LFOs
        T rate = modRate_.load(std::memory_order_relaxed);
        for (int i = 0; i < kFDNSize; ++i)
        {
            modLFOA_[i].prepare(sr, rate * (T(0.7) + T(0.05) * static_cast<T>(i)),
                                static_cast<uint32_t>(i * 7919 + 1));
            modLFOB_[i].prepare(sr, rate * (T(1.8) + T(0.11) * static_cast<T>(i)),
                                static_cast<uint32_t>(i * 6271 + 31337));
        }

        // DC block coefficient (~23 Hz, sample-rate independent)
        dcCoeff_ = T(1) - std::exp(T(-6.283185307179586) * T(23)
                                    / static_cast<T>(sr));

        // Noise modulation: LP cutoff ~3 Hz, depth scales with modDepth_
        noiseCoeff_ = T(1) - std::exp(T(-6.283185307179586) * T(3)
                                       / static_cast<T>(sr));
        // Eco quality refreshes the noise LP only once per control interval,
        // so the coefficient is scaled to keep the same ~3 Hz cutoff.
        noiseCoeffEco_ = T(1) - std::exp(T(-6.283185307179586) * T(3)
                                          * T(kEcoCtrlInterval)
                                          / static_cast<T>(sr));
        noiseState_ = 1;
        noiseLP_ = T(0);

        eco_ = quality_.load(std::memory_order_relaxed) == Quality::Eco;
        nLines_ = eco_ ? kEcoLines : kFDNSize;

        applyPreset(type_.load(std::memory_order_relaxed));
        // Clear pending flags — prepare() has already applied whatever the
        // caller configured before prepare, so no drain is needed on the first
        // processBlock.
        presetDirty_.store(false, std::memory_order_relaxed);
        paramsDirty_.store(false, std::memory_order_relaxed);
        toneDirty_.store(false, std::memory_order_relaxed);
        qualityDirty_.store(false, std::memory_order_relaxed);
        reset();
    }

    /**
     * @brief Processes an audio block in-place with zero allocations.
     *
     * Applies the complete FDN reverberation algorithm, handling mono, stereo,
     * and mid/side signal edge cases. Thread-safe and designed for the hot path.
     *
     * @param buffer View of the audio buffers (supports mono or stereo).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), 2);
        const int nS  = buffer.getNumSamples();
        if (nCh == 0 || nS == 0) return;

        // C2/C3: drain deferred parameter changes on the audio thread. All
        // mutations of non-atomic topology arrays (fdnDelayLens_, diffCoeffs_,
        // …) happen here, never from GUI-thread setters. acquire-ordered loads
        // synchronize with the release-stores in setType/setXxx.
        if (qualityDirty_.exchange(false, std::memory_order_acquire))
        {
            const bool eco =
                quality_.load(std::memory_order_relaxed) == Quality::Eco;
            if (eco != eco_)
            {
                eco_ = eco;
                nLines_ = eco ? kEcoLines : kFDNSize;
                if (spec_.sampleRate > 0)
                {
                    updateDelayLengths();  // re-derives per-line delays + decay
                    generateERTapsForType(type_.load(std::memory_order_relaxed));
                }
                // Engine topology changed: old delay/filter state is stale.
                reset();
            }
        }

        if (presetDirty_.exchange(false, std::memory_order_acquire))
        {
            Type t = type_.load(std::memory_order_relaxed);
            applyPreset(t);
            // C3 fix: wipe all delay buffers and filter state — topology just
            // changed, so old state is stale and can spike the output.
            reset();
            // The preset-rebuild already ran updateDelayLengths /
            // updateDiffCoeffs / updateModulation, so consume paramsDirty_
            // without doing the work twice.
            paramsDirty_.store(false, std::memory_order_relaxed);
        }
        else if (paramsDirty_.exchange(false, std::memory_order_acquire))
        {
            // Non-topology parameter change: refresh coefficient arrays but
            // DO NOT wipe delay buffers (would click on every knob tweak).
            if (spec_.sampleRate > 0)
            {
                updateDelayLengths();  // also runs updateDecayParams
                updateDiffCoeffs();
                updateModulation();

                T md = modDepth_.load(std::memory_order_relaxed);
                modDepthA_.store(md * T(30), std::memory_order_relaxed);
                modDepthB_.store(md * T(15), std::memory_order_relaxed);

                T hd = highDecayMult_.load(std::memory_order_relaxed);
                T damp = std::clamp((T(1) - hd) / T(0.9), T(0), T(1));
                damping_.store(damp, std::memory_order_relaxed);
            }
        }

        if (toneDirty_.exchange(false, std::memory_order_acquire))
        {
            T hpHz = toneLowCutHz_.load(std::memory_order_relaxed);
            T lpHz = toneHighCutHz_.load(std::memory_order_relaxed);
            if (hpHz <= T(0) || spec_.sampleRate <= 0)
            {
                toneHPActive_ = false;
            }
            else
            {
                toneHPActive_ = true;
                toneHPBiquad_.setCoeffs(BiquadCoeffs<T>::makeHighPass(
                    spec_.sampleRate,
                    static_cast<double>(std::clamp(hpHz, T(20), T(500)))));
            }
            if (lpHz <= T(0) || spec_.sampleRate <= 0)
            {
                toneLPActive_ = false;
            }
            else
            {
                toneLPActive_ = true;
                toneLPBiquad_.setCoeffs(BiquadCoeffs<T>::makeLowPass(
                    spec_.sampleRate,
                    static_cast<double>(std::clamp(lpHz, T(2000), T(16000)))));
            }
        }

        mixer_.pushDry(buffer);
        refreshCachedParams();

        for (int i = 0; i < nS; ++i)
        {
            T monoIn;
            if (nCh >= 2)
            {
                T L = buffer.getChannel(0)[i];
                T R = buffer.getChannel(1)[i];
                T sum = L + R;
                T env = std::abs(L) + std::abs(R);
                
                if (std::abs(sum) < T(1e-5) * env)
                {
                    // Pure side condition: decode the side channel instead of hard-switching
                    // Preserves phase coherency and prevents toggle distortion.
                    monoIn = (L - R) * T(0.5);
                }
                else
                {
                    monoIn = sum * T(0.5);
                }
            }
            else
            {
                monoIn = buffer.getChannel(0)[i];
            }

            auto [outL, outR] = processSampleInternal(monoIn);

            if (nCh >= 2)
            {
                buffer.getChannel(0)[i] = outL;
                buffer.getChannel(1)[i] = outR;
            }
            else
            {
                buffer.getChannel(0)[i] = (outL + outR) * T(0.5);
            }
        }

        mixer_.mixWet(buffer, mix_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Processes a single sample and returns a stereo pair.
     *
     * @param input The mono input sample to reverberate.
     * @return std::pair<T, T> A pair containing the {Left, Right} reverberated output.
     */
    [[nodiscard]] std::pair<T, T> processSample(T input) noexcept
    {
        refreshCachedParams();
        return processSampleInternal(input);
    }

    /**
     * @brief Resets all internal delay buffers, filters, and LFO phases.
     *
     * Prevents feedback clicks or ringing when transport stops or topology changes.
     */
    void reset() noexcept
    {
        preDelayBuf_.reset();
        erBuf_.reset();
        erToLateBuf_.reset();
        for (auto& buf : diffBufs_) buf.reset();
        for (auto& dl : fdnDelays_) dl.reset();
        for (auto& buf : parAPBufs_) buf.reset();
        for (auto& buf : diffuserStep2_) buf.reset();
        for (auto& buf : fbAPBufsA_) buf.reset();
        for (auto& buf : fbAPBufsB_) buf.reset();
        for (auto& buf : intAPBufsA_) buf.reset();
        for (auto& buf : intAPBufsB_) buf.reset();
        for (auto& buf : outDiffBufsL_) buf.reset();
        for (auto& buf : outDiffBufsR_) buf.reset();

        absState_.fill(T(0));
        bassState_.fill(T(0));
        dcZ_.fill(T(0));
        prevFeedback_.fill(T(0));
        apInterpState_.fill(T(0));
        erLPStateL_.fill(T(0));
        erLPStateR_.fill(T(0));
        modACache_.fill(T(0));
        modBCache_.fill(T(0));
        noiseND_ = T(0);
        ctrlPhase_ = 0;

        for (auto& lfo : modLFOA_) lfo.reset();
        for (auto& lfo : modLFOB_) lfo.reset();

        toneLPBiquad_.reset();
        toneHPBiquad_.reset();
        mixer_.reset();
    }

    // =========================================================================
    // Level 1: Simple API
    // =========================================================================

    void setType(Type type) noexcept
    {
        // C2 fix: don't touch non-atomic state from GUI thread. Publish the
        // new type and raise the preset-dirty flag; the audio thread will
        // run applyPreset() + reset() at the top of its next processBlock.
        type_.store(type, std::memory_order_relaxed);
        presetDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Selects the engine quality / CPU cost trade-off.
     *
     * Quality::Full (default) runs the complete 16-line engine and is
     * bit-identical to previous releases. Quality::Eco reduces the engine to
     * roughly a quarter of the CPU cost for embedded and other constrained
     * targets, keeping the same control set and decay calibration:
     *
     * - 8 FDN lines instead of 16 (alternate base delays keep the full
     *   29.7-160 ms span, and per-line decay gains keep T60 exact)
     * - modulation LFOs and noise updated at control rate (every 16 samples;
     *   steps are orders of magnitude below audibility at reverb mod rates)
     * - linear interpolation in modulated allpasses (Dattorro-standard)
     *   instead of 4-point Hermite
     * - single-stage input scatter (16 unique echo paths instead of 256)
     * - no extra output taps (multi-tap / allpass taps); output density
     *   comes from the sign-weighted line sum plus output diffusion
     * - early reflections capped at 12 taps
     *
     * The audible difference is a somewhat lower tail density (most
     * noticeable on long, exposed cathedral tails); short and mid rooms stay
     * very close to Full.
     *
     * Like setType(), this is an engine-mode switch, not an automation
     * target: it is applied at the start of the next processBlock() and
     * clears the reverb state (the running tail is dropped).
     *
     * @param q Quality::Full or Quality::Eco.
     */
    void setQuality(Quality q) noexcept
    {
        quality_.store(q, std::memory_order_relaxed);
        qualityDirty_.store(true, std::memory_order_release);
    }

    void setDecay(T seconds) noexcept
    {
        decayTime_.store(std::clamp(seconds, T(0.1), T(30)),
                         std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    // =========================================================================
    // Level 2: Intermediate API
    // =========================================================================

    void setSize(T size) noexcept
    {
        size_.store(std::clamp(size, T(0.01), T(1)), std::memory_order_relaxed);
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
        T clamped = std::clamp(amount, T(0), T(1));
        damping_.store(clamped, std::memory_order_relaxed);
        highDecayMult_.store(T(1) - clamped * T(0.9), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    void setPreDelay(T ms) noexcept
    {
        T clamped = std::clamp(ms, T(0), T(200));
        preDelayMs_.store(clamped, std::memory_order_relaxed);
        // preDelaySamples_ is already atomic and only consumed in the audio
        // loop; storing it from any thread is safe and cheap.
        if (spec_.sampleRate > 0)
            preDelaySamples_.store(static_cast<int>(
                static_cast<T>(spec_.sampleRate) * clamped / T(1000)),
                std::memory_order_relaxed);
    }

    void setDiffusion(T amount) noexcept
    {
        diffusion_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    void setModulation(T amount) noexcept
    {
        T clamped = std::clamp(amount, T(0), T(1));
        modDepth_.store(clamped, std::memory_order_relaxed);
        modDepthA_.store(clamped * T(30), std::memory_order_relaxed);
        modDepthB_.store(clamped * T(15), std::memory_order_relaxed);
    }

    /**
     * @brief Sets stereo width of the late reverb tail.
     *
     * Uses M/S processing: 0 = mono, 1 = natural stereo, 2 = extra wide.
     * Applied after output diffusion, before combining with early reflections.
     *
     * @param width Stereo width (0.0 - 2.0). Default: 1.0.
     */
    void setWidth(T width) noexcept
    {
        width_.store(std::clamp(width, T(0), T(2)), std::memory_order_relaxed);
    }

    void setErToLateDelay(T ms) noexcept
    {
        T clamped = std::clamp(ms, T(0), T(200));
        erToLateMs_.store(clamped, std::memory_order_relaxed);
        if (spec_.sampleRate > 0)
            erToLateSamples_.store(static_cast<int>(
                static_cast<T>(spec_.sampleRate) * clamped / T(1000)),
                std::memory_order_relaxed);
    }

    // =========================================================================
    // Level 3: Expert API — Frequency-Dependent Decay
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
        T clamped = std::clamp(mult, T(0.05), T(1));
        highDecayMult_.store(clamped, std::memory_order_relaxed);
        damping_.store((T(1) - clamped) / T(0.9), std::memory_order_relaxed);
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
        bassDecayMult_.store(std::clamp(mult, T(0.3), T(3)),
                             std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Frequency above which HF decay multiplier applies.
     * @param hz Crossover in Hz (1000 - 16000). Default: 5000.
     */
    void setHighCrossover(T hz) noexcept
    {
        highCrossover_.store(std::clamp(hz, T(1000), T(16000)),
                             std::memory_order_relaxed);
    }

    /**
     * @brief Frequency below which bass decay multiplier applies.
     * @param hz Crossover in Hz (50 - 500). Default: 200.
     */
    void setBassCrossover(T hz) noexcept
    {
        bassCrossover_.store(std::clamp(hz, T(50), T(500)),
                             std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    // =========================================================================
    // Level 3: Expert API — Tone & Levels
    // =========================================================================

    void setEarlyLevel(T dB) noexcept
    {
        earlyLevel_.store(decibelsToGain(std::clamp(dB, T(-60), T(6))), std::memory_order_relaxed);
    }

    void setLateLevel(T dB) noexcept
    {
        lateLevel_.store(decibelsToGain(std::clamp(dB, T(-60), T(6))), std::memory_order_relaxed);
    }

    void setModRate(T hz) noexcept
    {
        modRate_.store(std::clamp(hz, T(0.1), T(5)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets a post-reverb low-cut filter on the wet signal (12 dB/oct).
     * @param hz Cutoff in Hz (0 = off, 20-500 typical).
     */
    void setToneLowCut(T hz) noexcept
    {
        // Queue target; audio thread rebuilds the biquad inside processBlock.
        toneLowCutHz_.store(hz, std::memory_order_relaxed);
        toneDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets a post-reverb high-cut filter on the wet signal (12 dB/oct).
     * @param hz Cutoff in Hz (0 = off, 2000-16000 typical).
     */
    void setToneHighCut(T hz) noexcept
    {
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
    [[nodiscard]] T getHighDecayMultiplier() const noexcept { return highDecayMult_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBassDecayMultiplier() const noexcept { return bassDecayMult_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWidth() const noexcept { return width_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHighCrossover() const noexcept { return highCrossover_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBassCrossover() const noexcept { return bassCrossover_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("ARVB"), 1);
        w.write("type", static_cast<int32_t>(type_.load(std::memory_order_relaxed)));
        w.write("quality", static_cast<int32_t>(quality_.load(std::memory_order_relaxed)));
        w.write("decay", decayTime_.load(std::memory_order_relaxed));
        w.write("size", size_.load(std::memory_order_relaxed));
        w.write("damping", damping_.load(std::memory_order_relaxed));
        w.write("diffusion", diffusion_.load(std::memory_order_relaxed));
        w.write("modDepth", modDepth_.load(std::memory_order_relaxed));
        w.write("modRate", modRate_.load(std::memory_order_relaxed));
        w.write("preDelay", preDelayMs_.load(std::memory_order_relaxed));
        w.write("erToLate", erToLateMs_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("width", width_.load(std::memory_order_relaxed));
        w.write("highDecay", highDecayMult_.load(std::memory_order_relaxed));
        w.write("bassDecay", bassDecayMult_.load(std::memory_order_relaxed));
        w.write("highXover", highCrossover_.load(std::memory_order_relaxed));
        w.write("bassXover", bassCrossover_.load(std::memory_order_relaxed));
        w.write("earlyDb", static_cast<float>(20.0 * std::log10(std::max(
            static_cast<double>(earlyLevel_.load(std::memory_order_relaxed)), 1e-6))));
        w.write("lateDb", static_cast<float>(20.0 * std::log10(std::max(
            static_cast<double>(lateLevel_.load(std::memory_order_relaxed)), 1e-6))));
        w.write("toneLowCut", toneLowCutHz_.load(std::memory_order_relaxed));
        w.write("toneHighCut", toneHighCutHz_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("ARVB")) return false;
        setType(static_cast<Type>(r.read("type", 0)));
        // Older blobs have no "quality" key: default 0 = Full (unchanged sound).
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

    static constexpr int kFDNSize      = 16;
    static constexpr int kDiffStages   = 8;
    static constexpr int kMaxERTaps    = 40;
    static constexpr int kNumMultiTaps = 7;
    static constexpr int kOutDiffStages = 2;

    // Eco quality engine reductions (see setQuality)
    static constexpr int kEcoLines        = 8;   ///< FDN lines in Eco quality
    static constexpr int kEcoCtrlInterval = 16;  ///< modulation refresh period (samples)
    static constexpr int kEcoERTaps       = 12;  ///< early-reflection tap cap in Eco

    static constexpr T kInputGain = T(1) / T(8);

    // Output normalization: 16 main + 7 multi-tap + fbAP taps + intAP taps
    static constexpr T kOutputNorm = T(1) / T(5.0);

    // Eco output normalization: 8 sign-weighted lines only (no extra taps).
    // Calibrated so the Eco wet tail matches Full loudness (measured RMS).
    static constexpr T kOutputNormEco = T(0.32);

    // Orthogonal stereo output sign vectors (inner product = 0)
    static constexpr int kOutSignL_[kFDNSize] = {
         1, -1,  1,  1, -1,  1, -1, -1,
         1, -1, -1,  1, -1,  1,  1, -1
    };
    static constexpr int kOutSignR_[kFDNSize] = {
         1,  1, -1,  1,  1, -1, -1, -1,
        -1,  1, -1,  1, -1, -1,  1,  1
    };

    // Multi-tap output: Dattorro-style decorrelation from different delay positions
    static constexpr int    kMultiTapLineL_[kNumMultiTaps] = {0, 2, 5, 7, 9, 12, 14};
    static constexpr int    kMultiTapLineR_[kNumMultiTaps] = {1, 3, 4, 8, 10, 13, 15};
    static constexpr double kMultiTapFracL_[kNumMultiTaps] = {0.37, 0.67, 0.23, 0.81, 0.44, 0.59, 0.31};
    static constexpr double kMultiTapFracR_[kNumMultiTaps] = {0.43, 0.71, 0.29, 0.63, 0.47, 0.53, 0.37};
    static constexpr int    kMultiTapSignL_[kNumMultiTaps] = {+1, -1, +1, -1, +1, -1, +1};
    static constexpr int    kMultiTapSignR_[kNumMultiTaps] = {+1, +1, -1, +1, -1, +1, -1};

    // FDN base delay times in ms (at size=1.0)
    static constexpr double kBaseDelaysMs_[kFDNSize] = {
        29.7, 34.1, 39.3, 45.2, 52.0, 58.1, 64.9, 72.3,
        80.4, 89.0, 98.3, 108.7, 119.9, 132.3, 145.7, 160.1
    };

    // Input diffusion allpass delays (ms) — 8 stages (Dattorro-style, ~34ms total)
    static constexpr double kDiffDelaysMs_[kDiffStages] = {
        1.03, 1.47, 2.19, 3.13, 4.23, 5.59, 7.19, 9.47
    };

    static constexpr double kDiffBaseCoeffs_[kDiffStages] = {
        0.75, 0.75, 0.72, 0.72, 0.70, 0.70, 0.68, 0.68
    };

    // Feedback allpass stages per FDN line (echo density multiplier)
    static constexpr int kFbAPStages = 2;

    // Feedback allpass delays as fraction of FDN delay (Dattorro-style proportional)
    static constexpr double kFbAPRatioA_ = 0.25;  // 25% of FDN delay
    static constexpr double kFbAPRatioB_ = 0.35;  // 35% of FDN delay

    // Parallel allpass diffuser — step 1 delays (ms, per channel, different IRs)
    static constexpr double kParAPDelaysMs_[kFDNSize] = {
        5.3, 6.1, 7.1, 7.9, 8.9, 9.7, 10.7, 11.7,
        12.3, 13.3, 14.3, 15.1, 16.1, 17.1, 18.1, 19.1
    };

    // Multi-channel diffuser — step 2 per-channel delays (ms)
    static constexpr double kDiffuserStep2Ms_[kFDNSize] = {
        15.7, 17.9, 20.1, 22.3, 24.7, 26.9, 29.3, 31.1,
        33.7, 35.3, 37.1, 39.3, 41.1, 42.9, 44.3, 45.7
    };

    // Output diffusion allpass delays (ms, decorrelated L/R)
    static constexpr double kOutDiffDelaysMsL_[kOutDiffStages] = {1.47, 2.31};
    static constexpr double kOutDiffDelaysMsR_[kOutDiffStages] = {1.63, 2.47};

    // =========================================================================
    // Smooth Random LFO (Lexicon-style modulation)
    // =========================================================================

    /**
     * @brief Hermite-interpolated random noise generator for organic modulation.
     *
     * Generates band-limited random values via cubic Hermite (Catmull-Rom)
     * interpolation between xorshift32 random targets. Produces smooth,
     * non-periodic modulation — the key to Lexicon-quality reverb character.
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

        void setRate(T rate, double sr) noexcept
        {
            phaseInc_ = rate / static_cast<T>(sr);
        }

        T next() noexcept { return nextStride(1); }

        /**
         * @brief Advances the LFO by `stride` samples in one call.
         *
         * Used by Eco quality to run modulation at control rate: the phase
         * advances by stride * phaseInc_ so the effective LFO frequency is
         * unchanged. Rates are clamped well below 1/stride of the sample
         * rate, so at most one Hermite target is consumed per call.
         */
        T nextStride(int stride) noexcept
        {
            phase_ += phaseInc_ * static_cast<T>(stride);
            if (phase_ >= T(1))
            {
                phase_ -= T(1);
                h0_ = h1_; h1_ = h2_; h2_ = h3_;
                h3_ = nextRandom();
            }
            // Cubic Hermite (Catmull-Rom) interpolation
            T d = phase_;
            T c0 = h1_;
            T c1 = T(0.5) * (h2_ - h0_);
            T c2 = h0_ - T(2.5) * h1_ + T(2) * h2_ - T(0.5) * h3_;
            T c3 = T(0.5) * (h3_ - h0_) + T(1.5) * (h1_ - h2_);
            return ((c3 * d + c2) * d + c1) * d + c0;
        }

        void reset() noexcept { phase_ = T(0); h0_ = h1_ = h2_ = h3_ = T(0); }

    private:
        T nextRandom() noexcept
        {
            state_ ^= state_ << 13;
            state_ ^= state_ >> 17;
            state_ ^= state_ << 5;
            return static_cast<T>(state_)
                   / static_cast<T>(0xFFFFFFFFu) * T(2) - T(1);
        }
    };

    // --- State ---------------------------------------------------------------

    AudioSpec spec_ {};

    // FDN delay lines
    std::array<RingBuffer<T>, kFDNSize> fdnDelays_;
    std::array<int, kFDNSize> fdnDelayLens_ {};

    // Jot absorption filter state (per line)
    std::array<T, kFDNSize> absB0_ {};       // feedforward: g_mid * (1 - damp)
    std::array<T, kFDNSize> absA1_ {};       // feedback: damp coefficient
    std::array<T, kFDNSize> absState_ {};    // filter state

    // Bass shelf state (per line)
    std::array<T, kFDNSize> bassRatio_ {};   // g_bass / g_mid
    std::array<T, kFDNSize> bassState_ {};   // 1-pole LP state

    // DC blocker state
    std::array<T, kFDNSize> dcZ_ {};

    // Feedback allpass (2 stages per FDN line, density multiplier)
    std::array<RingBuffer<T>, kFDNSize> fbAPBufsA_;
    std::array<RingBuffer<T>, kFDNSize> fbAPBufsB_;
    std::array<int, kFDNSize> fbAPDelaysA_ {};
    std::array<int, kFDNSize> fbAPDelaysB_ {};
    T fbAPCoeff_ = T(0.6);

    // Internal serial allpasses (per FDN line, pre-write + post-read)
    std::array<RingBuffer<T>, kFDNSize> intAPBufsA_;  // pre-write
    std::array<RingBuffer<T>, kFDNSize> intAPBufsB_;  // post-read
    std::array<int, kFDNSize> intAPDelaysA_ {};
    std::array<int, kFDNSize> intAPDelaysB_ {};
    static constexpr T intAPCoeff_ = T(0.5);  // Infinity2 uses 0.5
    static constexpr double kIntAPRatioA_ = 0.15;  // 15% of FDN delay
    static constexpr double kIntAPRatioB_ = 0.20;  // 20% of FDN delay

    // Feedback IIR smoothing (Verbity technique: eliminates discrete-echo quality)
    std::array<T, kFDNSize> prevFeedback_ {};
    T fbSmooth_ = T(0.3);

    // Smooth random modulation (2 per line = 32 total)
    std::array<SmoothRandomLFO, kFDNSize> modLFOA_;
    std::array<SmoothRandomLFO, kFDNSize> modLFOB_;

    // Allpass interpolation state for modulated FDN reads (preserves HF)
    std::array<T, kFDNSize> apInterpState_ {};

    // Filtered noise for modulation randomization (Progenitor-style)
    uint32_t noiseState_ = 1;
    T noiseLP_ = T(0);
    T noiseCoeff_ = T(0);
    T noiseDepth_ = T(0);

    // Quality engine state (audio-thread only, derived from quality_ at the
    // dirty-flag drain). Full: 16 lines, per-sample modulation. Eco: 8 lines,
    // control-rate modulation refreshed every kEcoCtrlInterval samples.
    bool eco_   = false;
    int  nLines_ = kFDNSize;
    std::array<T, kFDNSize> modACache_ {};   // per-line LFO A value (samples)
    std::array<T, kFDNSize> modBCache_ {};   // per-line LFO B value (samples)
    T   noiseND_ = T(0);                     // filtered noise * noiseDepth_
    int ctrlPhase_ = 0;                      // Eco control-rate phase counter
    T   noiseCoeffEco_ = T(0);               // noise LP coeff at control rate

    // Input diffusion
    std::array<RingBuffer<T>, kDiffStages> diffBufs_;
    std::array<int, kDiffStages> diffDelays_ {};
    std::array<T, kDiffStages> diffCoeffs_ {};

    // Parallel allpass diffuser — step 1 (16 parallel allpass, different delays)
    std::array<RingBuffer<T>, kFDNSize> parAPBufs_;
    std::array<int, kFDNSize> parAPDelays_ {};
    T parAPCoeff_ = T(0.65);

    // Multi-channel diffuser — step 2 (16 per-channel delay buffers)
    std::array<RingBuffer<T>, kFDNSize> diffuserStep2_;
    std::array<int, kFDNSize> diffuserStep2Delays_ {};

    // Output diffusion (L/R decorrelated)
    std::array<RingBuffer<T>, kOutDiffStages> outDiffBufsL_;
    std::array<RingBuffer<T>, kOutDiffStages> outDiffBufsR_;
    std::array<int, kOutDiffStages> outDiffDelaysL_ {};
    std::array<int, kOutDiffStages> outDiffDelaysR_ {};
    T outDiffCoeff_ = T(0.45);

    // Early reflections
    RingBuffer<T> erBuf_;
    std::array<int, kMaxERTaps> erTapsL_ {}, erTapsR_ {};
    std::array<T, kMaxERTaps> erGainsL_ {}, erGainsR_ {};
    std::array<T, kMaxERTaps> erAbsCoeffs_ {};
    std::array<T, kMaxERTaps> erLPStateL_ {};
    std::array<T, kMaxERTaps> erLPStateR_ {};
    int numERTaps_ = 0;

    // Pre-delay
    RingBuffer<T> preDelayBuf_;
    std::atomic<int> preDelaySamples_ { 0 };

    // ER-to-late gap
    RingBuffer<T> erToLateBuf_;
    std::atomic<int> erToLateSamples_ { 0 };

    // Tone correction EQ (Biquad 12 dB/oct)
    Biquad<T, 2> toneLPBiquad_;
    Biquad<T, 2> toneHPBiquad_;
    bool toneLPActive_ = false;
    bool toneHPActive_ = false;

    // Mixer
    DryWetMixer<T> mixer_;

    // --- Parameters ----------------------------------------------------------
    //
    // Thread-safety model (C2/C3 fix):
    //  - All GUI-thread setters mutate ONLY atomic shadow fields below and set
    //    `paramsDirty_` (or `presetDirty_` for setType).
    //  - The audio thread drains the dirty flags at the top of `processBlock()`
    //    and calls the update helpers there — so every non-atomic array that
    //    participates in the audio path (fdnDelayLens_, diffCoeffs_, …) is
    //    mutated exclusively from the audio thread. No races, no torn reads.

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
    std::atomic<T> width_        { T(1) };     // Stereo width: 0=mono, 1=natural, 2=wide

    // Frequency-dependent decay parameters
    std::atomic<T> highDecayMult_ { T(0.5) };   // HF T60 multiplier (0.05-1.0)
    std::atomic<T> bassDecayMult_ { T(1.2) };   // bass T60 multiplier (0.3-3.0)
    std::atomic<T> highCrossover_ { T(5000) };  // Hz
    std::atomic<T> bassCrossover_ { T(200) };   // Hz

    std::atomic<Quality> quality_ { Quality::Full };

    // Deferred-apply flags (audio thread drains these at top of processBlock)
    std::atomic<bool> presetDirty_ { false };  // setType() → rebuild topology + reset
    std::atomic<bool> paramsDirty_ { false };  // any other setter → refresh coeffs
    std::atomic<bool> toneDirty_   { false };  // tone EQ cutoff changes
    std::atomic<bool> qualityDirty_ { false }; // setQuality() -> resize engine + reset
    std::atomic<T> toneLowCutHz_  { T(-1) };   // <0 = off, queued value for audio thread
    std::atomic<T> toneHighCutHz_ { T(-1) };

    // --- Computed coefficients ------------------------------------------------

    T bassLPCoeff_ = T(0.026);   // bass crossover filter coeff
    T dcCoeff_     = T(0.003);   // DC blocker coeff (~23Hz)
    std::atomic<T> modDepthA_ { T(1) };       // slow LFO depth in samples
    std::atomic<T> modDepthB_ { T(0.5) };     // fast LFO depth in samples

    // =========================================================================
    // Processing helpers
    // =========================================================================

    /// Lowpass-filtered white noise for modulation randomization.
    T nextFilteredNoise() noexcept
    {
        noiseState_ = noiseState_ * 196314165u + 907633515u;
        T white = static_cast<T>(static_cast<int32_t>(noiseState_))
                  / T(2147483648.0);
        noiseLP_ += noiseCoeff_ * (white - noiseLP_);
        return noiseLP_;
    }

    /// Control-rate variant (Eco): same generator, coefficient scaled so the
    /// ~3 Hz cutoff is preserved when called every kEcoCtrlInterval samples.
    T nextFilteredNoiseEco() noexcept
    {
        noiseState_ = noiseState_ * 196314165u + 907633515u;
        T white = static_cast<T>(static_cast<int32_t>(noiseState_))
                  / T(2147483648.0);
        noiseLP_ += noiseCoeffEco_ * (white - noiseLP_);
        return noiseLP_;
    }

    T processAllpass(RingBuffer<T>& buf, int delay, T coeff, T input) noexcept
    {
        T delayed = buf.read(delay);
        T temp = input + coeff * delayed;
        T output = delayed - coeff * temp;
        buf.push(temp);
        return output;
    }

    T processAllpassModulated(RingBuffer<T>& buf, int baseDelay, T modAmount,
                              T coeff, T input, bool linearInterp) noexcept
    {
        T readPos = static_cast<T>(baseDelay) + modAmount;
        readPos = std::max(readPos, T(1));
        // Eco quality reads with 2-point linear interpolation (the industry
        // standard for diffusion allpasses with small excursions); Full keeps
        // the default 4-point Hermite read.
        T delayed = linearInterp
            ? buf.template readInterpolated<InterpMethod::Linear>(readPos)
            : buf.readInterpolated(readPos);
        T temp = input + coeff * delayed;
        T output = delayed - coeff * temp;
        buf.push(temp);
        return output;
    }

    /**
     * @brief In-place Hadamard 16x16 via Fast Walsh-Hadamard butterfly.
     *
     * 4-stage butterfly, 64 add/sub, normalized by 1/sqrt(16)=1/4.
     * Used in diffusion stages where maximum inter-channel mixing is desired.
     */
    static void hadamardInPlace(std::array<T, kFDNSize>& x) noexcept
    {
        for (int i = 0; i < kFDNSize; i += 2)
        { T a = x[i], b = x[i+1]; x[i] = a + b; x[i+1] = a - b; }

        for (int i = 0; i < kFDNSize; i += 4)
            for (int j = 0; j < 2; ++j)
            { T a = x[i+j], b = x[i+j+2]; x[i+j] = a + b; x[i+j+2] = a - b; }

        for (int i = 0; i < kFDNSize; i += 8)
            for (int j = 0; j < 4; ++j)
            { T a = x[i+j], b = x[i+j+4]; x[i+j] = a + b; x[i+j+4] = a - b; }

        for (int j = 0; j < 8; ++j)
        { T a = x[j], b = x[j+8]; x[j] = a + b; x[j+8] = a - b; }

        for (auto& v : x) v *= T(0.25);
    }

    /**
     * @brief In-place Hadamard 8x8 on the first 8 elements (Eco quality).
     *
     * 3-stage butterfly, normalized by 1/sqrt(8). Elements 8..15 untouched.
     */
    static void hadamard8InPlace(std::array<T, kFDNSize>& x) noexcept
    {
        for (int i = 0; i < 8; i += 2)
        { T a = x[i], b = x[i+1]; x[i] = a + b; x[i+1] = a - b; }

        for (int i = 0; i < 8; i += 4)
            for (int j = 0; j < 2; ++j)
            { T a = x[i+j], b = x[i+j+2]; x[i+j] = a + b; x[i+j+2] = a - b; }

        for (int j = 0; j < 4; ++j)
        { T a = x[j], b = x[j+4]; x[j] = a + b; x[j+4] = a - b; }

        constexpr T norm = T(0.35355339059327373);  // 1/sqrt(8)
        for (int i = 0; i < 8; ++i) x[i] *= norm;
    }

    /**
     * @brief In-place Householder reflection for 16 channels.
     *
     * H = I - (2/N) * ones. Each output = input - (2/N) * sum.
     * Provides moderate cross-feeding without locking delays together,
     * as recommended by Signalsmith for FDN feedback mixing.
     */
    static void householderInPlace(std::array<T, kFDNSize>& x, int n) noexcept
    {
        T sum = T(0);
        for (int i = 0; i < n; ++i) sum += x[i];
        T factor = sum * T(2) / static_cast<T>(n);
        for (int i = 0; i < n; ++i) x[i] -= factor;
    }

    /// Block-cached copies of the atomic parameters: 16 FDN lines reading
    /// ~10 relaxed atomics per SAMPLE added measurable overhead; one refresh
    /// per block (refreshCachedParams) is bit-identical for parameters that
    /// only change at block rate anyway.
    struct CachedParams
    {
        int preDelaySamples = 0;
        int erToLateSamples = 0;
        T earlyLevel = T(1);
        T lateLevel  = T(1);
        T width      = T(1);
        T modDepthA  = T(1);
        T modDepthB  = T(0.5);
    };
    CachedParams cachedParams_ {};

    /// Pulls the atomic parameters into the block-local cache (audio thread).
    void refreshCachedParams() noexcept
    {
        cachedParams_.preDelaySamples = preDelaySamples_.load(std::memory_order_relaxed);
        cachedParams_.erToLateSamples = erToLateSamples_.load(std::memory_order_relaxed);
        cachedParams_.earlyLevel = earlyLevel_.load(std::memory_order_relaxed);
        cachedParams_.lateLevel  = lateLevel_.load(std::memory_order_relaxed);
        cachedParams_.width      = width_.load(std::memory_order_relaxed);
        cachedParams_.modDepthA  = modDepthA_.load(std::memory_order_relaxed);
        cachedParams_.modDepthB  = modDepthB_.load(std::memory_order_relaxed);
    }

    /// Core per-sample processing — returns wet {L, R}.
    std::pair<T, T> processSampleInternal(T input) noexcept
    {
        // Block-cached params (see refreshCachedParams)
        const int preDelSamp = cachedParams_.preDelaySamples;
        const int erToLateSamp = cachedParams_.erToLateSamples;
        const T earlyLvl = cachedParams_.earlyLevel;
        const T lateLvl = cachedParams_.lateLevel;
        const T widthVal = cachedParams_.width;
        const T modDA = cachedParams_.modDepthA;
        const T modDB = cachedParams_.modDepthB;

        // --- Pre-delay ---
        preDelayBuf_.push(input);
        T delayed = (preDelSamp > 0)
            ? preDelayBuf_.read(preDelSamp) : input;

        // --- Input diffusion: 8 cascaded modulated allpass ---
        T diffused = delayed;
        T diffNoise = nextFilteredNoise();
        for (int d = 0; d < kDiffStages; ++d)
        {
            T diffPol = (d & 1) ? T(-1) : T(1);
            T diffMod = diffNoise * T(2) * diffPol;  // +/-2 samples excursion
            diffused = processAllpassModulated(diffBufs_[d], diffDelays_[d],
                                               diffMod, diffCoeffs_[d], diffused,
                                               eco_);
        }

        // --- Early reflections with progressive absorption ---
        erBuf_.push(diffused);
        T earlyL = T(0), earlyR = T(0);
        for (int t = 0; t < numERTaps_; ++t)
        {
            T rawL = erBuf_.read(erTapsL_[t]) * erGainsL_[t];
            T rawR = erBuf_.read(erTapsR_[t]) * erGainsR_[t];
            // Progressive 1-pole LP: later taps are darker (wall absorption)
            erLPStateL_[t] += erAbsCoeffs_[t] * (rawL - erLPStateL_[t]);
            erLPStateR_[t] += erAbsCoeffs_[t] * (rawR - erLPStateR_[t]);
            earlyL += erLPStateL_[t];
            earlyR += erLPStateR_[t];
        }
        earlyL *= earlyLvl;
        earlyR *= earlyLvl;

        // --- ER-to-late gap ---
        erToLateBuf_.push(diffused);
        T fdnInputRaw = (erToLateSamp > 0)
            ? erToLateBuf_.read(erToLateSamp) : diffused;

        // --- Parallel allpass diffuser (2-step, 16² = 256 echo paths) ---
        // Step 1: parallel allpass with different delays create unique IRs.
        // Eco: single step, 8 channels (16 echo paths via one Hadamard).
        const int n = nLines_;
        std::array<T, kFDNSize> diffCh {};
        for (int d = 0; d < n; ++d)
            diffCh[d] = processAllpass(parAPBufs_[d], parAPDelays_[d],
                                        parAPCoeff_, fdnInputRaw);
        if (!eco_)
        {
            hadamardInPlace(diffCh);
            // Step 2: per-channel delay + Hadamard → 256 unique paths
            for (int d = 0; d < kFDNSize; ++d)
                diffuserStep2_[d].push(diffCh[d]);
            for (int d = 0; d < kFDNSize; ++d)
                diffCh[d] = diffuserStep2_[d].read(diffuserStep2Delays_[d]);
            hadamardInPlace(diffCh);
        }
        else
        {
            hadamard8InPlace(diffCh);
        }

        // =================================================================
        // FDN Core
        // =================================================================

        // Refresh modulation values. Full: per sample (bit-identical to the
        // previous inline computation: same LFO call order and arithmetic).
        // Eco: every kEcoCtrlInterval samples (control rate); at reverb mod
        // rates (0.1-5 Hz) the position steps are microscopic (< 1e-2 samples)
        // and far below audibility.
        if (!eco_)
        {
            T noise = nextFilteredNoise();
            noiseND_ = noise * noiseDepth_;
            for (int d = 0; d < n; ++d)
            {
                modACache_[d] = modLFOA_[d].next() * modDA;
                modBCache_[d] = modLFOB_[d].next() * modDB;
            }
        }
        else
        {
            if (ctrlPhase_ == 0)
            {
                noiseND_ = nextFilteredNoiseEco() * noiseDepth_;
                for (int d = 0; d < n; ++d)
                {
                    modACache_[d] = modLFOA_[d].nextStride(kEcoCtrlInterval) * modDA;
                    modBCache_[d] = modLFOB_[d].nextStride(kEcoCtrlInterval) * modDB;
                }
            }
            if (++ctrlPhase_ >= kEcoCtrlInterval) ctrlPhase_ = 0;
        }

        // Read from the delay lines with LFO + noise modulated delay
        std::array<T, kFDNSize> reads {};
        for (int d = 0; d < n; ++d)
        {
            // Noise with alternating polarity (Progenitor-style)
            T polarity = (d & 1) ? T(-1) : T(1);
            T noiseMod = noiseND_ * polarity;
            T readPos = static_cast<T>(fdnDelayLens_[d]) + modACache_[d]
                        + modBCache_[d] + noiseMod;
            readPos = std::max(readPos, T(1));
            // Allpass interpolation: preserves HF over hundreds of feedback
            // iterations (linear/cubic causes cumulative dulling).
            // Formula: z1 = older + frac*(newer - z1)  [Progenitor2-style]
            {
                int readInt = static_cast<int>(readPos);
                T frac = readPos - static_cast<T>(readInt);
                T newer = fdnDelays_[d].read(readInt);
                T older = fdnDelays_[d].read(readInt + 1);
                apInterpState_[d] = older + frac * (newer - apInterpState_[d]);
                reads[d] = apInterpState_[d];
            }
            // Post-read serial allpass (density multiplier, Infinity2-style)
            reads[d] = processAllpass(intAPBufsB_[d], intAPDelaysB_[d],
                                       intAPCoeff_, reads[d]);
        }

        // Householder mixing (moderate coupling, keeps lines independent)
        std::array<T, kFDNSize> mixed = reads;
        householderInPlace(mixed, n);

        // Per-line: Jot absorption → bass shelf → feedback allpass → write
        for (int d = 0; d < n; ++d)
        {
            T val = mixed[d];

            // --- Jot absorption filter (1st-order shelving, Jot 1991) ---
            // y[n] = b0 * x[n] + a1 * y[n-1]
            // At DC: gain = g_mid. At Nyquist: gain = g_high.
            absState_[d] = absB0_[d] * val + absA1_[d] * absState_[d];
            val = absState_[d];

            // --- Bass shelf (independent LF control) ---
            bassState_[d] += bassLPCoeff_ * (val - bassState_[d]);
            val += bassState_[d] * (bassRatio_[d] - T(1));

            // --- Feedback allpass (2 stages, modulated, Dattorro-style) ---
            {
                T fbMod = modBCache_[d] * T(0.3);
                val = processAllpassModulated(fbAPBufsA_[d], fbAPDelaysA_[d],
                                              fbMod, fbAPCoeff_, val, eco_);
                val = processAllpassModulated(fbAPBufsB_[d], fbAPDelaysB_[d],
                                              fbMod * T(-0.7), fbAPCoeff_, val,
                                              eco_);
            }

            // --- DC blocker (~23 Hz) ---
            dcZ_[d] += dcCoeff_ * (val - dcZ_[d]);
            val -= dcZ_[d];

            // --- Soft saturation (Branchless fast approximation) ---
            // Extract the overshoot beyond [-1, 1] without branching
            T exceed = std::max(T(0), val - T(1)) - std::max(T(0), T(-1) - val);
            
            // Limit base value strictly to [-1, 1]
            val -= exceed; 
            
            // Apply fast rational soft-clip: x / (1 + |x|) ONLY to the exceeding portion
            // Approximates tanh curve infinitely closer to limits without unpredictable CPU branch hits
            val += exceed / (T(1) + std::abs(exceed));

            // --- Pre-write serial allpass (density, Infinity2-style) ---
            val = processAllpass(intAPBufsA_[d], intAPDelaysA_[d],
                                  intAPCoeff_, val);

            // --- Feedback IIR smoothing (Verbity technique) ---
            val = val * (T(1) - fbSmooth_) + prevFeedback_[d] * fbSmooth_;
            prevFeedback_[d] = val;

            // --- Write back with per-line diffuser output ---
            fdnDelays_[d].push(val + diffCh[d] * kInputGain);
        }

        // --- Stereo output ---

        // Main: orthogonal sign-weighted from all line reads (the first 8
        // sign entries are also mutually orthogonal, so Eco keeps the L/R
        // decorrelation property)
        T lateL = T(0), lateR = T(0);
        for (int d = 0; d < n; ++d)
        {
            lateL += reads[d] * static_cast<T>(kOutSignL_[d]);
            lateR += reads[d] * static_cast<T>(kOutSignR_[d]);
        }

        if (!eco_)
        {
        // Multi-tap: Dattorro-style reads from different delay positions
        for (int t = 0; t < kNumMultiTaps; ++t)
        {
            int posL = std::max(1, static_cast<int>(
                fdnDelayLens_[kMultiTapLineL_[t]] * kMultiTapFracL_[t]));
            int posR = std::max(1, static_cast<int>(
                fdnDelayLens_[kMultiTapLineR_[t]] * kMultiTapFracR_[t]));
            lateL += fdnDelays_[kMultiTapLineL_[t]].read(posL)
                     * static_cast<T>(kMultiTapSignL_[t]) * T(0.7);
            lateR += fdnDelays_[kMultiTapLineR_[t]].read(posR)
                     * static_cast<T>(kMultiTapSignR_[t]) * T(0.7);
        }

        // Feedback allpass taps (Dattorro-style: read from within AP buffers)
        for (int d = 0; d < kFDNSize; d += 2)
        {
            int tapA = std::max(1, fbAPDelaysA_[d] * 2 / 3);
            int tapB = std::max(1, fbAPDelaysB_[d] * 3 / 5);
            lateL += fbAPBufsA_[d].read(tapA) * static_cast<T>(kOutSignL_[d]) * T(0.35);
            lateR += fbAPBufsB_[d].read(tapB) * static_cast<T>(kOutSignR_[d]) * T(0.35);
        }
        for (int d = 1; d < kFDNSize; d += 2)
        {
            int tapA = std::max(1, fbAPDelaysA_[d] * 3 / 5);
            int tapB = std::max(1, fbAPDelaysB_[d] * 2 / 3);
            lateL += fbAPBufsB_[d].read(tapB) * static_cast<T>(kOutSignL_[d]) * T(0.35);
            lateR += fbAPBufsA_[d].read(tapA) * static_cast<T>(kOutSignR_[d]) * T(0.35);
        }

        // Internal serial allpass taps (additional temporal smearing)
        for (int d = 0; d < kFDNSize; d += 2)
        {
            int tapA = std::max(1, intAPDelaysA_[d] * 2 / 3);
            int tapB = std::max(1, intAPDelaysB_[d] * 3 / 5);
            lateL += intAPBufsA_[d].read(tapA) * static_cast<T>(kOutSignL_[d]) * T(0.2);
            lateR += intAPBufsB_[d].read(tapB) * static_cast<T>(kOutSignR_[d]) * T(0.2);
        }
        for (int d = 1; d < kFDNSize; d += 2)
        {
            int tapA = std::max(1, intAPDelaysA_[d] * 3 / 5);
            int tapB = std::max(1, intAPDelaysB_[d] * 2 / 3);
            lateL += intAPBufsB_[d].read(tapB) * static_cast<T>(kOutSignL_[d]) * T(0.2);
            lateR += intAPBufsA_[d].read(tapA) * static_cast<T>(kOutSignR_[d]) * T(0.2);
        }
        } // !eco_ (extra output taps)

        const T outNorm = eco_ ? kOutputNormEco : kOutputNorm;
        lateL *= lateLvl * outNorm;
        lateR *= lateLvl * outNorm;

        // --- Output diffusion (L/R decorrelated allpass) ---
        for (int s = 0; s < kOutDiffStages; ++s)
        {
            lateL = processAllpass(outDiffBufsL_[s], outDiffDelaysL_[s],
                                   outDiffCoeff_, lateL);
            lateR = processAllpass(outDiffBufsR_[s], outDiffDelaysR_[s],
                                   outDiffCoeff_, lateR);
        }

        // --- Stereo width (M/S on late tail only) ---
        {
            T mid  = (lateL + lateR) * T(0.5);
            T side = (lateL - lateR) * T(0.5);
            side *= widthVal;
            lateL = mid + side;
            lateR = mid - side;
        }

        // --- Combine early + late ---
        T outL = earlyL + lateL;
        T outR = earlyR + lateR;

        // --- Tone correction EQ (Biquad 12 dB/oct) ---
        if (toneHPActive_)
        {
            outL = toneHPBiquad_.processSample(outL, 0);
            outR = toneHPBiquad_.processSample(outR, 1);
        }
        if (toneLPActive_)
        {
            outL = toneLPBiquad_.processSample(outL, 0);
            outR = toneLPBiquad_.processSample(outR, 1);
        }

        return { outL, outR };
    }

    // =========================================================================
    // Coefficient update helpers
    // =========================================================================

    /**
     * @brief Computes per-line Jot absorption filter coefficients.
     *
     * Implements Jot/Chaigne (1991): each delay line gets a 1st-order
     * shelving IIR that produces frequency-dependent decay.
     *
     * H(z) = b0 / (1 - a1 * z^-1)
     *
     * where |H(1)| = g_mid (decay at DC/mid) and |H(-1)| = g_high (decay at Nyquist).
     */
    void updateDecayParams() noexcept
    {
        T decay = decayTime_.load(std::memory_order_relaxed);
        if (spec_.sampleRate <= 0 || decay <= T(0)) return;

        T sr = static_cast<T>(spec_.sampleRate);
        T hdMult = highDecayMult_.load(std::memory_order_relaxed);
        T bdMult = bassDecayMult_.load(std::memory_order_relaxed);
        T t60Mid  = decay;
        T t60High = decay * hdMult;
        T t60Bass = decay * bdMult;

        t60High = std::max(t60High, T(0.05));
        t60Bass = std::max(t60Bass, T(0.05));

        for (int d = 0; d < kFDNSize; ++d)
        {
            // Total loop delay = FDN delay + feedback APs + internal APs
            T M = static_cast<T>(fdnDelayLens_[d] + fbAPDelaysA_[d] + fbAPDelaysB_[d]
                                  + intAPDelaysA_[d] + intAPDelaysB_[d]);
            if (M < T(1)) M = T(1);

            // Per-loop gains: g = 0.001^(M / (T60 * sr))
            T gMid  = std::pow(T(0.001), M / (t60Mid * sr));
            T gHigh = std::pow(T(0.001), M / (t60High * sr));
            T gBass = std::pow(T(0.001), M / (t60Bass * sr));

            // Jot damping coefficient: smooth shelving from gMid (DC) to gHigh (Nyquist)
            T damp = (gMid - gHigh) / (gMid + gHigh + T(1e-10));
            damp = std::clamp(damp, T(0), T(0.999));

            // Precomputed filter coefficients
            absB0_[d] = gMid * (T(1) - damp);   // feedforward
            absA1_[d] = damp;                     // feedback (pole)

            // Bass ratio for independent LF control. The bass shelf multiplies the
            // per-line DC loop gain by bassRatio, so cap it so gMid*bassRatio stays
            // below 1 — otherwise extreme decay + high bass multiplier make the low
            // end self-sustain (a non-decaying drone) instead of ringing out.
            bassRatio_[d] = gBass / (gMid + T(1e-10));
            const T maxBassRatio = T(0.98) / std::max(gMid, T(1e-6));
            bassRatio_[d] = std::clamp(bassRatio_[d], T(0.1), std::min(T(3), maxBassRatio));
        }

        // Bass crossover filter coefficient
        T bassCut = bassCrossover_.load(std::memory_order_relaxed);
        bassLPCoeff_ = T(1) - std::exp(T(-6.283185307179586) * bassCut / sr);
    }

    void updateDelayLengths()
    {
        double sr = spec_.sampleRate;
        // Nonlinear size mapping: size=0 -> 0.35, size=1 -> 1.0
        double sz = 0.35 + 0.65 * static_cast<double>(
            size_.load(std::memory_order_relaxed));

        for (int d = 0; d < kFDNSize; ++d)
        {
            // Eco (8 lines) picks every other base delay so the active lines
            // still span the full 29.7-160 ms range (mode density stays even).
            const int srcIdx = eco_ ? std::min(d * 2, kFDNSize - 1) : d;
            int raw = std::max(1, static_cast<int>(
                kBaseDelaysMs_[srcIdx] * sz * sr / 1000.0));
            fdnDelayLens_[d] = nearestPrime(raw);
        }

        for (int d = 0; d < kDiffStages; ++d)
            diffDelays_[d] = nearestPrime(std::max(1,
                static_cast<int>(kDiffDelaysMs_[d] * sr / 1000.0)));

        for (int d = 0; d < kFDNSize; ++d)
        {
            fbAPDelaysA_[d] = nearestPrime(std::max(1,
                static_cast<int>(fdnDelayLens_[d] * kFbAPRatioA_)));
            fbAPDelaysB_[d] = nearestPrime(std::max(1,
                static_cast<int>(fdnDelayLens_[d] * kFbAPRatioB_)));
            intAPDelaysA_[d] = nearestPrime(std::max(1,
                static_cast<int>(fdnDelayLens_[d] * kIntAPRatioA_)));
            intAPDelaysB_[d] = nearestPrime(std::max(1,
                static_cast<int>(fdnDelayLens_[d] * kIntAPRatioB_)));
        }

        for (int s = 0; s < kOutDiffStages; ++s)
        {
            outDiffDelaysL_[s] = nearestPrime(std::max(1,
                static_cast<int>(kOutDiffDelaysMsL_[s] * sr / 1000.0)));
            outDiffDelaysR_[s] = nearestPrime(std::max(1,
                static_cast<int>(kOutDiffDelaysMsR_[s] * sr / 1000.0)));
        }

        // Parallel allpass diffuser — step 1 delays
        for (int d = 0; d < kFDNSize; ++d)
            parAPDelays_[d] = nearestPrime(std::max(1,
                static_cast<int>(kParAPDelaysMs_[d] * sr / 1000.0)));

        // Multi-channel diffuser — step 2 delays
        for (int d = 0; d < kFDNSize; ++d)
            diffuserStep2Delays_[d] = nearestPrime(std::max(1,
                static_cast<int>(kDiffuserStep2Ms_[d] * sr / 1000.0)));

        updateDecayParams();
    }

    void updateDiffCoeffs() noexcept
    {
        T diff = diffusion_.load(std::memory_order_relaxed);
        for (int d = 0; d < kDiffStages; ++d)
            diffCoeffs_[d] = static_cast<T>(kDiffBaseCoeffs_[d]) * diff;
        outDiffCoeff_ = T(0.45) * diff;
        fbAPCoeff_ = T(0.3) + T(0.4) * diff;  // Dattorro range 0.3-0.7
        // Feedback IIR smoothing: higher diffusion = more smoothing
        fbSmooth_ = T(0.1) + T(0.25) * diff;   // range 0.1-0.35
    }

    void updateModulation() noexcept
    {
        if (spec_.sampleRate <= 0) return;
        T rate = modRate_.load(std::memory_order_relaxed);
        for (int i = 0; i < kFDNSize; ++i)
        {
            modLFOA_[i].setRate(rate * (T(0.7) + T(0.05) * static_cast<T>(i)),
                                spec_.sampleRate);
            modLFOB_[i].setRate(rate * (T(1.8) + T(0.11) * static_cast<T>(i)),
                                spec_.sampleRate);
        }
        // Noise depth: 40% of LFO depth (breaks periodic patterns)
        noiseDepth_ = modDepthA_.load(std::memory_order_relaxed) * T(0.4);
    }

    // --- Early reflection generation -----------------------------------------

    /// Regenerates the ER tap set for a reverb type, honouring the Eco cap.
    /// The taps are re-spread over the type's full min..max window, so a
    /// reduced count keeps the temporal coverage (and the 1/sqrt(N) gain
    /// normalization stays correct).
    void generateERTapsForType(Type type) noexcept
    {
        const int cap = eco_ ? kEcoERTaps : kMaxERTaps;
        switch (type)
        {
            case Type::Room:      generateERTaps(1.5, 35.0, std::min(30, cap));   break;
            case Type::Hall:      generateERTaps(5.0, 110.0, std::min(40, cap));  break;
            case Type::Chamber:   generateERTaps(3.0, 60.0, std::min(35, cap));   break;
            case Type::Plate:     numERTaps_ = 0;                                 break;
            case Type::Spring:    generateERTaps(1.0, 25.0, std::min(15, cap));   break;
            case Type::Cathedral: generateERTaps(10.0, 160.0, std::min(40, cap)); break;
        }
    }

    void generateERTaps(double minMs, double maxMs, int numTaps) noexcept
    {
        numERTaps_ = std::min(numTaps, kMaxERTaps);
        if (numERTaps_ <= 0) return;

        double sr = spec_.sampleRate;
        double ratio = (maxMs > minMs) ? maxMs / minMs : 1.0;
        double sqrtN = std::sqrt(static_cast<double>(numERTaps_));

        for (int t = 0; t < numERTaps_; ++t)
        {
            double frac = (numERTaps_ > 1)
                ? static_cast<double>(t) / static_cast<double>(numERTaps_ - 1) : 0.0;

            double msL = minMs * std::pow(ratio, frac);
            erTapsL_[t] = std::max(1, static_cast<int>(msL * sr / 1000.0));

            double jitter = 1.0 + 0.13 * std::sin(static_cast<double>(t) * 2.39996323);
            double msR = msL * jitter;
            msR = std::clamp(msR, minMs * 0.8, maxMs * 1.15);
            erTapsR_[t] = std::max(1, static_cast<int>(msR * sr / 1000.0));

            if (erTapsR_[t] == erTapsL_[t])
                erTapsR_[t] = std::max(1, erTapsR_[t] + ((t & 1) ? 1 : -1));

            double rawGain = std::pow(0.92, static_cast<double>(t));
            erGainsL_[t] = static_cast<T>(rawGain / sqrtN);
            erGainsR_[t] = static_cast<T>(rawGain / sqrtN * 0.95);
        }

        // Progressive frequency absorption: early taps bright, late taps dark
        // Cutoff sweeps exponentially from 15kHz (tap 0) to 3kHz (last tap)
        for (int t = 0; t < numERTaps_; ++t)
        {
            T f = static_cast<T>(t) / static_cast<T>(std::max(1, numERTaps_ - 1));
            T cutoff = T(15000) * std::pow(T(3000) / T(15000), f);
            erAbsCoeffs_[t] = T(1) - std::exp(T(-6.283185307179586) * cutoff
                                               / static_cast<T>(sr));
        }
    }

    // Sieve of Eratosthenes up to kPrimeTableMax, computed once the first
    // time `nearestPrime` is called. Keeps parameter-update calls cheap
    // (previously O(sqrt(n)) per lookup × ~20 calls per update). For
    // `n >= kPrimeTableMax` we fall back to the original trial division —
    // that branch is only reached at extreme sample rates / reverb sizes (M4).
    static constexpr int kPrimeTableMax = 131072;  // 2^17 — covers up to ~2.7 s @48kHz

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

    static int nearestPrime(int n) noexcept
    {
        if (n <= 2) return 2;
        if (n < kPrimeTableMax)
        {
            const auto& sieve = getPrimeSieve();
            while (n < kPrimeTableMax && !sieve[static_cast<size_t>(n)])
                ++n;
            if (n < kPrimeTableMax) return n;
            // fall through to trial division for values at/above the table
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

    // Small helper: atomic-store all preset params in one go. Keeps the big
    // preset switch readable and avoids repeating `.store(..., mo_relaxed)`.
    struct PresetValues {
        T size, decay, hdMult, bdMult, hxover, bxover;
        T diff, modDepth, modRate, earlyLvl, lateLvl, erToLate;
        bool toneLP, toneHP;
    };

    void commitPreset(const PresetValues& p) noexcept
    {
        size_.store(p.size, std::memory_order_relaxed);
        decayTime_.store(p.decay, std::memory_order_relaxed);
        highDecayMult_.store(p.hdMult, std::memory_order_relaxed);
        bassDecayMult_.store(p.bdMult, std::memory_order_relaxed);
        highCrossover_.store(p.hxover, std::memory_order_relaxed);
        bassCrossover_.store(p.bxover, std::memory_order_relaxed);
        diffusion_.store(p.diff, std::memory_order_relaxed);
        modDepth_.store(p.modDepth, std::memory_order_relaxed);
        modRate_.store(p.modRate, std::memory_order_relaxed);
        earlyLevel_.store(p.earlyLvl, std::memory_order_relaxed);
        lateLevel_.store(p.lateLvl, std::memory_order_relaxed);
        erToLateMs_.store(p.erToLate, std::memory_order_relaxed);
        toneLPActive_ = p.toneLP;
        toneHPActive_ = p.toneHP;
    }

    void applyPreset(Type type)
    {
        switch (type)
        {
            case Type::Room:
                commitPreset({T(0.22), T(0.5),  T(0.40), T(1.1),
                              T(5000), T(250),  T(0.72), T(0.14), T(1.2),
                              T(1),    T(0.8),  T(0),    false,   false});
                break;

            case Type::Hall:
                commitPreset({T(0.68), T(2.2),  T(0.32), T(1.3),
                              T(4500), T(200),  T(0.84), T(0.22), T(0.55),
                              T(0.7),  T(1),    T(15),   false,   false});
                break;

            case Type::Chamber:
                commitPreset({T(0.38), T(1.2),  T(0.38), T(1.1),
                              T(5000), T(250),  T(0.78), T(0.18), T(0.8),
                              T(0.9),  T(0.9),  T(8),    false,   false});
                break;

            case Type::Plate:
                commitPreset({T(0.14), T(1.5),  T(0.55), T(0.8),
                              T(7000), T(150),  T(0.94), T(0.32), T(1.4),
                              T(0),    T(1),    T(0),    false,   false});
                numERTaps_ = 0;
                break;

            case Type::Spring:
                commitPreset({T(0.11), T(0.9),  T(0.28), T(1.0),
                              T(4000), T(200),  T(0.38), T(0.08), T(0.35),
                              T(0.5),  T(1),    T(0),    false,   false});
                break;

            case Type::Cathedral:
                commitPreset({T(0.98), T(5.0),  T(0.24), T(1.5),
                              T(3500), T(150),  T(0.91), T(0.18), T(0.35),
                              T(0.5),  T(1),    T(25),   false,   false});
                break;
        }

        // Sync damping_ from highDecayMult_
        T hd = highDecayMult_.load(std::memory_order_relaxed);
        T damp = std::clamp((T(1) - hd) / T(0.9), T(0), T(1));
        damping_.store(damp, std::memory_order_relaxed);

        updateDiffCoeffs();
        T md = modDepth_.load(std::memory_order_relaxed);
        modDepthA_.store(md * T(30), std::memory_order_relaxed);
        modDepthB_.store(md * T(15), std::memory_order_relaxed);

        if (spec_.sampleRate > 0)
        {
            updateDelayLengths(); // also calls updateDecayParams()
            updateModulation();

            T er = erToLateMs_.load(std::memory_order_relaxed);
            erToLateSamples_.store(static_cast<int>(
                static_cast<T>(spec_.sampleRate) * er / T(1000)),
                std::memory_order_relaxed);

            T rate = modRate_.load(std::memory_order_relaxed);
            for (int i = 0; i < kFDNSize; ++i)
            {
                modLFOA_[i].prepare(spec_.sampleRate,
                    rate * (T(0.7) + T(0.05) * static_cast<T>(i)),
                    static_cast<uint32_t>(i * 7919 + 1));
                modLFOB_[i].prepare(spec_.sampleRate,
                    rate * (T(1.8) + T(0.11) * static_cast<T>(i)),
                    static_cast<uint32_t>(i * 6271 + 31337));
            }

            generateERTapsForType(type);
        }
    }
};

} // namespace dspark
