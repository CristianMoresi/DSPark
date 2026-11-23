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
 *   +--> [Early reflections: 64 velvet-noise reflections per input, density
 *   |     rising with time; the first few discrete (raw input), the rest
 *   |     from the diffused feed; each reaches both outputs with the time
 *   |     and level differences of its direction (ITD, head shadow);
 *   |     grouped air/wall absorption]
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
 *   |      +- slowly turning Givens rotations of line pairs (lossless)
 *   |      +- short in-loop allpass (echo density), scaled with the size
 *   |      +- safety soft limit, write back + injection
 *   |       |
 *   |       v
 *   |    [Output: orthogonal L/R sign sums of the taps -> 2 allpass
 *   |     diffusers per side -> coherent low band -> width]
 *   v
 * [Early + late] -> [20 Hz high-pass] -> [Tone EQ] -> DryWetMixer -> Output
 * ```
 * Type::Spring replaces the FDN with twelve spring tanks (see runSprings()).
 *
 * Design notes:
 * - **True stereo, like a listener's ears.** Each input channel is a
 *   source with its own pre-delay, diffusion and early reflections. Every
 *   reflection arrives from a direction and reaches both outputs as it
 *   reaches both ears: the far side later (Woodworth ITD, up to 0.66 ms),
 *   quieter and, beyond 17 degrees, duller (head shadow). The floor and
 *   ceiling reflections share the source's direction (30 degrees for a
 *   hard-panned input), the next ones come mostly from its side of the
 *   room, the later ones from everywhere. Measured against dummy-head
 *   responses of the Detmold Konzerthaus (35 seats, sources across the
 *   stage, direct sound removed), the early field matches the hall's:
 *   interaural cross-correlation 0.25-0.35 (hall 0.22-0.42), and for a
 *   source on the left the left side leads by 0.27 ms (hall 0.15-0.40)
 *   and is 2-3 dB louder over 80 ms (hall 0.7-4.9 dB), while the late
 *   tail is centred.
 * - **Realistic early field.** The early reflections are a velvet-noise
 *   sequence (Valimaki et al.): sparse +-1 taps whose density rises with
 *   time, like a room's echo density, spectrally flat and free of the
 *   periodic ring of chained allpasses. setDiffusion() moves the split
 *   between the first discrete reflections (crisp, localisable) and the
 *   smooth diffuse ones.
 * - **One continuous decay.** The early field follows the room's own
 *   decay: its energy density falls at the mid-band T60, its highs at the
 *   HF T60 (a high shelf per absorption group), and its level is derived
 *   from the late field's energy flow (exp(-t / tau) over the mean round
 *   trip), not set apart. The two therefore join on one exponential, and
 *   the early decay time matches the T30 (0.94-1.02 over the presets at
 *   0.5-6 s of decay; measured rooms 0.84-1.03) without per-preset
 *   tuning.
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
 *   early decay time is 0.94-1.02 of the T30, rooms 0.84-1.03). This rests
 *   on five choices:
 *   - 32 lines with about 3.7 s of delay at full size: the modal density
 *     (modes per Hz equals the total delay in seconds) keeps the modes
 *     overlapping, and the Hadamard matrix plus a line rotation feeds
 *     every line from every other with equal weight (a slow pairwise
 *     rotation keeps that mix moving, see Transparent modulation);
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
 * - **Measured interaural coherence.** A diffuse field is coherent between
 *   the ears at low frequencies and incoherent above about 500 Hz. The late
 *   tail's side signal is high-passed (2nd order, 355 Hz, Q 0.74, fitted
 *   to the Konzerthaus seats), so its L/R coherence is 1.0 at 63 Hz, 0.95
 *   at 125 Hz, 0.55-0.7 at 250 Hz and about 0.1 from 500 Hz up (hall:
 *   0.97, 0.92, 0.54-0.79, 0.07-0.24): no phasey low end, full width on
 *   top. Below 20 Hz, where no room rings, a high-pass removes what is
 *   left.
 * - **Transparent modulation.** Two mechanisms share the work of smearing
 *   the tail's remaining resonances. Each line's length wanders with its
 *   own smooth random LFO, updated at control rate with a
 *   per-sample linear ramp and read through a first-order allpass
 *   interpolator, so there is no cumulative HF loss. Its fractional delay
 *   covers [0.1, 1.1): where the read index steps, the two configurations
 *   that meet there have nearly the same response (they would be identical
 *   at an offset of 0), so the step leaves little noise. With the usual
 *   [0.5, 1.5) range each step changed the high-frequency phase delay and
 *   left a noise floor 15-20 dB higher. Depth is set in
 *   time and scales with the room size, so small rooms do not warble. A
 *   wandering delay spreads a tone in proportion to its frequency, so the
 *   line wander is kept light and the mixing matrix itself moves too:
 *   after the Hadamard mix, each pair of lines (i, i XOR 4) turns through a
 *   slow Givens rotation (0.3-0.6 Hz, about 1 rad in the Hall; Schlecht and
 *   Habets' time-varying feedback matrix). The matrix stays orthogonal, so
 *   the loop stays lossless and every T60 exact, and modes trade energy
 *   with a spread that is the same at every frequency. The pairs are
 *   chosen so both lines weigh the same in the stereo output: the
 *   rotation cannot tilt the late field's interaural coherence. A steady
 *   tone keeps all but this fraction of its energy within +/-3 Hz (1 kHz /
 *   3 kHz): Room -51 / -50 dB, Chamber -69 / -48 dB, Hall -55 / -44 dB,
 *   Cathedral -54 / -45 dB (line wander alone: Hall -33 dB, Cathedral
 *   -31 dB at 3 kHz); Plate, lusher by design, -29 / -11 dB.
 * - **Smooth size changes.** setSize() opens a second read head at the new
 *   line and in-loop allpass lengths and cross-fades to it at constant
 *   power over a quarter of a second, so the size can be automated without
 *   clicks and without the pitch glide of lengths that slide (changes under
 *   0.2 % of a line's length still slide).
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
 *   cache. A stereo hall costs about 2.5% of one core at 48 kHz.
 * - **Eco quality** (setQuality): 16 lines, 24 early reflections per input and one
 *   spring per side, with the same decay calibration and loudness (within
 *   0.5 dB), at about 55% of the CPU.
 *
 * Around the tank, all off by default (each one is described at its setter):
 * a parametric equalizer of the decay time (setDecayBand()), tail hold
 * (setTailHold()), a dip under the dry signal and a relative gate driven by the dry signal
 * (setDryDip(), setGate()), source depth (setSourceDepth()), tank leveling
 * (setTankLeveling()), pre-delay and gate hold at a tempo (setTempo(),
 * setPreDelayNote(), setGateHoldNote()), input width and a width law that
 * keeps the power (setInputWidth(), setWidthLaw()), a parametric equalizer of
 * the wet signal (setPostEqBand()), modulation by random segments
 * (setModulationStyle()), line lengths in a geometric series
 * (setSpaceLayout()) and an octave-up shimmer (setShimmer()).
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
 * - Griesinger: random modulation of reverberators
 * - Valimaki et al. (2012, IEEE): "50 Years of Artificial Reverberation"
 * - Abel & Huang (2006): normalized echo density
 *
 * Dependencies: DryWetMixer.h, DspMath.h, Biquad.h, BiquadCascade.h,
 *               FilterDesign.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h,
 *               StateBlob.h, SimdOps.h, KeyEqualizer.h, TailShaper.h.
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
#include "../Core/BiquadCascade.h"
#include "../Core/FilterDesign.h"
#include "../Core/StateBlob.h"
#include "../Core/SimdOps.h"
#include "../Core/KeyEqualizer.h"
#include "../Core/TailShaper.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

    /** @brief Shape of a band of the decay equalizer (setDecayBand()). */
    enum class DecayShape { Off, Bell, LowShelf, HighShelf };

    /** @brief One band of the decay equalizer. */
    struct DecayBand
    {
        DecayShape shape = DecayShape::Off;
        T frequency = T(1000);   ///< Centre or corner, 20 Hz to 20 kHz.
        T q = T(0.7071);         ///< Width of a bell, 0.1 to 10 (the shelves ignore it).
        T multiplier = T(1);     ///< Decay time in the band against the rest, 0.1 to 4.
    };

    /** @brief A note value for the tempo-synced times. */
    enum class NoteValue { Off, Whole, Half, Quarter, Eighth, Sixteenth, ThirtySecond };

    /** @brief The feel of a note value. */
    enum class NoteFeel { Straight, Dotted, Triplet };

    /** @brief How the output width acts (setWidthLaw()). */
    enum class WidthLaw { Side, PowerKept };

    /** @brief How the lines' lengths wander (setModulationStyle()). */
    enum class ModulationStyle { Smooth, Segments };

    /** @brief How the line lengths are laid out (setSpaceLayout()). */
    enum class SpaceLayout { Prime, Geometric };

    static constexpr int kDecayBands = 6;   ///< Bands of the decay equalizer.
    static constexpr int kPostBands = KeyEqualizer<T, 2>::kBands;   ///< Bands of the wet equalizer.

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
        // A stereo buffer is reverberated in true stereo even on a mono spec,
        // so the mixer keeps a dry copy of both channels it blends.
        AudioSpec mixSpec = spec;
        mixSpec.numChannels = std::max(spec.numChannels, 2);
        mixer_.prepare(mixSpec);
        const double sr = spec.sampleRate;
        tail_.prepare(sr);
        refreshGateHold();
        postEq_.prepare(sr);
        wall_.prepare(sr);
        {
            auto& p = plate_;
            p.scale = sr / PlateTank::kRate;
            for (int b = 0; b < 2; ++b)
            {
                const auto s = static_cast<std::size_t>(b);
                p.lAp1[s] = p.at(PlateTank::kAp1[s]);
                p.lD1[s] = p.at(PlateTank::kD1[s]);
                p.lAp2[s] = p.at(PlateTank::kAp2[s]);
                p.lD2[s] = p.at(PlateTank::kD2[s]);
                p.ap1[s].prepare(p.lAp1[s] + p.at(PlateTank::kExcursion) + 4);
                p.d1[s].prepare(p.lD1[s] + 1);
                p.ap2[s].prepare(p.lAp2[s] + 1);
                p.d2[s].prepare(p.lD2[s] + 1);
            }
            p.loopSec = 0.5 * (p.lAp1[0] + p.lD1[0] + p.lAp2[0] + p.lD2[0] + p.lAp1[1] + p.lD1[1] + p.lAp2[1] + p.lD2[1]) / sr;
        }
        for (int j = 0; j < SecondSlope::kLines; ++j)
        {
            const auto s = static_cast<std::size_t>(j);
            slope2_.len[s] = nearestPrime(static_cast<int>(std::lround(kSecondMs[s] * 1e-3 * sr)));
            slope2_.line[s].prepare(slope2_.len[s] + 1);
        }
        {
            int size = 1;
            while (size < static_cast<int>(2.0 * kShimmerMaxMs * 0.001 * sr) + 8) size <<= 1;
            shimBuf_.assign(static_cast<size_t>(size), T(0));
            shimMask_ = size - 1;
        }
        levelRelease_ = std::exp(-1.0 / (0.5 * sr));
        levelHoldSamples_ = static_cast<int>(std::lround(0.03 * sr));

        // Re-derive the sample counts stored at set-time: after a re-prepare
        // at a different rate they would keep the OLD rate's sample count.
        preDelaySamples_.store(msToSamples(static_cast<T>(effectivePreDelayMs())),
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

        subsonicFilter_.setCoeffsNow(BiquadCoeffs::makeHighPass(sr, kSubsonicHz));
        coherenceFilter_.setCoeffsNow(BiquadCoeffs::makeHighPass(
            sr, std::min(kCoherenceHz, 0.45 * sr), kCoherenceQ));
        shadowCoeff_ = static_cast<T>(1.0 - std::exp(-6.283185307179586 * kShadowHz / sr));
        erGlidePerSample_ = static_cast<T>(1.0 / (0.02 * sr));
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
     * channels beyond two are left untouched. A block longer than the
     * prepared maxBlockSize is processed as consecutive blocks of at most
     * that size, with the same result.
     *
     * @param buffer View of the audio buffers (mono or stereo).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), 2);
        const int nS  = buffer.getNumSamples();
        if (nCh == 0 || nS == 0 || !prepared_) return;

        // The dry copy holds maxBlockSize samples, so a longer block would
        // lose its dry signal past that point: split it.
        if (nS > spec_.maxBlockSize)
        {
            for (int start = 0; start < nS; start += spec_.maxBlockSize)
                processBlock(buffer.getSubView(start, std::min(spec_.maxBlockSize, nS - start)));
            return;
        }

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

        // The wet signal's equalizer.
        if (postEq_.sync())
        {
            // A frame at a time, advanced by samples: a band on the middle
            // or on the sides takes both channels at once, and the
            // coefficients of a band on the move do not depend on the size
            // of the host's block.
            for (int i = 0; i < nS; ++i)
            {
                T frame[2] = {};
                for (int ch = 0; ch < nCh; ++ch) frame[ch] = buffer.getChannel(ch)[i];
                postEq_.processFrame(frame, nCh);
                for (int ch = 0; ch < nCh; ++ch) buffer.getChannel(ch)[i] = frame[ch];
                postEq_.advance(1);
            }
        }

        // The band limit, on the whole wet signal.
        wall_.retarget(static_cast<double>(bandLimit_.load(std::memory_order_relaxed)));
        if (wall_.active())
        {
            T* l = buffer.getChannel(0);
            T* r = nCh >= 2 ? buffer.getChannel(1) : nullptr;
            for (int i = 0; i < nS; ++i)
            {
                T a = l[i], b = r ? r[i] : l[i];
                wall_.frame(a, b);
                l[i] = a;
                if (r) r[i] = b;
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
        if (postEq_.sync())
        {
            T frame[2] = { outL, outR };
            postEq_.processFrame(frame, 2);
            outL = frame[0];
            outR = frame[1];
            postEq_.advance(1);
        }
        wall_.retarget(static_cast<double>(bandLimit_.load(std::memory_order_relaxed)));
        if (wall_.active()) wall_.frame(outL, outR);
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
            erShadowLP_[c].fill(T(0));
            erGain_[c] = erGainTarget_[c];
            erShelfGain_ = erShelfGainTarget_;
        }
        lines_.clear();
        loopAP_.clear();
        subsonicFilter_.reset();
        coherenceFilter_.reset();
        std::fill(springHist_.begin(), springHist_.end(), T(0));
        springW_ = 0;
        for (auto& st : springLPState_) st.fill(T(0));
        apY1_.fill(T(0));
        jotX1_.fill(T(0));
        jotY1_.fill(T(0));
        bassX1_.fill(T(0));
        bassY1_.fill(T(0));
        for (auto& z : dqZ1_) z.fill(T(0));
        for (auto& z : dqZ2_) z.fill(T(0));
        std::fill(shimBuf_.begin(), shimBuf_.end(), T(0));
        shimW_ = 0;
        shimPhase_ = { 0.0, 0.5 };
        shimLen_ = { 0.045, 0.045 };
        shimRng_ = 0x51ED27u;
        lastLateMono_ = T(0);
        levelGr_ = 0.0;
        levelHold_ = 0;
        distLp_ = { T(0), T(0) };
        distMix_ = sourceDepth_.load(std::memory_order_relaxed) > T(0) ? T(1) : T(0);
        tail_.reset();
        postEq_.reset();
        wall_.reset();
        for (auto& l : slope2_.line) l.clear();
        for (int b = 0; b < 2; ++b)
        {
            plate_.ap1[static_cast<std::size_t>(b)].clear();
            plate_.d1[static_cast<std::size_t>(b)].clear();
            plate_.ap2[static_cast<std::size_t>(b)].clear();
            plate_.d2[static_cast<std::size_t>(b)].clear();
        }
        plate_.lp.fill(T(0));
        plate_.phase = 0.0;
        slope2_.x1.fill(T(0));
        slope2_.y1.fill(T(0));
        slope2_.gainNow = slope2_.gain;
        slope2_.keepNow = slope2_.keep;
        holdNow_ = tailHold_.load(std::memory_order_relaxed) ? T(1) : T(0);
        quietSamples_ = 1e300;
        snapAbsorption();
        applyHoldCoeffs();

        const double sr = spec_.sampleRate > 0 ? spec_.sampleRate : 48000.0;
        const T rate = modRate_.load(std::memory_order_relaxed);
        for (int i = 0; i < kMaxLines; ++i)
        {
            lfo_[i].prepare(sr, rate * lfoRateFactor(i), lfoSeed(i));
            if (i < kMaxLines / 2)
            {
                rotLfo_[i].prepare(sr, rotRate() * rotRateFactor(i), rotSeed(i));
                rotC_[i] = T(1);
                rotS_[i] = T(0);
            }
            seg_[i].prepare(lfoSeed(i) ^ 0x9E3779B9u);
            lenCur_[i] = static_cast<T>(lenTarget_[i]);
            loopAPCur_[i] = static_cast<T>(loopAPLen_[i]);
            pos_[i] = static_cast<double>(std::max(lenCur_[i], T(kMinReadPos)));
            posInc_[i] = 0.0;
        }
        ctrlPhase_ = 0;
        rotTick_ = 0;   // the rotation refresh phase restarts with the LFOs
        loopAPGliding_ = false;
        sizeFade_ = false;

        // Parameter glides snap to their targets with the state.
        snapGlides();
        fresh_ = true;

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
     * when those values should override the selected preset. The preset's
     * parameter values are written here, on the calling thread, so a setter
     * called afterwards always overrides them. The engine change is applied
     * at the start of the next processBlock(); the running tail is cleared.
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
        commitPreset(presetValues(static_cast<Type>(t)));
        presetDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Selects the engine quality / CPU cost trade-off.
     *
     * Quality::Full (default) runs the complete engine. Quality::Eco runs
     * at about 55% of its CPU for embedded and other constrained targets,
     * with the same controls, decay calibration and loudness:
     *
     * - 16 FDN lines instead of 32 (every other base delay, so the lines
     *   still span the full range; per-line decay gains keep every T60
     *   exact)
     * - 24 early reflections per input instead of 64
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

    /**
     * @brief Mid-frequency reverberation time in seconds (0.1 - 30).
     *
     * The ISO 3382-1 T_mid: the mean of the 500 Hz and 1 kHz octave-band T60s,
     * the figure a hall is quoted by. The bass and high bands decay at the
     * multiples setBassDecayMultiplier() and setHighDecayMultiplier() set,
     * relative to the same calibration. Measured over the six types at their
     * default damping, T_mid lands within 3% of this value.
     */
    void setDecay(T seconds) noexcept
    {
        if (!std::isfinite(seconds)) return;
        decayTime_.store(std::clamp(seconds, T(0.1), T(30)),
                         std::memory_order_relaxed);
        markUserParam(kUserDecay);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Dry/wet mix (0 = dry, 1 = wet). The wet signal is calibrated to
     *        the dry one's level: on pink noise at full mix it sits within
     *        about a decibel of it at every type's preset, and within
     *        -1 to +3.5 dB as the decay goes from 12 s down to 0.3 s and the
     *        size changes (its energy grows with the decay; a gain from the
     *        decay and the mean round trip takes that back).
     */
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
     *        of their base lengths. A change cross-fades two read heads, the
     *        old lengths to the new, over a quarter of a second (no click and
     *        no glide of pitch; the next change waits for the fade), so the
     *        size can be automated.
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

    /** @brief Pre-delay before the early reflections, in ms (0 - 500); a
     *  note value set by setPreDelayNote() takes its place. */
    void setPreDelay(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        T clamped = std::clamp(ms, T(0), T(kMaxPreDelayMs));
        preDelayMs_.store(clamped, std::memory_order_relaxed);
        refreshPreDelay();
    }

    /**
     * @brief Diffusion (0 - 1): the strength of the input, in-loop and
     *        output allpass diffusers, and how many of the first early
     *        reflections stay discrete (7 at 0, 1 at 1). High values give a
     *        smooth, dense onset; low values keep crisp, localisable early
     *        echoes. For Type::Spring it sets the strength of the chirp.
     *        The allpass coefficients glide over about 20 ms.
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
     * early reflections are added. Changes glide over about 20 ms.
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

    /**
     * @brief Early reflections level in dB (-60 to +6). 0 dB is the
     *        physical balance: the early field then carries the late
     *        field's energy density and decay, so the two join on one
     *        exponential at any size and decay time. Changes glide over
     *        about 20 ms.
     */
    void setEarlyLevel(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        earlyLevel_.store(decibelsToGain(std::clamp(dB, T(-60), T(6))), std::memory_order_relaxed);
        markUserParam(kUserEarly);
    }

    /** @brief Late tail level in dB (-60 to +6). Changes glide over about 20 ms. */
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
     *
     * Cutoff changes glide over about 20 ms, and switching the filter on or
     * off crossfades, so it can be automated.
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
     *
     * Cutoff changes glide over about 20 ms, and switching the filter on or
     * off crossfades, so it can be automated.
     * @param hz Cutoff in Hz (0 = off, 2000-16000 typical).
     */
    void setToneHighCut(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        toneHighCutHz_.store(hz, std::memory_order_relaxed);
        toneDirty_.store(true, std::memory_order_release);
    }

    // =========================================================================
    // Level 3: Around the tank (all off by default)
    // =========================================================================

    /**
     * @brief One band of the decay equalizer: the decay time in a band of
     *        frequencies, as a multiple of what the other controls give there.
     *
     * Each line carries the band as a filter in its loop, whose gain is
     * -60 x (round trip / T60) x (1 / multiplier - 1) dB at the band (T60 is
     * the decay there without the band): a bell at its centre, a low shelf at
     * DC, a high shelf at Nyquist. Where bands that lengthen the decay would
     * lift a loop's gain over 0.9995 at some frequency, their boosts are
     * scaled down together until it does not. A multiplier of 1 or the shape
     * Off takes the band out of the loop.
     *
     * @param index Band, 0 to kDecayBands - 1 (others are ignored).
     * @param band  The settings (non-finite fields are ignored).
     */
    void setDecayBand(int index, const DecayBand& band) noexcept
    {
        if (index < 0 || index >= kDecayBands) return;
        auto& d = decayBands_[static_cast<size_t>(index)];
        d.shape.store(std::clamp(static_cast<int>(band.shape), 0, 3), std::memory_order_relaxed);
        if (std::isfinite(band.frequency)) d.frequency.store(std::clamp(band.frequency, T(20), T(20000)), std::memory_order_relaxed);
        if (std::isfinite(band.q)) d.q.store(std::clamp(band.q, T(0.1), T(10)), std::memory_order_relaxed);
        if (std::isfinite(band.multiplier)) d.multiplier.store(std::clamp(band.multiplier, T(0.1), T(4)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @return The settings of a band of the decay equalizer (an Off band out of range). */
    [[nodiscard]] DecayBand getDecayBand(int index) const noexcept
    {
        DecayBand b;
        if (index < 0 || index >= kDecayBands) return b;
        const auto& d = decayBands_[static_cast<size_t>(index)];
        b.shape = static_cast<DecayShape>(d.shape.load(std::memory_order_relaxed));
        b.frequency = d.frequency.load(std::memory_order_relaxed);
        b.q = d.q.load(std::memory_order_relaxed);
        b.multiplier = d.multiplier.load(std::memory_order_relaxed);
        return b;
    }

    /**
     * @brief Holds the tail: the input to the tank and the early field fade
     *        out over about 20 ms while every loop's absorption and decay band
     *        fade to unity, so the tail holds with its highs. Off fades back.
     */
    void setTailHold(bool on) noexcept { tailHold_.store(on, std::memory_order_relaxed); }
    /** @return Whether the tail is held (as set). */
    [[nodiscard]] bool isTailHeld() const noexcept { return tailHold_.load(std::memory_order_relaxed); }

    /**
     * @brief Dips the wet signal under the dry one while it plays (see TailShaper):
     *        by up to `dB` (0 to 48; 0, the default, is off), in proportion to
     *        the dry level against its own recent peak.
     */
    void setDryDip(T dB) noexcept { tail_.setDryDip(dB); }

    /**
     * @brief Gate on the wet signal (see TailShaper): it closes once the dry
     *        signal has stayed `thresholdDb` under its recent peak (-80 to 0;
     *        below -80, the default, the gate is off) for the hold time.
     */
    void setGate(T thresholdDb) noexcept { tail_.setGate(thresholdDb); }

    /** @brief Hold of the gate, 0 to 4000 ms (default 100); a note value set by setGateHoldNote() takes its place. */
    void setGateHold(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        gateHoldMs_.store(std::clamp(ms, T(0), T(4000)), std::memory_order_relaxed);
        refreshGateHold();
    }

    /** @brief Release of the gate, 5 to 4000 ms (default 150). */
    void setGateRelease(T ms) noexcept { tail_.setGateRelease(ms); }

    /**
     * @brief Depth of the source in the room, 0 to 1 (0, the default, is off): one
     *        control that moves the balance from the early reflections (down
     *        to half) to the tail (up by half), shortens the pre-delay (to a
     *        fifth at 1), adds input diffusion (with its square, up to 0.25)
     *        and darkens the wet signal with a one-pole low-pass from 18 kHz
     *        down (to 2.25 kHz at 1, further with a darker HF decay).
     *        Non-finite values are ignored.
     */
    void setSourceDepth(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        sourceDepth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Leveling of the tank, 0 to 1 (0, the default, is off): a compressor on the
     *        tank's input (ratio 1 + 3 x amount over -30 dBFS with a 24 dB
     *        knee), its gain compensated at -18 dBFS: quiet passages feed the
     *        tail as much as loud ones. It reads the peak of each sample (as
     *        the power of a sine with that peak); its reduction deepens at
     *        once, holds 30 ms and comes back with a time constant of half a
     *        second, so it neither moves within a cycle of a low note nor
     *        lets the start of a hit through. Non-finite values are ignored.
     */
    void setTankLeveling(T amount) noexcept
    {
        if (std::isfinite(amount)) tankLeveling_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Soft saturation after the leveling compressor (off by default). */
    void setTankSaturation(bool on) noexcept { tankSat_.store(on, std::memory_order_relaxed); }

    /** @brief Tempo for the note values, 20 to 400 BPM (default 120). */
    void setTempo(T bpm) noexcept
    {
        if (!std::isfinite(bpm)) return;
        tempo_.store(std::clamp(bpm, T(20), T(400)), std::memory_order_relaxed);
        refreshPreDelay();
        refreshGateHold();
    }

    /** @brief Pre-delay as a note value at the tempo (Off, the default, uses setPreDelay()); up to 500 ms. */
    void setPreDelayNote(NoteValue note, NoteFeel feel = NoteFeel::Straight) noexcept
    {
        preNote_.store(std::clamp(static_cast<int>(note), 0, 6), std::memory_order_relaxed);
        preFeel_.store(std::clamp(static_cast<int>(feel), 0, 2), std::memory_order_relaxed);
        refreshPreDelay();
    }

    /** @brief Gate hold as a note value at the tempo (Off, the default, uses setGateHold()); up to 4 s. */
    void setGateHoldNote(NoteValue note, NoteFeel feel = NoteFeel::Straight) noexcept
    {
        gateNote_.store(std::clamp(static_cast<int>(note), 0, 6), std::memory_order_relaxed);
        gateFeel_.store(std::clamp(static_cast<int>(feel), 0, 2), std::memory_order_relaxed);
        refreshGateHold();
    }

    /**
     * @brief Width of the input, 0 to 1 (1, the default, is the input as it
     *        is): the side of the input is scaled before the tank, so a wide
     *        source excites the room from nearer the centre.
     */
    void setInputWidth(T width) noexcept
    {
        if (std::isfinite(width)) inputWidth_.store(std::clamp(width, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief How setWidth() acts on the tail: Side (the default) scales the
     *        side; PowerKept also raises the middle as the side falls
     *        (sqrt(2 - width^2)), so the tail's power stays the same from 0 to
     *        1.41.
     */
    void setWidthLaw(WidthLaw law) noexcept
    {
        widthLaw_.store(std::clamp(static_cast<int>(law), 0, 1), std::memory_order_relaxed);
    }

    /**
     * @brief One band of the wet signal's equalizer (KeyEqualizer bands).
     *
     * A band that keeps its form glides to a new setting over 20 ms, designed
     * again on the way; a change of shape crosses to a second bank of
     * filters over the same time. A band can sit on both channels of the wet
     * pair, on its middle or on its sides (Band::placement).
     */
    void setPostEqBand(int index, const typename KeyEqualizer<T, 2>::Band& band) noexcept { postEq_.setBand(index, band); }

    /**
     * @brief Keeps the loudness of the wet signal through its equalizer (off
     *        by default; see KeyEqualizer::setLoudnessCompensation()). State
     *        key `postloudness`.
     */
    void setPostEqLoudness(bool on) noexcept { postEq_.setLoudnessCompensation(on); }

    /** @return Whether the wet equalizer keeps the loudness. */
    [[nodiscard]] bool getPostEqLoudness() const noexcept { return postEq_.getLoudnessCompensation(); }
    /** @return The settings of a band of the wet signal's equalizer. */
    [[nodiscard]] typename KeyEqualizer<T, 2>::Band getPostEqBand(int index) const noexcept { return postEq_.getBand(index); }

    /**
     * @brief How the line lengths wander: Smooth (the default) is the smooth
     *        random modulation of every line; Segments moves every other line
     *        in segments that alternate sides, toward random positions at
     *        random speeds (each segment 0.5 to 1.5 cycles of the line's
     *        rate), rounded by a critically damped follower so the pitch has
     *        no corners, with the depth held under 2 % of the line's length.
     */
    void setModulationStyle(ModulationStyle style) noexcept
    {
        modStyle_.store(std::clamp(static_cast<int>(style), 0, 1), std::memory_order_relaxed);
    }

    /**
     * @brief How the line lengths are laid out: Prime (the default) takes the
     *        tuned base lengths; Geometric spaces them evenly in log over the
     *        same range (about 6.6 per octave, the lines of the two sides
     *        interleaved half a step apart), then to the nearest prime. The
     *        read heads cross-fade to the new layout as on a change of size.
     */
    void setSpaceLayout(SpaceLayout layout) noexcept
    {
        layout_.store(std::clamp(static_cast<int>(layout), 0, 1), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief The plate's own tank (off by default; Type::Plate only): the
     *        figure-eight of Dattorro's plate instead of the delay network.
     *        Two branches feed each other, each an all-pass that the
     *        modulation moves, a delay, the damping and the decay, a second
     *        all-pass and a delay; each side of the output is read at seven
     *        points inside the other branch and its own. Decay, high decay
     *        and its crossover, modulation and its rate follow the controls;
     *        a plate has one size, so Size does not change it, and Bass Decay
     *        does not reach it. The wet level is that of the network within
     *        0.25 dB; the decay is the one asked within 5 % from 2 s up (a
     *        branch takes a third of a second, so at 1.5 s it is 12 % short).
     */
    void setPlateTank(bool on) noexcept
    {
        plateTank_.store(on, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @return Whether a Plate runs its own tank. */
    [[nodiscard]] bool isPlateTank() const noexcept { return plateTank_.load(std::memory_order_relaxed); }

    /**
     * @brief A second slope, 0 to 1 (0, the default, is off): beside the
     *        tank, four lines of its own decay in 0.3 of its times, so the
     *        first decibels of the tail fall faster than the rest, as in a
     *        hall coupled to a smaller space. At 1 the early decay time falls
     *        to 0.87 (Hall) to 0.90 (Room) of what it was, the T30 moves under
     *        3 % and the energy of the tail under 0.2 dB. Not for Spring. Non-finite values are
     *        ignored.
     */
    void setDoubleSlope(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        doubleSlope_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @return The second slope, 0 to 1. */
    [[nodiscard]] T getDoubleSlope() const noexcept { return doubleSlope_.load(std::memory_order_relaxed); }

    /**
     * @brief Limits the band of the wet signal, as early digital reverbs
     *        did: an elliptic wall (order 12, 0.05 dB of ripple, 100 dB down
     *        a fifth of an octave above the corner) on the output, outside
     *        the tank, so the decay does not change. 0, the default, is off;
     *        otherwise 1 kHz up to 0.4 of the rate. A move fades over 20 ms
     *        (one during a fade waits for it). Non-finite values are ignored.
     */
    void setBandLimit(T hz) noexcept
    {
        if (std::isfinite(hz)) bandLimit_.store(hz <= T(0) ? T(0) : std::max(hz, T(1000)), std::memory_order_relaxed);
    }

    /** @return The corner of the band limit in hertz (0 when off). */
    [[nodiscard]] T getBandLimit() const noexcept { return bandLimit_.load(std::memory_order_relaxed); }

    /**
     * @brief Octave-up shimmer, 0 to 1 (0, the default, is off): the tail is
     *        pitched up an octave by two read heads with windows of random
     *        length (30 to 60 ms) and a raised-cosine crossfade, and fed back
     *        into the tank. Non-finite values are ignored.
     */
    void setShimmer(T amount) noexcept
    {
        if (std::isfinite(amount)) shimmer_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getDryDip() const noexcept { return tail_.getDryDip(); }
    [[nodiscard]] T getGate() const noexcept { return tail_.getGate(); }
    [[nodiscard]] T getGateHold() const noexcept { return gateHoldMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getGateRelease() const noexcept { return tail_.getGateRelease(); }
    [[nodiscard]] T getSourceDepth() const noexcept { return sourceDepth_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTankLeveling() const noexcept { return tankLeveling_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isThicknessSaturated() const noexcept { return tankSat_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTempo() const noexcept { return tempo_.load(std::memory_order_relaxed); }
    [[nodiscard]] NoteValue getPreDelayNote() const noexcept { return static_cast<NoteValue>(preNote_.load(std::memory_order_relaxed)); }
    [[nodiscard]] NoteValue getGateHoldNote() const noexcept { return static_cast<NoteValue>(gateNote_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getInputWidth() const noexcept { return inputWidth_.load(std::memory_order_relaxed); }
    [[nodiscard]] WidthLaw getWidthLaw() const noexcept { return static_cast<WidthLaw>(widthLaw_.load(std::memory_order_relaxed)); }
    [[nodiscard]] ModulationStyle getModulationStyle() const noexcept { return static_cast<ModulationStyle>(modStyle_.load(std::memory_order_relaxed)); }
    [[nodiscard]] SpaceLayout getSpaceLayout() const noexcept { return static_cast<SpaceLayout>(layout_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getShimmer() const noexcept { return shimmer_.load(std::memory_order_relaxed); }

    /** @return The pre-delay in use, in ms (the note value's time when one is set). */
    [[nodiscard]] T getEffectivePreDelay() const noexcept { return static_cast<T>(effectivePreDelayMs()); }

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
        char key[24];
        for (int b = 0; b < kDecayBands; ++b)
        {
            const DecayBand band = getDecayBand(b);
            std::snprintf(key, sizeof(key), "db%d.shape", b); w.write(key, static_cast<int32_t>(band.shape));
            std::snprintf(key, sizeof(key), "db%d.freq", b);  w.write(key, static_cast<float>(band.frequency));
            std::snprintf(key, sizeof(key), "db%d.q", b);     w.write(key, static_cast<float>(band.q));
            std::snprintf(key, sizeof(key), "db%d.mult", b);  w.write(key, static_cast<float>(band.multiplier));
        }
        w.write("tailHold", isTailHeld());
        w.write("dryDip", static_cast<float>(getDryDip()));
        w.write("gate", static_cast<float>(getGate()));
        w.write("gateHold", static_cast<float>(getGateHold()));
        w.write("gateRelease", static_cast<float>(getGateRelease()));
        w.write("sourceDepth", static_cast<float>(getSourceDepth()));
        w.write("tankLeveling", static_cast<float>(getTankLeveling()));
        w.write("tankSat", isThicknessSaturated());
        w.write("tempo", static_cast<float>(getTempo()));
        w.write("preNote", preNote_.load(std::memory_order_relaxed));
        w.write("preFeel", preFeel_.load(std::memory_order_relaxed));
        w.write("gateNote", gateNote_.load(std::memory_order_relaxed));
        w.write("gateFeel", gateFeel_.load(std::memory_order_relaxed));
        w.write("inWidth", static_cast<float>(getInputWidth()));
        w.write("widthLaw", widthLaw_.load(std::memory_order_relaxed));
        w.write("modStyle", modStyle_.load(std::memory_order_relaxed));
        w.write("layout", layout_.load(std::memory_order_relaxed));
        w.write("shimmer", static_cast<float>(getShimmer()));
        w.write("bandLimit", static_cast<float>(getBandLimit()));
        w.write("doubleSlope", static_cast<float>(getDoubleSlope()));
        w.write("plateTank", isPlateTank());
        postEq_.writeState(w, "post");
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
        // A missing or non-positive cutoff restores the filter as off, like
        // the blob's source instance.
        setToneLowCut(static_cast<T>(r.read("toneLowCut", -1.0f)));
        setToneHighCut(static_cast<T>(r.read("toneHighCut", -1.0f)));
        char key[24];
        for (int b = 0; b < kDecayBands; ++b)
        {
            DecayBand band;
            std::snprintf(key, sizeof(key), "db%d.shape", b); band.shape = static_cast<DecayShape>(r.read(key, 0));
            std::snprintf(key, sizeof(key), "db%d.freq", b);  band.frequency = static_cast<T>(r.read(key, 1000.0f));
            std::snprintf(key, sizeof(key), "db%d.q", b);     band.q = static_cast<T>(r.read(key, 0.7071f));
            std::snprintf(key, sizeof(key), "db%d.mult", b);  band.multiplier = static_cast<T>(r.read(key, 1.0f));
            setDecayBand(b, band);
        }
        setTailHold(r.read("tailHold", false));
        setDryDip(static_cast<T>(r.read("dryDip", 0.0f)));
        setGate(static_cast<T>(r.read("gate", -200.0f)));
        setGateHold(static_cast<T>(r.read("gateHold", 100.0f)));
        setGateRelease(static_cast<T>(r.read("gateRelease", 150.0f)));
        setSourceDepth(static_cast<T>(r.read("sourceDepth", 0.0f)));
        setTankLeveling(static_cast<T>(r.read("tankLeveling", 0.0f)));
        setTankSaturation(r.read("tankSat", false));
        setTempo(static_cast<T>(r.read("tempo", 120.0f)));
        setPreDelayNote(static_cast<NoteValue>(r.read("preNote", 0)), static_cast<NoteFeel>(r.read("preFeel", 0)));
        setGateHoldNote(static_cast<NoteValue>(r.read("gateNote", 0)), static_cast<NoteFeel>(r.read("gateFeel", 0)));
        setInputWidth(static_cast<T>(r.read("inWidth", 1.0f)));
        setWidthLaw(static_cast<WidthLaw>(r.read("widthLaw", 0)));
        setModulationStyle(static_cast<ModulationStyle>(r.read("modStyle", 0)));
        setSpaceLayout(static_cast<SpaceLayout>(r.read("layout", 0)));
        setShimmer(static_cast<T>(r.read("shimmer", 0.0f)));
        setBandLimit(static_cast<T>(r.read("bandLimit", 0.0f)));
        setDoubleSlope(static_cast<T>(r.read("doubleSlope", 0.0f)));
        setPlateTank(r.read("plateTank", false));
        postEq_.readState(r, "post");
        return true;
    }

protected:
    // --- Constants -----------------------------------------------------------

    static constexpr int kMaxLines   = 32;
    static constexpr int kEcoLines   = 16;
    static constexpr int kInStages   = 4;   ///< input diffusers per channel feeding the late field (Full)
    static constexpr int kMaxERTaps  = 128; ///< early taps per output side: 64 reflections per input (Full)
    static constexpr int kEcoERTaps  = 48;
    static constexpr int kShadowBins = 2;   ///< head-shadow classes of the early taps (open, shadowed)
    static constexpr int kMaxDiscreteER = 7; ///< raw (discrete) early taps at diffusion 0
    /// Early-tap slots per output side: every reflection, plus a second slot
    /// for each of the first kMaxDiscreteER per input, so the raw and the
    /// diffused reading of those both exist and setDiffusion() crossfades
    /// between them instead of switching the source.
    static constexpr int kMaxERSlots = kMaxERTaps + 2 * kMaxDiscreteER;
    static constexpr int kERGroups   = 4;   ///< absorption groups (early -> late)
    static constexpr int kCtrl       = 16;  ///< modulation control period (samples)
    static constexpr int kChunk      = 64;  ///< sparse-FIR processing chunk (samples)

    static constexpr double kMaxPreDelayMs = 500.0;
    static constexpr double kShimmerMinMs = 30.0;    ///< Shortest window of the shimmer's heads.
    static constexpr double kShimmerMaxMs = 60.0;    ///< Longest window of the shimmer's heads.
    static constexpr T      kShimmerFeed = T(0.08);  ///< Share of the shimmer fed back at full amount.
    static constexpr double kMaxDecayLoopGain = 0.9995;   ///< Highest loop gain the decay bands may reach.
    static constexpr double kMaxErToLateMs = 200.0;
    static constexpr double kMaxErMs       = 170.0;
    static constexpr double kMaxInjectMs   = 8.0;
    static constexpr double kMaxInDiffMs   = 2.0;
    static constexpr double kMaxLoopApMs   = 5.5;
    static constexpr double kCoherenceHz   = 355.0; ///< side high-pass of the late field (2nd order, Q 0.74)
    static constexpr double kCoherenceQ    = 0.74;
    static constexpr double kShadowHz      = 1000.0; ///< head-shadow corner of the early taps
    static constexpr double kShadowHF      = 0.5;    ///< high-frequency gain of a shadowed (far-ear) tap
    static constexpr double kModMaxMs      = 2.0;   ///< peak line wander at modulation 1, size 1
    /// Time-varying mixing matrix: Givens rotations of line pairs, their
    /// angle scaled by the modulation amount (the Hall's 0.10 gives 1 rad)
    /// at a fixed slow rate - a rotation spreads every frequency alike, so
    /// it stays slow whatever the line-wander rate.
    static constexpr double kRotMaxRad     = 10.0;  ///< peak rotation angle at modulation 1 (rad)
    static constexpr double kRotRateHz     = 0.35;  ///< base rate of the pair rotations
    /// Rotation partner of line i is i XOR 4: both carry the same L*R sign
    /// in the output sums and the same input side, so turning them never
    /// tilts the stereo image (checked in the test suite).
    static constexpr int kRotPairStride = 4;
    static constexpr double kGlideMs       = 60.0;  ///< size-change glide time constant
    static constexpr double kMaxGlideSpeed = 0.04;  ///< max length change per sample (4% Doppler)
    static constexpr double kSizeFadeOver    = 0.002;  ///< a length change beyond this share cross-fades two read heads
    static constexpr double kSizeFadeSeconds = 0.25;   ///< the cross-fade of the two read heads
    static constexpr double kSubsonicHz    = 20.0;  ///< wet-output high-pass (2nd order): no room rings below it
    static constexpr T      kMinReadPos    = T(2);
    /// The modulated reads cover the fractional delay [kApLow, kApLow + 1).
    /// Where the read index steps, (M, kApLow + 1) hands over to
    /// (M + 1, kApLow). At kApLow = 0 those would be the same filter (a pure
    /// one-sample delay against an allpass with eta = 1, which cancels to
    /// unity), so near 0 the handover changes the phase delay very little at
    /// any frequency; with [0.5, 1.5) it changed by up to a sample near
    /// Nyquist and every step left high-frequency noise in the tail. At 0.1
    /// the pole stays at |eta| <= 0.82 (at most 10 samples of group delay,
    /// at Nyquist only).
    static constexpr double kApLow = 0.1;
    /// First-order allpass coefficient for a fractional delay d in
    /// [kApLow, kApLow + 1): (eta + z^-1) / (1 + eta z^-1) delays d at DC.
    static T allpassCoeff(double d) noexcept { return static_cast<T>((1.0 - d) / (1.0 + d)); }
    static constexpr T      kSoftLimit     = T(2);  ///< in-loop safety limiter threshold

    /// Wet output scale: calibrated so the steady-state level matches the
    /// previous engine's (and Full matches Eco within a fraction of a dB).
    static constexpr T kOutGain = T(0.25);
    /// The wet signal's level at full mix against the dry one (see
    /// updateDecayParams()): measured on pink noise, the types with a delay
    /// network land within 0.7 dB of the dry level at their presets, the
    /// spring with its own constant (its loops disperse).
    static constexpr double kWetCalibration = 4.73;
    static constexpr double kWetCalibrationSpring = 2.85;

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

    static constexpr double kMeanInjectMs = 4.0;   ///< mean of kInjectMs_
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
    /// The band limit: an elliptic wall on the wet pair, in double, in two
    /// banks faded over 20 ms when the corner moves.
    struct Wall
    {
        static constexpr int kSections = 6;
        using Cascade = BiquadCascade<T, kSections, 2>;

        void prepare(double sampleRate)
        {
            rate = sampleRate;
            step = 1.0 / (0.02 * sampleRate);
            for (auto& b : bank)
            {
                b.setNumSections(kSections);
                b.reset();
            }
            hz = 0.0;
            fade = 1.0;
            fromOn = toOn = false;
        }

        void reset() noexcept
        {
            for (auto& b : bank) b.reset();
            fade = 1.0;
            fromOn = toOn;
        }

        [[nodiscard]] bool active() const noexcept { return toOn || fade < 1.0; }

        /// The corner wanted now (0 = off); a fade under way finishes first.
        void retarget(double want) noexcept
        {
            if (want > 0.0) want = std::min(want, 0.4 * rate);
            if (want == hz || fade < 1.0) return;
            fromOn = toOn;
            live = 1 - live;
            toOn = want > 0.0;
            if (toOn)
            {
                BiquadCoeffs c[kSections];
                // By the bilinear transform with the corner mapped exactly: the
                // ripple of the pass band stays the prototype's (a fit of each
                // section moves it by half a decibel), the stop band only gets deeper.
                const AnalogCascade proto = FilterDesign::ellipticHighCut(want);
                const int n = std::min(proto.count, kSections);
                for (int s = 0; s < n; ++s) c[s] = FilterDesign::bilinear(proto.sections[static_cast<std::size_t>(s)], rate, want);
                for (int s = 0; s < kSections; ++s) bank[live].setCoeffs(s, s < n ? c[s] : BiquadCoeffs {});
                bank[live].reset();
            }
            hz = want;
            fade = 0.0;
        }

        void frame(T& l, T& r) noexcept
        {
            const double x[2] = { static_cast<double>(l), static_cast<double>(r) };
            double to[2] = { x[0], x[1] };
            if (toOn)
                for (int c = 0; c < 2; ++c) to[c] = bank[live].processSample(x[c], c);
            if (fade < 1.0)
            {
                double from[2] = { x[0], x[1] };
                if (fromOn)
                    for (int c = 0; c < 2; ++c) from[c] = bank[1 - live].processSample(x[c], c);
                fade = std::min(1.0, fade + step);
                for (int c = 0; c < 2; ++c) to[c] = from[c] + fade * (to[c] - from[c]);
            }
            l = static_cast<T>(to[0]);
            r = static_cast<T>(to[1]);
        }

        Cascade bank[2];
        double rate = 48000.0, hz = 0.0, fade = 1.0, step = 0.0;
        int live = 0;
        bool fromOn = false, toOn = false;
    };

    Wall wall_;
    std::atomic<T> bandLimit_ { T(0) };

    /// The second slope: four lines of their own beside the tank (0.3 of its
    /// decay times), each with
    /// the tank's kind of absorption (a shelf at the high crossover).
    struct SecondSlope
    {
        static constexpr int kLines = 4;
        std::array<Line, kLines> line;
        std::array<int, kLines> len {};
        std::array<T, kLines> b0 {}, b1 {}, a1 {}, x1 {}, y1 {};
        T gain = T(0), keep = T(1);          ///< its share of the output, what the tank keeps (targets)
        T gainNow = T(0), keepNow = T(1);    ///< where they are now (they glide)
        double loopSec = 0.0;                ///< mean loop length
        bool on = false;
    };
    /// The plate's own tank: Dattorro's figure-eight (lengths at 29761 Hz,
    /// scaled to the rate). Branch 0 is the left one of his figure.
    struct PlateTank
    {
        static constexpr double kRate = 29761.0;
        static constexpr std::array<int, 2> kAp1 { 672, 908 }, kD1 { 4453, 4217 }, kAp2 { 1800, 2656 }, kD2 { 3720, 3163 };
        static constexpr double kExcursion = 16.0;   ///< Of the modulated all-passes at full modulation.
        static constexpr T kDiffusion1 = T(-0.7), kDiffusion2 = T(0.5);
        std::array<Line, 2> ap1, d1, ap2, d2;
        std::array<int, 2> lAp1 {}, lD1 {}, lAp2 {}, lD2 {};
        std::array<T, 2> gain {}, lp {};
        T damp = T(0);
        double excursion = 0.0, phase = 0.0, phaseStep = 0.0;
        double loopSec = 0.0;          ///< Mean branch length.
        double scale = 1.0;            ///< The rate over Dattorro's.
        bool on = false;

        [[nodiscard]] int at(double samples29k) const noexcept { return std::max(1, static_cast<int>(std::lround(samples29k * scale))); }
    };
    PlateTank plate_;
    std::atomic<bool> plateTank_ { false };
    static constexpr double kPlateEnergy = 0.50;  ///< The plate's tail energy over the tail formula (measured, 1.5 to 8 s: within 0.25 dB).

    SecondSlope slope2_;
    std::atomic<T> doubleSlope_ { T(0) };
    static constexpr std::array<double, SecondSlope::kLines> kSecondMs { 23.3, 29.1, 37.9, 44.7 };
    static constexpr double kSecondRatio = 0.3;   ///< its decay times, as a share of the tank's
    static constexpr double kSecondShare = 0.30;  ///< its share of the tail's energy at 1

    struct SmoothRandomLFO
    {
        // Double state whatever T is: the increment is tiny against the phase,
        // so a float accumulator rounds every step and drifts in rate, and a
        // rotation of up to kRotMaxRad turns that drift into an audible change.
        double h0_ = 0.0, h1_ = 0.0, h2_ = 0.0, h3_ = 0.0;
        double phase_ = 0.0;
        double phaseInc_ = 0.0;
        uint32_t state_ = 1;

        void prepare(double sr, T rate, uint32_t seed) noexcept
        {
            phaseInc_ = static_cast<double>(rate) / sr;
            state_ = seed ? seed : 1;
            h0_ = nextRandom(); h1_ = nextRandom();
            h2_ = nextRandom(); h3_ = nextRandom();
            phase_ = 0.0;
        }

        void setRate(T rate, double sr) noexcept { phaseInc_ = static_cast<double>(rate) / sr; }

        /// Advances by `stride` samples (rates stay far below sr / stride, so
        /// at most one new target is drawn per call).
        T nextStride(int stride) noexcept
        {
            phase_ += phaseInc_ * static_cast<double>(stride);
            if (phase_ >= 1.0)
            {
                phase_ -= 1.0;
                h0_ = h1_; h1_ = h2_; h2_ = h3_;
                h3_ = nextRandom();
            }
            const double d = phase_;
            const double c1 = 0.5 * (h2_ - h0_);
            const double c2 = h0_ - 2.5 * h1_ + 2.0 * h2_ - 0.5 * h3_;
            const double c3 = 0.5 * (h3_ - h0_) + 1.5 * (h1_ - h2_);
            return static_cast<T>(((c3 * d + c2) * d + c1) * d + h1_);
        }

    private:
        double nextRandom() noexcept
        {
            state_ ^= state_ << 13;
            state_ ^= state_ >> 17;
            state_ ^= state_ << 5;
            return static_cast<double>(state_) / static_cast<double>(0xFFFFFFFFu) * 2.0 - 1.0;
        }
    };

    /**
     * @brief Modulation by segments: a straight move toward the other side,
     *        a random share (0.3 to 0.95) of the way to its limit, at a random
     *        speed (each segment 0.5 to 1.5 cycles of the rate), then the
     *        next; the output follows the segments through a critically
     *        damped second-order filter (at twice the rate, 0.2 to 20 Hz), so
     *        the delay and its rate of change are continuous.
     */
    struct SegmentLfo
    {
        double pos = 0.0, vel = 0.0, goal = 0.0, target = 0.0, step = 0.0, side = 1.0;
        uint32_t state = 1;

        void prepare(uint32_t seed) noexcept
        {
            state = seed ? seed : 1;
            pos = vel = goal = target = step = 0.0;
            side = 1.0;
        }

        double next(int stride, double rateHz, double sr) noexcept
        {
            if (step == 0.0)
            {
                // Toward the other side, a random share of the way to its limit.
                target = goal + (side - goal) * (0.3 + 0.65 * random());
                side = -side;
                const double seconds = (0.5 + random()) / std::max(rateHz, 0.01);
                step = std::abs(target - goal) / std::max(1.0, seconds * sr);
                if (step == 0.0) step = 1e-9;
            }
            const double move = step * static_cast<double>(stride);
            if (std::abs(target - goal) <= move) { goal = target; step = 0.0; }
            else goal += target > goal ? move : -move;
            // A critically damped follower of the segments: the delay and its
            // rate of change stay continuous (no step of pitch at a corner).
            const double dt = static_cast<double>(stride) / sr;
            const double w = std::min(2.0 * 3.14159265358979323846 * std::clamp(2.0 * rateHz, 0.2, 20.0), 0.5 / dt);
            vel += (w * w * (goal - pos) - 2.0 * w * vel) * dt;
            pos += vel * dt;
            return pos;
        }

    private:
        double random() noexcept
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return static_cast<double>(state) / 4294967296.0;
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
    static constexpr T rotRate() noexcept { return static_cast<T>(kRotRateHz); }
    static constexpr T rotRateFactor(int i) noexcept
    {
        return T(0.83) + T(0.051) * static_cast<T>(i);
    }
    static constexpr uint32_t rotSeed(int i) noexcept
    {
        return static_cast<uint32_t>(i) * 104729u + 7u;
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
    std::array<T, kInStages> inAPCoeffTarget_ {};   ///< inAPCoeff_ glides here (setDiffusion)
    std::array<Line, 2> ring_;             ///< allpass-diffused input, feeds the late field
    std::array<std::array<Line, kOutStages>, 2> outAP_;   ///< late-field output diffusers
    std::array<std::array<int, kOutStages>, 2> outAPLen_ {};
    T outAPCoeff_ = T(0.5);
    T outAPCoeffTarget_ = T(0.5);

    // Early reflections (per side): velvet taps, each reading its own side
    // (erChan_ 0) or the other (1)
    int numERTaps_ = 0;    ///< reflections per output side
    int numERSlots_ = 0;   ///< tap slots per output side (see kMaxERSlots)
    Type erType_ = Type::Room;
    std::array<std::array<int, kMaxERSlots>, 2> erTap_ {}, erChan_ {}, erBin_ {};
    std::array<std::array<T, kMaxERSlots>, 2> erGain_ {};         ///< current (gliding) tap gains
    std::array<std::array<T, kMaxERSlots>, 2> erGainTarget_ {};   ///< gains for the current size/decay
    std::array<T, kERGroups> erShelfGainTarget_ {};
    T erGlidePerSample_ = T(0.001);                               ///< glide rate of the two (1 / samples)
    std::array<std::array<int, kERGroups + 1>, 2> erGroupStart_ {};   ///< slot index per side
    std::array<T, kERGroups> erShelfGain_ {};   ///< HF gain of each absorption group
    T erShelfCoeff_ = T(0.5);                   ///< one-pole coefficient at the high crossover
    std::array<std::array<T, kERGroups>, 2> erLP_ {};
    std::array<std::array<T, kERGroups>, 2> erShadowLP_ {};
    double meanLoopSec_ = 0.1;   ///< mean FDN round trip (s), for the early/late energy match
    T wetLevel_ = T(1);          ///< the wet signal's level against the decay (see updateDecayParams())
    T shadowCoeff_ = T(0);

    // FDN
    DelayBank lines_;
    std::array<int, kMaxLines> lenTarget_ {};    ///< prime line lengths (samples)
    std::array<T, kMaxLines> lenCur_ {};         ///< gliding length
    // Double whatever T is: a float position on a line thousands of samples long
    // resolves only about 0.001 samples, which is jitter on every modulated read.
    std::array<double, kMaxLines> pos_ {};       ///< current read position
    std::array<double, kMaxLines> posInc_ {};    ///< per-sample ramp
    // A change of size under way: the second read heads (see beginSizeFade()).
    std::array<double, kMaxLines> pos2_ {}, posInc2_ {};
    std::array<T, kMaxLines> apY2_ {}, lenNew_ {};
    std::array<int, kMaxLines> loopAPFrom_ {}, loopAPTo_ {};
    double fadeAt_ = 0.0, fadeStep_ = 0.0;       ///< the fade, 0 to 1, and its step per sample
    bool sizeFade_ = false;
    std::array<T, kMaxLines> apY1_ {};           ///< allpass interpolator state
    std::array<int, kMaxLines> injTap_ {};
    std::array<T, kMaxLines> injGain_ {};        ///< sign / sqrt(lines)
    std::array<T, kMaxLines> outSignL_ {}, outSignR_ {};
    std::array<SmoothRandomLFO, kMaxLines> lfo_;
    // Time-varying mixing: one slow Givens rotation per line pair after the
    // Hadamard mix (orthogonal, so the loop stays lossless).
    std::array<SmoothRandomLFO, kMaxLines / 2> rotLfo_;
    alignas(64) std::array<T, kMaxLines / 2> rotC_ {};
    alignas(64) std::array<T, kMaxLines / 2> rotS_ {};
    T rotDepth_ = T(0);
    unsigned rotTick_ = 0;
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
    T loopAPCoeffTarget_ = T(0.4);

    // Absorption: Jot shelf (mid/high) and bass shelf, per line
    std::array<T, kMaxLines> jotB0_ {}, jotB1_ {}, jotA1_ {}, jotX1_ {}, jotY1_ {};
    std::array<T, kMaxLines> bassB0_ {}, bassB1_ {}, bassA1_ {}, bassX1_ {}, bassY1_ {};

    // Output stage
    // These low-frequency recursions need the shared double-precision core.
    // Coefficients are owned by prepare(); no cross-thread publications.
    Biquad<T, 1> coherenceFilter_;
    Biquad<T, 2> subsonicFilter_;

    // Tone correction EQ (Biquad 12 dB/oct)
    Biquad<T, 2> toneLPBiquad_;
    Biquad<T, 2> toneHPBiquad_;
    bool toneLPActive_ = false;   ///< filter running (on, or still fading out)
    bool toneHPActive_ = false;
    // Cutoff changes glide in the log domain, chunk by chunk; switching a
    // filter on or off crossfades between the filtered and the plain signal.
    double toneLPCurHz_ = 0.0, toneLPTargetHz_ = 0.0;
    double toneHPCurHz_ = 0.0, toneHPTargetHz_ = 0.0;
    bool toneLPOn_ = false, toneHPOn_ = false;
    T toneLPMix_ = T(0), toneHPMix_ = T(0);   ///< share of the filtered signal

    // Gliding output gains (setEarlyLevel/setLateLevel/setWidth targets).
    T earlyCur_ = T(1), lateCur_ = T(1), widthCur_ = T(1);
    bool fresh_ = true;   ///< no sample processed since prepare()/reset()

    DryWetMixer<T> mixer_;

    // --- Parameters ----------------------------------------------------------
    //
    // Thread-safety model: every setter mutates only the atomic shadows below
    // and raises a dirty flag; the audio thread drains the flags at the top of
    // processBlock()/processSample() and rebuilds the non-atomic coefficient
    // arrays there, so nothing on the audio path is written from other threads.
    // The parameter atomics have one writer, the control thread: setType()
    // writes the preset's values itself, and the audio thread only reads them.

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

    // Per-param user-override mask. setType() clears it and writes the preset
    // on the calling thread, so a setter issued after it simply overwrites the
    // preset's value (the documented quick-start is
    // `setType(Hall); setDecay(2.0f);`). prepare() re-applies the selected
    // preset and skips every atomic whose bit is set, so the values configured
    // before prepare() survive it.
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
        T inputWidth = T(1);
        T tankLeveling = T(0);
        bool tankSat = false;
        bool powerKept = false;
        bool segments = false;
        T shimmer    = T(0);
        T sourceDepth = T(0);
    };
    CachedParams cachedParams_ {};

    // Around the tank (settings).
    struct PublishedDecayBand
    {
        std::atomic<int> shape { 0 };
        std::atomic<T> frequency { T(1000) };
        std::atomic<T> q { T(0.7071) };
        std::atomic<T> multiplier { T(1) };
    };
    std::array<PublishedDecayBand, kDecayBands> decayBands_;
    std::atomic<bool> tailHold_ { false }, tankSat_ { false };
    std::atomic<T> sourceDepth_ { T(0) }, tankLeveling_ { T(0) }, tempo_ { T(120) }, inputWidth_ { T(1) };
    std::atomic<T> shimmer_ { T(0) }, gateHoldMs_ { T(100) };
    std::atomic<int> preNote_ { 0 }, preFeel_ { 0 }, gateNote_ { 0 }, gateFeel_ { 0 };
    std::atomic<int> widthLaw_ { 0 }, modStyle_ { 0 }, layout_ { 0 };
    TailShaper<T> tail_;
    KeyEqualizer<T, 2> postEq_;

    // ... and their state (audio thread).
    std::array<std::array<T, kMaxLines>, 3> jotBase_ {}, bassBase_ {};
    std::array<std::array<std::array<T, kMaxLines>, kDecayBands>, 5> dqBase_ {};
    // The designed absorption the above glide to after a change of decay,
    // size or band while the tank holds signal (see updateDecayParams()).
    std::array<std::array<T, kMaxLines>, 3> jotGoal_ {}, bassGoal_ {};
    std::array<std::array<std::array<T, kMaxLines>, kDecayBands>, 5> dqGoal_ {};
    bool absorbGlide_ = false;
    /// The loop gains' ramp across a chunk: where they stood and their step per sample.
    std::array<std::array<T, kMaxLines>, 6> absFrom_ {}, absStep_ {};
    std::array<std::array<std::array<T, kMaxLines>, kDecayBands>, 5> dqFrom_ {}, dqStep_ {};
    double quietSamples_ = 1e300;    ///< samples since input last entered the tank
    double tankEmptyAfter_ = 0.0;    ///< quiet samples after which the tank is under -120 dB
    std::array<std::array<T, kMaxLines>, kDecayBands> dqB0_ {}, dqB1_ {}, dqB2_ {}, dqA1_ {}, dqA2_ {};
    std::array<std::array<T, kMaxLines>, kDecayBands> dqZ1_ {}, dqZ2_ {};
    std::array<int, kDecayBands> dqList_ {};
    int dqCount_ = 0;
    T holdNow_ = T(0);
    double levelGr_ = 0.0, levelRelease_ = 0.0;   ///< Tank leveling: the reduction (dB) and its release.
    int levelHold_ = 0, levelHoldSamples_ = 0;      ///< Tank leveling: hold left and its length.
    std::array<T, 2> distLp_ {};
    T distCoeff_ = T(1), distMix_ = T(0);
    std::array<SegmentLfo, kMaxLines> seg_ {};
    std::vector<T> shimBuf_;
    int shimMask_ = 0, shimW_ = 0;
    std::array<double, 2> shimPhase_ { 0.0, 0.5 };
    std::array<double, 2> shimLen_ { 0.045, 0.045 };
    uint32_t shimRng_ = 0x51ED27u;
    T lastLateMono_ = T(0);

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
            // setType() already wrote the preset's values; build the engine.
            applyTopology(type_.load(std::memory_order_relaxed));
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
        // Before the first sample after prepare() or reset() there is nothing
        // to glide from: settings made then apply at once.
        if (fresh_) snapGlides();
    }

    /// Puts every gliding parameter (levels, width, diffusion, tone) at its target.
    void snapGlides() noexcept
    {
        const T d = sourceDepth_.load(std::memory_order_relaxed);
        earlyCur_ = earlyLevel_.load(std::memory_order_relaxed) * wetLevel_ * (d > T(0) ? T(1) - T(0.5) * d : T(1));
        lateCur_  = lateLevel_.load(std::memory_order_relaxed) * wetLevel_ * (d > T(0) ? T(1) + T(0.5) * d : T(1));
        widthCur_ = width_.load(std::memory_order_relaxed);
        inAPCoeff_ = inAPCoeffTarget_;
        loopAPCoeff_ = loopAPCoeffTarget_;
        outAPCoeff_ = outAPCoeffTarget_;
        snapTone();
    }

    /// One chunk of the shared parameter glide (about 20 ms): returns the
    /// value at the end of the chunk, exactly the target once within 1e-6.
    static T glideTo(T cur, T target, T amount) noexcept
    {
        const T next = cur + amount * (target - cur);
        return std::abs(target - next) <= T(1e-6) * std::max(T(1), std::abs(target)) ? target : next;
    }

    void drainTone() noexcept
    {
        if (!(toneDirty_.load(std::memory_order_acquire)
              && toneDirty_.exchange(false, std::memory_order_acquire)))
            return;
        if (!(spec_.sampleRate > 0)) return;
        const T hpHz = toneLowCutHz_.load(std::memory_order_relaxed);
        const T lpHz = toneHighCutHz_.load(std::memory_order_relaxed);
        toneHPOn_ = hpHz > T(0);
        if (toneHPOn_)
        {
            toneHPTargetHz_ = static_cast<double>(std::clamp(hpHz, T(20), T(500)));
            if (!toneHPActive_)
            {
                // Off before: start at the target, from a clean state, while
                // the crossfade brings the filtered signal in.
                toneHPActive_ = true;
                toneHPCurHz_ = toneHPTargetHz_;
                toneHPBiquad_.setCoeffsNow(BiquadCoeffs::makeHighPass(spec_.sampleRate, toneHPCurHz_));
                toneHPBiquad_.reset();
            }
        }
        toneLPOn_ = lpHz > T(0);
        if (toneLPOn_)
        {
            toneLPTargetHz_ = static_cast<double>(std::clamp(lpHz, T(2000), T(16000)));
            if (!toneLPActive_)
            {
                toneLPActive_ = true;
                toneLPCurHz_ = toneLPTargetHz_;
                toneLPBiquad_.setCoeffsNow(BiquadCoeffs::makeLowPass(spec_.sampleRate, toneLPCurHz_));
                toneLPBiquad_.reset();
            }
        }
    }

    /// One glide step of a running cutoff toward its target (log domain).
    static void glideHz(double& cur, double target, double amount) noexcept
    {
        const double ratio = target / cur;
        if (std::abs(ratio - 1.0) < 1e-4) cur = target;
        else cur *= std::pow(ratio, amount);
    }

    /// Moves the tone cutoffs and on/off crossfades by one chunk (audio
    /// thread); the crossfade values at the chunk's start are returned
    /// through hp0/lp0 for the per-sample ramp.
    void advanceTone(T amount, T& hp0, T& lp0) noexcept
    {
        hp0 = toneHPMix_;
        lp0 = toneLPMix_;
        const double a = static_cast<double>(amount);
        if (toneHPActive_)
        {
            if (toneHPOn_ && toneHPCurHz_ != toneHPTargetHz_)
            {
                glideHz(toneHPCurHz_, toneHPTargetHz_, a);
                toneHPBiquad_.setCoeffsNow(BiquadCoeffs::makeHighPass(spec_.sampleRate, toneHPCurHz_));
            }
            toneHPMix_ = glideTo(toneHPMix_, toneHPOn_ ? T(1) : T(0), amount);
            if (!toneHPOn_ && toneHPMix_ == T(0) && hp0 == T(0)) toneHPActive_ = false;
        }
        if (toneLPActive_)
        {
            if (toneLPOn_ && toneLPCurHz_ != toneLPTargetHz_)
            {
                glideHz(toneLPCurHz_, toneLPTargetHz_, a);
                toneLPBiquad_.setCoeffsNow(BiquadCoeffs::makeLowPass(spec_.sampleRate, toneLPCurHz_));
            }
            toneLPMix_ = glideTo(toneLPMix_, toneLPOn_ ? T(1) : T(0), amount);
            if (!toneLPOn_ && toneLPMix_ == T(0) && lp0 == T(0)) toneLPActive_ = false;
        }
    }

    /// Puts the tone filters at their targets at once (prepare, reset).
    void snapTone() noexcept
    {
        if (!(spec_.sampleRate > 0)) return;
        toneHPActive_ = toneHPOn_;
        toneHPMix_ = toneHPOn_ ? T(1) : T(0);
        toneHPCurHz_ = toneHPTargetHz_;
        if (toneHPActive_)
            toneHPBiquad_.setCoeffsNow(BiquadCoeffs::makeHighPass(spec_.sampleRate, toneHPCurHz_));
        toneLPActive_ = toneLPOn_;
        toneLPMix_ = toneLPOn_ ? T(1) : T(0);
        toneLPCurHz_ = toneLPTargetHz_;
        if (toneLPActive_)
            toneLPBiquad_.setCoeffsNow(BiquadCoeffs::makeLowPass(spec_.sampleRate, toneLPCurHz_));
    }

    /// Pulls the atomic parameters into the block-local cache (audio thread).
    void refreshCachedParams() noexcept
    {
        cachedParams_.preDelaySamples = preDelaySamples_.load(std::memory_order_relaxed);
        cachedParams_.erToLateSamples = erToLateSamples_.load(std::memory_order_relaxed);
        cachedParams_.earlyLevel = earlyLevel_.load(std::memory_order_relaxed) * wetLevel_;
        cachedParams_.lateLevel  = lateLevel_.load(std::memory_order_relaxed) * wetLevel_;
        cachedParams_.width      = width_.load(std::memory_order_relaxed);
        cachedParams_.inputWidth = inputWidth_.load(std::memory_order_relaxed);
        cachedParams_.tankLeveling  = tankLeveling_.load(std::memory_order_relaxed);
        cachedParams_.tankSat   = tankSat_.load(std::memory_order_relaxed);
        cachedParams_.powerKept  = widthLaw_.load(std::memory_order_relaxed) == 1;
        cachedParams_.segments   = modStyle_.load(std::memory_order_relaxed) == 1;
        cachedParams_.shimmer    = shimmer_.load(std::memory_order_relaxed);
        cachedParams_.sourceDepth   = sourceDepth_.load(std::memory_order_relaxed);
        const T d = cachedParams_.sourceDepth;
        if (d > T(0))
        {
            // Source depth: from the early field to the tail, a shorter gap.
            cachedParams_.earlyLevel *= T(1) - T(0.5) * d;
            cachedParams_.lateLevel  *= T(1) + T(0.5) * d;
            cachedParams_.preDelaySamples = static_cast<int>(static_cast<T>(cachedParams_.preDelaySamples)
                                                             * (T(1) - T(0.8) * d));
        }
        tail_.sync();
    }

    /**
     * @brief Control-rate update: glides the line lengths toward their
     *        targets, draws the next modulation values and sets the
     *        per-sample read-position ramps for the next kCtrl samples.
     */
    void controlTick() noexcept
    {
        const int n = nLines_;
        // The pair angles move by under 0.005 rad per 64 samples: refresh
        // their sine and cosine every fourth control tick.
        if ((++rotTick_ & 3) == 0)
            for (int k = 0; k < n / 2; ++k)
            {
                const T theta = rotLfo_[k].nextStride(4 * kCtrl) * rotDepth_;
                // Form the rotation in double before rounding coefficients to T.
                // Single-precision libm differences are otherwise recirculated
                // through the feedback network on every sample.
                rotC_[k] = static_cast<T>(std::cos(static_cast<double>(theta)));
                rotS_[k] = static_cast<T>(std::sin(static_cast<double>(theta)));
            }
        const bool segments = cachedParams_.segments;
        const double sr = spec_.sampleRate;
        const double rate = static_cast<double>(modRate_.load(std::memory_order_relaxed));
        // A change of size beyond kSizeFadeOver of a line's length opens a
        // second read head at the new lengths and cross-fades to it (a glide
        // of the lengths is a glide of pitch); a smaller one glides. While a
        // fade runs the next change waits.
        if (!sizeFade_)
        {
            T widest = T(0);
            for (int i = 0; i < n; ++i)
                widest = std::max(widest, std::abs(static_cast<T>(lenTarget_[i]) - lenCur_[i]) / std::max(lenCur_[i], T(1)));
            if (widest > static_cast<T>(kSizeFadeOver)) beginSizeFade();
        }
        for (int i = 0; i < n; ++i)
        {
            if (!sizeFade_)
            {
                const T diff = static_cast<T>(lenTarget_[i]) - lenCur_[i];
                lenCur_[i] += std::clamp(diff * glideCoeff_, -maxGlideStep_, maxGlideStep_);
            }
            // Segments: every other line, its depth held under 2 % of its length.
            const T wander = segments
                ? ((i & 1) ? T(0) : static_cast<T>(seg_[i].next(kCtrl, rate * static_cast<double>(lfoRateFactor(i)), sr)))
                : lfo_[i].nextStride(kCtrl);
            const auto aim = [&](T length) {
                const T m = wander * (segments ? std::min(modDepthSamples_, T(0.02) * length) : modDepthSamples_);
                return std::clamp(static_cast<double>(length) + static_cast<double>(m),
                                  static_cast<double>(kMinReadPos), static_cast<double>(maxReadPos_));
            };
            posInc_[i] = (aim(lenCur_[i]) - pos_[i]) * (1.0 / static_cast<double>(kCtrl));
            if (sizeFade_) posInc2_[i] = (aim(lenNew_[i]) - pos2_[i]) * (1.0 / static_cast<double>(kCtrl));
        }
        if (loopAPGliding_ && !sizeFade_)
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

    /**
     * @brief Opens the second read head of every line at the new lengths
     *        (with the wander the first one carries) and of every in-loop
     *        allpass at its new length; the two heads then cross-fade at
     *        constant power over kSizeFadeSeconds (far apart, what they read
     *        is uncorrelated).
     */
    void beginSizeFade() noexcept
    {
        for (int i = 0; i < nLines_; ++i)
        {
            const auto l = static_cast<std::size_t>(i);
            lenNew_[l] = static_cast<T>(lenTarget_[l]);
            pos2_[l] = std::clamp(static_cast<double>(lenNew_[l]) + (pos_[l] - static_cast<double>(lenCur_[l])),
                                  static_cast<double>(kMinReadPos), static_cast<double>(maxReadPos_));
            posInc2_[l] = 0.0;
            apY2_[l] = lines_.at(i, static_cast<int>(pos2_[l] - kApLow) + 1);
            loopAPFrom_[l] = static_cast<int>(loopAPCur_[l] + T(0.5));
            loopAPTo_[l] = loopAPLen_[l];
        }
        fadeAt_ = 0.0;
        sizeFade_ = true;
    }

    /** @brief The fade is over: the second heads become the first. */
    void endSizeFade() noexcept
    {
        bool allpassesMoved = false;
        for (int i = 0; i < nLines_; ++i)
        {
            const auto l = static_cast<std::size_t>(i);
            pos_[l] = pos2_[l];
            posInc_[l] = posInc2_[l];
            apY1_[l] = apY2_[l];
            lenCur_[l] = lenNew_[l];
            loopAPCur_[l] = static_cast<T>(loopAPTo_[l]);
            allpassesMoved = allpassesMoved || loopAPTo_[l] != loopAPLen_[l];
        }
        loopAPGliding_ = allpassesMoved;
        sizeFade_ = false;
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
    void runFdn(T& lateL, T& lateR, int injLag, T loopAPCoeff, T inGain, T extra) noexcept
    {
        constexpr int W = std::min(simd::kVecWidth<T>, N);
        using O = simd::Vec<T, W>;
        using V = typename O::V;
        static_assert(N % W == 0, "line count must be a multiple of the SIMD width");

        alignas(64) std::array<T, N + 1> r;   // +1: wrap slot for the rotation
        alignas(64) std::array<T, N> v;
        alignas(64) std::array<T, N> tmp;
        alignas(64) std::array<int, N> idx;

        // Allpass-interpolated modulated reads: eta = (1 - d) / (1 + d) for
        // the fractional delay d in [kApLow, kApLow + 1) (see allpassRead).
        // The arithmetic runs in separate (vectorizable) passes around the
        // scalar gather.
        for (int i = 0; i < N; ++i)
        {
            const double p = pos_[i] += posInc_[i];
            const int M = static_cast<int>(p - kApLow);
            v[i] = allpassCoeff(p - static_cast<double>(M));
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
        // A change of size under way: the second head, at the new lengths,
        // takes over from the first at constant power.
        const bool fading = sizeFade_;
        const T fadeNew = fading ? static_cast<T>(std::sin(1.5707963267948966 * fadeAt_)) : T(0);
        const T fadeOld = fading ? static_cast<T>(std::cos(1.5707963267948966 * fadeAt_)) : T(1);
        if (fading) [[unlikely]]
            for (int i = 0; i < N; ++i)
            {
                const double p = pos2_[i] += posInc2_[i];
                const int M = static_cast<int>(p - kApLow);
                const T y = allpassCoeff(p - static_cast<double>(M)) * (lines_.at(i, M) - apY2_[i]) + lines_.at(i, M + 1);
                apY2_[i] = y;
                r[i] = fadeOld * r[i] + fadeNew * y;
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
        if (dqCount_ > 0) applyDecayBands<N, W, O>(r.data());

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
        // Slow Givens rotation of each line pair: the mixing matrix drifts
        // while staying orthogonal (lossless), so modes exchange energy
        // without the pitch deviation of delay modulation, whose spread
        // grows with frequency.
        // Pairs (i, i + 4) inside each block of 8: contiguous, branch-free,
        // so the four rotations of a block run as one vector operation.
        static_assert(kRotPairStride == 4 && N % 8 == 0, "rotation pairs are (i, i + 4) in blocks of 8");
        for (int blk = 0; blk < N; blk += 8)
        {
            T* lo = v.data() + blk;
            T* hi = lo + 4;
            const T* c = rotC_.data() + blk / 2;
            const T* sn = rotS_.data() + blk / 2;
            for (int q = 0; q < 4; ++q)
            {
                const T a = lo[q], b = hi[q];
                lo[q] = c[q] * a - sn[q] * b;
                hi[q] = sn[q] * a + c[q] * b;
            }
        }

        {
            if (fading) [[unlikely]]
            {
                for (int i = 0; i < N; ++i)
                    tmp[i] = fadeOld * loopAP_.at(i, loopAPFrom_[i]) + fadeNew * loopAP_.at(i, loopAPTo_[i]);
            }
            else if (loopAPGliding_) [[unlikely]]
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
            const V g = O::set1(loopAPCoeff);
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
            lines_.write(i, x + injGain_[i] * (inGain * ring_[(i >> 1) & 1].at(injLag + injTap_[i]) + extra));
        }
        lines_.advance();
        if (fading) [[unlikely]] advanceSizeFade();
    }

    /**
     * @brief The plate tank's decay and damping. A branch takes its gain once
     *        a pass, from its own length; the damping is a one-pole low-pass
     *        whose loss at the high crossover makes the high decay asked for.
     *        Its tail sets the wet level as the network's does.
     */
    void designPlate(double decay, double t60H, double crossoverHz) noexcept
    {
        auto& p = plate_;
        const double sr = spec_.sampleRate;
        p.on = plateTank_.load(std::memory_order_relaxed) && erType_ == Type::Plate && !spring_;
        if (!p.on) return;
        double damp = 0.0;
        for (int b = 0; b < 2; ++b)
        {
            const auto s = static_cast<std::size_t>(b);
            const double L = static_cast<double>(p.lAp1[s] + p.lD1[s] + p.lAp2[s] + p.lD2[s]);
            p.gain[s] = static_cast<T>(std::pow(0.001, L / (decay * sr)));
            // |(1 - d) / (1 - d e^-jw)| at the crossover equals what the high
            // decay takes over the mid one in a pass: the lower root of
            // (1 - q^2) d^2 - 2 (1 - q^2 c) d + (1 - q^2) = 0.
            const double q = std::pow(0.001, L / sr * (1.0 / t60H - 1.0 / decay));
            if (q < 1.0)
            {
                const double c = std::cos(2.0 * 3.14159265358979323846 * crossoverHz / sr), q2 = q * q;
                const double A = 1.0 - q2, B = 1.0 - q2 * c;
                damp = std::max(damp, (B - std::sqrt(std::max(0.0, B * B - A * A))) / A);
            }
        }
        p.damp = static_cast<T>(std::clamp(damp, 0.0, 0.99));
        p.excursion = static_cast<double>(modDepth_.load(std::memory_order_relaxed)) * PlateTank::kExcursion * p.scale;
        p.phaseStep = static_cast<double>(modRate_.load(std::memory_order_relaxed)) / sr;
        // The taps read inside the branches, so the first pass counts in full.
        const double tail = 1.0 / -std::expm1(-std::min(2.0 * 6.907755278982137 * p.loopSec / std::max(1e-3, decay), 50.0));
        wetLevel_ = static_cast<T>(kWetCalibration / std::sqrt(std::max(1e-6, kPlateEnergy * tail)));
    }

    /** @brief One sample of the plate tank: the late pair, from the diffused feed. */
    void runPlate(T& lateL, T& lateR, int injLag, T inGain, T extra) noexcept
    {
        auto& p = plate_;
        const T in = T(0.5) * (ring_[0].at(injLag + 1) + ring_[1].at(injLag + 1)) * inGain + extra;
        const T hold = holdNow_;
        const std::array<T, 2> ends { p.d2[0].at(p.lD2[0]), p.d2[1].at(p.lD2[1]) };
        p.phase += p.phaseStep;
        if (p.phase >= 1.0) p.phase -= 1.0;
        for (int b = 0; b < 2; ++b)
        {
            const auto s = static_cast<std::size_t>(b);
            T x = in + ends[1 - s];
            // The first all-pass, its length moved by the modulation (the
            // two branches a quarter of a turn apart), read between samples.
            const double turn = p.phase + 0.25 * b;
            const double pos = static_cast<double>(p.lAp1[s])
                + p.excursion * static_cast<double>(fastSin(static_cast<T>(6.283185307179586 * (turn - std::floor(turn)) - 3.141592653589793)));
            const int k = static_cast<int>(pos);
            const T fr = static_cast<T>(pos - k);
            const T dl = p.ap1[s].at(k) + fr * (p.ap1[s].at(k + 1) - p.ap1[s].at(k));
            const T w1 = x + PlateTank::kDiffusion1 * dl;
            p.ap1[s].push(w1);
            x = dl - PlateTank::kDiffusion1 * w1;
            const T v = p.d1[s].at(p.lD1[s]);
            p.d1[s].push(x);
            // Damping and decay (both open while the tail is held).
            const T damp = p.damp * (T(1) - hold);
            p.lp[s] = v + damp * (p.lp[s] - v);
            x = p.lp[s] * (p.gain[s] + (T(1) - p.gain[s]) * hold);
            const T d2 = p.ap2[s].at(p.lAp2[s]);
            const T w2 = x + PlateTank::kDiffusion2 * d2;
            p.ap2[s].push(w2);
            p.d2[s].push(d2 - PlateTank::kDiffusion2 * w2);
        }
        // Dattorro's taps (at 29761 Hz), each side mostly from the other branch.
        auto tap = [&p](const Line& l, double at29k) { return l.at(p.at(at29k)); };
        lateL += T(0.6) * (tap(p.d1[1], 266) + tap(p.d1[1], 2974) - tap(p.ap2[1], 1913) + tap(p.d2[1], 1996)
                           - tap(p.d1[0], 1990) - tap(p.ap2[0], 187) - tap(p.d2[0], 1066));
        lateR += T(0.6) * (tap(p.d1[0], 353) + tap(p.d1[0], 3627) - tap(p.ap2[0], 1228) + tap(p.d2[0], 2673)
                           - tap(p.d1[1], 2111) - tap(p.ap2[1], 335) - tap(p.d2[1], 121));
    }

    /**
     * @brief The second slope's absorption and shares. Fed and summed as the
     *        tank is (unit energy in, every line out), a network's tail holds
     *        1 / (e^(13.8 x its mean loop / T60) - 1) of energy; the share asked for sets
     *        its gain from the two, and the tank keeps the rest, so the sum
     *        has the tail's energy.
     */
    void designSecondSlope(double decay, double t60H, double Kh) noexcept
    {
        auto& s = slope2_;
        const double sr = spec_.sampleRate;
        const double d2 = kSecondRatio * decay, h2 = kSecondRatio * t60H;
        double loop = 0.0;
        for (int j = 0; j < SecondSlope::kLines; ++j)
        {
            const auto l = static_cast<std::size_t>(j);
            const double M = static_cast<double>(s.len[l]);
            loop += M;
            const double gM = std::pow(0.001, M / (d2 * sr));
            const double gH = std::min(std::pow(0.001, M / (h2 * sr)), gM);
            const double ratio = std::sqrt(gM / gH);
            const double a = Kh * ratio, b = Kh / ratio, inv = 1.0 / (1.0 + b);
            s.b0[l] = static_cast<T>(gH * (1.0 + a) * inv);
            s.b1[l] = static_cast<T>(gH * (a - 1.0) * inv);
            s.a1[l] = static_cast<T>((b - 1.0) * inv);
        }
        s.loopSec = loop / (SecondSlope::kLines * sr);
        const double amount = static_cast<double>(doubleSlope_.load(std::memory_order_relaxed));
        s.on = amount > 0.0 && !spring_;
        if (!s.on)
        {
            s.gain = T(0);
            s.keep = T(1);
            return;
        }
        const double share = kSecondShare * amount;
        // Each network's tail: g^2 / (1 - g^2) per unit in, g^2 = e^-(13.8 loop / T60).
        const auto tail = [](double loopSec, double t60) { return 1.0 / std::expm1(std::min(2.0 * 6.907755278982137 * loopSec / std::max(1e-3, t60), 50.0)); };
        const double own = tail(s.loopSec, kSecondRatio * decay) / std::max(1e-9, tail(meanLoopSec_, decay));   // its energy over the tank's
        s.gain = static_cast<T>(std::sqrt(share / ((1.0 - share) * own)));
        s.keep = static_cast<T>(std::sqrt(1.0 - share));
    }

    /** @brief One sample of the second slope, added to the late pair. */
    void runSecondSlope(T& lateL, T& lateR, int injLag) noexcept
    {
        auto& s = slope2_;
        constexpr T kGlide = T(0.001);
        s.gainNow += kGlide * (s.gain - s.gainNow);
        s.keepNow += kGlide * (s.keep - s.keepNow);
        T y[SecondSlope::kLines];
        for (int j = 0; j < SecondSlope::kLines; ++j)
        {
            const auto l = static_cast<std::size_t>(j);
            const T x = s.line[l].at(s.len[l]);
            const T v = s.b0[l] * x + s.b1[l] * s.x1[l] - s.a1[l] * s.y1[l];
            s.x1[l] = x;
            s.y1[l] = v;
            y[j] = v;
        }
        const T outL = y[0] - y[1] + y[2] - y[3];
        const T outR = y[0] + y[1] - y[2] - y[3];
        hadamardSmall<SecondSlope::kLines>(y);
        for (int j = 0; j < SecondSlope::kLines; ++j)
        {
            const auto l = static_cast<std::size_t>(j);
            const T feed = ring_[static_cast<std::size_t>(j & 1)].at(injLag + 1 + 7 * j);
            s.line[l].push(T(0.5) * (y[j] + ((j & 2) ? -feed : feed)));
        }
        lateL = s.keepNow * lateL + s.gainNow * outL;
        lateR = s.keepNow * lateR + s.gainNow * outR;
    }

    /** @brief One sample of a size fade; at its end the second heads become the first. */
    void advanceSizeFade() noexcept
    {
        fadeAt_ += fadeStep_;
        if (fadeAt_ >= 1.0) endSizeFade();
    }

    /**
     * @brief The decay bands in the loop, one TDF-II biquad per line and band
     *        (the per-line arithmetic runs across the lines in SIMD).
     */
    template <int N, int W, typename O>
    void applyDecayBands(T* r) noexcept
    {
        using V = typename O::V;
        for (int k = 0; k < dqCount_; ++k)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(k)]);
            T* z1 = dqZ1_[b].data();
            T* z2 = dqZ2_[b].data();
            const T* b0 = dqB0_[b].data();
            const T* b1 = dqB1_[b].data();
            const T* b2 = dqB2_[b].data();
            const T* a1 = dqA1_[b].data();
            const T* a2 = dqA2_[b].data();
            for (int i = 0; i < N; i += W)
            {
                const V x = O::load(r + i);
                const V y = O::madd(O::load(b0 + i), x, O::load(z1 + i));
                O::store(z1 + i, O::sub(O::madd(O::load(b1 + i), x, O::load(z2 + i)), O::mul(O::load(a1 + i), y)));
                O::store(z2 + i, O::sub(O::mul(O::load(b2 + i), x), O::mul(O::load(a2 + i), y)));
                O::store(r + i, y);
            }
        }
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

        // A change of size under way: a second head at the new lengths takes
        // over from the first at constant power (see beginSizeFade()).
        const bool fading = sizeFade_;
        const T fadeNew = fading ? static_cast<T>(std::sin(1.5707963267948966 * fadeAt_)) : T(0);
        const T fadeOld = fading ? static_cast<T>(std::cos(1.5707963267948966 * fadeAt_)) : T(1);
        alignas(64) std::array<T, Springs> x;
        for (int s = 0; s < Springs; ++s)
        {
            const double p = pos_[s] += posInc_[s];
            const int M = static_cast<int>(p - kApLow);
            const T eta = allpassCoeff(p - static_cast<double>(M));
            const T y = eta * (lines_.at(s, M) - apY1_[s]) + lines_.at(s, M + 1);
            apY1_[s] = y;
            x[s] = y;
            if (fading) [[unlikely]]
            {
                const double p2 = pos2_[s] += posInc2_[s];
                const int M2 = static_cast<int>(p2 - kApLow);
                const T y2 = allpassCoeff(p2 - static_cast<double>(M2)) * (lines_.at(s, M2) - apY2_[s]) + lines_.at(s, M2 + 1);
                apY2_[s] = y2;
                x[s] = fadeOld * y + fadeNew * y2;
            }
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
        if (dqCount_ > 0) applyDecayBands<Springs, W, O>(x.data());
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
        if (fading) [[unlikely]] advanceSizeFade();
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

    /** @brief Pushes `count` samples into a line, in pieces that stop at its end. */
    static void writeSpan(Line& l, const T* src, int count) noexcept
    {
        for (int s = 0; s < count;)
        {
            const int m = std::min(count - s, l.mask + 1 - l.w);
            T* DSPARK_RESTRICT dst = l.buf.data() + l.w;
            for (int k = 0; k < m; ++k) dst[k] = src[s + k];
            l.w = (l.w + m) & l.mask;
            s += m;
        }
    }

    /**
     * @brief One Schroeder allpass stage over a chunk: w = x + g d, out =
     *        d - g w, with g = g0 + gDelta ramp[n] at sample n of the chunk.
     *
     * The stage runs in pieces that read only samples older than themselves
     * (no longer than its delay), touch neither end of the ring, and read
     * and write apart (no longer than the ring minus the delay), so each
     * piece is plain array arithmetic on the line itself.
     */
    static void diffuseSpans(Line& ap, int len, T g0, T gDelta, const T* ramp, T* x, int count) noexcept
    {
        const int size = ap.mask + 1;
        const int delay = std::max(1, len);
        for (int s = 0; s < count;)
        {
            const int readAt = (ap.w - delay) & ap.mask;
            const int m = std::max(1, std::min({ delay, count - s, size - readAt, size - ap.w, size - delay }));
            const T* DSPARK_RESTRICT rd = ap.buf.data() + readAt;
            T* DSPARK_RESTRICT wr = ap.buf.data() + ap.w;
            T* DSPARK_RESTRICT xs = x + s;
            const T* rs = ramp + s;
            for (int k = 0; k < m; ++k)
            {
                const T g = g0 + gDelta * rs[k];
                const T d = rd[k];
                const T w = xs[k] + g * d;
                wr[k] = w;
                xs[k] = d - g * w;
            }
            ap.w = (ap.w + m) & ap.mask;
            s += m;
        }
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

        // One step of the ~20 ms parameter glides for this chunk: diffusion
        // coefficients and tone cutoffs per chunk, output gains per sample
        // (ramped below), so automating them does not click.
        const T gc = std::min(T(1), erGlidePerSample_ * static_cast<T>(count));
        const std::array<T, kInStages> inAP0 = inAPCoeff_;
        const T loopAP0 = loopAPCoeff_, outAP0 = outAPCoeff_;
        for (int st = 0; st < kInStages; ++st)
            inAPCoeff_[static_cast<std::size_t>(st)] = glideTo(inAPCoeff_[static_cast<std::size_t>(st)],
                                                             inAPCoeffTarget_[static_cast<std::size_t>(st)], gc);
        loopAPCoeff_ = glideTo(loopAPCoeff_, loopAPCoeffTarget_, gc);
        outAPCoeff_ = glideTo(outAPCoeff_, outAPCoeffTarget_, gc);
        T toneHP0 = T(0), toneLP0 = T(0);
        advanceTone(gc, toneHP0, toneLP0);
        const T early0 = earlyCur_, late0 = lateCur_, width0 = widthCur_;
        earlyCur_ = glideTo(early0, cachedParams_.earlyLevel, gc);
        lateCur_  = glideTo(late0, cachedParams_.lateLevel, gc);
        widthCur_ = glideTo(width0, cachedParams_.width, gc);
        const T rampStep = T(1) / static_cast<T>(count);
        fresh_ = false;

        // Tail hold: the absorption fades to unity and the input out.
        const T hold0 = holdNow_;
        holdNow_ = glideTo(holdNow_, tailHold_.load(std::memory_order_relaxed) ? T(1) : T(0), gc);
        // The loop gains move toward their design (and toward unity while
        // the tail is held) sample by sample across the chunk: a step of
        // them is a step of every line's output.
        const bool absorbMoved = absorbGlide_;
        const bool absorbRamp = holdNow_ != hold0 || absorbMoved;
        if (absorbRamp)
        {
            holdAbsorption();
            if (absorbMoved) glideAbsorption(gc);
            applyHoldCoeffs();
            rampAbsorption(rampStep);
        }
        {
            // Input entering the tank (for the empty-tank test of updateDecayParams()).
            T peak = T(0);
            for (int n = 0; n < count; ++n) peak = std::max(peak, std::max(std::abs(inL[n]), std::abs(inR[n])));
            quietSamples_ = peak > T(1e-6) ? 0.0 : quietSamples_ + count;
        }
        const bool holding = holdNow_ != T(0) || hold0 != T(0);
        const T in0 = T(1) - hold0, in1 = T(1) - holdNow_;

        // Input width: the side of the input before the tank.
        alignas(64) T narrowL[kChunk];
        alignas(64) T narrowR[kChunk];
        if (cachedParams_.inputWidth != T(1))
        {
            const T w = cachedParams_.inputWidth;
            for (int n = 0; n < count; ++n)
            {
                const T mid = (inL[n] + inR[n]) * T(0.5), side = (inL[n] - inR[n]) * T(0.5) * w;
                narrowL[n] = mid + side;
                narrowR[n] = mid - side;
            }
            inL = narrowL;
            inR = narrowR;
        }

        // Source depth: the low-pass on the wet signal, crossfaded in and out.
        {
            const T d = cachedParams_.sourceDepth;
            distMix_ = glideTo(distMix_, d > T(0) ? T(1) : T(0), gc);
            if (d > T(0))
            {
                const double hd = static_cast<double>(highDecayMult_.load(std::memory_order_relaxed));
                const double darkness = static_cast<double>(d) * (1.25 - 0.5 * hd);
                const double fc = std::min(18000.0 * std::exp2(-3.0 * darkness), 0.45 * spec_.sampleRate);
                distCoeff_ = static_cast<T>(1.0 - std::exp(-6.283185307179586 * fc / spec_.sampleRate));
            }
        }

        // --- Pre-delay, then the allpass diffusers of the late-field feed ---
        alignas(64) T raw[2][kChunk];
        for (int c = 0; c < 2; ++c)
        {
            Line& pre = preDelay_[static_cast<std::size_t>(c)];
            const T* in = c == 0 ? inL : inR;
            for (int n = 0; n < count; ++n) pre.push(in[n]);
            std::fill(raw[c], raw[c] + count, T(0));
            addTap(raw[c], pre, pd + 1 + last, T(1), count);
        }

        // Tank leveling: a compressor on the tank's input, both sides linked.
        if (cachedParams_.tankLeveling > T(0))
        {
            const double ratio = 1.0 + 3.0 * static_cast<double>(cachedParams_.tankLeveling);
            const double slope = 1.0 / ratio - 1.0;
            constexpr double kThreshold = -30.0, kKnee = 24.0;
            const double makeup = -slope * 12.0;   // -18 dBFS sits on the knee's top
            const bool saturate = cachedParams_.tankSat;
            for (int n = 0; n < count; ++n)
            {
                const double peak = std::max(static_cast<double>(raw[0][n]) * raw[0][n],
                                             static_cast<double>(raw[1][n]) * raw[1][n]);
                const double over = 10.0 * std::log10(0.5 * peak + 1e-30) - kThreshold;
                const double gr = over <= -0.5 * kKnee ? 0.0
                                : over >= 0.5 * kKnee ? slope * over
                                : slope * (over + 0.5 * kKnee) * (over + 0.5 * kKnee) / (2.0 * kKnee);
                // Deeper at once; a crest within 0.01 dB of the reduction held
                // holds it again; after the hold it comes back slowly.
                if (gr <= levelGr_ + 0.01)
                {
                    levelGr_ = std::min(levelGr_, gr);
                    levelHold_ = levelHoldSamples_;
                }
                else if (levelHold_ > 0) --levelHold_;
                else levelGr_ = gr + levelRelease_ * (levelGr_ - gr);
                const T g = static_cast<T>(std::pow(10.0, (levelGr_ + makeup) / 20.0));
                for (int c = 0; c < 2; ++c)
                {
                    T x = raw[c][n] * g;
                    if (saturate) x = std::tanh(T(1.5) * x) * T(1.0 / 1.5);
                    raw[c][n] = x;
                }
            }
        }

        // The input diffusers, one stage at a time over the chunk: a stage
        // reads its line for a whole span before writing it, which holds
        // while the span is no longer than its delay, so each span runs as
        // array arithmetic (the same operations, in the same order, as one
        // sample through all four stages).
        alignas(64) T ramp[kChunk];
        for (int n = 0; n < count; ++n) ramp[n] = static_cast<T>(n + 1) * rampStep;
        for (int c = 0; c < 2; ++c)
        {
            alignas(64) T x[kChunk];
            std::copy_n(raw[c], count, x);
            auto& aps = inAP_[static_cast<std::size_t>(c)];
            for (int st = 0; st < kInStages; ++st)
            {
                const T g0 = inAP0[static_cast<std::size_t>(st)];
                diffuseSpans(aps[static_cast<std::size_t>(st)], inAPLen_[static_cast<std::size_t>(c)][static_cast<std::size_t>(st)],
                             g0, inAPCoeff_[static_cast<std::size_t>(st)] - g0, ramp, x, count);
            }
            writeSpan(ring_[static_cast<std::size_t>(c)], x, count);
        }

        // The early gains follow size and decay changes in a ~20 ms glide
        // (per chunk), so automating them does not click.
        {
            for (int s = 0; s < 2; ++s)
                for (int k = 0; k < numERSlots_; ++k)
                    erGain_[s][k] = glideTo(erGain_[s][k], erGainTarget_[s][k], gc);
            for (int g = 0; g < kERGroups; ++g)
                erShelfGain_[g] += gc * (erShelfGainTarget_[g] - erShelfGain_[g]);
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
                // Each tap lands in its head-shadow class: open (the ear
                // facing the reflection) or shadowed (the far ear).
                alignas(64) T acc[kShadowBins][kChunk];
                for (auto& a : acc) std::fill(a, a + count, T(0));
                for (int k = erGroupStart_[s][g]; k < erGroupStart_[s][g + 1]; ++k)
                    if (erGain_[s][k] != T(0))   // an idle crossfade slot adds nothing
                        addTap(acc[erBin_[s][k]], *src[erChan_[s][k]], erTap_[s][k] + lagOff[erChan_[s][k]],
                               erGain_[s][k], count);
                {
                    // The shadowed class keeps its lows, loses half its highs.
                    constexpr T hf = static_cast<T>(kShadowHF) - T(1);
                    T z = erShadowLP_[static_cast<std::size_t>(s)][static_cast<std::size_t>(g)];
                    for (int n = 0; n < count; ++n)
                    {
                        const T x = acc[1][n];
                        z += shadowCoeff_ * (x - z);
                        acc[0][n] += x + hf * (x - z);
                    }
                    erShadowLP_[static_cast<std::size_t>(s)][static_cast<std::size_t>(g)] = z;
                }
                // Frequency-dependent absorption: a first-order high shelf
                // at the high crossover, the group's highs attenuated as the
                // late field's are by that time (its HF T60).
                const T c = erShelfCoeff_;
                const T hg = erShelfGain_[static_cast<std::size_t>(g)];
                T y = lp[g];
                for (int n = 0; n < count; ++n)
                {
                    const T x = acc[0][n];
                    y += c * (x - y);
                    early[s][n] += y + hg * (x - y);
                }
                lp[g] = y;
            }
        }

        for (int n = 0; n < count; ++n)
        {
            if (ctrlPhase_ == 0) controlTick();
            ctrlPhase_ = (ctrlPhase_ + 1) & (kCtrl - 1);
            if (absorbRamp) stepAbsorption();

            // --- Late field ---
            T lateL = T(0), lateR = T(0);
            const T a = static_cast<T>(n + 1) * rampStep;
            const T inGain = holding ? in0 + (in1 - in0) * a : T(1);
            if (spring_)
            {
                const T rawN[2] = { raw[0][n] * inGain, raw[1][n] * inGain };
                if (eco_) runSprings<kEcoSprings, kEcoSpringStages>(lateL, lateR, rawN);
                else      runSprings<kSprings, kSpringStages>(lateL, lateR, rawN);
            }
            if (!spring_)
            {
                const T loopG = loopAP0 + (loopAPCoeff_ - loopAP0) * a;
                const T extra = cachedParams_.shimmer > T(0) ? shimmerStep(lastLateMono_) * cachedParams_.shimmer * kShimmerFeed : T(0);
                if (plate_.on)  runPlate(lateL, lateR, gap + last - n, inGain, extra);
                else if (eco_) runFdn<kEcoLines>(lateL, lateR, gap + last - n, loopG, inGain, extra);
                else      runFdn<kMaxLines>(lateL, lateR, gap + last - n, loopG, inGain, extra);
                lastLateMono_ = (lateL + lateR) * T(0.5);
                if (slope2_.on || slope2_.gainNow != T(0)) runSecondSlope(lateL, lateR, gap + last - n);
            }

            const T earlyNow = early0 + (earlyCur_ - early0) * a;
            auto [yl, yr] = outputStage(lateL, lateR, early[0][n], early[1][n],
                                        holding ? earlyNow * inGain : earlyNow,
                                        late0 + (lateCur_ - late0) * a,
                                        width0 + (widthCur_ - width0) * a,
                                        toneHP0 + (toneHPMix_ - toneHP0) * a,
                                        toneLP0 + (toneLPMix_ - toneLP0) * a,
                                        outAP0 + (outAPCoeff_ - outAP0) * a);
            if (tail_.isActive())
            {
                const T g = tail_.next(std::max(std::abs(inL[n]), std::abs(inR[n])));
                yl *= g;
                yr *= g;
            }
            outL[n] = yl;
            outR[n] = yr;
        }
        if (absorbRamp) applyHoldCoeffs();   // the chunk's end, without the ramp's rounding
    }

    /**
     * @brief One sample of the octave-up shimmer: two read heads, half a
     *        window apart, sweep the recent input at twice its speed, each
     *        under a raised-cosine window; a head draws its next window length
     *        (30 to 60 ms) as it wraps.
     */
    T shimmerStep(T input) noexcept
    {
        shimBuf_[static_cast<std::size_t>(shimW_)] = input;
        const double sr = spec_.sampleRate;
        T sum = T(0), weight = T(0);
        for (int h = 0; h < 2; ++h)
        {
            const auto hh = static_cast<std::size_t>(h);
            const double ph = shimPhase_[hh];
            const double len = shimLen_[hh] * sr;
            // The delay falls by one sample per sample: the head reads at
            // twice the speed of the writing, an octave up.
            const double delay = 1.0 + len * (1.0 - ph);
            const int di = static_cast<int>(delay);
            const T f = static_cast<T>(delay - di);
            const T s0 = shimBuf_[static_cast<std::size_t>((shimW_ - di) & shimMask_)];
            const T s1 = shimBuf_[static_cast<std::size_t>((shimW_ - di - 1) & shimMask_)];
            const T w = static_cast<T>(0.5 - 0.5 * std::cos(6.283185307179586 * ph));
            sum += w * (s0 + f * (s1 - s0));
            weight += w;
            shimPhase_[hh] += 1.0 / std::max(1.0, len);
            if (shimPhase_[hh] >= 1.0)
            {
                shimPhase_[hh] -= 1.0;
                shimLen_[hh] = drawShimmerLength();
            }
        }
        shimW_ = (shimW_ + 1) & shimMask_;
        // The windows of heads of different lengths do not sum to one: the
        // sum is normalized, and attenuated where both heads are near a wrap.
        return weight > T(0.25) ? sum / weight : sum * T(4);
    }

    double drawShimmerLength() noexcept
    {
        shimRng_ ^= shimRng_ << 13;
        shimRng_ ^= shimRng_ >> 17;
        shimRng_ ^= shimRng_ << 5;
        const double u = static_cast<double>(shimRng_) / 4294967296.0;
        return 0.001 * (kShimmerMinMs + (kShimmerMaxMs - kShimmerMinMs) * u);
    }

    /** @brief The absorption and decay bands in use: the designed ones moved toward unity by the tail hold. */
    void applyHoldCoeffs() noexcept
    {
        const T f = holdNow_, k = T(1) - f;
        for (int i = 0; i < kMaxLines; ++i)
        {
            const auto l = static_cast<std::size_t>(i);
            jotB0_[l] = jotBase_[0][l] + (T(1) - jotBase_[0][l]) * f;
            jotB1_[l] = jotBase_[1][l] * k;
            jotA1_[l] = jotBase_[2][l] * k;
            bassB0_[l] = bassBase_[0][l] + (T(1) - bassBase_[0][l]) * f;
            bassB1_[l] = bassBase_[1][l] * k;
            bassA1_[l] = bassBase_[2][l] * k;
            for (int q = 0; q < dqCount_; ++q)
            {
                const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
                dqB0_[b][l] = dqBase_[0][b][l] + (T(1) - dqBase_[0][b][l]) * f;
                dqB1_[b][l] = dqBase_[1][b][l] * k;
                dqB2_[b][l] = dqBase_[2][b][l] * k;
                dqA1_[b][l] = dqBase_[3][b][l] * k;
                dqA2_[b][l] = dqBase_[4][b][l] * k;
            }
        }
    }

    /// The loop gains as they stand, kept for the ramp of a chunk.
    void holdAbsorption() noexcept
    {
        absFrom_[0] = jotB0_; absFrom_[1] = jotB1_; absFrom_[2] = jotA1_;
        absFrom_[3] = bassB0_; absFrom_[4] = bassB1_; absFrom_[5] = bassA1_;
        for (int q = 0; q < dqCount_; ++q)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
            dqFrom_[0][b] = dqB0_[b]; dqFrom_[1][b] = dqB1_[b]; dqFrom_[2][b] = dqB2_[b];
            dqFrom_[3][b] = dqA1_[b]; dqFrom_[4][b] = dqA2_[b];
        }
    }

    /// From the gains held to the ones just applied in a chunk: the step per
    /// sample, with the gains back where the chunk starts.
    void rampAbsorption(T perSample) noexcept
    {
        const auto ramp = [perSample](auto& now, auto& from, auto& step) {
            for (std::size_t i = 0; i < now.size(); ++i)
            {
                step[i] = (now[i] - from[i]) * perSample;
                now[i] = from[i];
            }
        };
        ramp(jotB0_, absFrom_[0], absStep_[0]); ramp(jotB1_, absFrom_[1], absStep_[1]); ramp(jotA1_, absFrom_[2], absStep_[2]);
        ramp(bassB0_, absFrom_[3], absStep_[3]); ramp(bassB1_, absFrom_[4], absStep_[4]); ramp(bassA1_, absFrom_[5], absStep_[5]);
        for (int q = 0; q < dqCount_; ++q)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
            ramp(dqB0_[b], dqFrom_[0][b], dqStep_[0][b]); ramp(dqB1_[b], dqFrom_[1][b], dqStep_[1][b]);
            ramp(dqB2_[b], dqFrom_[2][b], dqStep_[2][b]); ramp(dqA1_[b], dqFrom_[3][b], dqStep_[3][b]);
            ramp(dqA2_[b], dqFrom_[4][b], dqStep_[4][b]);
        }
    }

    /// One sample of the loop gains' ramp.
    void stepAbsorption() noexcept
    {
        const auto add = [](auto& now, const auto& step) {
            for (std::size_t i = 0; i < now.size(); ++i) now[i] += step[i];
        };
        add(jotB0_, absStep_[0]); add(jotB1_, absStep_[1]); add(jotA1_, absStep_[2]);
        add(bassB0_, absStep_[3]); add(bassB1_, absStep_[4]); add(bassA1_, absStep_[5]);
        for (int q = 0; q < dqCount_; ++q)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
            add(dqB0_[b], dqStep_[0][b]); add(dqB1_[b], dqStep_[1][b]); add(dqB2_[b], dqStep_[2][b]);
            add(dqA1_[b], dqStep_[3][b]); add(dqA2_[b], dqStep_[4][b]);
        }
    }

    /// Per-sample output stage: coherent low band, width, early + late, DC
    /// block, tone EQ.
    std::pair<T, T> outputStage(T lateL, T lateR, T earlyL, T earlyR,
                                T earlyLevel, T lateLevel, T width,
                                T toneHPMix, T toneLPMix, T outAPCoeff) noexcept
    {
        // --- Output: coherent low band, width, early + late, subsonic high-pass, tone ---
        const T lateLvl = lateLevel * kOutGain;
        lateL *= lateLvl;
        lateR *= lateLvl;
        if (!spring_)
        {
            // Output diffusion (the springs' chirps stay crisp).
            T* lr[2] = { &lateL, &lateR };
            const T g = outAPCoeff;
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
            // (the wavelength dwarfs the head) and incoherent above about
            // 500 Hz. The side signal is high-passed (Butterworth-like at
            // kCoherenceHz), which reproduces the interaural coherence of a
            // measured concert hall seat: about 0.97 at 125 Hz, 0.65 at
            // 250 Hz, under 0.1 from 500 Hz up.
            T mid  = (lateL + lateR) * T(0.5);
            T side = (lateL - lateR) * T(0.5);
            const double hp = coherenceFilter_.processSampleCore(static_cast<double>(side), 0);
            side = static_cast<T>(hp * static_cast<double>(width));
            if (cachedParams_.powerKept) mid *= std::sqrt(std::max(T(0), T(2) - width * width));
            lateL = mid + side;
            lateR = mid - side;
        }

        T out[2] = { earlyL * earlyLevel + lateL,
                     earlyR * earlyLevel + lateR };
        for (int c = 0; c < 2; ++c)
            out[c] = subsonicFilter_.processSample(out[c], c);
        if (distMix_ > T(0))
            for (int c = 0; c < 2; ++c)
            {
                T& z = distLp_[static_cast<std::size_t>(c)];
                z += distCoeff_ * (out[c] - z);
                out[c] += distMix_ * (z - out[c]);
            }
        // Tone EQ, crossfaded in and out when it is switched (a settled
        // filter's share is exactly 1 or 0).
        if (toneHPActive_)
            for (int c = 0; c < 2; ++c)
            {
                const T f = toneHPBiquad_.processSample(out[c], c);
                out[c] = toneHPMix == T(1) ? f : out[c] + toneHPMix * (f - out[c]);
            }
        if (toneLPActive_)
            for (int c = 0; c < 2; ++c)
            {
                const T f = toneLPBiquad_.processSample(out[c], c);
                out[c] = toneLPMix == T(1) ? f : out[c] + toneLPMix * (f - out[c]);
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
        // damping_ and erToLateSamples_ are kept current by their setters and
        // by commitPreset(): the audio thread only reads parameter atomics.
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
        const bool geometric = layout_.load(std::memory_order_relaxed) == 1;

        for (int i = 0; i < kMaxLines; ++i)
        {
            // Eco picks every other base delay, so its 16 lines still span
            // the full range (even mode density).
            const int src = eco_ ? std::min(i * (kMaxLines / kEcoLines) + 1, kMaxLines - 1) : i;
            // Geometric layout: the same range, even in log.
            const double baseMs = geometric
                ? kBaseDelaysMs_[0] * std::pow(kBaseDelaysMs_[kMaxLines - 1] / kBaseDelaysMs_[0],
                                               static_cast<double>(src) / (kMaxLines - 1))
                : kBaseDelaysMs_[src];
            lenTarget_[i] = nearestPrime(ms(baseMs * sz));
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
        // Targets: the running coefficients glide to them chunk by chunk
        // (processChunk), and snap with reset().
        const double dist = static_cast<double>(sourceDepth_.load(std::memory_order_relaxed));
        for (int st = 0; st < kInStages; ++st)
            inAPCoeffTarget_[static_cast<std::size_t>(st)] = dist > 0.0
                ? static_cast<T>(std::min(0.8, kInDiffCoeffs_[st] * diff + 0.25 * dist * dist))
                : static_cast<T>(kInDiffCoeffs_[st] * diff);
        loopAPCoeffTarget_ = static_cast<T>(0.075 + 0.175 * diff);
        outAPCoeffTarget_ = static_cast<T>(0.6 * diff);

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
        rotDepth_ = spring_ ? T(0)
                            : modDepth_.load(std::memory_order_relaxed) * static_cast<T>(kRotMaxRad);
        for (int k = 0; k < kMaxLines / 2; ++k)
            rotLfo_[k].setRate(rotRate() * rotRateFactor(k), sr);
        glideCoeff_ = static_cast<T>(1.0 - std::exp(-kCtrl / (kGlideMs * sr / 1000.0)));
        fadeStep_ = 1.0 / (kSizeFadeSeconds * sr);
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
        const double decay = calibratedMidDecay();
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
        double loopSum = 0.0;
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
            if (i < nLines_) loopSum += M;
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
        meanLoopSec_ = loopSum / std::max(1, nLines_) / sr;
        for (int i = 0; i < kMaxLines; ++i)
        {
            const auto l = static_cast<std::size_t>(i);
            jotGoal_[0][l] = jotB0_[l]; jotGoal_[1][l] = jotB1_[l]; jotGoal_[2][l] = jotA1_[l];
            bassGoal_[0][l] = bassB0_[l]; bassGoal_[1][l] = bassB1_[l]; bassGoal_[2][l] = bassA1_[l];
        }
        const auto prevList = dqList_;
        const int prevCount = dqCount_;
        designDecayBands(decay, t60H, t60B, Kh, Kb);
        // The wet signal's energy is that of the tail (the passes round the
        // tank, band by band as heard) plus that of the early field: this
        // gain takes both back, so the wet signal at full mix has the dry
        // signal's level whatever the type, decay, damping and size.
        wetLevel_ = static_cast<T>((spring_ ? kWetCalibrationSpring : kWetCalibration)
            / std::sqrt(std::max(1e-6, heardTailEnergy(decay, t60H, t60B, Kh, Kb) + earlyEnergy())));
        designSecondSlope(decay, t60H, Kh);
        designPlate(decay, t60H, fh);

        // With signal in the tank the loop gains glide to the new design (a
        // step of them is a step of every line's output: a click); an empty
        // tank, after prepare() or once its content is under -120 dB, takes
        // the design at once.
        const bool empty = fresh_ || (holdNow_ == T(0) && quietSamples_ > tankEmptyAfter_);
        double longest = std::max({ decay, t60H, t60B });
        for (int q = 0; q < dqCount_; ++q)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
            longest = std::max(longest, decay * static_cast<double>(getDecayBand(static_cast<int>(b)).multiplier));
            const bool known = std::find(prevList.begin(), prevList.begin() + prevCount, static_cast<int>(b))
                               != prevList.begin() + prevCount;
            if (!known && !empty)
                for (int i = 0; i < kMaxLines; ++i)
                {
                    // A band coming in glides from a flat filter, its state cleared.
                    const auto l = static_cast<std::size_t>(i);
                    dqBase_[0][b][l] = T(1);
                    for (int r = 1; r < 5; ++r) dqBase_[static_cast<std::size_t>(r)][b][l] = T(0);
                    dqZ1_[b][l] = dqZ2_[b][l] = T(0);
                }
        }
        // Content decays 60 dB per T60; the pre-delay and the ER-to-late gap
        // (up to 0.6 s together) hold input back before the tank.
        tankEmptyAfter_ = (2.0 * longest + 0.6) * sr;
        if (empty) snapAbsorption();
        else absorbGlide_ = true;
        applyHoldCoeffs();
    }

    /// The absorption at its design at once (no glide).
    void snapAbsorption() noexcept
    {
        jotBase_ = jotGoal_;
        bassBase_ = bassGoal_;
        for (int q = 0; q < dqCount_; ++q)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
            for (int r = 0; r < 5; ++r)
                dqBase_[static_cast<std::size_t>(r)][b] = dqGoal_[static_cast<std::size_t>(r)][b];
        }
        absorbGlide_ = false;
    }

    /// One chunk of the absorption glide (the ~20 ms one of every parameter);
    /// a first-order shelf and a biquad stay stable along the way, since
    /// their stable coefficient sets are convex.
    void glideAbsorption(T gc) noexcept
    {
        bool moving = false;
        const auto step = [&](T& cur, T goal) {
            cur = glideTo(cur, goal, gc);
            moving = moving || cur != goal;
        };
        for (int r = 0; r < 3; ++r)
            for (int i = 0; i < kMaxLines; ++i)
            {
                const auto rr = static_cast<std::size_t>(r), l = static_cast<std::size_t>(i);
                step(jotBase_[rr][l], jotGoal_[rr][l]);
                step(bassBase_[rr][l], bassGoal_[rr][l]);
            }
        for (int q = 0; q < dqCount_; ++q)
        {
            const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(q)]);
            for (int r = 0; r < 5; ++r)
                for (int i = 0; i < kMaxLines; ++i)
                    step(dqBase_[static_cast<std::size_t>(r)][b][static_cast<std::size_t>(i)],
                         dqGoal_[static_cast<std::size_t>(r)][b][static_cast<std::size_t>(i)]);
        }
        absorbGlide_ = moving;
    }

    /** |H(e^jw)| of ((1 + a) + (a - 1) z^-1) / ((1 + b) + (b - 1) z^-1). */
    [[nodiscard]] static double shelfGain(double a, double b, double w) noexcept
    {
        const double c = std::cos(w), sn = std::sin(w);
        const double nr = (1.0 + a) + (a - 1.0) * c, ni = -(a - 1.0) * sn;
        const double dr = (1.0 + b) + (b - 1.0) * c, di = -(b - 1.0) * sn;
        return std::sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
    }

    /**
     * @brief The energy of the tail for a unit of energy injected, as heard.
     *        At each frequency the first pass leaves each line attenuated by
     *        that line's own loop gain (the short lines first and louder),
     *        and what follows is the geometric series of the mean round trip:
     *        mean(g^2) / (1 - exp(-x)), x = 2 ln(1000) (mean round trip) /
     *        T60, which tends to T60 / (2 ln(1000) round trip) for a long
     *        decay. A spring is its own loop and its first pass is heard
     *        unattenuated: 1 + 1 / (exp(x) - 1). The loop gains are those
     *        just designed (the absorption, the bass shelf and the decay
     *        bands); the frequencies weigh as programme is heard (pink,
     *        through the loudness weighting: the low end out under about
     *        40 Hz and 4 dB more over 2 kHz).
     */
    [[nodiscard]] double heardTailEnergy(double decay, double t60H, double t60B, double Kh, double Kb) const noexcept
    {
        const double sr = spec_.sampleRate, kPi = 3.14159265358979323846;
        constexpr int kPoints = 24;
        const double top = std::min(16000.0, 0.45 * sr);
        double sum = 0.0, weight = 0.0;
        for (int p = 0; p < kPoints; ++p)
        {
            const double f = 31.25 * std::pow(top / 31.25, static_cast<double>(p) / (kPoints - 1));
            const double f2 = f * f, f4 = f2 * f2, hp = f4 / (f4 + 38.0 * 38.0 * 38.0 * 38.0);
            const double heard = hp * (1.0 + 1.51 * f2 / (f2 + 1700.0 * 1700.0));
            const double w = 2.0 * kPi * f / sr;
            double t60 = 0.0, firstPass = 0.0;
            for (int i = 0; i < nLines_; ++i)
            {
                const auto l = static_cast<std::size_t>(i);
                const double M = spring_
                    ? static_cast<double>(lenTarget_[l]) + springStages_ * springK_
                          * (1.0 - static_cast<double>(springA_)) / (1.0 + static_cast<double>(springA_))
                    : static_cast<double>(lenTarget_[l] + loopAPLen_[l]);
                const double gM = std::pow(0.001, M / (decay * sr));
                const double gH = std::min(std::pow(0.001, M / (t60H * sr)), gM);
                const double gB = std::min(std::pow(0.001, M / (t60B * sr)), 0.9995);
                const double rh = std::sqrt(gM / gH), rb = std::sqrt(gB / gM);
                double g = gH * shelfGain(Kh * rh, Kh / rh, w) * shelfGain(Kb * rb, Kb / rb, w);
                for (int k = 0; k < dqCount_; ++k)
                {
                    const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(k)]);
                    BiquadCoeffs c;
                    c.b0 = static_cast<double>(dqGoal_[0][b][l]);
                    c.b1 = static_cast<double>(dqGoal_[1][b][l]);
                    c.b2 = static_cast<double>(dqGoal_[2][b][l]);
                    c.a1 = static_cast<double>(dqGoal_[3][b][l]);
                    c.a2 = static_cast<double>(dqGoal_[4][b][l]);
                    g *= biquadGain(c, w);
                }
                g = std::clamp(g, 1e-9, 0.9999999);
                t60 += -3.0 * M / (sr * std::log10(g));
                firstPass += g * g;
            }
            t60 /= static_cast<double>(std::max(1, nLines_));
            firstPass /= static_cast<double>(std::max(1, nLines_));
            const double x = std::min(2.0 * 6.907755278982137 * meanLoopSec_ / std::max(1e-3, t60), 50.0);
            sum += heard * (spring_ ? 1.0 + 1.0 / std::expm1(x) : firstPass / -std::expm1(-x));
            weight += heard;
        }
        return sum / weight;
    }

    /** @brief The early reflections' window of a type, in ms (none: both zero). */
    static constexpr std::array<double, 2> earlyWindow(Type type) noexcept
    {
        switch (type)
        {
            case Type::Room:      return { 1.5, 35.0 };
            case Type::Hall:      return { 5.0, 110.0 };
            case Type::Chamber:   return { 3.0, 60.0 };
            case Type::Cathedral: return { 10.0, 160.0 };
            case Type::Plate:
            case Type::Spring:    break;
        }
        return { 0.0, 0.0 };
    }

    /**
     * @brief The energy of the early field in the tail's units: the tail's
     *        own energy flow over the early window (see generateERTaps()).
     */
    [[nodiscard]] double earlyEnergy() const noexcept
    {
        const auto window = earlyWindow(erType_);
        if (!(window[1] > window[0]) || meanLoopSec_ <= 0.0) return 0.0;
        const double tau = static_cast<double>(decayTime_.load(std::memory_order_relaxed)) / (2.0 * 6.907755278982137);
        const double tInj = static_cast<double>(erToLateSamples_.load(std::memory_order_relaxed)) / spec_.sampleRate
                            + kMeanInjectMs * 1e-3;
        return tau / meanLoopSec_
             * (std::exp(-(window[0] * 1e-3 - tInj) / tau) - std::exp(-(window[1] * 1e-3 - tInj) / tau));
    }

    /** |H(e^jw)| of a biquad. */
    [[nodiscard]] static double biquadGain(const BiquadCoeffs& q, double w) noexcept
    {
        const double c1 = std::cos(w), s1 = std::sin(w), c2 = std::cos(2.0 * w), s2 = std::sin(2.0 * w);
        const double nr = q.b0 + q.b1 * c1 + q.b2 * c2, ni = -(q.b1 * s1 + q.b2 * s2);
        const double dr = 1.0 + q.a1 * c1 + q.a2 * c2, di = -(q.a1 * s1 + q.a2 * s2);
        return std::sqrt((nr * nr + ni * ni) / std::max(1e-300, dr * dr + di * di));
    }

    /**
     * @brief Designs the decay bands per line (audio thread at the drain, or
     *        prepare()): each band's gain from the line's own loop gain there,
     *        and the boosts scaled down together where they would lift the
     *        loop over kMaxDecayLoopGain.
     */
    void designDecayBands(double decay, double t60H, double t60B, double Kh, double Kb) noexcept
    {
        const double sr = spec_.sampleRate;
        dqCount_ = 0;
        std::array<DecayBand, kDecayBands> bands {};
        for (int k = 0; k < kDecayBands; ++k)
        {
            const DecayBand b = getDecayBand(k);
            if (b.shape == DecayShape::Off || std::abs(static_cast<double>(b.multiplier) - 1.0) < 1e-3) continue;
            bands[static_cast<std::size_t>(dqCount_)] = b;
            dqList_[static_cast<std::size_t>(dqCount_++)] = k;
        }
        if (dqCount_ == 0) return;

        constexpr int kGrid = 32;
        std::array<double, kGrid> grid {};
        for (int g = 0; g < kGrid; ++g)
            grid[static_cast<std::size_t>(g)] = 2.0 * 3.14159265358979323846
                * 20.0 * std::pow(0.45 * sr / 20.0, static_cast<double>(g) / (kGrid - 1)) / sr;
        for (int i = 0; i < kMaxLines; ++i)
        {
            const double M = spring_
                ? static_cast<double>(lenTarget_[i]) + springStages_ * springK_
                      * (1.0 - static_cast<double>(springA_)) / (1.0 + static_cast<double>(springA_))
                : static_cast<double>(lenTarget_[i] + loopAPLen_[i]);
            const double gM = std::pow(0.001, M / (decay * sr));
            const double gH = std::min(std::pow(0.001, M / (t60H * sr)), gM);
            const double gB = std::min(std::pow(0.001, M / (t60B * sr)), 0.9995);
            const double rh = std::sqrt(gM / gH), rb = std::sqrt(gB / gM);
            const auto loopGain = [&](double w) { return gH * shelfGain(Kh * rh, Kh / rh, w) * shelfGain(Kb * rb, Kb / rb, w); };

            std::array<double, kDecayBands> gains {};
            for (int k = 0; k < dqCount_; ++k)
            {
                const DecayBand& b = bands[static_cast<std::size_t>(k)];
                const double ref = b.shape == DecayShape::LowShelf ? gB
                                 : b.shape == DecayShape::HighShelf ? gH
                                 : loopGain(2.0 * 3.14159265358979323846 * static_cast<double>(b.frequency) / sr);
                gains[static_cast<std::size_t>(k)] = 20.0 * std::log10(std::clamp(ref, 1e-6, 0.999999))
                                                   * (1.0 / static_cast<double>(b.multiplier) - 1.0);
            }
            std::array<BiquadCoeffs, kDecayBands> coeffs {};
            const auto design = [&](double boostScale) {
                for (int k = 0; k < dqCount_; ++k)
                {
                    const DecayBand& b = bands[static_cast<std::size_t>(k)];
                    double g = gains[static_cast<std::size_t>(k)];
                    if (g > 0.0) g *= boostScale;
                    const double f = std::min(static_cast<double>(b.frequency), 0.45 * sr);
                    coeffs[static_cast<std::size_t>(k)] =
                          b.shape == DecayShape::LowShelf ? BiquadCoeffs::makeLowShelf(sr, f, g)
                        : b.shape == DecayShape::HighShelf ? BiquadCoeffs::makeHighShelf(sr, f, g)
                        : BiquadCoeffs::makePeak(sr, f, static_cast<double>(b.q), g);
                }
            };
            const auto peakLoop = [&]() {
                double top = 0.0;
                for (double w : grid)
                {
                    double v = loopGain(w);
                    for (int k = 0; k < dqCount_; ++k) v *= biquadGain(coeffs[static_cast<std::size_t>(k)], w);
                    top = std::max(top, v);
                }
                return top;
            };
            design(1.0);
            if (peakLoop() > kMaxDecayLoopGain)
            {
                double lo = 0.0, hi = 1.0;
                for (int it = 0; it < 12; ++it)
                {
                    const double mid = 0.5 * (lo + hi);
                    design(mid);
                    (peakLoop() > kMaxDecayLoopGain ? hi : lo) = mid;
                }
                design(lo);
            }
            for (int k = 0; k < dqCount_; ++k)
            {
                const auto b = static_cast<std::size_t>(dqList_[static_cast<std::size_t>(k)]);
                const auto l = static_cast<std::size_t>(i);
                const BiquadCoeffs& c = coeffs[static_cast<std::size_t>(k)];
                dqGoal_[0][b][l] = static_cast<T>(c.b0);
                dqGoal_[1][b][l] = static_cast<T>(c.b1);
                dqGoal_[2][b][l] = static_cast<T>(c.b2);
                dqGoal_[3][b][l] = static_cast<T>(c.a1);
                dqGoal_[4][b][l] = static_cast<T>(c.a2);
            }
        }
    }

    /** @brief The pre-delay in ms: the note value's time at the tempo, or the one set. */
    [[nodiscard]] double effectivePreDelayMs() const noexcept
    {
        const int note = preNote_.load(std::memory_order_relaxed);
        if (note == 0) return static_cast<double>(preDelayMs_.load(std::memory_order_relaxed));
        return std::min(noteMs(note, preFeel_.load(std::memory_order_relaxed)), kMaxPreDelayMs);
    }

    void refreshPreDelay() noexcept
    {
        if (spec_.sampleRate > 0)
            preDelaySamples_.store(msToSamples(static_cast<T>(effectivePreDelayMs())), std::memory_order_relaxed);
    }

    void refreshGateHold() noexcept
    {
        const int note = gateNote_.load(std::memory_order_relaxed);
        const double ms = note == 0 ? static_cast<double>(gateHoldMs_.load(std::memory_order_relaxed))
                                    : std::min(noteMs(note, gateFeel_.load(std::memory_order_relaxed)), 4000.0);
        tail_.setGateHold(static_cast<T>(ms));
    }

    /** @brief A note value's time at the tempo, in ms. */
    [[nodiscard]] double noteMs(int note, int feel) const noexcept
    {
        static constexpr double kBeats[7] = { 0.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125 };
        const double beat = 60000.0 / static_cast<double>(tempo_.load(std::memory_order_relaxed));
        const double factor = feel == 1 ? 1.5 : feel == 2 ? 2.0 / 3.0 : 1.0;
        return beat * kBeats[static_cast<std::size_t>(std::clamp(note, 0, 6))] * factor;
    }

    /**
     * @brief The loop's DC decay that puts the ISO 3382 mid-frequency
     *        reverberation time on the decay setting.
     *
     * setDecay() promises T_mid, the mean of the 500 Hz and 1 kHz octave
     * T60s (ISO 3382-1, the number a room is quoted by). The absorption
     * shelves are anchored at DC and at Nyquist, and a first-order shelf
     * spans several octaves: the HF shelf already shortens 1 kHz and the bass
     * shelf still lengthens 500 Hz. With the DC anchor at the setting, T_mid
     * measured 11.7% short on Cathedral and 3 to 9% short elsewhere. So the
     * DC anchor is solved for instead: the loop T60 at 500 Hz and 1 kHz is
     * evaluated from the exact shelf responses at the mean loop length, and
     * the DC decay is rescaled until their mean equals the setting (the HF
     * and bass targets stay proportional to it, as the multipliers say).
     * Four fixed-point steps converge to far below a percent.
     */
    [[nodiscard]] double calibratedMidDecay() const noexcept
    {
        const double sr = spec_.sampleRate;
        const double target = static_cast<double>(decayTime_.load(std::memory_order_relaxed));
        if (!(sr > 0.0) || nLines_ < 1) return target;
        const double hd = static_cast<double>(highDecayMult_.load(std::memory_order_relaxed));
        const double bd = static_cast<double>(bassDecayMult_.load(std::memory_order_relaxed));
        const double kPi = 3.14159265358979323846;
        const double Kh = std::tan(kPi * std::clamp(static_cast<double>(
            highCrossover_.load(std::memory_order_relaxed)), 100.0, 0.45 * sr) / sr);
        const double Kb = std::tan(kPi * std::clamp(static_cast<double>(
            bassCrossover_.load(std::memory_order_relaxed)), 10.0, 0.45 * sr) / sr);
        auto loopLen = [&](int i) {
            return spring_
                ? static_cast<double>(lenTarget_[i]) + springStages_ * springK_
                      * (1.0 - static_cast<double>(springA_)) / (1.0 + static_cast<double>(springA_))
                : static_cast<double>(lenTarget_[i] + loopAPLen_[i]);
        };

        // |H(e^jw)| of ((1 + a) + (a - 1) z^-1) / ((1 + b) + (b - 1) z^-1).
        auto shelf = [](double a, double b, double w) {
            const double c = std::cos(w), sn = std::sin(w);
            const double nr = (1.0 + a) + (a - 1.0) * c, ni = -(a - 1.0) * sn;
            const double dr = (1.0 + b) + (b - 1.0) * c, di = -(b - 1.0) * sn;
            return std::sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
        };
        // Decay rate (1/T60) of the line mix at hz: each line's shelves are
        // designed from its own length, so the rate between the anchors
        // differs from line to line; the tail decays at their mean rate.
        auto t60At = [&](double d, double hz) {
            const double w = 2.0 * kPi * hz / sr;
            double rate = 0.0;
            int used = 0;
            for (int i = 0; i < nLines_; ++i)
            {
                const double M = loopLen(i);
                if (!(M > 0.0)) continue;
                const double gM = std::pow(0.001, M / (d * sr));
                const double gH = std::min(std::pow(0.001, M / (std::max(0.05, d * hd) * sr)), gM);
                const double gB = std::min(std::pow(0.001, M / (std::max(0.05, d * bd) * sr)), 0.9995);
                const double rh = std::sqrt(gM / gH), rb = std::sqrt(gB / gM);
                const double g = gH * shelf(Kh * rh, Kh / rh, w) * shelf(Kb * rb, Kb / rb, w);
                if (!(g > 0.0 && g < 1.0)) continue;
                rate += -sr * std::log10(g) / (3.0 * M);
                ++used;
            }
            return (used > 0 && rate > 0.0) ? used / rate : d;
        };
        double d = target;
        for (int it = 0; it < 4; ++it)
        {
            const double tm = 0.5 * (t60At(d, 500.0) + t60At(d, 1000.0));
            if (!(tm > 0.0)) break;
            d *= target / tm;
        }
        return std::clamp(d, 0.05, 120.0);
    }

    // --- Early reflection generation -----------------------------------------

    /// Regenerates the early-reflection taps for a reverb type (Eco caps the count).
    void generateERTapsForType(Type type) noexcept
    {
        const int cap = eco_ ? kEcoERTaps : kMaxERTaps;
        switch (type)
        {
            case Type::Room:
            case Type::Hall:
            case Type::Chamber:
            case Type::Cathedral:
            {
                const auto window = earlyWindow(type);
                generateERTaps(window[0], window[1], cap);
                break;
            }
            case Type::Plate:
            case Type::Spring:    generateERTaps(0.0, 0.0, 0);                   break;
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
        numERTaps_ = std::clamp(numTaps, 0, kMaxERTaps) / 2 * 2;
        numERSlots_ = 0;
        for (auto& side : erGroupStart_) side.fill(0);
        if (numERTaps_ == 0) return;

        const double sr = spec_.sampleRate;
        const int R = numERTaps_ / 2;   // reflections per input channel
        constexpr double p = 1.6;
        // Diffusion sets how many of the first reflections stay discrete
        // (crisp, room-like) instead of reading the diffused feed (smooth).
        const int discrete = 1 + static_cast<int>(std::lround(
            (kMaxDiscreteER - 1) * (1.0 - static_cast<double>(diffusion_.load(std::memory_order_relaxed)))));

        const double decay = static_cast<double>(decayTime_.load(std::memory_order_relaxed));
        constexpr double kLn1000 = 6.907755278982137;
        // One record per reflection; a candidate for the discrete set keeps
        // two slots (raw and diffused feed), weighted by `raw`.
        struct Tap { double ms; int chan; int bin; double gain; bool candidate; bool raw; };
        std::array<std::array<Tap, kMaxERTaps>, 2> taps {};
        std::array<int, 2> used { 0, 0 };
        for (int c = 0; c < 2; ++c)   // input channel: a source on that side
        {
            std::array<double, kMaxERTaps / 2> gv {}, msv {}, lat {};
            double dcSum = 0.0;
            for (int k = 0; k < R; ++k)
            {
                const double u = (k + 0.15 + 0.7 * hash01(k, c + 20)) / R;
                // Warped time in [0, 1): a floor of uniform density plus a
                // share rising like t^(p - 1).
                const double w = 0.3 * u + 0.7 * std::pow(u, 1.0 / p);
                msv[k] = minMs + (maxMs - minMs) * w;
                const double density = 1.0 / (0.3 + 0.7 / p * std::pow(std::max(u, 1e-3), 1.0 / p - 1.0));   // dk/dw
                const double sign = k < 4 || hash01(k, c + 40) < 0.5 ? 1.0 : -1.0;
                // Energy per unit time follows the room's own decay (the
                // late field's rate, so the two join without a swell or a
                // dip at any decay time): sparse taps are individually
                // louder.
                gv[k] = sign * std::exp(-kLn1000 * msv[k] * 1e-3 / decay) / std::sqrt(density);
                if (k >= discrete) dcSum += gv[k];
                // Arrival direction as the lateral coordinate (sine of the
                // lateral angle, uniform over a diffuse field), positive
                // toward the source's side: the first two (floor and
                // ceiling) share the source's own direction (30 degrees, a
                // hard-panned input), the next come mostly from the source's
                // side of the room, the later ones from everywhere.
                const double side = hash01(k, c + 50) < 0.92 - 0.42 * w ? 1.0 : -1.0;
                lat[k] = k < 2 ? 0.5 : side * (0.25 + 0.75 * hash01(k, c + 60));
            }
            // No DC: the early field's low end is left to the (coherent) late
            // field, and the output high-pass never has to drain a step.
            if (R > discrete)
                for (int k = discrete; k < R; ++k) gv[k] -= dcSum / (R - discrete);
            double energy = 0.0;
            for (int k = 0; k < R; ++k) energy += gv[k] * gv[k];
            // The early field joins the late one on a single exponential:
            // the late field's energy flow at time t after its injection is
            // exp(-t / tau) / (mean round trip) (the output taps follow the
            // absorption, so even its first echoes sit on that curve), and
            // the early reflections carry that same density over their
            // window. Half per input, so a mono source keeps it.
            const double tau = decay / (2.0 * kLn1000);
            const double tInj = static_cast<double>(erToLateSamples_.load(std::memory_order_relaxed)) / sr
                                + kMeanInjectMs * 1e-3;
            const double target = static_cast<double>(kOutGain) * static_cast<double>(kOutGain)
                                  * tau / meanLoopSec_
                                  * (std::exp(-(minMs * 1e-3 - tInj) / tau) - std::exp(-(maxMs * 1e-3 - tInj) / tau));
            const double norm = std::sqrt(0.5 * target / energy);

            // Every reflection reaches both outputs like both ears: the far
            // one later (Woodworth ITD, up to 0.66 ms), quieter (up to
            // -6 dB) and, beyond a lateral angle of 17 degrees, duller (half
            // its highs); the near one up to 1.6 dB louder.
            for (int e = 0; e < 2; ++e)
            {
                for (int k = 0; k < R; ++k)
                {
                    const double x = e == c ? lat[k] : -lat[k];   // toward this ear
                    const double phi = std::asin(std::min(1.0, std::abs(x)));
                    const double itdMs = x < 0.0 ? 1000.0 * 0.0875 / 343.0 * (phi + std::sin(phi)) : 0.0;
                    const int bin = x < -0.3 ? 1 : 0;
                    const double ild = x > 0.0 ? 1.0 + 0.2 * x : 1.0 + 0.5 * x;
                    taps[e][used[e]++] = { msv[k] + itdMs, e == c ? 0 : 1, bin, gv[k] * norm * ild,
                                           k < kMaxDiscreteER, k < discrete };
                }
            }
        }
        for (int e = 0; e < 2; ++e)
        {
            // Time order, so the absorption groups run from early to late.
            std::sort(taps[e].begin(), taps[e].begin() + used[e],
                      [](const Tap& a, const Tap& b) { return a.ms < b.ms; });
            // Slots in time order. The absorption groups split the
            // reflections (not the slots) evenly, as before the crossfade
            // slots existed, so a crossfade slot shares its reflection's group.
            int slot = 0, group = 1;
            erGroupStart_[e][0] = 0;
            for (int k = 0; k < used[e]; ++k)
            {
                while (group < kERGroups && k == used[e] * group / kERGroups)
                    erGroupStart_[e][group++] = slot;
                const Tap& t = taps[e][k];
                const int lag = std::max(1, static_cast<int>(t.ms * sr / 1000.0));
                const auto put = [&](int chan, double gain) {
                    erTap_[e][slot] = lag;
                    erChan_[e][slot] = chan;
                    erBin_[e][slot] = t.bin;
                    erGainTarget_[e][slot] = static_cast<T>(gain);
                    ++slot;
                };
                if (t.candidate)
                {
                    put(t.chan + 2, t.raw ? t.gain : 0.0);   // raw input: discrete
                    put(t.chan, t.raw ? 0.0 : t.gain);       // diffused feed
                }
                else
                {
                    put(t.chan, t.gain);
                }
            }
            while (group <= kERGroups) erGroupStart_[e][group++] = slot;
            numERSlots_ = slot;   // the same on both sides
        }

        // Each absorption group loses its highs as the late field does by
        // the group's mean arrival time: 60 dB per HF T60 beyond the mid band.
        const double fh = std::clamp(static_cast<double>(highCrossover_.load(std::memory_order_relaxed)),
                                     100.0, 0.45 * sr);
        erShelfCoeff_ = static_cast<T>(1.0 - std::exp(-6.283185307179586 * fh / sr));
        const double t60H = std::max(0.05, calibratedMidDecay()
                                             * static_cast<double>(highDecayMult_.load(std::memory_order_relaxed)));
        for (int g = 0; g < kERGroups; ++g)
        {
            double tSum = 0.0;
            int cnt = 0;
            // Mean arrival time of the group's reflections (each counted
            // once, whatever its slots).
            for (int e = 0; e < 2; ++e)
            {
                int last = -1;
                for (int k = erGroupStart_[e][g]; k < erGroupStart_[e][g + 1]; ++k)
                {
                    const bool second = k > erGroupStart_[e][g] && erTap_[e][k] == last
                                        && erChan_[e][k] + 2 == erChan_[e][k - 1];
                    last = erTap_[e][k];
                    if (second) continue;
                    tSum += erTap_[e][k] / sr;
                    ++cnt;
                }
            }
            const double t = cnt > 0 ? tSum / cnt : 0.0;
            erShelfGainTarget_[g] = static_cast<T>(std::pow(10.0, -3.0 * t * std::max(0.0, 1.0 / t60H - 1.0 / decay)));
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

    /// Writes the preset's values into the parameter atomics, skipping every
    /// one whose user bit is set. Control or setup thread only: the audio
    /// thread never writes parameter atomics, so nothing it does can override
    /// a setter that follows setType().
    void commitPreset(const PresetValues& p) noexcept
    {
        const uint32_t m = userParamMask_.load(std::memory_order_acquire);
        if (!(m & kUserSize))      size_.store(p.size, std::memory_order_relaxed);
        if (!(m & kUserDecay))     decayTime_.store(p.decay, std::memory_order_relaxed);
        if (!(m & kUserDamping))
        {
            highDecayMult_.store(p.hdMult, std::memory_order_relaxed);
            damping_.store(std::clamp((T(1) - p.hdMult) / T(0.9), T(0), T(1)),
                           std::memory_order_relaxed);
        }
        if (!(m & kUserBassDecay)) bassDecayMult_.store(p.bdMult, std::memory_order_relaxed);
        if (!(m & kUserHighXover)) highCrossover_.store(p.hxover, std::memory_order_relaxed);
        if (!(m & kUserBassXover)) bassCrossover_.store(p.bxover, std::memory_order_relaxed);
        if (!(m & kUserDiffusion)) diffusion_.store(p.diff, std::memory_order_relaxed);
        if (!(m & kUserModDepth))  modDepth_.store(p.modDepth, std::memory_order_relaxed);
        if (!(m & kUserModRate))   modRate_.store(p.modRate, std::memory_order_relaxed);
        if (!(m & kUserEarly))     earlyLevel_.store(p.earlyLvl, std::memory_order_relaxed);
        if (!(m & kUserLate))      lateLevel_.store(p.lateLvl, std::memory_order_relaxed);
        if (!(m & kUserErToLate))  erToLateMs_.store(p.erToLate, std::memory_order_relaxed);
        if (spec_.sampleRate > 0)
            erToLateSamples_.store(msToSamples(erToLateMs_.load(std::memory_order_relaxed)),
                                   std::memory_order_relaxed);
    }

    /// The parameter values of each preset.
    static PresetValues presetValues(Type type) noexcept
    {
        //               size     decay    hdMult   bdMult   hxover   bxover
        //               diff     modDep   modRate  early    late     erToLate
        switch (type)
        {
            case Type::Room:
                return { T(0.22), T(0.5),  T(0.40), T(1.1),  T(5000), T(250),
                         T(0.72), T(0.07), T(0.5),  T(1),    T(0.8),  T(0) };
            case Type::Hall:
                return { T(0.68), T(2.2),  T(0.32), T(1.3),  T(4500), T(200),
                         T(0.84), T(0.10), T(0.55), T(1),    T(1),    T(15) };
            case Type::Chamber:
                return { T(0.38), T(1.2),  T(0.38), T(1.1),  T(5000), T(250),
                         T(0.78), T(0.08), T(0.6),  T(1),    T(0.9),  T(8) };
            case Type::Plate:
                return { T(0.14), T(1.5),  T(0.55), T(0.8),  T(7000), T(150),
                         T(0.94), T(0.16), T(1.4),  T(0),    T(1),    T(0) };
            case Type::Spring:
                return { T(0.11), T(0.9),  T(0.28), T(1.0),  T(4000), T(200),
                         T(0.6),  T(0.08), T(0.35), T(0.5),  T(1),    T(0) };
            case Type::Cathedral:
                break;
        }
        return { T(0.98), T(5.0),  T(0.24), T(1.5),  T(3500), T(150),
                 T(0.91), T(0.12), T(0.35), T(1),    T(1),    T(25) };
    }

    /// Engine structure of a preset: spring tanks or FDN, its early field,
    /// and the coefficients (audio thread at the drain, or prepare()).
    void applyTopology(Type type) noexcept
    {
        spring_ = type == Type::Spring;
        erType_ = type;
        refreshTopology();
        if (prepared_)
            updateAll();
    }

    /// Setup-thread preset load: values (keeping the user's overrides), then
    /// the engine.
    void applyPreset(Type type) noexcept
    {
        commitPreset(presetValues(type));
        applyTopology(type);
    }
};

} // namespace dspark
