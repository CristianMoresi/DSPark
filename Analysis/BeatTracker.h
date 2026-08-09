// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file BeatTracker.h
 * @brief Beat and tempo tracking: offline dynamic programming plus a causal
 *        resonator bank, over the shared onset-strength envelope.
 *
 * Two paths over one front end. Both read the onset-strength envelope of
 * Analysis/OnsetDetector.h -- the continuous curve, not the picked events --
 * so the two paths and the onset detector agree by construction on what a
 * transient is and where it sits.
 *
 * OFFLINE (analyze()). Dynamic-programming beat tracking after Ellis, "Beat
 * Tracking by Dynamic Programming", Journal of New Music Research 36(1), 2007.
 * The beat sequence maximising
 *
 *     C*(t) = O(t) + max_tau { alpha * F(t - tau, tau_p) + C*(tau) },
 *     F(dt, tau_p) = -(log(dt / tau_p))^2
 *
 * is recovered by backtracking, where O is the conditioned envelope, tau_p is
 * the target beat period and alpha is the tightness. The logarithm is natural,
 * and F is symmetric on log-time: an interval k times too long costs exactly
 * what one k times too short costs.
 *
 * The target period is a single number while the METRICAL LEVEL is being
 * decided, and a per-frame number once it has been. One global period is the
 * right model for the first question because it is the stable one, and the
 * wrong model for laying a grid on music whose tempo moves, because the
 * interval cost would then measure the distance from the average rather than
 * from the tempo being played. The per-frame period comes from sweeping the
 * causal path's own resonator bank over the envelope twice, once each way, and
 * combining: one direction carries the filter's memory as a lag, the other
 * carries the mirror image of that lag, and the two together are centred on
 * the frame they describe. That is only possible offline, and it is why the
 * offline path can follow a tempo the causal path can only chase. Measured on
 * a 100-to-140 BPM ramp, it takes the worst local tempo error from 40.2% to
 * 0.18% and the grid F-measure from 0.725 to 0.994.
 *
 * Two further stages sit between the recursion and the returned grid, and both
 * exist because of where the recursion is weak rather than because they sound
 * like a good idea. The recursion has to guess the phase of its first links
 * before it has accumulated any evidence, so its earliest beats can sit a
 * frame or two off a transient that is plainly visible: the grid is therefore
 * snapped to nearby envelope peaks, within a window too narrow to reach an
 * off-beat, and then continued off both ends for as long as the envelope keeps
 * supporting it. Together they take the grid F-measure on clean clicks from
 * 0.9315 to 1.0000 and the worst beat error from 29.1 ms to 5.4 ms.
 *
 * TIGHTNESS. The default is 400 -- the value the paper names as its default
 * (Figure 5 caption) and the one its author's reference implementation
 * hard-codes (beat2.m line 56, beatdyn.m line 56). It is a default, not a
 * constant of nature, and the published values disagree with each other by
 * most of an order of magnitude: the same paper's best score on its own tuning
 * set is at alpha = 680 (section 4.2), while librosa shipped 400 through 0.3.x
 * and 100 from 0.4.0 onward. Both use the natural logarithm and the same form,
 * so those numbers are directly comparable and the disagreement is real rather
 * than a change of units. What that disagreement is worth here is measured
 * rather than assumed, and the answer is: very little. See setTightness().
 *
 * CAUSAL (processBlock()). A bank of one-pole complex resonators over the same
 * envelope -- baseline-removed the same way, by a trailing mean rather than a
 * centred one -- one per candidate period, geometrically spaced at 128 per
 * octave.
 * Resonator tau accumulates
 *
 *     z[n] = lambda(tau) * exp(i * 2*pi/tau) * z[n-1] + o[n],
 *
 * so |z| is the envelope's exponentially-windowed Fourier magnitude at the
 * pulse frequency and arg(z) is that pulse's phase -- the comb filter and the
 * local-pulse phase estimate are the same object read two ways. lambda gives
 * every resonator the same memory in BEATS (kIntegrationBeats of its own
 * period), so slow and fast candidates are compared over equal musical spans
 * rather than equal wall-clock spans. Beats are emitted where the winning
 * resonator's phase completes a turn. No backtracking, no future context.
 *
 * COHERENCE, and what the confidence number means. For a candidate period tau
 * the tracker measures
 *
 *     C(tau) = |sum_n o[n] * exp(-i*2*pi*n/tau)| / sum_n o[n]      in [0, 1],
 *
 * the fraction of the envelope's mass that is in phase with a pulse at that
 * period. It is 1 when every bit of onset strength falls on the pulse and 0
 * when the strength is spread evenly in phase. Under onset jitter of standard
 * deviation s it takes the closed form exp(-2*pi^2*s^2/tau^2) -- the
 * characteristic function of the jitter distribution -- so the number can be
 * checked against theory rather than only against itself. Measured against
 * that closed form on a 120 BPM pulse jittered from 0 to 40 ms, the worst
 * disagreement over the sweep is 0.011.
 *
 * The reported confidence is C at the delivered tempo, reduced only where a
 * different metrical level explained the signal nearly as well (see THE
 * SECONDARY HYPOTHESIS). It answers how much of what the signal did is
 * explained by the grid that was returned. It is NOT a probability that the
 * tempo is right.
 *
 * THE OCTAVE, which is the error that matters. Half and double tempo explain a
 * pulse train almost as well as the truth does, and a tolerance band will
 * absorb the confusion instead of reporting it. C separates the two directions
 * because it is not symmetric between them: at HALF the true tempo alternate
 * pulses land in antiphase and C collapses, while at DOUBLE the tempo every
 * true pulse still lands in phase and C stays high. So C rejects half by
 * itself and cannot reject double at all, and the discriminator both paths use
 * is
 *
 *     score(tau) = W(tau) * C(tau) * prod_m (1 - C(m*tau)),   m in {2, 3},
 *
 * where each extra factor asks whether the candidate's own subharmonic is just
 * as coherent as the candidate: if it is, the candidate is a harmonic of the
 * real pulse and the real pulse is the slower one. Two and three between them
 * catch every integer multiple up to four, and they are exactly the relations
 * a tempo evaluation calls a metrical-level confusion. Leaving the m = 3
 * factor out is not an academic point: on a 60 BPM pulse the causal path then
 * locks to 180 BPM and reports a confidence of 0.994 while doing it. W is the
 * log-Gaussian tempo prior of the same paper (eq. 6), centred on 0.5 s per
 * beat with a width of 1.4 octaves; it is the weakest factor on purpose, being
 * the only one that does not look at the signal.
 *
 * THE SECONDARY HYPOTHESIS is a competitor, not a label. It is the
 * best-scoring candidate at least a quarter of an octave from the winner --
 * the metrical level a listener could plausibly have tapped instead -- and it
 * is 0 when the envelope offers no such alternative inside the searched range,
 * which is the honest answer for a pulse near the bottom of that range. It
 * does two things. It is reported, so a caller that disagrees about the level
 * has somewhere to go. And it discounts the confidence when it scores close to
 * the winner, because a grid that fits is not the same claim as a grid that is
 * the only one that fits: on a three-against-two polyrhythm, where both pulses
 * are real, that takes the reported confidence from 0.52 to between 0.05 and
 * 0.40, while leaving every unambiguous case in the acceptance corpus
 * untouched. See kAmbiguityFloor for why it has a floor under it.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with anything else).
 * - processBlock() / pushSamples(): audio thread, stream owner. Allocation
 *   free, lock free, no throw.
 * - getRunningTempoBpm() / getConfidence(): any thread, lock free. The two
 *   travel as ONE packed 64-bit word because they are a set: a confidence
 *   belongs to the tempo it was measured at, and a caller that gates on
 *   confidence before trusting a tempo must never see one of them refreshed
 *   without the other. Read them together with getTempoAndConfidence() when
 *   both are wanted; the two separate getters unpack the same single load and
 *   so are consistent individually, but two separate calls can straddle an
 *   update.
 * - beatNow() / getLastBeatSample(): any thread, lock free, two independent
 *   words. They can only disagree in one direction and it is harmless: both
 *   are written in the same processing call, so a reader that catches the
 *   latch of one call and the sample of the next is handed a LATER real beat,
 *   never a stale one and never a position that was not a beat.
 * - setTempoRange() / setTightness(): control thread, lock free. The range is
 *   two indices with an invariant between them, so it travels as one packed
 *   word as well; setTempoRange() never allocates, because the resonator bank
 *   is built once at prepare() over the whole supported span and the range
 *   only selects which part of it is searched.
 * - analyze() / reset(): setup thread. analyze() allocates and CLEARS the
 *   causal state, so it must not run while the audio path is running.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O, so it is unaffected by DSPARK_NO_FILE_IO.
 *
 * Dependencies: DspMath.h, AudioBuffer.h, AudioSpec.h, OnsetDetector.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "OnsetDetector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class BeatTracker
 * @brief Tempo and beat tracking with an offline grid and a causal readout.
 *
 * Role: analysis readout. It consumes const audio and never mutates it. All
 * heap use for the causal path happens in prepare(); processBlock() allocates
 * nothing, takes no lock and throws nothing. analyze() is offline and does
 * allocate, once per call, before its inner loops.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class BeatTracker final
{
public:
    /** @brief What analyze() returns: one tempo, one grid, one number saying
     *         how much of the signal that grid explains, and the metrical
     *         alternative that lost. */
    struct Result
    {
        T tempoBpm = T(0);           ///< Delivered tempo, fitted to the grid.
        T confidence = T(0);         ///< Coherence of the envelope with the grid, [0,1].
        T secondaryTempoBpm = T(0);  ///< The competing hypothesis that lost.
        std::vector<int64_t> beatSamples; ///< Beat positions, ascending, in samples.
    };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Allocates all causal state and prepares the shared front end.
     *
     * Not real-time safe. An invalid spec (non-finite or non-positive sample
     * rate) is ignored and the previous configuration is preserved.
     *
     * The resonator bank is built once here, over the whole supported tempo
     * span (kBankMinBpm..kBankMaxBpm) rather than over the range in force, so
     * that setTempoRange() can stay allocation free and callable from the
     * control thread while audio runs. The span reaches an octave BELOW the
     * slowest searchable tempo on purpose: the octave discriminator needs the
     * candidate's own subharmonic, which is a resonator that must exist.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!(spec.sampleRate > 0.0) || !std::isfinite(spec.sampleRate))
            return;

        prepared_.store(false, std::memory_order_relaxed);

        sampleRate_ = spec.sampleRate;
        onset_.prepare(spec);
        hop_ = onset_.getHopSize();
        if (hop_ < 1) hop_ = 1;
        warmupFrames_ = onset_.getWarmupFrames();
        frameRate_ = sampleRate_ / static_cast<double>(hop_);
        baselineCoef_ = 1.0 - std::exp(-1.0 / std::max(1.0, kBaselineSeconds * frameRate_));

        latencySamples_.store(static_cast<int64_t>(onset_.getEnvelopeLatencySamples())
                                  + static_cast<int64_t>(hop_),
                              std::memory_order_relaxed);

        buildBank();
        setTempoRange(T(kDefaultMinBpm), T(kDefaultMaxBpm));
        tightness_.store(kDefaultTightness, std::memory_order_relaxed);

        // Offline scratch that does not depend on the signal length.
        candidates_.reserve(64);

        resetState();
        prepared_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Restricts the searched tempo range, in BPM. Lock free, no alloc.
     *
     * Clamped into the bank's supported span and to min <= max; a non-finite
     * or inverted request is ignored and the previous range kept. Takes effect
     * on the next analysis frame for the causal path and on the next call for
     * analyze().
     */
    void setTempoRange(T minBpm, T maxBpm) noexcept
    {
        if (!std::isfinite(minBpm) || !std::isfinite(maxBpm)) return;
        if (!(minBpm > T(0)) || !(maxBpm > T(0)) || !(minBpm <= maxBpm)) return;
        if (bankSize_ < 2) return;

        const double lo = std::clamp(static_cast<double>(minBpm),
                                     kBankMinBpm, kBankMaxBpm);
        const double hi = std::clamp(static_cast<double>(maxBpm),
                                     kBankMinBpm, kBankMaxBpm);

        // Bank index runs with PERIOD, so the fastest tempo is the lowest
        // index: the high BPM bound selects the low index bound.
        int iLo = bpmToIndex(hi);
        int iHi = bpmToIndex(lo);
        iLo = std::clamp(iLo, 0, bankSize_ - 1);
        iHi = std::clamp(iHi, 0, bankSize_ - 1);
        if (iLo > iHi) std::swap(iLo, iHi);

        // The two indices are a set with an invariant (lo <= hi), so they are
        // published as one word: half of one range combined with half of the
        // next is a range nobody asked for, and it would be searched.
        activeRange_.store(packRange(iLo, iHi), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dynamic-programming tightness (Ellis alpha).
     *
     * Weight of the interval-consistency term against the onset evidence:
     * larger holds the grid closer to the estimated period, smaller lets it
     * follow the envelope. Offline path only -- the causal path has no
     * transition cost to weigh. Non-finite values are ignored; values are
     * clamped to a positive floor because a tightness of zero removes the
     * term the recursion is built on and leaves the grid free to place a beat
     * at every envelope peak.
     *
     * Default 400. Measured sensitivity on this implementation, swept over
     * alpha = 25, 50, 100, 200, 400, 680, 1000, 2000, 4000:
     *
     * - Clean 120 BPM clicks and the same track jittered by 20 ms: the grid is
     *   BIT-IDENTICAL across the whole sweep. F-measure 1.0000 and worst beat
     *   error 4.72 ms (clean) / 4.97 ms (jittered) at every value.
     * - A 100-to-140 BPM ramp is the only case that moves, and it moves the
     *   way the term predicts: F 1.0000 for alpha <= 100, 0.9939 for 200 to
     *   1000, 0.9877 for 2000 and above. One beat in 81 across a factor of
     *   forty in alpha.
     *
     * So on this corpus the tightness that scores best is the LOOSEST end,
     * which is not the shipped default. The default stays at 400 because that
     * is the value the algorithm's source names and ships, and because the
     * whole measured spread is a single beat; a default chosen instead to win
     * one synthetic ramp by a fortieth of a percent would be tuned to this
     * suite rather than to music. Callers whose material has a genuinely
     * elastic tempo should lower it.
     */
    void setTightness(T alpha) noexcept
    {
        if (!std::isfinite(alpha)) return;
        tightness_.store(std::max(kMinTightness, static_cast<double>(alpha)),
                         std::memory_order_relaxed);
    }

    /** @brief Tightness currently in force. */
    [[nodiscard]] T getTightness() const noexcept
    {
        return static_cast<T>(tightness_.load(std::memory_order_relaxed));
    }

    // -- Offline -------------------------------------------------------------

    /**
     * @brief Tracks tempo and beats over a whole mono buffer (channel 0).
     *
     * Allocates; not an audio-thread call, and not concurrent with the audio
     * path -- it drives the same front end, so it CLEARS the causal state on
     * entry and leaves it cleared.
     *
     * Returns an all-zero Result with an empty grid when the input is shorter
     * than the analysis needs (kMinAnalysisBeats beats at the slowest
     * searched tempo), rather than a tempo fitted to nothing.
     */
    Result analyze(AudioBufferView<const T> whole)
    {
        Result out;
        if (!prepared_.load(std::memory_order_relaxed)) return out;
        if (whole.getNumChannels() < 1 || whole.getNumSamples() <= 0) return out;

        resetState();
        buildEnvelope(whole);

        int iLo = 0, iHi = 0;
        unpackRange(activeRange_.load(std::memory_order_relaxed), iLo, iHi);

        const int n = static_cast<int>(env_.size());
        const double maxPeriod = bankPeriod_[static_cast<size_t>(iHi)];
        if (n < static_cast<int>(kMinAnalysisBeats * maxPeriod))
        {
            resetState();
            return out;
        }

        conditionEnvelope();

        // Two candidate periods from the tempo-prior-weighted autocorrelation,
        // at least a quarter octave apart so that the runner-up is a different
        // metrical reading and not the same peak one lag over.
        double tau1 = 0.0, tau2 = 0.0;
        pickCandidates(iLo, iHi, tau1, tau2);
        if (!(tau1 > 0.0)) { resetState(); return out; }

        // The metrical level is decided against one period for the whole
        // signal, which is the stable model, and only then is the grid laid
        // down against a period that is allowed to move.
        // A period that moves is only worth estimating where there is a pulse
        // to estimate it from. When the winner does not clear the fundamental
        // floor -- the caller restricted the range past the real pulse -- a
        // per-frame estimate over that range is tracking noise, and it drags
        // the delivered grid with it: measured at 96 BPM on 160 BPM material
        // searched over 60 to 100, where the answer a listener would give is
        // the 80 BPM the correlation points at.
        std::vector<int64_t> win;
        localPeriod_.clear();
        if (coherence(tau1) >= kMinFundamentalCoherence)
            computeLocalPeriods(tau1);
        const bool ok = (buildGrid(tau1, win) && win.size() >= 2);
        localPeriod_.clear();
        if (!ok)
        {
            if (!buildGrid(tau1, win) || win.size() < 2) { resetState(); return out; }
        }

        // The delivered tempo is fitted to the delivered grid, not read off
        // the autocorrelation lag: the grid spans the whole signal, so a
        // straight-line fit through it resolves the period far below the
        // frame spacing the lag is quantised to.
        const double slope = fitBeatSlope(win);
        out.beatSamples = win;
        out.tempoBpm = static_cast<T>((slope > 0.0) ? 60.0 * sampleRate_ / slope
                                                    : periodToBpm(tau1));
        out.secondaryTempoBpm = static_cast<T>((tau2 > 0.0) ? periodToBpm(tau2) : 0.0);

        // Confidence is the coherence of the delivered grid, reduced only if
        // the alternative explained the signal nearly as well. Coherence alone
        // answers "does this grid fit"; it cannot answer "and is it the only
        // grid that does", and on material that genuinely carries two pulses
        // -- three against two, most obviously -- a tracker that reports one
        // of them at full confidence is asserting something it has no evidence
        // for. See kAmbiguityFloor for why the reduction has a floor under it
        // instead of being applied in proportion.
        const double base = coherence((slope > 0.0) ? slope / static_cast<double>(hop_)
                                                    : tau1);
        out.confidence = static_cast<T>(std::clamp(base * ambiguityDiscount(secondaryShare_),
                                                   0.0, 1.0));

        resetState();
        return out;
    }

    // -- Audio path (causal, RT-safe) ---------------------------------------

    /** @brief Feeds a mono block; reads channel 0 only. Const, never mutated.
     *         Lock free and allocation free. Safe no-op before prepare(). */
    void processBlock(AudioBufferView<const T> in) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed) || in.getNumChannels() < 1)
            return;
        const T* ch0 = in.getChannel(0);
        pushSamples(std::span<const T>(ch0, static_cast<size_t>(in.getNumSamples())));
    }

    /**
     * @brief Feeds a mono stream of samples. Lock free, allocation free.
     *
     * The stream is handed to the front end in pieces that end exactly on its
     * analysis-frame boundaries, so this call knows precisely which frames it
     * caused and can read each one's envelope value as it appears. The result
     * does not depend on how the caller blocks the stream.
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;

        bool fired = false;
        size_t off = 0;
        while (off < samples.size())
        {
            const int64_t sinceFrame = samplesPushed_ % static_cast<int64_t>(hop_);
            const size_t toBoundary = static_cast<size_t>(static_cast<int64_t>(hop_)
                                                          - sinceFrame);
            const size_t take = std::min(samples.size() - off, toBoundary);

            onset_.pushSamples(samples.subspan(off, take));
            samplesPushed_ += static_cast<int64_t>(take);
            off += take;

            if (static_cast<size_t>(take) == toBoundary)
            {
                if (advanceFrame()) fired = true;
            }
        }

        beatLatched_.store(fired, std::memory_order_relaxed);
    }

    // -- Readout (lock-free) -------------------------------------------------

    /** @brief Running tempo estimate in BPM, 0 before the first frame. */
    [[nodiscard]] T getRunningTempoBpm() const noexcept
    {
        float bpm = 0.0f, conf = 0.0f;
        unpackTempo(tempoAndConfidence_.load(std::memory_order_relaxed), bpm, conf);
        return static_cast<T>(bpm);
    }

    /** @brief Coherence of the envelope with the running tempo, in [0,1].
     *         See the file header for exactly what the number measures. */
    [[nodiscard]] T getConfidence() const noexcept
    {
        float bpm = 0.0f, conf = 0.0f;
        unpackTempo(tempoAndConfidence_.load(std::memory_order_relaxed), bpm, conf);
        return static_cast<T>(conf);
    }

    /**
     * @brief Running tempo and its confidence, from ONE load.
     *
     * The pair is published as a single word, so this hands back two numbers
     * that were measured at the same instant. Calling the two single getters
     * one after the other can straddle an update and pair a fresh tempo with
     * the confidence of the previous one, which is the reading that would let
     * a caller trust a tempo the tracker had already stopped believing.
     */
    void getTempoAndConfidence(T& bpmOut, T& confidenceOut) const noexcept
    {
        float bpm = 0.0f, conf = 0.0f;
        unpackTempo(tempoAndConfidence_.load(std::memory_order_relaxed), bpm, conf);
        bpmOut = static_cast<T>(bpm);
        confidenceOut = static_cast<T>(conf);
    }

    /** @brief True if a beat was reported during the most recent call. */
    [[nodiscard]] bool beatNow() const noexcept
    {
        return beatLatched_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Sample index the most recent causal beat is attributed to; -1
     *        before the first one.
     *
     * This is where the beat WAS, in the caller's own timeline, not when it
     * was announced. The two differ by getLatencySamples() and a caller that
     * aligns anything to the grid wants this one: the announcement carries the
     * analysis delay, the attribution does not.
     */
    [[nodiscard]] int64_t getLastBeatSample() const noexcept
    {
        return lastBeatSample_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Delay between a beat and the call in which it is announced.
     *
     * Half the analysis frame less the front end's group-delay compensation,
     * plus one hop for the frame the phase turn is observed on. Fixed for a
     * given sample rate and independent of block size.
     */
    [[nodiscard]] int getLatencySamples() const noexcept
    {
        return static_cast<int>(latencySamples_.load(std::memory_order_relaxed));
    }

    /** @brief Analysis frames per second of the shared envelope. */
    [[nodiscard]] double getFrameRate() const noexcept { return frameRate_; }

    /** @brief Read-only access to the front end, for callers that also want
     *         onsets and would otherwise run a second copy of the same STFT. */
    [[nodiscard]] const OnsetDetector<T>& getOnsetDetector() const noexcept
    {
        return onset_;
    }

    /** @brief Clears all streaming state. Not concurrent with pushSamples(). */
    void reset() noexcept { resetState(); }

private:
    // -- Constants -----------------------------------------------------------

    /// Searchable tempo span. A caller may set any range inside it.
    static constexpr double kBankMinBpm = 20.0;
    static constexpr double kBankMaxBpm = 480.0;
    /// The bank itself runs slower than the slowest searchable tempo, because
    /// the harmonic test below has to read resonators BELOW every candidate:
    /// a factor of three past the bottom of the searchable range.
    static constexpr double kHarmonicReach = 3.0;
    static constexpr int kBinsPerOctave = 128; ///< Bank resolution, ~0.54%.
    static constexpr double kDefaultMinBpm = 40.0;
    static constexpr double kDefaultMaxBpm = 240.0;

    /// Which subharmonics the metrical-level test reads. A candidate that is
    /// really a harmonic of the true pulse has a subharmonic just as coherent
    /// as itself, and the true pulse does not. Two and three are the divisors
    /// that matter: every octave error is caught by the two, every triplet
    /// error by the three, and between them they catch every integer multiple
    /// up to four inclusive. They are also exactly the relations a tempo
    /// evaluation calls a metrical-level confusion rather than a wrong answer.
    static constexpr int kNumHarmonics = 2;
    static constexpr double kHarmonicDivisors[kNumHarmonics] = { 2.0, 3.0 };

    /// Memory of every resonator, in units of its own period. Equal memory in
    /// BEATS rather than in seconds, so a 60 BPM and a 240 BPM candidate are
    /// judged over the same musical span; eight is long enough for the phase
    /// to be well determined and short enough to follow a tempo that moves.
    static constexpr double kIntegrationBeats = 8.0;

    /// Tempo prior (Ellis eq. 6): a Gaussian on log period, centred on half a
    /// second per beat with a width in octaves.
    static constexpr double kPriorCentreSeconds = 0.5;
    static constexpr double kPriorWidthOctaves = 1.4;

    /// Dynamic-programming tightness. See setTightness() for what the number
    /// is, where it comes from and why it is a default rather than a constant.
    static constexpr double kDefaultTightness = 400.0;
    static constexpr double kMinTightness = 1e-3;

    /// Least share of the onset mass a candidate must carry in phase before
    /// it is treated as a pulse the signal actually has, rather than as a
    /// numerical residue. Below it the metrical-level question is not on the
    /// table and the strongest periodicity is the best available answer; see
    /// pickCandidates(). A tenth separates the two cases by a wide margin on
    /// the acceptance corpus: the weakest genuine reading there carries 0.18
    /// of the mass, and a range deliberately set to exclude the real pulse
    /// leaves every candidate below 0.05.
    static constexpr double kMinFundamentalCoherence = 0.1;

    /// How competitive the metrical alternative has to be before it is
    /// allowed to reduce the confidence, as a fraction of the winner's score,
    /// and the discount is ramped from there to zero confidence at parity.
    /// A floor rather than a plain ratio, and the floor is what makes the
    /// number stay meaningful: every signal has a runner-up, and letting one
    /// that scores a fifth of the winner take a fifth off the confidence
    /// makes the confidence a function of the corpus's noise rather than of
    /// its rhythm. Measured on the acceptance corpus, a plain ratio moved the
    /// reading on material with no ambiguity at all -- a bed at a fiftieth of
    /// the click amplitude took it from 0.99 to 0.78 while the grid stayed
    /// exact -- and it broke the agreement with the closed form that makes
    /// the number checkable. With the floor the discount is inert on every
    /// unambiguous case and full strength on three against two.
    static constexpr double kAmbiguityFloor = 0.5;

    /// A candidate and its runner-up must be this far apart on log tempo to
    /// count as different readings; a quarter octave admits the half and
    /// double levels and rejects a neighbouring lag of the same peak.
    static constexpr double kCandidateSeparationOctaves = 0.25;

    /// Shortest signal analyze() will fit a tempo to, in beats at the slowest
    /// searched tempo.
    static constexpr double kMinAnalysisBeats = 4.0;

    /// Coherence window, in beats. Matches the causal memory so the offline
    /// and causal numbers mean the same thing; short enough that a tempo drift
    /// across the signal does not read as incoherence.
    static constexpr double kCoherenceWindowBeats = 8.0;

    /// Envelope conditioning: the baseline is the mean over this many seconds.
    static constexpr double kBaselineSeconds = 1.0;

    /// Smoothing of the conditioned envelope before the recursion, as a
    /// fraction of the beat period.
    static constexpr double kLocalScoreSigmaPeriods = 1.0 / 32.0;

    /// Interval search around the target period in the recursion.
    static constexpr double kSearchLowFactor = 0.5;
    static constexpr double kSearchHighFactor = 2.0;

    /// How far a beat may be moved onto a nearby envelope peak, and how far
    /// the grid extension may look for the next one, as a fraction of the beat
    /// period. An eighth of a period cannot reach an off-beat at any swing
    /// ratio -- the nearest one is a third of a period away -- so the step can
    /// correct a misplacement but never change which pulse a beat is on.
    static constexpr double kSnapFraction = 0.125;

    /// Grid trimming: a leading or trailing beat carrying less than this
    /// fraction of the grid's mean onset strength is an artefact of where the
    /// recursion had to start and stop, not a beat anyone played.
    static constexpr double kTrimFraction = 0.5;

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<std::int64_t>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<double>::is_always_lock_free,
                  "audio-thread stores must not lock");

    // -- Packing -------------------------------------------------------------

    /// Tempo and its confidence share one word. Two 32-bit floats, copied
    /// through their object representations rather than through a union or a
    /// pointer cast, which is the only form that is defined for every
    /// compiler this ships to.
    [[nodiscard]] static std::uint64_t packTempo(float bpm, float conf) noexcept
    {
        std::uint32_t a = 0, b = 0;
        std::memcpy(&a, &bpm, sizeof(a));
        std::memcpy(&b, &conf, sizeof(b));
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    }

    static void unpackTempo(std::uint64_t v, float& bpm, float& conf) noexcept
    {
        const std::uint32_t a = static_cast<std::uint32_t>(v >> 32);
        const std::uint32_t b = static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
        std::memcpy(&bpm, &a, sizeof(bpm));
        std::memcpy(&conf, &b, sizeof(conf));
    }

    [[nodiscard]] static std::uint32_t packRange(int lo, int hi) noexcept
    {
        return (static_cast<std::uint32_t>(lo & 0xFFFF) << 16)
             | static_cast<std::uint32_t>(hi & 0xFFFF);
    }

    static void unpackRange(std::uint32_t v, int& lo, int& hi) noexcept
    {
        lo = static_cast<int>((v >> 16) & 0xFFFFu);
        hi = static_cast<int>(v & 0xFFFFu);
    }

    // -- Bank ----------------------------------------------------------------

    [[nodiscard]] double periodToBpm(double periodFrames) const noexcept
    {
        return (periodFrames > 0.0) ? 60.0 * frameRate_ / periodFrames : 0.0;
    }

    [[nodiscard]] double bpmToPeriod(double bpm) const noexcept
    {
        return (bpm > 0.0) ? 60.0 * frameRate_ / bpm : 0.0;
    }

    /** @brief Nearest bank index to a tempo in BPM. */
    [[nodiscard]] int bpmToIndex(double bpm) const noexcept
    {
        const double p = bpmToPeriod(bpm);
        if (!(p > 0.0) || bankSize_ < 1) return 0;
        const double x = std::log2(p / bankPeriod_[0])
                       * static_cast<double>(kBinsPerOctave);
        return std::clamp(static_cast<int>(std::lround(x)), 0, bankSize_ - 1);
    }

    void buildBank()
    {
        const double pMin = bpmToPeriod(kBankMaxBpm);
        const double pMax = bpmToPeriod(kBankMinBpm) * kHarmonicReach;
        const double octaves = std::log2(pMax / pMin);
        bankSize_ = static_cast<int>(std::lround(octaves
                                                 * static_cast<double>(kBinsPerOctave))) + 1;
        bankSize_ = std::max(bankSize_, 2 * kBinsPerOctave + 2);

        bankPeriod_.assign(static_cast<size_t>(bankSize_), 0.0);
        bankCosCoef_.assign(static_cast<size_t>(bankSize_), 0.0);
        bankSinCoef_.assign(static_cast<size_t>(bankSize_), 0.0);
        bankDecay_.assign(static_cast<size_t>(bankSize_), 0.0);
        bankPrior_.assign(static_cast<size_t>(bankSize_), 0.0);
        zRe_.assign(static_cast<size_t>(bankSize_), 0.0);
        zIm_.assign(static_cast<size_t>(bankSize_), 0.0);
        zMass_.assign(static_cast<size_t>(bankSize_), 0.0);
        bankCoherence_.assign(static_cast<size_t>(bankSize_), 0.0);

        const double centre = kPriorCentreSeconds * frameRate_;
        for (int i = 0; i < bankSize_; ++i)
        {
            const double p = pMin * std::pow(2.0, static_cast<double>(i)
                                                  / static_cast<double>(kBinsPerOctave));
            bankPeriod_[static_cast<size_t>(i)] = p;

            const double w = 2.0 * static_cast<double>(pi<double>) / p;
            const double lambda = std::exp(-1.0 / (kIntegrationBeats * p));
            bankDecay_[static_cast<size_t>(i)] = lambda;
            bankCosCoef_[static_cast<size_t>(i)] = lambda * std::cos(w);
            bankSinCoef_[static_cast<size_t>(i)] = lambda * std::sin(w);

            const double lg = std::log2(p / centre) / kPriorWidthOctaves;
            bankPrior_[static_cast<size_t>(i)] = std::exp(-0.5 * lg * lg);
        }

        // Index distance to each subharmonic. The bank is geometric, so a
        // ratio in period is a constant offset in index; rounding it to the
        // nearest bin is worth nothing at all here, because a resonator with a
        // memory of eight periods is a hundred times broader than one bin.
        for (int k = 0; k < kNumHarmonics; ++k)
            harmonicOffset_[k] = static_cast<int>(std::lround(
                std::log2(kHarmonicDivisors[k]) * static_cast<double>(kBinsPerOctave)));
    }

    // -- Causal path ---------------------------------------------------------

    /**
     * @brief One analysis frame of the causal path.
     * @return true if a beat was reported on this frame.
     *
     * Allocation free and lock free: every buffer it touches was sized at
     * prepare() and the loop bounds are the bank size, which does not depend
     * on the signal.
     */
    bool advanceFrame() noexcept
    {
        const typename OnsetDetector<T>::OdfFrame f = onset_.getLastOdfFrame();
        ++frameIndex_;

        // The first frames read the step from silence into the input rather
        // than the input, and a periodicity estimator weights an impulse at
        // time zero more heavily than any other, because every candidate
        // period reaches it.
        if (frameIndex_ <= warmupFrames_) return false;

        const double raw = (std::isfinite(f.value) && f.value > T(0))
                               ? static_cast<double>(f.value) : 0.0;
        lastFrameRef_ = f.referenceSample;

        // Same conditioning the offline path applies, in the only form a
        // causal path can have it: a trailing mean instead of a centred one.
        // Without it the log-compressed envelope's positive floor counts as
        // onset mass that is spread evenly in phase, so the coherence ratio
        // reads the noise floor of the material rather than its pulse -- on a
        // bed at a twentieth of the click amplitude that alone took the
        // published confidence from 0.98 to 0.25 while the tempo stayed
        // correct to a tenth of a percent, which is an under-reading a caller
        // gating on confidence would act on.
        baseline_ += (raw - baseline_) * baselineCoef_;
        const double o = std::max(0.0, raw - baseline_);

        for (int i = 0; i < bankSize_; ++i)
        {
            const size_t u = static_cast<size_t>(i);
            const double cr = bankCosCoef_[u];
            const double ci = bankSinCoef_[u];
            const double re = zRe_[u];
            const double im = zIm_[u];
            zRe_[u] = cr * re - ci * im + o;
            zIm_[u] = cr * im + ci * re;
            zMass_[u] = bankDecay_[u] * zMass_[u] + o;

            const double mass = zMass_[u];
            bankCoherence_[u] = (mass > kMassFloor)
                                    ? std::min(1.0, std::sqrt(zRe_[u] * zRe_[u]
                                                              + zIm_[u] * zIm_[u]) / mass)
                                    : 0.0;
        }

        int iLo = 0, iHi = 0;
        unpackRange(activeRange_.load(std::memory_order_relaxed), iLo, iHi);
        iLo = std::clamp(iLo, 0, bankSize_ - 1);
        iHi = std::clamp(iHi, iLo, bankSize_ - 1);

        int best = -1;
        double bestScore = 0.0;
        for (int i = iLo; i <= iHi; ++i)
        {
            const double s = scoreAt(i);
            if (s > bestScore) { bestScore = s; best = i; }
        }

        if (best < 0)
        {
            // Nothing in the searched range explains the envelope yet. Keep
            // the previous published pair rather than publishing a tempo of
            // zero and a confidence of zero, which a caller cannot tell from
            // a real reading of silence.
            return false;
        }

        // The runner-up at a different metrical level, and how nearly it
        // explained the envelope as well as the winner did. Same discount the
        // offline path applies, for the same reason and on the same scale: a
        // grid that fits is not the same claim as a grid that is the only one
        // that fits, and a caller gating on one number must be told the
        // difference.
        double runnerUp = 0.0;
        const int apart = std::max(1, static_cast<int>(kCandidateSeparationOctaves
                                                       * kBinsPerOctave));
        for (int i = iLo; i <= iHi; ++i)
        {
            if (std::abs(i - best) < apart) continue;
            runnerUp = std::max(runnerUp, scoreAt(i));
        }
        const double share = (bestScore > 0.0)
            ? std::clamp(runnerUp / bestScore, 0.0, 1.0) : 0.0;

        const size_t ub = static_cast<size_t>(best);
        const double period = refinePeriod(best);
        const double conf = std::clamp(bankCoherence_[ub] * ambiguityDiscount(share),
                                       0.0, 1.0);

        tempoAndConfidence_.store(packTempo(static_cast<float>(periodToBpm(period)),
                                            static_cast<float>(conf)),
                                  std::memory_order_relaxed);

        // Phase. arg(z) advances by one turn per period, so the frames since
        // the last pulse are arg(z) scaled by the period; the turn completing
        // -- that quantity falling back through zero -- is the beat.
        const double w = 2.0 * static_cast<double>(pi<double>) / bankPeriod_[ub];
        double sinceBeat = std::atan2(zIm_[ub], zRe_[ub]) / w;
        const double p = bankPeriod_[ub];
        while (sinceBeat < 0.0) sinceBeat += p;
        while (sinceBeat >= p) sinceBeat -= p;

        ++framesSinceBeat_;
        bool beat = false;
        // The turn completes when the frames-since-pulse quantity falls back
        // through zero. The refractory is not a smoother: the winning bin can
        // step between neighbours from one frame to the next, and the two
        // neighbours do not carry exactly the same phase, so without it a
        // one-bin step near a turn reports the same beat twice.
        if (havePhase_ && (prevSinceBeat_ - sinceBeat) > 0.5 * p
            && static_cast<double>(framesSinceBeat_) >= kBeatRefractory * p)
        {
            const int64_t backFrames = static_cast<int64_t>(std::lround(sinceBeat));
            const int64_t ref = lastFrameRef_ - backFrames * static_cast<int64_t>(hop_);
            lastBeatSample_.store(ref, std::memory_order_relaxed);
            framesSinceBeat_ = 0;
            beat = true;
        }
        prevSinceBeat_ = sinceBeat;
        lastBestIndex_ = best;
        havePhase_ = true;

        return beat;
    }

    /** @brief Parabolic refinement of the winning period, on log period, over
     *  the octave-adjudicated score. The bank is geometric, so the three
     *  points are equally spaced in the variable being interpolated. */
    [[nodiscard]] double refinePeriod(int i) const noexcept
    {
        if (i <= 0 || i >= bankSize_ - 1) return bankPeriod_[static_cast<size_t>(i)];
        const double a = scoreAt(i - 1);
        const double b = scoreAt(i);
        const double c = scoreAt(i + 1);
        const double den = a - 2.0 * b + c;
        if (!(std::abs(den) > 0.0)) return bankPeriod_[static_cast<size_t>(i)];
        double d = 0.5 * (a - c) / den;
        d = std::clamp(d, -0.5, 0.5);
        return bankPeriod_[static_cast<size_t>(i)]
             * std::pow(2.0, d / static_cast<double>(kBinsPerOctave));
    }

    [[nodiscard]] double scoreAt(int i) const noexcept
    {
        const size_t u = static_cast<size_t>(i);
        double s = bankPrior_[u] * bankCoherence_[u];
        for (int k = 0; k < kNumHarmonics; ++k)
        {
            const int sub = i + harmonicOffset_[k];
            if (sub < bankSize_)
                s *= (1.0 - bankCoherence_[static_cast<size_t>(sub)]);
        }
        return s;
    }

    // -- Offline stages ------------------------------------------------------

    /** @brief Runs the shared front end over the whole buffer and keeps the
     *  envelope with each frame's position in the caller's timeline. */
    void buildEnvelope(AudioBufferView<const T> whole)
    {
        env_.clear();
        envRef_.clear();

        const int n = whole.getNumSamples();
        const T* x = whole.getChannel(0);
        const size_t frames = static_cast<size_t>(n / std::max(1, hop_) + 2);
        env_.reserve(frames);
        envRef_.reserve(frames);

        onset_.reset();
        int64_t pos = 0;
        int64_t frame = 0;
        while (pos < static_cast<int64_t>(n))
        {
            const int64_t take = std::min<int64_t>(hop_, static_cast<int64_t>(n) - pos);
            onset_.pushSamples(std::span<const T>(x + pos, static_cast<size_t>(take)));
            pos += take;
            if (take == static_cast<int64_t>(hop_))
            {
                ++frame;
                if (frame > warmupFrames_)
                {
                    const typename OnsetDetector<T>::OdfFrame f = onset_.getLastOdfFrame();
                    const double v = std::isfinite(f.value)
                                         ? static_cast<double>(f.value) : 0.0;
                    env_.push_back(std::max(0.0, v));
                    envRef_.push_back(f.referenceSample);
                }
            }
        }
        onset_.reset();
    }

    /**
     * @brief Removes the envelope's own floor, then scales it to unit spread.
     *
     * The onset function is log-compressed and therefore sits on a positive,
     * signal-dependent floor. A constant added to every frame is evidence for
     * a beat at every frame: the recursion would read it as onset strength
     * spread evenly over time, which is exactly the thing a beat grid is
     * supposed to be able to rule out. Subtracting a slow local mean removes
     * it without touching anything that happens at beat rate, since the
     * baseline window is longer than any period searched.
     */
    void conditionEnvelope()
    {
        const int n = static_cast<int>(env_.size());
        if (n < 1) return;

        const int half = std::max(1, static_cast<int>(std::lround(0.5 * kBaselineSeconds
                                                                  * frameRate_)));
        scratchA_.assign(static_cast<size_t>(n) + 1, 0.0);
        for (int i = 0; i < n; ++i)
            scratchA_[static_cast<size_t>(i) + 1] = scratchA_[static_cast<size_t>(i)]
                                                  + env_[static_cast<size_t>(i)];

        double sumSq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const int lo = std::max(0, i - half);
            const int hi = std::min(n, i + half + 1);
            const double mean = (scratchA_[static_cast<size_t>(hi)]
                                 - scratchA_[static_cast<size_t>(lo)])
                              / static_cast<double>(hi - lo);
            const double v = std::max(0.0, env_[static_cast<size_t>(i)] - mean);
            env_[static_cast<size_t>(i)] = v;
            sumSq += v * v;
        }

        const double rms = std::sqrt(sumSq / static_cast<double>(n));
        if (rms > 0.0)
        {
            const double g = 1.0 / rms;
            for (int i = 0; i < n; ++i) env_[static_cast<size_t>(i)] *= g;
        }
    }

    /**
     * @brief Coherence of the conditioned envelope with a pulse of the given
     *        period: the share of onset mass that is in phase with it.
     *
     * Measured over windows of kCoherenceWindowBeats and averaged in
     * magnitude, not in phase. Over one long window a period wrong by a
     * fraction of a percent would decohere against itself simply by drifting,
     * and the number would report the estimator's resolution instead of the
     * signal's regularity.
     */
    [[nodiscard]] double coherence(double periodFrames) const
    {
        const int n = static_cast<int>(env_.size());
        if (n < 2 || !(periodFrames > 1.0)) return 0.0;

        const int win = std::max(4, static_cast<int>(std::lround(kCoherenceWindowBeats
                                                                 * periodFrames)));
        if (win > n) return coherenceWindow(0, n, periodFrames);

        const int step = std::max(1, win / 2);
        double acc = 0.0;
        int count = 0;
        for (int start = 0; start + win <= n; start += step)
        {
            acc += coherenceWindow(start, start + win, periodFrames);
            ++count;
        }
        if (count == 0) return coherenceWindow(0, n, periodFrames);
        return acc / static_cast<double>(count);
    }

    [[nodiscard]] double coherenceWindow(int from, int to, double periodFrames) const
    {
        const double w = 2.0 * static_cast<double>(pi<double>) / periodFrames;
        double re = 0.0, im = 0.0, mass = 0.0;
        for (int i = from; i < to; ++i)
        {
            const double v = env_[static_cast<size_t>(i)];
            const double a = w * static_cast<double>(i);
            re += v * std::cos(a);
            im -= v * std::sin(a);
            mass += v;
        }
        if (!(mass > kMassFloor)) return 0.0;
        return std::min(1.0, std::sqrt(re * re + im * im) / mass);
    }

    /**
     * @brief The two best periods to realise as beat grids.
     *
     * Two stages, because they answer different questions. The
     * tempo-prior-weighted autocorrelation GENERATES candidates: its peaks are
     * the lags at which the envelope repeats, and that is all a correlation
     * can tell, since a pulse train repeats at every multiple of its period
     * just as well as at the period itself. The octave discriminator then
     * RANKS them, which is where the metrical level is actually decided.
     *
     * Ranking on the correlation height alone is what produces half-tempo
     * readings at fast tempi: every multiple of the true period is a peak of
     * equal height, so the choice falls to the prior, and the prior prefers
     * the multiple nearest half a second per beat. Above about 200 BPM that
     * multiple is the wrong one, every time, by construction rather than by
     * bad luck.
     *
     * @p tau1 receives the best-ranked candidate and @p tau2 the best-ranked
     * one at least kCandidateSeparationOctaves away from it, or 0 if the
     * envelope offers no distinct second reading -- which is the honest answer
     * for a pulse at the slow end of the range, whose alternatives are all
     * outside it.
     */
    void pickCandidates(int iLo, int iHi, double& tau1, double& tau2)
    {
        tau1 = 0.0;
        tau2 = 0.0;
        secondaryShare_ = 0.0;

        const int n = static_cast<int>(env_.size());
        const int lagLo = std::max(1, static_cast<int>(std::floor(
                                          bankPeriod_[static_cast<size_t>(iLo)])));
        const int lagHi = std::min(n / 2, static_cast<int>(std::ceil(
                                              bankPeriod_[static_cast<size_t>(iHi)])));
        if (lagHi <= lagLo) return;

        acf_.assign(static_cast<size_t>(lagHi + 1), 0.0);
        const double centre = kPriorCentreSeconds * frameRate_;
        for (int lag = lagLo; lag <= lagHi; ++lag)
        {
            double s = 0.0;
            const int m = n - lag;
            for (int i = 0; i < m; ++i)
                s += env_[static_cast<size_t>(i)] * env_[static_cast<size_t>(i + lag)];
            s /= static_cast<double>(m);
            const double lg = std::log2(static_cast<double>(lag) / centre)
                            / kPriorWidthOctaves;
            acf_[static_cast<size_t>(lag)] = s * std::exp(-0.5 * lg * lg);
        }

        // Peaks, not merely the argmax: the runner-up must be a peak of its
        // own, or it is the same peak read one lag over.
        candidates_.clear();
        for (int lag = lagLo + 1; lag < lagHi; ++lag)
        {
            const double v = acf_[static_cast<size_t>(lag)];
            if (v > acf_[static_cast<size_t>(lag - 1)]
                && v >= acf_[static_cast<size_t>(lag + 1)] && v > 0.0)
                candidates_.push_back(lag);
        }
        if (candidates_.empty()) return;

        // Rank by the octave discriminator, not by correlation height.
        candScore_.assign(candidates_.size(), 0.0);
        for (size_t k = 0; k < candidates_.size(); ++k)
        {
            const double tau = refineLag(candidates_[k]);
            const double lg = std::log2(tau / centre) / kPriorWidthOctaves;
            candScore_[k] = std::exp(-0.5 * lg * lg) * coherence(tau);
            for (const double m : kHarmonicDivisors)
                candScore_[k] *= (1.0 - coherence(m * tau));
        }

        size_t k1 = 0;
        for (size_t k = 1; k < candidates_.size(); ++k)
            if (candScore_[k] > candScore_[k1]) k1 = k;

        // The discriminator can only choose between candidates that look like
        // a fundamental. When the caller has restricted the range so that the
        // signal's own period falls outside it, every candidate left inside is
        // one of that period's subharmonics -- and a subharmonic is exactly
        // what the coherence term is built to score at nothing, so the ranking
        // would be choosing between residues. That is not a failure of the
        // caller's request: a listener held to 60..100 BPM on a 160 BPM pulse
        // hears 80, and 80 is a peak of the correlation. So when no candidate
        // clears the floor, the correlation decides on its own.
        bool anyFundamental = false;
        for (size_t k = 0; k < candidates_.size(); ++k)
            if (coherence(refineLag(candidates_[k])) >= kMinFundamentalCoherence)
            { anyFundamental = true; break; }
        if (!anyFundamental)
        {
            k1 = 0;
            for (size_t k = 1; k < candidates_.size(); ++k)
                if (acf_[static_cast<size_t>(candidates_[k])]
                    > acf_[static_cast<size_t>(candidates_[k1])]) k1 = k;
            for (size_t k = 0; k < candidates_.size(); ++k)
                candScore_[k] = acf_[static_cast<size_t>(candidates_[k])];
        }
        tau1 = refineLag(candidates_[k1]);

        int k2 = -1;
        for (size_t k = 0; k < candidates_.size(); ++k)
        {
            const double sep = std::abs(std::log2(static_cast<double>(candidates_[k])
                                                  / static_cast<double>(candidates_[k1])));
            if (sep < kCandidateSeparationOctaves) continue;
            if (k2 < 0 || candScore_[k] > candScore_[static_cast<size_t>(k2)])
                k2 = static_cast<int>(k);
        }
        if (k2 >= 0)
        {
            tau2 = refineLag(candidates_[static_cast<size_t>(k2)]);
            const double top = candScore_[k1];
            secondaryShare_ = (top > 0.0)
                ? std::clamp(candScore_[static_cast<size_t>(k2)] / top, 0.0, 1.0)
                : 0.0;
        }
    }

    /** @brief Parabolic refinement of an autocorrelation peak, in lag. */
    [[nodiscard]] double refineLag(int lag) const noexcept
    {
        if (lag <= 0 || lag + 1 >= static_cast<int>(acf_.size()))
            return static_cast<double>(lag);
        const double a = acf_[static_cast<size_t>(lag - 1)];
        const double b = acf_[static_cast<size_t>(lag)];
        const double c = acf_[static_cast<size_t>(lag + 1)];
        const double den = a - 2.0 * b + c;
        if (!(std::abs(den) > 0.0)) return static_cast<double>(lag);
        const double d = std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
        return static_cast<double>(lag) + d;
    }

    /**
     * @brief Per-frame beat period, for music whose tempo moves.
     *
     * The same resonator bank the causal path runs, swept over the envelope
     * once forward and once backward and combined in the log-period domain.
     * One direction alone carries its own memory as a lag -- a forward sweep
     * reports where the tempo HAS been -- and running the identical filter the
     * other way produces the mirror-image lag, so the two together are
     * symmetric in time and centred on the frame being described. That is only
     * possible offline, and it is the whole reason the offline path can follow
     * a tempo the causal path can only chase.
     *
     * The sweep is confined to half an octave either side of the period the
     * global analysis settled on: the metrical level has already been decided
     * by then, and letting a local estimate re-open it would let one bar of
     * dense playing move the grid to double time.
     */
    void computeLocalPeriods(double globalTau)
    {
        const int n = static_cast<int>(env_.size());
        localPeriod_.assign(static_cast<size_t>(n), globalTau);
        if (n < 8 || bankSize_ < 2) return;

        // Confined twice over: to half an octave either side of the period the
        // global analysis settled on, AND to the range the caller asked to be
        // searched. The first keeps one dense bar from moving the grid to
        // double time; the second is the caller's contract, and a local
        // estimate that escaped it would deliver a tempo outside the range
        // that was asked for -- measured at 110 BPM on 160 BPM material
        // searched over 60 to 100 before this clamp existed.
        int aLo = 0, aHi = 0;
        unpackRange(activeRange_.load(std::memory_order_relaxed), aLo, aHi);
        aLo = std::clamp(aLo, 0, bankSize_ - 1);
        aHi = std::clamp(aHi, aLo, bankSize_ - 1);

        const int gi = periodToIndex(globalTau);
        int lo = std::clamp(gi - kBinsPerOctave / 2, aLo, aHi);
        int hi = std::clamp(gi + kBinsPerOctave / 2, aLo, aHi);
        if (hi < lo) std::swap(lo, hi);
        if (hi <= lo) return;

        sweepA_.assign(static_cast<size_t>(n), globalTau);
        sweepB_.assign(static_cast<size_t>(n), globalTau);
        sweepPeriods(lo, hi, false, sweepA_);
        sweepPeriods(lo, hi, true, sweepB_);

        for (int i = 0; i < n; ++i)
        {
            const double a = sweepA_[static_cast<size_t>(i)];
            const double b = sweepB_[static_cast<size_t>(i)];
            localPeriod_[static_cast<size_t>(i)] =
                (a > 0.0 && b > 0.0) ? std::sqrt(a * b) : globalTau;
        }

        // Smooth over a couple of beats: a per-frame winner steps by whole
        // bank bins, and a target period that steps is a target period the
        // interval cost can be made to chase.
        const int half = std::max(1, static_cast<int>(std::lround(globalTau)));
        scratchA_.assign(static_cast<size_t>(n) + 1, 0.0);
        for (int i = 0; i < n; ++i)
            scratchA_[static_cast<size_t>(i) + 1] = scratchA_[static_cast<size_t>(i)]
                + std::log(localPeriod_[static_cast<size_t>(i)]);
        for (int i = 0; i < n; ++i)
        {
            const int a = std::max(0, i - half);
            const int b = std::min(n, i + half + 1);
            const double m = (scratchA_[static_cast<size_t>(b)]
                              - scratchA_[static_cast<size_t>(a)])
                           / static_cast<double>(b - a);
            localPeriod_[static_cast<size_t>(i)] = std::exp(m);
        }
    }

    /** @brief One resonator sweep over the envelope, in either direction,
     *  writing the winning period per frame. Uses the causal path's own bank
     *  state, which analyze() owns for the duration of the call. */
    void sweepPeriods(int lo, int hi, bool reverse, std::vector<double>& out)
    {
        const int n = static_cast<int>(env_.size());
        std::fill(zRe_.begin(), zRe_.end(), 0.0);
        std::fill(zIm_.begin(), zIm_.end(), 0.0);
        std::fill(zMass_.begin(), zMass_.end(), 0.0);

        for (int k = 0; k < n; ++k)
        {
            const int i = reverse ? (n - 1 - k) : k;
            const double o = env_[static_cast<size_t>(i)];

            int best = -1;
            double bestValue = 0.0;
            for (int b = lo; b <= hi; ++b)
            {
                const size_t u = static_cast<size_t>(b);
                const double re = zRe_[u], im = zIm_[u];
                zRe_[u] = bankCosCoef_[u] * re - bankSinCoef_[u] * im + o;
                zIm_[u] = bankCosCoef_[u] * im + bankSinCoef_[u] * re;
                zMass_[u] = bankDecay_[u] * zMass_[u] + o;
                const double mass = zMass_[u];
                const double c = (mass > kMassFloor)
                    ? std::sqrt(zRe_[u] * zRe_[u] + zIm_[u] * zIm_[u]) / mass : 0.0;
                bankCoherence_[u] = c;
                const double s = bankPrior_[u] * c;
                if (s > bestValue) { bestValue = s; best = b; }
            }
            out[static_cast<size_t>(i)] = (best >= 0)
                ? bankPeriod_[static_cast<size_t>(best)] : 0.0;
        }
    }

    /** @brief How much of the confidence survives an alternative that scored
     *  @p share of the winner. One until the alternative is genuinely
     *  competitive, then linear to zero at parity. */
    [[nodiscard]] static double ambiguityDiscount(double share) noexcept
    {
        if (!(share > kAmbiguityFloor)) return 1.0;
        return std::clamp(1.0 - (share - kAmbiguityFloor) / (1.0 - kAmbiguityFloor),
                          0.0, 1.0);
    }

    /** @brief Nearest bank index to a period in frames. */
    [[nodiscard]] int periodToIndex(double periodFrames) const noexcept
    {
        if (!(periodFrames > 0.0) || bankSize_ < 1) return 0;
        const double x = std::log2(periodFrames / bankPeriod_[0])
                       * static_cast<double>(kBinsPerOctave);
        return std::clamp(static_cast<int>(std::lround(x)), 0, bankSize_ - 1);
    }

    /** @brief Realises a candidate period as a beat grid in the caller's
     *  timeline. The metrical level is already settled by then -- it is
     *  decided by the ranking in pickCandidates(), on the whole envelope,
     *  before any grid exists. */
    bool buildGrid(double periodFrames, std::vector<int64_t>& gridOut)
    {
        gridOut.clear();
        std::vector<int> beatFrames;
        if (!runDp(periodFrames, beatFrames) || beatFrames.size() < 2) return false;
        gridOut.reserve(beatFrames.size());
        for (const int f : beatFrames)
            gridOut.push_back(beatSample(f));
        return true;
    }

    /**
     * @brief The dynamic program of the file header, plus backtrace, trim,
     *        grid extension and sub-frame refinement.
     * @return false if the signal is too short for the period asked for.
     *
     * The target period is a single number for the whole signal unless
     * localPeriod_ has been filled, in which case each frame is judged against
     * the period estimated where it sits. One global period is the right model
     * for deciding WHICH metrical level a signal is on, because it is the
     * stable one; it is the wrong model for laying the grid down on music
     * whose tempo moves, because the interval cost then measures the distance
     * from the average rather than from the tempo being played.
     */
    bool runDp(double periodFrames, std::vector<int>& beatFrames)
    {
        beatFrames.clear();
        const int n = static_cast<int>(env_.size());
        const bool varying = (localPeriod_.size() == env_.size());
        const int lo = std::max(1, static_cast<int>(std::lround(kSearchLowFactor
                                                                * periodFrames)));
        const int hi = std::max(lo + 1, static_cast<int>(std::lround(kSearchHighFactor
                                                                     * periodFrames)));
        if (n <= hi + 2) return false;

        buildLocalScore(periodFrames);

        const double alpha = tightness_.load(std::memory_order_relaxed);
        cumScore_.assign(static_cast<size_t>(n), 0.0);
        backlink_.assign(static_cast<size_t>(n), -1);

        for (int i = 0; i < n; ++i)
        {
            const double target = varying ? localPeriod_[static_cast<size_t>(i)]
                                          : periodFrames;
            const int dLo = varying
                ? std::max(1, static_cast<int>(std::lround(kSearchLowFactor * target)))
                : lo;
            const int dHiBound = varying
                ? std::max(dLo + 1, static_cast<int>(std::lround(kSearchHighFactor
                                                                 * target)))
                : hi;

            double best = 0.0;
            int bestJ = -1;
            const int dHi = std::min(dHiBound, i);
            for (int d = dLo; d <= dHi; ++d)
            {
                const int j = i - d;
                const double lr = std::log(static_cast<double>(d) / target);
                const double s = cumScore_[static_cast<size_t>(j)] - alpha * lr * lr;
                if (bestJ < 0 || s > best) { best = s; bestJ = j; }
            }
            cumScore_[static_cast<size_t>(i)] = local_[static_cast<size_t>(i)]
                                              + ((bestJ >= 0) ? best : 0.0);
            backlink_[static_cast<size_t>(i)] = bestJ;
        }

        int end = 0;
        for (int i = 1; i < n; ++i)
            if (cumScore_[static_cast<size_t>(i)] > cumScore_[static_cast<size_t>(end)])
                end = i;

        for (int i = end; i >= 0; i = backlink_[static_cast<size_t>(i)])
        {
            beatFrames.push_back(i);
            if (backlink_[static_cast<size_t>(i)] < 0) break;
        }
        std::reverse(beatFrames.begin(), beatFrames.end());

        trimBeats(beatFrames);
        if (beatFrames.size() < 2) return false;

        snapBeats(beatFrames, periodFrames);
        extendGrid(beatFrames);
        snapBeats(beatFrames, periodFrames);
        return beatFrames.size() >= 2;
    }

    /**
     * @brief Moves each beat to the nearest peak of the envelope, within a
     *        bounded window.
     *
     * The recursion places beats on frames; the frames are 5 ms apart and a
     * beat is not. It also has to guess the phase of its first few links
     * before it has accumulated any evidence to guess with, so the beats at
     * the start of the chain can sit a full frame or two off a transient that
     * is plainly visible in the envelope. Both are repaired by the same step,
     * and it is bounded to an eighth of a period so that it can correct a
     * misplacement without ever being able to drag a beat onto its neighbour
     * or onto an off-beat: at any swing ratio the nearest off-beat is at least
     * a third of a period away.
     */
    void snapBeats(std::vector<int>& beatFrames, double periodFrames) const
    {
        const int n = static_cast<int>(env_.size());
        const int win = std::max(1, static_cast<int>(std::lround(periodFrames
                                                                 * kSnapFraction)));
        int previous = -1;
        for (size_t k = 0; k < beatFrames.size(); ++k)
        {
            const int f = beatFrames[k];
            int best = f;
            double bestValue = local_[static_cast<size_t>(f)];
            const int lo = std::max(previous + 1, f - win);
            const int hi = std::min(n - 1, f + win);
            for (int i = lo; i <= hi; ++i)
            {
                if (local_[static_cast<size_t>(i)] > bestValue)
                {
                    bestValue = local_[static_cast<size_t>(i)];
                    best = i;
                }
            }
            beatFrames[k] = best;
            previous = best;
        }
    }

    /**
     * @brief Continues the grid off both ends of what the recursion returned,
     *        as far as the envelope keeps supporting it.
     *
     * The recursion needs history before its interval cost means anything, so
     * the beats at the very start of a signal are the ones it places worst,
     * and trimming then removes them -- along with any real beat that was
     * among them. Offline there is no reason to accept that: the period is
     * known once the middle of the signal has been tracked, so the two ends
     * are stepped out at that period and each candidate is kept only if the
     * envelope carries as much onset strength there as the beats already in
     * the grid do. Silence past the end of the music therefore ends the
     * extension immediately, and a real beat before the recursion had warmed
     * up is recovered.
     */
    void extendGrid(std::vector<int>& beatFrames) const
    {
        if (beatFrames.size() < 3) return;
        const int n = static_cast<int>(env_.size());

        std::vector<int> ibi;
        ibi.reserve(beatFrames.size());
        for (size_t k = 1; k < beatFrames.size(); ++k)
            ibi.push_back(beatFrames[k] - beatFrames[k - 1]);
        std::sort(ibi.begin(), ibi.end());
        const int step = ibi[ibi.size() / 2];
        if (step < 2) return;

        double mean = 0.0;
        for (const int f : beatFrames) mean += local_[static_cast<size_t>(f)];
        mean /= static_cast<double>(beatFrames.size());
        const double floorValue = kTrimFraction * mean;
        const int win = std::max(1, static_cast<int>(std::lround(step * kSnapFraction)));

        std::vector<int> front;
        for (int guess = beatFrames.front() - step; guess - win >= 0; guess -= step)
        {
            int best = -1;
            double bestValue = floorValue;
            for (int i = std::max(0, guess - win); i <= std::min(n - 1, guess + win); ++i)
                if (local_[static_cast<size_t>(i)] > bestValue)
                { bestValue = local_[static_cast<size_t>(i)]; best = i; }
            if (best < 0) break;
            front.push_back(best);
            guess = best;
        }
        std::reverse(front.begin(), front.end());

        std::vector<int> back;
        for (int guess = beatFrames.back() + step; guess + win <= n - 1; guess += step)
        {
            int best = -1;
            double bestValue = floorValue;
            for (int i = std::max(0, guess - win); i <= std::min(n - 1, guess + win); ++i)
                if (local_[static_cast<size_t>(i)] > bestValue)
                { bestValue = local_[static_cast<size_t>(i)]; best = i; }
            if (best < 0) break;
            back.push_back(best);
            guess = best;
        }

        if (front.empty() && back.empty()) return;
        std::vector<int> merged;
        merged.reserve(front.size() + beatFrames.size() + back.size());
        merged.insert(merged.end(), front.begin(), front.end());
        merged.insert(merged.end(), beatFrames.begin(), beatFrames.end());
        merged.insert(merged.end(), back.begin(), back.end());
        beatFrames.swap(merged);
    }

    /** @brief Conditioned envelope smoothed by a Gaussian a thirty-second of
     *  the beat period wide, so that a beat that lands a frame or two off a
     *  peak is not read as landing on nothing. */
    void buildLocalScore(double periodFrames)
    {
        const int n = static_cast<int>(env_.size());
        local_.assign(static_cast<size_t>(n), 0.0);

        const double sigma = std::max(0.5, kLocalScoreSigmaPeriods * periodFrames);
        const int half = std::max(1, static_cast<int>(std::lround(3.0 * sigma)));
        kernel_.assign(static_cast<size_t>(2 * half + 1), 0.0);
        double ksum = 0.0;
        for (int k = -half; k <= half; ++k)
        {
            const double x = static_cast<double>(k) / sigma;
            const double v = std::exp(-0.5 * x * x);
            kernel_[static_cast<size_t>(k + half)] = v;
            ksum += v;
        }
        if (ksum > 0.0)
            for (double& v : kernel_) v /= ksum;

        for (int i = 0; i < n; ++i)
        {
            double acc = 0.0;
            const int kLo = std::max(-half, -i);
            const int kHi = std::min(half, n - 1 - i);
            for (int k = kLo; k <= kHi; ++k)
                acc += env_[static_cast<size_t>(i + k)]
                     * kernel_[static_cast<size_t>(k + half)];
            local_[static_cast<size_t>(i)] = acc;
        }
    }

    /** @brief Drops leading and trailing beats the envelope does not support.
     *  The recursion must start and stop somewhere and it will place a beat
     *  in silence rather than leave the chain open; those are the two ends. */
    void trimBeats(std::vector<int>& beatFrames) const
    {
        if (beatFrames.size() < 3) return;

        double mean = 0.0;
        for (const int f : beatFrames) mean += local_[static_cast<size_t>(f)];
        mean /= static_cast<double>(beatFrames.size());
        const double floorValue = kTrimFraction * mean;

        size_t first = 0;
        while (first + 2 < beatFrames.size()
               && local_[static_cast<size_t>(beatFrames[first])] < floorValue)
            ++first;

        size_t last = beatFrames.size() - 1;
        while (last > first + 1
               && local_[static_cast<size_t>(beatFrames[last])] < floorValue)
            --last;

        if (first > 0 || last + 1 < beatFrames.size())
            beatFrames = std::vector<int>(beatFrames.begin()
                                              + static_cast<std::ptrdiff_t>(first),
                                          beatFrames.begin()
                                              + static_cast<std::ptrdiff_t>(last) + 1);
    }

    /**
     * @brief Position of a beat in the caller's timeline, to better than one
     *        analysis frame.
     *
     * Frames are one hop apart -- five milliseconds at the default settings --
     * and a grid quantised to them carries a sawtooth error of half a hop that
     * has nothing to do with the tracking. The envelope around a transient is
     * smooth, so fitting a parabola through the frame and its two neighbours
     * recovers where the peak really fell between them. The correction is
     * clamped to half a frame: beyond that the parabola is describing
     * something other than one peak.
     */
    [[nodiscard]] int64_t beatSample(int f) const noexcept
    {
        const int n = static_cast<int>(env_.size());
        const int64_t base = envRef_[static_cast<size_t>(f)];
        if (f <= 0 || f >= n - 1) return base;
        const double a = local_[static_cast<size_t>(f - 1)];
        const double b = local_[static_cast<size_t>(f)];
        const double c = local_[static_cast<size_t>(f + 1)];
        const double den = a - 2.0 * b + c;
        if (!(den < 0.0)) return base;   // not a peak: nothing to interpolate
        const double d = std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
        return base + static_cast<int64_t>(std::llround(d * static_cast<double>(hop_)));
    }

    /** @brief Least-squares samples-per-beat through the delivered grid. */
    [[nodiscard]] static double fitBeatSlope(const std::vector<int64_t>& beats) noexcept
    {
        const size_t n = beats.size();
        if (n < 2) return 0.0;
        const double dn = static_cast<double>(n);
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (size_t k = 0; k < n; ++k)
        {
            const double x = static_cast<double>(k);
            const double y = static_cast<double>(beats[k]);
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        const double den = dn * sxx - sx * sx;
        if (!(std::abs(den) > 0.0)) return 0.0;
        return (dn * sxy - sx * sy) / den;
    }

    // -- State ---------------------------------------------------------------

    void resetState() noexcept
    {
        onset_.reset();
        std::fill(zRe_.begin(), zRe_.end(), 0.0);
        std::fill(zIm_.begin(), zIm_.end(), 0.0);
        std::fill(zMass_.begin(), zMass_.end(), 0.0);
        std::fill(bankCoherence_.begin(), bankCoherence_.end(), 0.0);

        samplesPushed_ = 0;
        frameIndex_ = 0;
        lastFrameRef_ = 0;
        baseline_ = 0.0;
        prevSinceBeat_ = 0.0;
        framesSinceBeat_ = 0;
        lastBestIndex_ = -1;
        havePhase_ = false;

        tempoAndConfidence_.store(packTempo(0.0f, 0.0f), std::memory_order_relaxed);
        beatLatched_.store(false, std::memory_order_relaxed);
        lastBeatSample_.store(-1, std::memory_order_relaxed);
    }

    /// Shortest gap between two reported causal beats, as a fraction of the
    /// period in force.
    static constexpr double kBeatRefractory = 0.5;

    /// Below this the exponentially-windowed onset mass is numerically nothing
    /// and the coherence ratio would be a quotient of two rounding errors.
    static constexpr double kMassFloor = 1e-12;

    // -- Members -------------------------------------------------------------

    OnsetDetector<T> onset_;

    double sampleRate_ = 44100.0;
    double frameRate_ = 200.0;
    int hop_ = 221;
    int warmupFrames_ = 12;

    // Resonator bank (built at prepare, read on the audio thread).
    int bankSize_ = 0;
    std::vector<double> bankPeriod_;
    std::vector<double> bankCosCoef_, bankSinCoef_, bankDecay_, bankPrior_;
    int harmonicOffset_[kNumHarmonics] = { 0, 0 };
    std::vector<double> zRe_, zIm_, zMass_, bankCoherence_;

    // Causal stream state (audio thread only).
    int64_t samplesPushed_ = 0;
    int64_t frameIndex_ = 0;
    int64_t lastFrameRef_ = 0;
    double baseline_ = 0.0;
    double baselineCoef_ = 0.005;
    double prevSinceBeat_ = 0.0;
    int64_t framesSinceBeat_ = 0;
    int lastBestIndex_ = -1;
    bool havePhase_ = false;

    // Offline scratch (analyze only).
    std::vector<double> env_;
    std::vector<int64_t> envRef_;
    std::vector<double> local_, cumScore_, acf_, scratchA_, kernel_;
    std::vector<double> localPeriod_, sweepA_, sweepB_;
    std::vector<int> backlink_;
    std::vector<int> candidates_;
    std::vector<double> candScore_;
    double secondaryShare_ = 0.0;

    // Published state.
    std::atomic<bool> prepared_ { false };
    std::atomic<std::uint64_t> tempoAndConfidence_ { 0 };
    std::atomic<std::uint32_t> activeRange_ { 0 };
    std::atomic<double> tightness_ { kDefaultTightness };
    std::atomic<bool> beatLatched_ { false };
    std::atomic<std::int64_t> lastBeatSample_ { -1 };
    std::atomic<std::int64_t> latencySamples_ { 549 };
};

} // namespace dspark
