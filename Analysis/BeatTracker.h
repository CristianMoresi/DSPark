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
 * decided: one global period is the right model for that question because it
 * is the stable one. The grid is then laid in TWO PASSES, because holding a
 * grid and following a tempo ask opposite things of alpha. A recursion loose
 * enough to follow a tempo that moves is loose enough to take a beat off the
 * grid onto a louder note beside it and bring it back at the next one -- the
 * anticipated sixteenth of a samba, the swung eighth of a shuffle -- and one
 * tight enough to refuse that loses a tempo that leaves its period. So a
 * first, loose pass only follows. Its intervals say what period the music
 * keeps overall and what period it moves at around each beat, and the second
 * pass is held, at the tightness asked for, to the overall period wherever
 * the two agree, and follows the local one where they part. How far apart
 * they have to be depends on how well the intervals around the beat agree
 * with one another: spread out, a period a few hundredths off the overall
 * one is the overall one read badly, and the grid is held up to a twelfth
 * of the period away; all alike, it is a tempo of its own, and the grid
 * follows it from a fiftieth. An interval of the held pass costs at most a
 * fixed number of beats' worth of onset strength, so that a change of phase
 * the music made -- a bar of another length, an edit -- is taken at once
 * instead of being slid into over the beats around it. Where the tempo
 * swings faster than a period can be read over a few beats, the held grid
 * ends up carrying less than half of what the loose one carries, and the
 * loose grid is the one delivered. On a 100-to-140 BPM ramp the worst local
 * tempo error is 1.0% and every beat is found; on a tempo swinging by 15%
 * every sixteen beats or every nine, every beat is found.
 *
 * A tempo that changes at once is taken at the beat it changes on. The
 * period around the change is read on the side that is steady, and a second
 * tempo too far from the first for the loose pass to reach is looked for
 * between the two. Clicks that go from one tempo to another between 90 and
 * 140 BPM, by a twentieth or by two fifths, once or back and forth, with a
 * third of them or more at each tempo, are all beats, and none is further
 * from its click than on a steady train (7.6 ms against 7.7). A second
 * tempo kept for a quarter of the signal or less is not followed. See
 * layGrid().
 *
 * MEASURED against annotated beats, as the mean F-measure at +/-70 ms, on
 * three collections: 698 excerpts of dance music, 243 takes of a drum kit
 * alone and 360 of a guitar alone. One loose pass with its beats moved to
 * the nearest envelope peak, which is what this path did before / the two
 * passes / a reference tracker of the same family (a dynamic program over
 * an onset function), run in the same job: against the annotation as it is,
 * 0.776 / 0.830 / 0.784, 0.760 / 0.805 / 0.763 and 0.520 / 0.559 / 0.608;
 * against its best reading (as annotated, twice or half as fast, on the
 * off-beats, in threes), which takes the choice of metrical level out of
 * the number, 0.838 / 0.919 / 0.891, 0.860 / 0.925 / 0.917 and 0.668 /
 * 0.786 / 0.775. The share of the grid's intervals within an eighth of
 * their median went from 0.88, 0.92 and 0.78 to 0.998, 0.997 and 0.995.
 * What is left behind the reference is the level chosen for the guitar
 * alone: the annotated tempo in 53% of the pieces against 61%, and half of
 * it in 14% against 5%.
 *
 * The grid is no longer moved onto envelope peaks afterwards. The recursion
 * already places each beat on the frame where its running total is largest,
 * and a beat moved to the largest peak within an eighth of a period, twice
 * over, lands on the note beside the beat as readily as on the beat. Two
 * things that step used to cover are handled where they arise. A chain is
 * free to start at any frame, where it used to open with an interval from
 * the first frames of the signal to its first note (see runDp()). And the
 * two ends of the grid are continued for as long as the envelope supports a
 * beat, down to a quarter of what a beat of the grid carries: excerpts fade
 * and endings thin out, and at a half the grid stopped 1.6 beats short of
 * the annotated ones per piece of dance music, against 0.6 now.
 *
 * DENSE MIXES. On a loud modern master the onset envelope is mostly strums,
 * hats and syllables, because they own most of its quarter-tone bands, while
 * the kick that states the beat lights a handful of bands below 200 Hz. The
 * metrical level is therefore decided on a REGISTER-BALANCED envelope - the
 * front end's flux split into four registers (below 200 Hz, to 800 Hz, to
 * 3.2 kHz, above), each conditioned on its own and weighted by its own
 * periodicity - so the kick counts as much as the hats. Measured on 46
 * synthetic pop masters from 92 to 150 BPM (kick on every beat, snare on two
 * and four, strummed 3-3-2 guitar, syncopated bass, syllabic vocal,
 * saturation and limiting), the delivered level is right on all 46, and
 * tempoBpm agrees with the grid's median interval within its frame
 * quantisation (0.55%). Correlation lags that are only bumps of the
 * correlation's noise floor are not candidates at all, and neither is a lag
 * on the shoulder of a taller peak of the same reading (see
 * kNeighbourPeakFloor).
 *
 * TIGHTNESS. The default is 400, the value the method's paper names, and it
 * means something narrower here than there: it is the tightness of the held
 * pass, where the music keeps its period. The first pass runs at 25, and
 * where the tempo moves the delivered grid is laid at 25 as well. Measured
 * on the three collections above, against the best reading of the
 * annotation: from 0.859, 0.869 and 0.690 at 25 to 0.919, 0.925 and 0.786 at
 * 400, level with it at 800 and lower again at 1600 (0.912, 0.923, 0.784).
 * On a bed of clicks where the interval cost is live -- a tempo swinging by
 * 15% every sixteen beats and every nine, and a bar whose loudest note sits
 * a sixteenth before a beat that carries nothing -- the default finds every
 * beat of all three, and so does a grid four times tighter: where the tempo
 * moves the grid is not held, whatever it is held with elsewhere. With the
 * term deleted the third loses a quarter of its beats, and the one loose
 * pass with its beats moved to the nearest peak, which this path was
 * before, lost a fifth of them. See setTightness().
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
 * rather than equal wall-clock spans. The bank chooses the period; the beats
 * are placed by a cumulative beat score after Stark, Davies and Plumbley,
 * "Real-time beat-synchronous analysis of musical audio", DAFx 2009,
 *
 *     C[n] = (1 - a) * o[n] + a * max_d { W(d / tau) * C[n - d] },
 *
 * with a = 0.9 and W a log-Gaussian over [tau / 2, 2 tau], and the next beat is
 * the strongest point of C continued one period ahead with no further onsets.
 * No backtracking, no future context. The period the score follows is the
 * metrical level described below. The score reads o as the sum of two
 * functions of the front end, each conditioned the same way and divided by its
 * own running peak (10 s of memory): the log-band envelope, where every
 * register owns its share, and the linear-magnitude flux of the same spectrum
 * (OdfFrame::linear), where the loudest sound does. On syncopated material the
 * busiest sound is often on the off-beats and the loudest on the beat.
 *
 * Why the beats are not read from the resonator's phase, which would be free:
 * that phase is the circular mean of the onset mass, which is what the
 * coherence below needs, but as a beat position it is pulled towards any
 * regular off-beat by the off-beat's share of the mass. On swung, dotted and
 * syncopated clicks (off-beat at 0.6, 2/3 and 3/4 of the beat, 0.15 to 0.7 of
 * its level, 96 and 120 BPM) the phase turn put the beats a mean 11 to 105 ms
 * early and 374 of 672 within 70 ms; with the beat score all 660 are within
 * 70 ms, at a mean of 1.2 to 2.0 ms late. Where two pulse trains of the same
 * period score within 5% of each other the score cannot say which is the
 * beat, and the beat falls back to the phase turn: at 0.7 of the beat's
 * level, 3/4 through it, 120 BPM, that keeps 31 of 31 beats within 70 ms
 * (30 with the envelope alone in the score). An off-beat at the beat's own level is the
 * same pulse at two phases, and neither placement finds the annotated one.
 *
 * Which METRICAL LEVEL the beat clock follows is a third question, and the
 * bank's coherence is the wrong measure for it on syncopated material: where
 * the off-beats carry as much onset mass as the beats, the beat level holds a
 * tenth of the coherence the off-beat level does (0.05 to 0.16 against 0.92
 * on a groove with the kick on the off-beats and the snare on the backbeat),
 * and the bank follows the off-beats. What still names the beat there is that
 * the envelope REPEATS at it. So the level comes from a comb-filter stage
 * after Davies and Plumbley, "Context-dependent beat tracking of musical
 * audio", IEEE TASLP 15(3), 2007: every kCombUpdateFrames the last six seconds
 * of the envelope are autocorrelated (by FFT), each candidate period is scored
 * by the mean of the correlation at its first four multiples, plus half the
 * correlation at half the period, under a Rayleigh prior peaking at 0.40 s,
 * and successive estimates are linked by a Gaussian transition on log period.
 * The half period is the subdivision of a binary level: a tresillo or clave
 * (3-3-2 eighths) repeats strongly at three eighths, a level that has no
 * subdivision on the grid, and without it the comb took that level, four
 * thirds of the annotated beat on 31 of the 122 Salsa recordings (3 now). A comb reads repetition, which a pulse train
 * shows at every multiple of its period, so its choice is moved down to a
 * subharmonic that coherence shows to carry the onset mass in phase
 * (foldToFundamental()): a slow click train is not followed at twice its
 * rate. The prior pulls the other way above it: a 200 BPM click train
 * repeats at 100 BPM as well as at 200, so the comb's half is overruled when
 * every register repeats at the bank's period as well as at twice it
 * (sameEventsAtHalfPeriod()) -- alike events, not a groove whose off-beats
 * are other instruments. Until the comb has six seconds' worth, a bank level
 * held for four of its beats steers the clock.
 *
 * The published tempo stays the bank's winner, the pulse that carries the
 * onset mass in phase; getBeatTempoBpm() names the level the beats follow.
 * The two agree on most material and differ by a metrical factor exactly on
 * the syncopated case above. Measured, causal beat F-measure (+/-70 ms,
 * beats after the first 5 s): ISMIR 2004 ballroom collection plus its
 * additional excerpts (698 recordings), 0.740 and 0.757 on the two halves of
 * the recording groups (0.706 and 0.728 with the envelope alone in the score
 * and the comb without the half period, at a 0.45 s prior); first 30 s of the
 * Salsa Dataset (122 recordings, the odd ones used for the choices), 0.399 and
 * 0.420 on the two halves (0.321 and 0.331 before); three syncopated grooves
 * at 100, 124 and 140 BPM, 0.980, 0.984 and 0.986; click trains from 40 to 240
 * BPM unchanged. On the off-beat sweep above, 660 beats within 70 ms against
 * 659 before, at a mean of 1.2 to 2.0 ms late (2.6 to 4.0 before). A
 * syncopated groove swept from 88 to
 * 152 BPM in 2 BPM steps over three seeds reads 0.916 to 0.918 against 0.908
 * to 0.909 (98 BPM goes from double tempo to the beat; 118 to 120 BPM, where
 * both lose the beat's phase part of the time, reads 0.77 to 0.83 against
 * 0.77 to 0.87). Cost: 1.78 ms of CPU per second of audio on the Salsa
 * recordings (1.69 before); every 32 frames the estimate is a burst of two
 * 4096-point transforms and some 10^4 to 10^5 multiplies, allocation free.
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
 * The reported confidence is C at the delivered tempo, reduced where a
 * different metrical level explained the signal nearly as well (see THE
 * SECONDARY HYPOTHESIS). It answers how much of what the signal did is
 * explained by the grid that was returned, and how much of that explanation is
 * unique to it. It is NOT a probability that the tempo is right, and in
 * particular it CANNOT flag a metrical-level error: the waveform argument
 * under THE OCTAVE shows two pieces of music with different correct answers
 * that are the same signal, so any number computed from the signal is the same
 * for both. A caller who needs the level checked reads secondaryTempoBpm,
 * always, not the confidence. What the confidence does report faithfully is
 * the state where the signal genuinely does not determine the level: there the
 * two readings explain it equally, the discount is total, and the number goes
 * to zero.
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
 *     score(tau) = W(tau) * C(tau) * prod_m (1 - min(1, R_m(tau) * C(m*tau))),
 *     m in {2, 3, 5},
 *
 * where each extra factor asks whether the candidate's own subharmonic is just
 * as coherent as the candidate: if it is, the candidate is a harmonic of the
 * real pulse and the real pulse is the slower one. A candidate that is k times
 * the true pulse is caught whenever k is divisible by one of the divisors, so
 * two, three and five catch every k up to six -- every relation a range
 * spanning a factor of six can hold, which the default 40..240 does exactly.
 * None of this is academic: without the m = 3 factor a 60 BPM pulse locks the
 * causal path to 180 BPM at a confidence of 0.994, and without the m = 5
 * factor 40, 42, 44 and 46 BPM all publish five times too fast, inside the
 * default range. W is the log-Gaussian tempo prior of the paper (eq. 6),
 * centred on 0.5 s per beat with a width of 1.4 octaves; it is the weakest
 * factor on purpose, being the only one that does not look at the signal.
 *
 * R_m is the TAPPING PREFERENCE of the subharmonic against the candidate,
 * never below one, and it is there because of a limit that has to be stated
 * rather than engineered away. Take a click train at rate R with alternate
 * events attenuated to a fraction of the others. That single waveform is two
 * different pieces of music: a beat at R with a backbeat, and a beat at R/2
 * with straight eighths. Nothing in the envelope distinguishes them -- they
 * ARE the same envelope -- so no measurement can, and C(2*tau) is exactly the
 * depth of the attenuation in both readings. What separates them is only which
 * absolute rate a listener would tap, so that is what R_m carries: a sharper
 * preference than W, and asymmetric, because a level at twice the preferred
 * rate is a far less plausible tactus than one at half of it. It can only ever
 * make the test reject a candidate more readily, never protect a plain
 * harmonic, whose subharmonic is fully coherent and whose rejection stays
 * absolute.
 *
 * Where that leaves the double, measured on straight eighths swept from 0.05
 * to 1.0 of the beat's amplitude at 70..140 BPM: the tracker reports the beat
 * while the eighths stay below an onset-strength ratio of about 0.44 at 90
 * BPM, 0.72 at 100 and 0.90 at 120, against 0.28 to 0.37 without R_m. Above
 * that ratio it reports the eighth level, with the beat in
 * secondaryTempoBpm. Below 90 BPM it reports the eighth level from a ratio of
 * 0.28 up, and that is the tapping-preferred reading rather than a defect:
 * 140 BPM is a likelier tactus than 70. See getConfidence() for what the
 * confidence can and cannot tell a caller about this.
 *
 * THE SECONDARY HYPOTHESIS is a competitor, not a label. It is the
 * best-scoring candidate at least a quarter of an octave from the winner --
 * the metrical level a listener could plausibly have tapped instead -- and it
 * is 0 when the envelope offers no such alternative inside the searched range,
 * which is the honest answer for a pulse near the bottom of that range. It
 * does two things, and it does them from two different scores because they are
 * two different questions. What is REPORTED is the best alternative after the
 * tapping preference, since the point of reporting it is to name a level a
 * caller might have wanted instead. What DISCOUNTS the confidence is the best
 * explanation of the same envelope at any other level, tapping preference set
 * aside, because a grid that fits is not the same claim as a grid that is the
 * only one that fits -- and a discount taken from the adjudicated score would
 * measure the tracker's own decision rather than the signal, reporting
 * certainty that the preference had manufactured. On a three-against-two
 * polyrhythm, where both pulses are real, the discount takes the reported
 * confidence from 0.52 to between 0.02 and 0.38, while leaving every
 * unambiguous case in the acceptance corpus untouched. See kAmbiguityFloor for
 * why it has a floor under it.
 * An offline level change retains the displaced reading as its secondary
 * hypothesis and reevaluates ambiguity against the selected period.
 *
 * ON RECORDED MUSIC. Measured with analyze() against the tempo each
 * recording is annotated with, counting an estimate within 4% of it (and, in
 * the second figure, within 4% of twice, three times, half or a third of it):
 *
 *   ISMIR 2004 ballroom, original 454       76.2%   90.3%
 *   Salsa Dataset, 122 recordings          68.9%   81.1%
 *   Freesound Loops 4k, 3223 loops (*)      51.0%   69.7%
 *   Ballroom, additional 242 excerpts (**) 74.8%   88.8%
 *
 * Native-rate mono input; whole ballroom excerpts and the first 30 s of
 * Salsa (two recordings have too few annotated beats in that span).
 *   (*) all 3228 decoded WAV loops, less five annotated outside 40..240 BPM;
 *       each repeated to 20 s. These replace the earlier 3008-loop subset,
 *       so its 49.9% / 67.1% figures are not a matched comparison.
 *   (**) absent from the original fit; two further excerpts matching its
 *        recordings are excluded. Reserved while developing the slope fix.
 *
 * The binary level weights retain their original 454/122/3008 fit. The
 * rational alternatives use the reconstructed 454/122/3223 collections, with
 * recording/loop-pack groups kept together in five folds and equal dataset
 * weights. The first three rows reuse training material: they are descriptive
 * results, not independent generalisation estimates. The additional ballroom
 * excerpts were excluded from both fits but were used for earlier slope
 * validation; they are not a new untouched product-level test set.
 *
 * Grouped validation of the added scorer retains the binary coefficients.
 * Exact loop matches increase from 1630 to 1634 of 3223 and metrical-tolerance
 * matches from 2150 to 2224; ballroom and Salsa counts are unchanged. This
 * validates the extension only, since the binary scorer has training overlap.
 * The small exact-tempo change does not establish a general accuracy gain.
 * The five fold models are averaged as logits into one scorer before the
 * additional ballroom check: exact matches stay 181/242, metrical matches
 * increase from 214 to 215. The descriptive table uses this final scorer.
 *
 * Rational period alternatives include 1/3, 2/3, 3/2 and 3 of the proposal,
 * subject to range and pulse support. They do not identify a time signature.
 * The level a dance is published at is partly a convention, so callers should
 * offer secondaryTempoBpm when choosing the metrical level matters.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with anything else).
 * - processBlock() / pushSamples(): audio thread, stream owner. Allocation
 *   free, lock free, no throw.
 * - getBeatTempoBpm(): any thread, lock free, one word.
 * - getRunningTempoBpm() / getConfidence(): any thread, lock free. The two
 *   travel as ONE packed 64-bit word because they are a set: a confidence
 *   belongs to the tempo it was measured at, and a caller that gates on
 *   confidence before trusting a tempo must never see one of them refreshed
 *   without the other. Read them together with getTempoAndConfidence() when
 *   both are wanted; the two separate getters unpack the same single load and
 *   so are consistent individually, but two separate calls can straddle an
 *   update.
 * - beatNow() / getLastBeatSample(): any thread, lock free, two independent
 *   words, ORDERED against each other. The position is written first and the
 *   latch is released last; beatNow() acquires. READ THEM IN THAT ORDER: test
 *   the latch first, then read the position, and the position is the one
 *   written in the call that set the latch or in a later one. Without the
 *   pairing the two are relaxed loads of distinct objects and a weakly ordered
 *   machine is free to satisfy them in either order, which would let a caller
 *   be told a beat just happened and handed the previous beat's position -- a
 *   full beat period out, up to 1.5 s at the bottom of the tempo range. The
 *   reverse reader order (position first, latch second) is NOT made safe by
 *   the pairing and is not a memory-model question at all: the writer may run
 *   between the reader's two loads, so the reader holds a position from before
 *   that gap and a latch from after it. Measured, that order returns a stale
 *   position on essentially every beat it observes, by a full beat period and
 *   by more when the reader's two loads are further apart than one beat.
 * - setTempoRange() / setTightness(): control thread, lock free. The range is
 *   two indices with an invariant between them, so it travels as one packed
 *   word as well; setTempoRange() never allocates, because the resonator bank
 *   is built once at prepare() over the whole supported span and the range
 *   only selects which part of it is searched.
 * - analyze() / reset(): setup thread. analyze() allocates and CLEARS the
 *   causal state, so it must not run while the audio path is running.
 * - beginOffline() / pushOffline() / finishOffline(): setup thread, the
 *   incremental form of analyze() with the same rules: allocates, clears the
 *   causal state, and must not be interleaved with the audio path.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O, so it is unaffected by DSPARK_NO_FILE_IO.
 *
 * Dependencies: DspMath.h, AudioBuffer.h, AudioSpec.h, FFT.h, OnsetDetector.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/FFT.h"
#include "../Core/detail/AccountedAllocator.h"
#include "OnsetDetector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace dspark {

namespace detail { template <FloatType T> struct OfflineBeatEngine; }

/**
 * @class BeatTracker
 * @brief Tempo and beat tracking with an offline grid and a causal readout.
 *
 * Role: analysis readout. It consumes const audio and never mutates it. All
 * heap use for the causal path happens in prepare(); processBlock() allocates
 * nothing, takes no lock and throws nothing. analyze() is offline and allocates
 * input-dependent envelope and grid working storage.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class BeatTracker final
{
    template <typename U>
    using WorkVector = std::vector<U, detail::AccountedAllocator<U>>;
    friend struct detail::OfflineBeatEngine<T>;

public:
    BeatTracker()
    {
        const detail::AccountedAllocator<double> doubles(&allocationAccount_);
        for (auto* values : {&bankPeriod_, &bankCosCoef_, &bankSinCoef_, &bankDecay_,
                             &bankPrior_, &zRe_, &zIm_, &zMass_, &bankCoherence_,
                             &bankLevelRatio_, &env_, &envBal_, &local_, &cumScore_,
                             &acf_, &scratchA_, &kernel_, &localPeriod_,
                             &localTightness_, &candScore_, &candExplains_, &beatScore_,
                             &beatScratch_, &beatAhead_, &beatLogLag_, &beatWeightTable_,
                             &combRing_, &combSum_, &combTime_, &combSpec_, &combAcf_,
                             &combDelta_, &combNext_, &combStep_, &combPrior_, &combKernel_,
                             &regRing_})
            *values = WorkVector<double>(doubles);
        for (auto& values : envReg_) values = WorkVector<double>(doubles);
        envRef_ = WorkVector<int64_t>(detail::AccountedAllocator<int64_t>(&allocationAccount_));
        backlink_ = WorkVector<int>(detail::AccountedAllocator<int>(&allocationAccount_));
        candidates_ = WorkVector<int>(detail::AccountedAllocator<int>(&allocationAccount_));
        onset_.setLinearFlux(true); // the beat score reads OdfFrame::linear
    }

    /** @brief What analyze() returns: one tempo, one grid, one number saying
     *         how much of the signal that grid explains, and the metrical
     *         alternative that lost. */
    struct Result
    {
        T tempoBpm = T(0);           ///< Delivered tempo, fitted to the grid.
        T confidence = T(0);         ///< Coherence of the envelope with the grid, [0,1].
        T secondaryTempoBpm = T(0);  ///< The competing hypothesis that lost.
        std::vector<int64_t> beatSamples; ///< Beat positions, ascending, in samples, never negative.
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
     *
     * The tempo range and tightness survive it, whether they were set before
     * the first prepare() or between two.
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
        frameRate_ = sampleRate_ / static_cast<double>(hop_);
        baselineCoef_ = 1.0 - std::exp(-1.0 / std::max(1.0, kBaselineSeconds * frameRate_));
        peakDecay_ = std::exp(-1.0 / std::max(1.0, kScorePeakSeconds * frameRate_));

        latencySamples_.store(static_cast<int64_t>(onset_.getEnvelopeLatencySamples())
                                  + 2 * static_cast<int64_t>(hop_),
                              std::memory_order_relaxed);

        buildBank();
        buildComb();
        applyTempoRange();

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
     * analyze(). The published running tempo stays inside it.
     */
    void setTempoRange(T minBpm, T maxBpm) noexcept
    {
        if (!std::isfinite(minBpm) || !std::isfinite(maxBpm)) return;
        if (!(minBpm > T(0)) || !(maxBpm > T(0)) || !(minBpm <= maxBpm)) return;

        const double lo = std::clamp(static_cast<double>(minBpm),
                                     kBankMinBpm, kBankMaxBpm);
        const double hi = std::clamp(static_cast<double>(maxBpm),
                                     kBankMinBpm, kBankMaxBpm);
        requestedRange_.store(packTempo(static_cast<float>(lo), static_cast<float>(hi)),
                              std::memory_order_relaxed);
        applyTempoRange();
    }

    /// Selects the part of the bank the requested range covers.
    void applyTempoRange() noexcept
    {
        if (bankSize_ < 2) return;
        float requestedLo = 0.0f, requestedHi = 0.0f;
        unpackTempo(requestedRange_.load(std::memory_order_relaxed), requestedLo, requestedHi);
        const double lo = static_cast<double>(requestedLo);
        const double hi = static_cast<double>(requestedHi);

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
     * larger holds the grid closer to its period, smaller lets it follow the
     * envelope. Offline path only -- the causal path has no transition cost
     * to weigh. Non-finite values are ignored; values are clamped to a
     * positive floor because a tightness of zero removes the term the
     * recursion is built on and leaves the grid free to place a beat at every
     * envelope peak.
     *
     * WHAT IT HOLDS. The grid is laid in two passes (see layGrid()). The
     * first follows the tempo, at the lesser of this value and 25. The second
     * is held at this value to the period the first one kept, wherever the
     * first one kept it, and follows at the first pass's value where it did
     * not. So the number is the tightness of the passages at a steady tempo,
     * and a value of 25 or less lays both passes alike: one loose grid.
     *
     * Default 400, the value the method's paper names as its default.
     *
     * MEASURED on annotated music -- 698 excerpts of dance music, 243 takes
     * of a drum kit alone, 360 of a guitar alone -- as the mean F-measure at
     * +/-70 ms against the best reading of the annotation (as annotated,
     * twice or half as fast, on the off-beats, in threes):
     *
     *   alpha       25     100    200    400    800   1600
     *   dance     0.859  0.898  0.912  0.919  0.919  0.912
     *   drums     0.869  0.906  0.920  0.925  0.926  0.923
     *   guitar    0.690  0.752  0.773  0.786  0.787  0.784
     *
     * and against the annotation as it is, 0.789, 0.819, 0.828, 0.830, 0.832
     * and 0.826; 0.768, 0.794, 0.803, 0.805, 0.805 and 0.802; 0.525, 0.542,
     * 0.553, 0.559, 0.557 and 0.558. Below the default the grid leaves the
     * beat for the notes beside it: at 25 the share of its intervals within
     * an eighth of their median is 0.955, 0.955 and 0.893, against 0.998,
     * 0.997 and 0.995 at 400. The default sits where the three rows level
     * off, and the paper's value is on that level; 800 is level with it and
     * 1600 is lower on all three.
     *
     * And on clicks, where the truth is exact, F at +/-70 ms: a tempo
     * swinging by 15% every sixteen beats and every nine, and a bar whose
     * loudest note is a sixteenth before a beat that carries nothing, read
     * 1.000, 1.000 and 1.000 at the default, and at 1600 as well: the grid
     * is held only where the tempo holds. With the term deleted (0.001) the
     * third reads 0.765.
     *
     * A CAUTION ON MEASURING IT. On isochronous clicks this parameter does
     * next to nothing: every interval is the target period, so the cost it
     * weighs is zero, and the grids at alpha = 0.001 -- which deletes the
     * term outright -- and at alpha = 4000 name the same beats within two
     * analysis frames of each other. A sweep run only on steady clicks
     * measures nothing and will report no spread whatever the code does.
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

    /** @brief The tightness in force until setTightness() is called. */
    [[nodiscard]] static constexpr T defaultTightness() noexcept
    {
        return static_cast<T>(kDefaultTightness);
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
        if (!prepared_.load(std::memory_order_relaxed)) return {};
        if (whole.getNumChannels() < 1 || whole.getNumSamples() <= 0) return {};
        const int n = whole.getNumSamples();
        beginOffline(n);
        pushOffline(std::span<const T>(whole.getChannel(0), static_cast<size_t>(n)));
        return finishOffline();
    }

    /**
     * @brief Opens an incremental offline analysis.
     *
     * For material that arrives in pieces -- a file decoded block by block, a
     * recording too long to hold -- without concatenating it first. Feed it
     * with pushOffline() in blocks of any size and close it with
     * finishOffline(); the Result is bit-identical to analyze() over the
     * concatenation, whatever the blocking. Only the onset envelope is kept
     * between calls (a few values per 5 ms hop), not the audio.
     *
     * Same threading as analyze(): it drives the shared front end, so it
     * clears the causal state on entry and must not be interleaved with
     * processBlock() / pushSamples() or run while the audio path is running.
     * @param expectedSamples Optional length hint, used only to reserve the
     *        envelope up front.
     */
    void beginOffline(int64_t expectedSamples = 0)
    {
        env_.clear();
        envRef_.clear();
        for (auto& r : envReg_) r.clear();
        offlineOpen_ = false;
        if (!prepared_.load(std::memory_order_relaxed)) return;

        resetState();
        if (expectedSamples > 0)
        {
            const size_t frames = static_cast<size_t>(expectedSamples / std::max(1, hop_) + 2);
            env_.reserve(frames);
            envRef_.reserve(frames);
            for (auto& r : envReg_) r.reserve(frames);
        }
        offlinePushed_ = 0;
        offlineOpen_ = true;
    }

    /** @brief Feeds the next piece of an offline session (channel-0 samples).
     *         No-op outside one. */
    void pushOffline(std::span<const T> samples)
    {
        if (!offlineOpen_) return;
        size_t off = 0;
        while (off < samples.size())
        {
            // Pieces end on the front end's frame boundaries, so each frame's
            // envelope value is read as the frame is computed.
            const int64_t since = offlinePushed_ % static_cast<int64_t>(hop_);
            const size_t take = std::min(samples.size() - off,
                                         static_cast<size_t>(static_cast<int64_t>(hop_) - since));
            onset_.pushSamples(samples.subspan(off, take));
            offlinePushed_ += static_cast<int64_t>(take);
            off += take;
            if (offlinePushed_ % static_cast<int64_t>(hop_) == 0) appendEnvelopeFrame();
        }
    }

    /** @brief Closes the session and tracks tempo and beats over everything
     *         pushed, as analyze() does. All-zero Result when no session is
     *         open. */
    Result finishOffline()
    {
        Result out;
        if (!offlineOpen_) return out;
        offlineOpen_ = false;
        onset_.reset();

        return finishEnvelope();
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

        // Release, and the matching acquire is in beatNow(). The latch and the
        // beat's position are two separate words written in this same call,
        // and a caller that sees the latch set must not then read a position
        // from an earlier call: on a weakly ordered machine two relaxed loads
        // of distinct objects may be observed in either order, so without this
        // pairing a reader could be handed the PREVIOUS beat's position -- an
        // error of one whole beat period -- while being told a beat just
        // happened. The store is last on purpose: everything the frame wrote,
        // the position included, is ordered before it.
        beatLatched_.store(fired, std::memory_order_release);
    }

    // -- Readout (lock-free) -------------------------------------------------

    /**
     * @brief Tempo the causal beat clock runs at, in BPM; 0 before it has
     *        one. Lock-free.
     *
     * The beats follow the metrical level of the comb stage (see the file
     * header), which reads how the envelope repeats; getRunningTempoBpm()
     * reports the pulse that carries the most onset mass in phase. They
     * agree on most material and differ by a metrical factor where an
     * off-beat carries as much onset mass as the beat: then this one names
     * the rate the beats are announced at.
     */
    [[nodiscard]] T getBeatTempoBpm() const noexcept
    {
        return static_cast<T>(beatTempo_.load(std::memory_order_relaxed));
    }

    /** @brief Running tempo estimate in BPM, 0 before the first frame. */
    [[nodiscard]] T getRunningTempoBpm() const noexcept
    {
        float bpm = 0.0f, conf = 0.0f;
        unpackTempo(tempoAndConfidence_.load(std::memory_order_relaxed), bpm, conf);
        return static_cast<T>(bpm);
    }

    /**
     * @brief Coherence of the envelope with the running tempo, in [0,1].
     *
     * The share of onset strength that falls in phase with the delivered
     * grid, discounted by how nearly another metrical level explained the same
     * envelope. See the file header for the definition and for the closed form
     * it can be checked against.
     *
     * WHAT IT DOES NOT TELL YOU. It is not a detector of half and double
     * tempo, and no number computed from the signal can be: a click train with
     * alternate events attenuated is at once a backbeat at the fast rate and
     * straight eighths at the slow one, so the two cases that a caller would
     * want told apart are the same waveform. Measured on straight eighths, the
     * tracker reports the eighth level above the amplitude ratios in the file
     * header and reports it with a confidence of 0.99, because that grid does
     * explain nearly all of the onset strength -- which is what the number
     * says, and it is true. Check secondaryTempoBpm whenever the metrical
     * level matters; it holds the other reading at every one of those points.
     *
     * A LOW value means one of two things and they are worth separating: the
     * grid explains little of what happened, or another level explains it just
     * as well. The second is the case the discount produces, and there the
     * number falls to zero rather than to something middling -- measured 0.000
     * on straight eighths at the amplitude where the two readings are level,
     * against 0.99 on either side of it.
     */
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

    /**
     * @brief True if a beat was reported during the most recent call.
     *
     * Acquire, paired with the release the processing call ends on: a reader
     * that sees this true and then reads getLastBeatSample() is guaranteed the
     * position written in that same call or a later one, never an earlier one.
     *
     * That is the order to use, and it is the only safe one. Reading the two
     * the other way round -- position first, latch second -- is not covered by
     * the pairing: the two loads are separate, the processing call may run
     * between them, and the reader is then told a beat just happened while
     * holding the position of an earlier one. Measured, that order returns a
     * stale position on essentially every beat it observes, by one full beat
     * period and by more when the reader's two loads are further apart than
     * one beat. See the threading notes in the file header.
     */
    [[nodiscard]] bool beatNow() const noexcept
    {
        return beatLatched_.load(std::memory_order_acquire);
    }

    /**
     * @brief Sample index the most recent causal beat is attributed to; -1
     *        before the first one.
     *
     * This is where the beat WAS, in the caller's own timeline, not when it
     * was announced. The two differ by getLatencySamples() and a caller that
     * aligns anything to the grid wants this one: the announcement carries the
     * analysis delay, the attribution does not.
     *
     * Relaxed on its own. To pair it with beatNow(), call beatNow() FIRST and
     * this second; the reverse order can hand back the position of an earlier
     * beat than the one the latch announced. See beatNow().
     */
    [[nodiscard]] int64_t getLastBeatSample() const noexcept
    {
        return lastBeatSample_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Delay between a beat and the call in which it is announced.
     *
     * Half the analysis frame less the front end's group-delay compensation,
     * plus two hops: a beat is announced on the frame after the one it is
     * placed on and attributed one hop before that frame (see
     * advanceBeatClock()). Fixed for a given sample rate and independent of
     * block size: 770 samples at 44.1 kHz.
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

    /** @brief Clears all streaming state and abandons an open offline
     *         session. Not concurrent with pushSamples(). */
    void reset() noexcept
    {
        offlineOpen_ = false;
        resetState();
    }

private:
    // Shared complete-envelope engine. The legacy sample frontend and the
    // pooled offline feature path both execute this exact arithmetic.
    Result finishEnvelope(void* beatContext = nullptr,
                          void (*receiveBeats)(void*, std::span<const int64_t>) = nullptr)
    {
        Result out;
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
        const bool balanced = (envBal_.size() == env_.size());
        if (balanced) env_.swap(envBal_);
        pickCandidates(iLo, iHi, tau1, tau2);
        if (balanced) env_.swap(envBal_);
        if (!(tau1 > 0.0)) { resetState(); return out; }

        // The ranking above proposes a reading; which multiple of it is the
        // beat is settled by the level model (see chooseMetricalLevel()). Only
        // where the signal has a pulse of its own in the range: otherwise the
        // correlation decided on its own, and so it stays.
        if (fundamentalFound_) chooseMetricalLevel(iLo, iHi, tau1, tau2);

        // The metrical level is decided against one period for the whole
        // signal, which is the stable model, and only then is the grid laid
        // down: held to the period the music keeps, following where it leaves
        // it (see layGrid()).
        WorkVector<int64_t> win(detail::AccountedAllocator<int64_t>{&allocationAccount_});
        if (!layGrid(tau1, win))
        {
            resetState();
            return out;
        }

        // The delivered tempo is fitted to the delivered grid, not read off
        // the autocorrelation lag: the grid spans the whole signal, so a
        // straight-line fit through it resolves the period far below the
        // frame spacing the lag is quantised to.
        const double slope = fitBeatSlope(win);
        if (receiveBeats)
            receiveBeats(beatContext, {win.data(), win.size()});
        else
            out.beatSamples.assign(win.begin(), win.end());
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
    // -- Constants -----------------------------------------------------------

    /// Searchable tempo span. A caller may set any range inside it.
    static constexpr double kBankMinBpm = 20.0;
    static constexpr double kBankMaxBpm = 480.0;
    /// The bank itself runs slower than the slowest searchable tempo, because
    /// the harmonic test below has to read resonators BELOW every candidate.
    /// Reaching as far past the bottom of the searchable range as the largest
    /// divisor the test uses keeps that test DEFINED for the slowest
    /// candidates instead of silently skipped; it is not known to change any
    /// answer. Reverting the reach to 3 while keeping the divisors leaves the
    /// causal range sweep at 0 wrong of 30, and the widening costs nothing
    /// measurable, so it is defensive rather than load-bearing.
    static constexpr double kHarmonicReach = 5.0;
    static constexpr int kBinsPerOctave = 128; ///< Bank resolution, ~0.54%.
    static constexpr double kDefaultMinBpm = 40.0;
    static constexpr double kDefaultMaxBpm = 240.0;

    /// Which subharmonics the metrical-level test reads. A candidate that is
    /// really a harmonic of the true pulse has a subharmonic just as coherent
    /// as itself, and the true pulse does not. A candidate that is the true
    /// pulse times k is caught whenever k is divisible by one of these
    /// divisors, so two and three alone leave k = 5 open -- and k = 5 fits
    /// inside the default range at its bottom, where a resonator at a fifth of
    /// the true period receives every pulse in phase and the tempo prior
    /// prefers it: measured at 40, 42, 44 and 46 BPM, all four published five
    /// times too fast before the fifth divisor existed. Two, three and five
    /// between them catch every integer multiple up to six inclusive, which is
    /// every multiple a range spanning a factor of six can hold, and the
    /// default range spans exactly six.
    static constexpr int kNumHarmonics = 3;
    static constexpr double kHarmonicDivisors[kNumHarmonics] = { 2.0, 3.0, 5.0 };

    /// Memory of every resonator, in units of its own period. Equal memory in
    /// BEATS rather than in seconds, so a 60 BPM and a 240 BPM candidate are
    /// judged over the same musical span; eight is long enough for the phase
    /// to be well determined and short enough to follow a tempo that moves.
    static constexpr double kIntegrationBeats = 8.0;

    /// Tempo prior (Ellis eq. 6): a Gaussian on log period, centred on half a
    /// second per beat with a width in octaves.
    static constexpr double kPriorCentreSeconds = 0.5;
    static constexpr double kPriorWidthOctaves = 1.4;

    /// Tapping preference, used ONLY to decide between two readings that stand
    /// in an exact metrical relation to each other. It is a different question
    /// from the search prior above and it gets a different, sharper answer.
    ///
    /// The search prior has to be broad, because it ranks candidates all over
    /// the searched range and a narrow one would suppress real tempi at the
    /// ends of it. The metrical-level question is narrower: both readings
    /// explain the same events and differ only in which of them is called the
    /// beat, so the only thing left to decide it with is which rate a listener
    /// would tap -- and that preference is sharp, and it is not symmetric. A
    /// level twice the preferred rate is a much less plausible tactus than one
    /// half of it, which is why the width above the centre tempo is the tight
    /// one and the width below it is the search prior's own.
    ///
    /// The two widths are the only free parameters in the level the ranking
    /// proposes, and they were chosen by measurement over both directions of
    /// the metrical corpus, not by citation; the frontier they produce is
    /// stated as a measured amplitude in setTightness()'s neighbour,
    /// getConfidence(). On recorded music the proposal is then revisited by
    /// the level model (see kLevelWeights).
    static constexpr double kTactusFastOctaves = 0.45;
    static constexpr double kTactusSlowOctaves = kPriorWidthOctaves;

    /// The metrical-level model (see chooseMetricalLevel()): three readings,
    /// the proposed period halved, as proposed and doubled, each scored as a
    /// weighted sum of eighteen measurements, of the same eighteen less the
    /// proposed reading's, and of its log tempo squared, plus a bias per
    /// reading; the highest score is the beat.
    ///
    /// The eighteen, in order: log2 of the tempo over 120 BPM; the pulse share
    /// (pulseShare()) at the period of the full envelope, of the
    /// register-balanced envelope and of the four register envelopes; the
    /// share at twice the period of the full envelope and of the four
    /// registers; the repetition (repetition()) of the full and of the balanced
    /// envelope at the period; the share at three times the period of the full
    /// envelope and of the lowest register; the share of the balanced envelope
    /// at twice and at three times the period.
    ///
    /// Fitted by multinomial logistic regression (L2 penalty 0.003 on
    /// standardised measurements, folded back into these weights; a reading
    /// outside 40..240 BPM excluded from the softmax as it is from the choice)
    /// to three public tempo-annotated collections, each weighted equally
    /// whatever its size: the ISMIR 2004 ballroom set (454 excerpts), the
    /// Salsa Dataset (122 recordings, tempo from the beat annotations) and
    /// Freesound Loops 4k (3008 loops). These are the original fit's subsets,
    /// not the expanded evaluation set documented above. This binary ranking
    /// is retained as the reference for the rational alternatives below.
    static constexpr int kLevelCandidates = 3;
    static constexpr int kLevelMiddle = 1;
    static constexpr size_t kLevelRawFeatures = 18;
    static constexpr size_t kLevelShareAtPeriod = 1;   ///< Full envelope, at the period.
    static constexpr size_t kLevelBalancedShareAtPeriod = 2; ///< Balanced envelope.
    static constexpr size_t kLevelShareAtDouble = 7;   ///< Full envelope, at twice it.
    static constexpr size_t kLevelShareAtTriple = 14;  ///< Full envelope, at three times it.
    static constexpr std::array<double, kLevelCandidates> kLevelMultiples = { 0.5, 1.0, 2.0 };
    static constexpr std::array<double, kLevelCandidates> kLevelBias = {
        -0.975336646, +0.092188351, +0.883148294 };
    static constexpr std::array<double, 2 * kLevelRawFeatures + 1> kLevelWeights = {
        // Each reading on its own:
        -0.542395096, -0.584902910, +0.498236961,
        +2.173179030, +0.290865194, +0.064019696,
        -0.056958253, -1.551139582, +2.419398334,
        +0.999335380, +1.771192530, +1.715518634,
        -0.269329648, -0.242521186, +0.512883972,
        +2.417326254, -4.177049390, -0.658688841,
        // The same, relative to the proposed reading:
        -0.740402303, -0.614477620, +0.536878776,
        +2.283943101, +0.321349606, +0.070567309,
        -0.060601959, -1.661774996, +2.331723324,
        +1.031193638, +1.895745003, +1.908437878,
        -0.571387423, -0.456177537, +0.860604058,
        +4.361899112, -4.360371230, -0.993521862,
        // Log tempo squared: the tapping preference.
        -2.979577030 };
    /// Rational alternatives compare against the retained binary winner.
    /// A positive logit predicts an advantage in exact-tempo correctness;
    /// equal-correctness pairs contribute no preference to the fit. The same
    /// 37 features use differences from that binary winner, not the proposal.
    /// Five recording/loop-pack-grouped fits, with equal dataset weights and
    /// L2=0.003, are averaged as logits and folded into one runtime scorer.
    /// The reconstructed 454/122/3223 datasets and separate validation are
    /// described in the file documentation. No new signal buffer is needed.
    static constexpr std::array<double, 4> kTernaryMultiples = {
        1.0 / 3.0, 2.0 / 3.0, 1.5, 3.0 };
    static constexpr std::array<double, 4> kTernaryBias = {
        0.53602663631392855, -0.44218443943636193, -0.98844100123510192,
        -4.3780218173667675 };
    static constexpr std::array<double, 2 * kLevelRawFeatures + 1> kTernaryWeights = {
        0.135530296867787, 1.516476106187614, 0.50512741958750595,
        -3.2226909770906635, 0.76647827698580318, -0.5179245237669392,
        1.354793380662362, -2.8513739655680381, -2.7957191619709669,
        -0.93155463697868401, -0.59124401355810241, 0.57664680014957181,
        -0.83331592757939144, -0.92868430654211009, 0.72147135283011565,
        1.342529710852844, -0.78142066296951229, 2.2830394447846869,
        -0.71810354059356818, 0.89986068541218478, 0.41744186276874551,
        1.3685809837441467, 1.8839328472219159, -0.48244306625392142,
        0.89516613728223393, -0.60386251174497119, 1.8096407974850166,
        2.9213215771564656, 4.4113655210180074, 3.2066836727020585,
        2.0230309593047608, 2.7743701263252531, -2.8072203231223085,
        -0.60235479209693188, 0.10118126378289685, -2.5817564678948961,
        -0.67204847147376856 };

    /// Least contrast between alternate beats of the proposed reading --
    /// pulseShare() at twice its period over pulseShare() at its period --
    /// for the binary model to be consulted. The model reads the
    /// relations between metrical levels, and where alternate beats are
    /// indistinguishable there are none to read: the level above the proposal
    /// has no accent to find, and the level below it adds events the signal
    /// does not have. Every signal of the acceptance corpus (clicks, swing,
    /// quiet subdivisions) measures 0.07 or less, where the proposal is right
    /// by construction; 98% of the ballroom and salsa recordings and 85% of
    /// the loops measure more. The rational alternatives can also use the
    /// corresponding contrast at three times the period, and require their
    /// own pulse support before challenging the retained binary reading.
    static constexpr double kLevelContrastFloor = 0.1;
    /// Window of pulseShare(), in periods of the pulse it measures.
    static constexpr double kLevelWindowBeats = 8.0;

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

    /// Smallest prior-weighted correlation peak, as a fraction of the tallest,
    /// that is ranked as a tempo candidate (see pickCandidates()).
    static constexpr double kCandidatePeakFloor = 0.25;

    /// The same floor against a neighbour: a candidate within
    /// kNeighbourOctaves of one that correlates more is ranked only if it
    /// correlates this much of it (see pickCandidates()). The distance stops
    /// short of the nearest metrical relation, four thirds of a period at
    /// 0.42 of an octave. A floor and not "the tallest wins": a tempo that
    /// swings spreads its correlation over a run of peaks of similar height,
    /// the tallest at the two ends of the swing, and the ranking has to keep
    /// the middle ones to choose from (with the tallest alone, a tempo
    /// swinging by 15% lost a sixth of its grid and a 100-to-140 BPM ramp a
    /// quarter of its local tempo).
    static constexpr double kNeighbourPeakFloor = 0.6;
    static constexpr double kNeighbourOctaves = 0.35;

    /// Tightness of the first pass of layGrid(), the one that only has to
    /// follow the tempo: the value at which a grid keeps up with a tempo that
    /// swings by 15% within nine beats. A caller that asks for less gets what
    /// it asks for in both passes.
    static constexpr double kFollowTightness = 25.0;

    /// Beats either side over which the period the loose grid moves at is
    /// read (see holdToGrid()): the mean of the central half of nine
    /// intervals, so that a beat taken off the grid and the one that brings
    /// it back do not count.
    static constexpr int kPeriodReachBeats = 4;

    /// Share of the spread of the intervals around a beat that the intervals
    /// on one side of it alone have to stay under for the period to be read
    /// on that side (see holdToGrid()): a change of tempo made at once, where
    /// the nine intervals are two tempi and their central mean a period that
    /// none of them has. A tempo that moves evenly stays above it: on a
    /// steady ramp one side spreads 0.54 of what the whole does, and at the
    /// steepest point of a tempo that swings over sixteen beats, 0.47.
    static constexpr double kStepSpreadShare = 1.0 / 3.0;

    /// How far the loose grid may move from its own overall period, in log
    /// period, before the delivered grid stops being held to that period and
    /// follows instead: at this distance it is half held, at half of it
    /// almost wholly (0.94), at twice it almost not (0.06). This is the
    /// distance where the intervals of the loose grid are spread out; where
    /// they agree with one another it is shorter (kHoldSpreads).
    static constexpr double kHoldWidth = 0.08;

    /// The hold distance in spreads of the intervals the local period was
    /// read from (their mean distance from it, in log period), and the
    /// distance where they do not spread at all. A period read off intervals
    /// that sit a twentieth either side of it may be the overall one read
    /// badly; the same period read off intervals that all agree is a tempo
    /// of its own, however near the overall one, and a grid held away from
    /// it sits beside every beat where the tempo changes (16 ms beside the
    /// click at a change from 120 to 110 BPM).
    static constexpr double kHoldSpreads = 4.0;
    static constexpr double kSteadyHoldWidth = 0.02;

    /// Furthest the periods read off the loose grid may sit from the period
    /// the candidates gave and from one another, in octaves: a third keeps
    /// them short of the next reading (four thirds and three quarters of the
    /// period are 0.42 of an octave away).
    static constexpr double kHoldReachOctaves = 1.0 / 3.0;

    /// How far the period the loose grid keeps may sit from the period it was
    /// given, in octaves, before the loose pass is laid again around its own
    /// (see layGrid()).
    static constexpr double kRecentreOctaves = 0.03;

    /// A second tempo in the same signal (see layGrid()): a ranked candidate
    /// this many octaves from the period decided. Nearer than the first, the
    /// loose pass follows both tempi from either of them; further than the
    /// second, the two do not fit within kHoldReachOctaves of one period
    /// between them.
    static constexpr double kSecondTempoNear = 0.25;
    static constexpr double kSecondTempoFar = 2.0 * kHoldReachOctaves;

    /// What a ranked candidate has to correlate, of what the period decided
    /// does, to be looked at as a second tempo. Low, because a tempo kept
    /// for a third of the signal correlates well under the other, and being
    /// looked at decides nothing by itself: at 0.6, sixteen clicks at one
    /// tempo among thirty-two at another were not looked at (F 0.78 to 0.80
    /// at 15 ms).
    static constexpr double kSecondTempoPeakFloor = 0.3;

    /// What makes the loose grid laid between two tempi the grid that is
    /// read (see layGrid()): this share of its intervals at each of the two
    /// periods, within kHoldWidth and in runs of this many or more (twice
    /// kPeriodReachBeats: what a local period is read over), and each of its
    /// beats carrying this much more than a beat of the grid laid at the
    /// period decided. That grid follows one tempo and slides over the
    /// other, where one beat in three or four of its own finds a beat of the
    /// music under it.
    ///
    /// The runs are what tells a change of tempo from a rhythm that crosses
    /// the beat. On 1301 pieces of music at one tempo, a second tempo is
    /// looked at in 1169. Counted without runs, a third of the intervals at
    /// each was enough to take the grid between the two in four pieces --
    /// three beats laid over four of a tango, a ride cymbal in threes --
    /// and each was the worse for it. In runs, no piece has a quarter at
    /// each (the most is 0.23 and 0.23) and none is taken. On clicks,
    /// sixteen at one tempo among thirty-two at another are followed (0.34
    /// of the intervals); twelve among thirty-six are not (0.23).
    static constexpr double kSecondTempoShare = 0.25;
    static constexpr size_t kSecondTempoRun = 8;
    static constexpr double kSecondTempoGain = 1.15;

    /// Most an interval may cost while the held grid is laid, in beats of
    /// the loose grid (the mean of what they carry). An interval that breaks
    /// the period is either a beat taken off the grid, which this prices out,
    /// or a change of phase the music made, which has to be taken once the
    /// beats after it confirm it. Without a cap the grid slides into a
    /// half-beat change over eight intervals instead.
    static constexpr double kIntervalCapBeats = 8.0;

    /// Share of what the loose grid carries that the held grid has to carry
    /// to be the one delivered. On music a held grid carries less than the
    /// loose one because it leaves the notes off the beat alone, and is the
    /// better grid for it at every share measured; under half, the tempo is
    /// one the held grid did not follow.
    static constexpr double kHeldShare = 0.5;

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

    /// How far either side of the next expected beat the grid extension
    /// looks for it, as a fraction of the beat period.
    static constexpr double kExtendFraction = 0.125;

    /// The ends of the grid: a leading or trailing beat carrying less than
    /// this fraction of the grid's mean onset strength is not a beat anyone
    /// played, and the grid is continued past its ends only onto beats that
    /// carry as much. A quarter: excerpts fade and endings thin out, and at
    /// a half the grid stopped 1.6 beats short of the annotated beats per
    /// piece on a collection of dance music; at a quarter, 0.6.
    static constexpr double kSupportFraction = 0.25;

    /// The same floor for a beat in the first frames of the signal. A signal
    /// that is already sounding at its first sample shows the envelope an
    /// onset there that is the edge of the file and not a note; a beat on it
    /// has to carry what a beat carries. The frames are those whose analysis
    /// window reaches back past the first sample.
    static constexpr double kEdgeSupportFraction = 0.5;
    static constexpr double kEdgeSeconds = 0.06;

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<std::int64_t>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<float>::is_always_lock_free,
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

        // The tactus preference of each subharmonic against its own candidate,
        // tabulated here so the per-frame score stays a few multiplies.
        bankLevelRatio_.assign(static_cast<size_t>(kNumHarmonics)
                                   * static_cast<size_t>(bankSize_), 1.0);
        for (int k = 0; k < kNumHarmonics; ++k)
            for (int i = 0; i < bankSize_; ++i)
                bankLevelRatio_[static_cast<size_t>(k) * static_cast<size_t>(bankSize_)
                                + static_cast<size_t>(i)]
                    = levelRatio(bankPeriod_[static_cast<size_t>(i)], kHarmonicDivisors[k]);
        buildBeatScore();
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

        // The beat score reads both of the front end's onset functions,
        // each conditioned the same way and over its own running peak: the
        // log-compressed flux, where every register owns its share, and the
        // linear flux, where the loudest sound in the mix does. On
        // syncopated material the off-beats are often the busier sound and
        // the beat the louder one.
        const double rawLinear = (std::isfinite(f.linear) && f.linear > T(0))
                                     ? static_cast<double>(f.linear) : 0.0;
        linearBaseline_ += (rawLinear - linearBaseline_) * baselineCoef_;
        const double oLinear = std::max(0.0, rawLinear - linearBaseline_);
        scorePeak_ = std::max(o, scorePeak_ * peakDecay_);
        scorePeakLinear_ = std::max(oLinear, scorePeakLinear_ * peakDecay_);
        const double oBeat = o / std::max(kMassFloor, scorePeak_)
                           + oLinear / std::max(kMassFloor, scorePeakLinear_);

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

        // The beat score is recursive over the envelope itself, so it is
        // updated on every frame, before the period that steers it is chosen.
        updateBeatScore(oBeat);
        updateComb(o);
        pushRegisters(f);

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
            // a real reading of silence. The beat clock keeps running on the
            // period it already has.
            return advanceBeatClock();
        }
        // The beat clock follows the comb's metrical level once it has one,
        // and a held bank level before that; the published reading stays the
        // frame's own winner.
        int follow = holdMetricalLevel(best, bestScore, iLo, iHi);
        if (combBin_ >= 0)
        {
            // Re-decided on the frames the comb re-estimates; the test of
            // alike events reads the register history and is the costly part.
            if (combFresh_)
            {
                combFollow_ = foldToFundamental(combBin_, iLo, iHi);
                combAlike_ = sameEventsAtHalfPeriod(combFollow_, best);
                combFresh_ = false;
            }
            follow = combAlike_ ? best : std::clamp(combFollow_, iLo, iHi);
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
            runnerUp = std::max(runnerUp, scoreAt(i, false));
        }
        const double bestExplains = scoreAt(best, false);
        const double share = (bestExplains > 0.0)
            ? std::clamp(runnerUp / bestExplains, 0.0, 1.0) : 0.0;

        // The level is adjudicated with a preference that changes from bin to
        // bin, so the bin it selects can sit one bin off the coherence peak of
        // the very level it selected. The period and the phase are read from
        // the peak itself, which is where the signal actually is: two steps
        // are enough, because the two functions never differ by more than the
        // preference's slope over one bin, and it costs -0.27% -> -0.14% on
        // the worst reported tempo at the top of the default range.
        const int searched = best;
        best = walkToCoherencePeak(best);

        const size_t ub = static_cast<size_t>(best);
        const double period = refinePeriod(best);
        const double conf = std::clamp(bankCoherence_[ub] * ambiguityDiscount(share),
                                       0.0, 1.0);

        // The walk onto the coherence peak can step past the end of the range
        // it was searched in; the published reading stays inside what was asked.
        float rangeLo = 0.0f, rangeHi = 0.0f;
        unpackTempo(requestedRange_.load(std::memory_order_relaxed), rangeLo, rangeHi);
        tempoAndConfidence_.store(packTempo(std::clamp(static_cast<float>(periodToBpm(period)),
                                                       rangeLo, rangeHi),
                                            static_cast<float>(conf)),
                                  std::memory_order_relaxed);

        // The held level's refined period steers the beat clock. The
        // resonator's own phase is kept as the fallback for an ambiguous beat
        // score: arg(z) advances one turn per period, so the frames since the
        // resonator's last pulse are arg(z) scaled by the period, and its next
        // pulse is the rest of the turn away.
        const int held = (follow == searched) ? best : walkToCoherencePeak(follow);
        const size_t uh = static_cast<size_t>(held);
        const double p = bankPeriod_[uh];
        const double w = 2.0 * static_cast<double>(pi<double>) / p;
        double sinceBeat = std::atan2(zIm_[uh], zRe_[uh]) / w;
        while (sinceBeat < 0.0) sinceBeat += p;
        while (sinceBeat >= p) sinceBeat -= p;
        resonatorLead_ = p - sinceBeat;
        beatPeriod_ = (held == best) ? period : refinePeriod(held);
        haveBeatPeriod_ = true;
        beatTempo_.store(std::clamp(static_cast<float>(periodToBpm(beatPeriod_)), rangeLo, rangeHi),
                         std::memory_order_relaxed);
        return advanceBeatClock();
    }

    // -- Causal metrical level: comb-filter periodicity ----------------------

    /**
     * @brief Sizes the comb stage for the frame rate in force. Causal path
     *        only; prepare() calls it after the bank exists.
     *
     * Davies and Plumbley, "Context-dependent beat tracking of musical
     * audio", IEEE TASLP 15(3), 2007: the autocorrelation of the last few
     * seconds of the envelope, each candidate period scored by the mean of
     * its first four multiples, a Rayleigh prior on the lag and a Gaussian
     * transition between successive estimates. The coherence of the bank
     * asks how much onset mass lands in phase with one pulse; the comb asks
     * how strongly the envelope REPEATS at the period, which is what still
     * names the beat where an off-beat carries as much mass as the beat.
     */
    void buildComb()
    {
        combLength_ = std::max(64, static_cast<int>(std::lround(kCombSeconds * frameRate_)));
        int size = 64;
        while (size < 2 * combLength_) size <<= 1;
        combFftSize_ = size;
        combFft_ = std::make_unique<FFTReal<double>>(static_cast<size_t>(size));
        combRing_.assign(static_cast<size_t>(combLength_), 0.0);
        combSum_.assign(static_cast<size_t>(combLength_) + 1, 0.0);
        combTime_.assign(static_cast<size_t>(size), 0.0);
        combSpec_.assign(static_cast<size_t>(size) + 2, 0.0);
        combAcf_.assign(static_cast<size_t>(size), 0.0);
        combDelta_.assign(static_cast<size_t>(bankSize_), 0.0);
        combNext_.assign(static_cast<size_t>(bankSize_), 0.0);
        combStep_.assign(static_cast<size_t>(bankSize_), 0.0);
        combPrior_.assign(static_cast<size_t>(bankSize_), 0.0);
        const double beta = kCombPriorSeconds * frameRate_;
        for (int i = 0; i < bankSize_; ++i)
        {
            const double l = bankPeriod_[static_cast<size_t>(i)];
            combPrior_[static_cast<size_t>(i)] = l / (beta * beta) * std::exp(-0.5 * l * l / (beta * beta));
        }
        const double sigma = kCombTransitionOctaves * static_cast<double>(kBinsPerOctave);
        combSpan_ = static_cast<int>(std::ceil(3.0 * sigma));
        combKernel_.assign(static_cast<size_t>(2 * combSpan_ + 1), 0.0);
        regRing_.assign(static_cast<size_t>(kRegisters) * static_cast<size_t>(combLength_), 0.0);
        for (int d = -combSpan_; d <= combSpan_; ++d)
        {
            const double u = static_cast<double>(d) / sigma;
            combKernel_[static_cast<size_t>(d + combSpan_)] = std::exp(-0.5 * u * u);
        }
    }

    /**
     * @brief One frame of the comb stage: stores the envelope value and,
     *        every kCombUpdateFrames frames, re-estimates the level.
     * @param o Conditioned envelope value of this frame.
     *
     * Allocation free. The estimate runs on the last kCombSeconds of the
     * envelope with a local mean (kCombThresholdSeconds each side) removed
     * and negative values clipped, which keeps the correlation on the onsets
     * rather than on the envelope's slow swell. The autocorrelation is the
     * inverse transform of the power spectrum, zero-padded to twice the span
     * so it is linear, not circular, and unbiased for the overlap.
     */
    void updateComb(double o) noexcept
    {
        if (!combFft_) return;
        const int n = combLength_;
        combRing_[static_cast<size_t>(combWrite_ % n)] = o;
        ++combWrite_;
        if (--combCountdown_ > 0 || combWrite_ < static_cast<int64_t>(n / 2)) return;
        combCountdown_ = kCombUpdateFrames;

        const int m = static_cast<int>(std::min<int64_t>(combWrite_, n));
        for (int i = 0; i < m; ++i)
            combSum_[static_cast<size_t>(i) + 1] = combSum_[static_cast<size_t>(i)]
                + combRing_[static_cast<size_t>((combWrite_ - m + i) % n)];
        const int half = std::max(1, static_cast<int>(std::lround(kCombThresholdSeconds * frameRate_)));
        std::fill(combTime_.begin(), combTime_.end(), 0.0);
        for (int i = 0; i < m; ++i)
        {
            const int a = std::max(0, i - half), b = std::min(m - 1, i + half);
            const double mean = (combSum_[static_cast<size_t>(b) + 1] - combSum_[static_cast<size_t>(a)])
                              / static_cast<double>(b - a + 1);
            combTime_[static_cast<size_t>(i)] = std::max(0.0, combRing_[static_cast<size_t>((combWrite_ - m + i) % n)] - mean);
        }
        combFft_->forward(combTime_.data(), combSpec_.data());
        for (int k = 0; k <= combFftSize_ / 2; ++k)
        {
            const double re = combSpec_[static_cast<size_t>(2 * k)];
            const double im = combSpec_[static_cast<size_t>(2 * k + 1)];
            combSpec_[static_cast<size_t>(2 * k)] = re * re + im * im;
            combSpec_[static_cast<size_t>(2 * k + 1)] = 0.0;
        }
        combFft_->inverse(combSpec_.data(), combAcf_.data());
        for (int l = 0; l < m; ++l) combAcf_[static_cast<size_t>(l)] /= static_cast<double>(m - l);

        int iLo = 0, iHi = 0;
        unpackRange(activeRange_.load(std::memory_order_relaxed), iLo, iHi);
        iLo = std::clamp(iLo, 0, bankSize_ - 1);
        iHi = std::clamp(iHi, iLo, bankSize_ - 1);

        // Each multiple a of the period is read as the mean of the lags within
        // (a - 1) * kCombWidthSeconds of it: an expressive beat drifts, and
        // the further multiples have drifted further.
        double total = 0.0;
        for (int i = iLo; i <= iHi; ++i)
        {
            const double lag = bankPeriod_[static_cast<size_t>(i)];
            double v = 0.0;
            for (int a = 1; a <= kCombMultiples; ++a)
            {
                const int w = static_cast<int>(std::lround(static_cast<double>(a - 1) * kCombWidthSeconds * frameRate_));
                const int c = static_cast<int>(std::lround(static_cast<double>(a) * lag));
                double sum = 0.0;
                for (int b = c - w; b <= c + w; ++b)
                    if (b >= 0 && b < m) sum += combAcf_[static_cast<size_t>(b)];
                v += sum / static_cast<double>(2 * w + 1);
            }
            // Half the period: the beat's subdivision is on the grid of a
            // binary level, and on no grid of a level three quarters or
            // three halves of it (a clave or tresillo repeats there).
            {
                const double h = 0.5 * lag;
                const int c0 = static_cast<int>(h);
                const double fr = h - static_cast<double>(c0);
                if (c0 + 1 < m)
                    v += kCombHalfWeight * ((1.0 - fr) * combAcf_[static_cast<size_t>(c0)]
                                            + fr * combAcf_[static_cast<size_t>(c0) + 1]);
            }
            combNext_[static_cast<size_t>(i)] = std::max(0.0, v) * combPrior_[static_cast<size_t>(i)];
            total += combNext_[static_cast<size_t>(i)];
        }
        if (!(total > 0.0)) return;

        // Transition: the best predecessor under a Gaussian on log period,
        // times this estimate's normalised evidence. The first estimate has
        // no predecessor and takes the evidence alone.
        const bool first = combBin_ < 0;
        double bestV = 0.0, norm = 0.0;
        int bestI = -1;
        for (int j = iLo; j <= iHi; ++j)
        {
            double carry = first ? 1.0 : 0.0;
            if (!first)
                for (int i = std::max(iLo, j - combSpan_); i <= std::min(iHi, j + combSpan_); ++i)
                    carry = std::max(carry, combDelta_[static_cast<size_t>(i)]
                                            * combKernel_[static_cast<size_t>(i - j + combSpan_)]);
            const double v = carry * combNext_[static_cast<size_t>(j)] / total;
            combStep_[static_cast<size_t>(j)] = v;
            norm += v;
            if (v > bestV) { bestV = v; bestI = j; }
        }
        if (!(norm > 0.0) || bestI < 0) return;
        std::fill(combDelta_.begin(), combDelta_.end(), 0.0);
        for (int j = iLo; j <= iHi; ++j)
            combDelta_[static_cast<size_t>(j)] = combStep_[static_cast<size_t>(j)] / norm;
        combBin_ = bestI;
        combFresh_ = true;
    }

    /**
     * @brief The comb's choice, moved down to the slower pulse it is a
     *        multiple of when that pulse is as coherent as the choice.
     * @param i  Bank index the comb chose.
     * @param iLo First bank index searched.
     * @param iHi Last bank index searched.
     * @return The bank index the beat clock follows.
     *
     * The comb reads repetition, and a pulse train repeats at every multiple
     * of its period just as well; its Rayleigh prior then prefers the
     * multiple nearest 0.40 s, which on a slow click train is twice the true
     * rate. Coherence settles that direction exactly as in scoreAt(): a
     * candidate whose subharmonic carries the onset mass in phase is a
     * harmonic of it. Only a subharmonic holding at least kCombFoldFloor of
     * the mass, and kCombFoldRatio of the candidate's own, moves the choice.
     */
    [[nodiscard]] int foldToFundamental(int i, int iLo, int iHi) const noexcept
    {
        int j = std::clamp(i, iLo, iHi);
        for (int guard = 0; guard < kNumHarmonics; ++guard)
        {
            bool moved = false;
            for (int k = 0; k < kNumHarmonics && !moved; ++k)
            {
                const int sub = j + harmonicOffset_[k];
                if (sub > iHi) continue;
                int peak = sub;
                for (int q = std::max(0, sub - 3); q <= std::min(bankSize_ - 1, sub + 3); ++q)
                    if (bankCoherence_[static_cast<size_t>(q)] > bankCoherence_[static_cast<size_t>(peak)])
                        peak = q;
                const double c = bankCoherence_[static_cast<size_t>(peak)];
                if (c > kCombFoldFloor && c >= kCombFoldRatio * bankCoherence_[static_cast<size_t>(j)])
                {
                    j = std::clamp(peak, iLo, iHi);
                    moved = true;
                }
            }
            if (!moved) break;
        }
        return j;
    }

    /// Stores this frame's register envelopes, baseline-removed as the
    /// envelope is, for sameEventsAtHalfPeriod().
    void pushRegisters(const typename OnsetDetector<T>::OdfFrame& f) noexcept
    {
        if (regRing_.empty()) return;
        const int n = combLength_;
        for (int g = 0; g < kRegisters; ++g)
        {
            const T r = f.registers[static_cast<size_t>(g)];
            const double v = (std::isfinite(r) && r > T(0)) ? static_cast<double>(r) : 0.0;
            regBaseline_[static_cast<size_t>(g)] += (v - regBaseline_[static_cast<size_t>(g)]) * baselineCoef_;
            regRing_[static_cast<size_t>(g) * static_cast<size_t>(n) + static_cast<size_t>((combWrite_ - 1) % n)]
                = std::max(0.0, v - regBaseline_[static_cast<size_t>(g)]);
        }
    }

    /**
     * @brief True when the comb's level is twice the bank's period and every
     *        register repeats at the bank's period as well as at twice it.
     * @param level Bank index the comb stage chose.
     * @param fast  Bank index of the bank's coherence winner.
     *
     * A pulse train whose events are all alike repeats at its own period and
     * at every multiple, so the comb's Rayleigh prior alone would take a
     * 200 BPM click train to 100. A syncopated groove is told apart by its
     * registers: there the low and middle registers repeat at the beat and
     * not at the off-beat, because the events on the off-beats are other
     * instruments. Kept to the one relation that occurs, twice the period.
     */
    [[nodiscard]] bool sameEventsAtHalfPeriod(int level, int fast) const noexcept
    {
        if (regRing_.empty() || level < 0 || fast < 0) return false;
        const double ratio = bankPeriod_[static_cast<size_t>(level)]
                           / bankPeriod_[static_cast<size_t>(fast)];
        if (std::abs(std::log2(ratio) - 1.0) > 0.125) return false;
        const int n = combLength_;
        const int m = static_cast<int>(std::min<int64_t>(combWrite_, n));
        const int lag = static_cast<int>(std::lround(bankPeriod_[static_cast<size_t>(fast)]));
        if (2 * lag + 2 >= m) return false;
        double worst = 1.0;
        bool any = false;
        for (int g = 0; g < kRegisters; ++g)
        {
            const double* x = &regRing_[static_cast<size_t>(g) * static_cast<size_t>(n)];
            const auto at = [&](int i) { return x[static_cast<size_t>((combWrite_ - m + i) % n)]; };
            const auto repetition = [&](int l) {
                double best = 0.0;
                for (int q = l - 2; q <= l + 2; ++q)
                {
                    double xy = 0.0, xx = 0.0, yy = 0.0;
                    for (int i = 0; i + q < m; ++i)
                    {
                        const double a = at(i), b = at(i + q);
                        xy += a * b; xx += a * a; yy += b * b;
                    }
                    if (xx > 0.0 && yy > 0.0) best = std::max(best, xy / std::sqrt(xx * yy));
                }
                return best;
            };
            const double slow = repetition(2 * lag);
            if (!(slow > kIsochronyFloor)) continue;
            any = true;
            worst = std::min(worst, repetition(lag) / slow);
        }
        return any && worst >= kIsochronyRatio;
    }

    /// Steps from a bank index onto the coherence peak of its own level, at
    /// most two bins (see advanceFrame()).
    [[nodiscard]] int walkToCoherencePeak(int i) const noexcept
    {
        for (int guard = 0; guard < 2; ++guard)
        {
            if (i > 0 && bankCoherence_[static_cast<size_t>(i - 1)]
                         > bankCoherence_[static_cast<size_t>(i)]) --i;
            else if (i + 1 < bankSize_
                     && bankCoherence_[static_cast<size_t>(i + 1)]
                        > bankCoherence_[static_cast<size_t>(i)]) ++i;
            else break;
        }
        return i;
    }

    /**
     * @brief Holds the metrical level the beat clock follows.
     * @param best      Winner of this frame's level search.
     * @param bestScore Its score.
     * @param iLo       First bank index searched.
     * @param iHi       Last bank index searched.
     * @return The bank index to follow on this frame.
     *
     * A candidate at a different level (a quarter of an octave or more away)
     * takes over only after it has outscored the level in force on every frame
     * for kLevelHoldBeats of its own periods; within the level in force the
     * winner is followed as it is, so the held level still tracks a tempo that
     * moves. The beat clock continues the interval this choice gives it, and a
     * level that flickers for a frame or two to a different candidate restarts
     * that continuation from the wrong interval.
     */
    [[nodiscard]] int holdMetricalLevel(int best, double bestScore, int iLo, int iHi) noexcept
    {
        const int apart = std::max(1, static_cast<int>(kCandidateSeparationOctaves
                                                       * kBinsPerOctave));
        if (heldLevel_ < iLo || heldLevel_ > iHi) { heldLevel_ = -1; levelChallengeFrames_ = 0; }
        if (heldLevel_ < 0 || std::abs(best - heldLevel_) < apart)
        {
            heldLevel_ = best;
            levelChallengeFrames_ = 0;
            return best;
        }
        // Best candidate within the level in force.
        const int half = apart / 2;
        int held = -1;
        double heldScore = 0.0;
        for (int i = std::max(iLo, heldLevel_ - half); i <= std::min(iHi, heldLevel_ + half); ++i)
        {
            const double sc = scoreAt(i);
            if (held < 0 || sc > heldScore) { heldScore = sc; held = i; }
        }
        if (bestScore > heldScore) ++levelChallengeFrames_;
        else levelChallengeFrames_ = 0;
        if (static_cast<double>(levelChallengeFrames_)
            >= kLevelHoldBeats * bankPeriod_[static_cast<size_t>(best)])
        {
            heldLevel_ = best;
            levelChallengeFrames_ = 0;
        }
        else if (held >= 0)
        {
            heldLevel_ = held;
        }
        return heldLevel_;
    }

    // -- Causal beat placement -------------------------------------------------

    /// Interval weight of the beat score, W(d / tau), read from a table over
    /// ln(d / tau). Zero outside [tau / 2, 2 tau].
    [[nodiscard]] double beatIntervalWeight(int d, double lnTau) const noexcept
    {
        const double u = beatLogLag_[static_cast<size_t>(d)] - lnTau;
        const double x = (u + kBeatTableSpan) * (static_cast<double>(kBeatTableSize - 1)
                                                 / (2.0 * kBeatTableSpan));
        const int k = static_cast<int>(x + 0.5);
        if (k < 0 || k >= kBeatTableSize) return 0.0;
        return beatWeightTable_[static_cast<size_t>(k)];
    }

    /// Sizes the beat score for the longest period the bank can deliver.
    void buildBeatScore()
    {
        const double pMax = bpmToPeriod(kBankMinBpm);
        int len = 64;
        while (static_cast<double>(len) < 3.0 * pMax + 8.0) len <<= 1;
        beatScoreLen_ = len;
        beatScore_.assign(static_cast<size_t>(len), 0.0);
        beatScratch_.assign(static_cast<size_t>(2 * len), 0.0);
        beatAhead_.assign(static_cast<size_t>(len) + 2, 0.0);
        beatLogLag_.assign(static_cast<size_t>(len) + 1, 0.0);
        for (int d = 1; d <= len; ++d)
            beatLogLag_[static_cast<size_t>(d)] = std::log(static_cast<double>(d));
        beatWeightTable_.assign(static_cast<size_t>(kBeatTableSize), 0.0);
        constexpr double ln2 = 0.6931471805599453;
        for (int k = 0; k < kBeatTableSize; ++k)
        {
            const double u = -kBeatTableSpan + 2.0 * kBeatTableSpan * static_cast<double>(k)
                                                   / static_cast<double>(kBeatTableSize - 1);
            const double e = kBeatIntervalSharpness * u;
            beatWeightTable_[static_cast<size_t>(k)]
                = (std::abs(u) <= ln2 + 1e-9) ? std::exp(-0.5 * e * e) : 0.0;
        }
    }

    /**
     * @brief One frame of the cumulative beat score,
     *        C[n] = (1 - a) o[n] + a max_d W(d / tau) C[n - d].
     * @param o Conditioned envelope value of this frame.
     *
     * Stark, Davies and Plumbley, "Real-time beat-synchronous analysis of
     * musical audio", DAFx 2009, with their a = 0.9 and their log-Gaussian
     * interval weight of sharpness 5 over [tau / 2, 2 tau].
     */
    void updateBeatScore(double o) noexcept
    {
        const int mask = beatScoreLen_ - 1;
        double best = 0.0;
        if (haveBeatPeriod_)
        {
            const double tau = beatPeriod_;
            const double lnTau = std::log(tau);
            const int lo = std::max(1, static_cast<int>(std::lround(0.5 * tau)));
            const int hi = std::min(static_cast<int>(std::lround(2.0 * tau)),
                                    static_cast<int>(std::min<int64_t>(beatScoreFrames_,
                                                                       beatScoreLen_ - 1)));
            for (int d = lo; d <= hi; ++d)
            {
                const double v = beatIntervalWeight(d, lnTau)
                               * beatScore_[static_cast<size_t>((beatScoreWrite_ - d) & mask)];
                if (v > best) best = v;
            }
        }
        beatScore_[static_cast<size_t>(beatScoreWrite_ & mask)]
            = (1.0 - kBeatScoreFeedback) * o + kBeatScoreFeedback * best;
        ++beatScoreWrite_;
        ++beatScoreFrames_;
    }

    /**
     * @brief Chooses the next beat from the beat score continued one period
     *        into the future with no further onsets.
     *
     * The continuation is evaluated over one period starting half a period
     * after the beat just placed, and its strongest point is the next beat.
     * The preference for continuing the beat already in force is very slight
     * (a Gaussian kPredictionSpread periods wide): a narrower one keeps
     * whichever of two equal-period pulse trains the clock happened to start
     * on, and the clock must be free to move to the stronger one.
     *
     * Where a pulse train at another phase scores within kAmbiguousScore of
     * the strongest, the score cannot tell which train is the beat, and the
     * beat is placed where the resonator's phase turn falls instead. That
     * phase is the circular mean of the onset mass, so it sits between the
     * two trains, nearer the stronger one.
     */
    void predictNextBeat() noexcept
    {
        const double tau = beatPeriod_;
        const double lnTau = std::log(tau);
        const int span = std::max(2, static_cast<int>(std::lround(tau)));
        const int lo = std::max(1, static_cast<int>(std::lround(0.5 * tau)));
        const int hi = static_cast<int>(std::lround(2.0 * tau));
        const int past = std::min(hi + 1, static_cast<int>(std::min<int64_t>(beatScoreFrames_,
                                                                              beatScoreLen_ - 1)));
        const int mask = beatScoreLen_ - 1;
        for (int i = 0; i < past; ++i)
            beatScratch_[static_cast<size_t>(i)]
                = beatScore_[static_cast<size_t>((beatScoreWrite_ - past + i) & mask)];

        int bestK = span / 2;
        double bestV = -1.0;
        const double centre = 0.5 * tau;
        const double spread = kPredictionSpread * tau;
        for (int k = 1; k <= span; ++k)
        {
            const int at = past - 1 + k;
            double b = 0.0;
            for (int d = lo; d <= hi; ++d)
            {
                const int j = at - d;
                if (j < 0) break;
                const double v = beatIntervalWeight(d, lnTau) * beatScratch_[static_cast<size_t>(j)];
                if (v > b) b = v;
            }
            beatScratch_[static_cast<size_t>(at)] = kBeatScoreFeedback * b;
            const double u = (static_cast<double>(k) - centre) / spread;
            const double q = b * std::exp(-0.5 * u * u);
            beatAhead_[static_cast<size_t>(k)] = q;
            if (q > bestV) { bestV = q; bestK = k; }
        }

        if (bestV > 0.0)
        {
            double rival = 0.0;
            for (int k = 1; k <= span; ++k)
            {
                const double v = beatAhead_[static_cast<size_t>(k)];
                if (k > 1 && v < beatAhead_[static_cast<size_t>(k - 1)]) continue;
                if (k < span && v < beatAhead_[static_cast<size_t>(k + 1)]) continue;
                if (std::abs(wrapToPeriod(static_cast<double>(k - bestK), tau))
                    > kRivalSeparation * tau)
                    rival = std::max(rival, v);
            }
            if (rival >= (1.0 - kAmbiguousScore) * bestV)
            {
                double lead = resonatorLead_;
                while (lead < 0.5) lead += tau;
                int k = static_cast<int>(std::lround(lead));
                const int step = std::max(1, static_cast<int>(std::lround(tau)));
                while (k > span) k -= step;
                bestK = std::max(1, k);
            }
        }
        beatCountdown_ = bestK;
        framesToPrediction_ = bestK + static_cast<int>(std::lround(0.5 * tau));
    }

    /// x folded into [-period / 2, period / 2].
    [[nodiscard]] static double wrapToPeriod(double x, double period) noexcept
    {
        double d = std::fmod(x, period);
        if (d > 0.5 * period) d -= period;
        if (d < -0.5 * period) d += period;
        return d;
    }

    /**
     * @brief Advances the beat clock one frame.
     * @return true if a beat was reported on this frame.
     *
     * A beat is announced on the frame after the one it is placed on and is
     * attributed one hop before that frame's position. The beat score adds
     * the linear flux to the log-band envelope, and on a transient the linear
     * flux peaks about one hop later (the loud part of the attack enters the
     * window after the first of it): measured, the attribution to the frame
     * itself put the beats a median 7.0 ms late on clean clicks and a mean
     * 8.3 ms after the annotated beat on ballroom recordings, against 3.6 and
     * 4.3 ms with the envelope alone; one hop earlier gives 2.1 and 3.7 ms.
     * Those two hops are the hop terms of getLatencySamples().
     */
    bool advanceBeatClock() noexcept
    {
        bool beat = false;
        if (beatPending_)
        {
            lastBeatSample_.store(std::max<int64_t>(0, pendingBeatRef_), std::memory_order_relaxed);
            beatPending_ = false;
            beat = true;
        }
        if (!haveBeatPeriod_) return beat;
        if (framesToPrediction_ < 0)
        {
            if (static_cast<double>(beatScoreFrames_) >= beatPeriod_) predictNextBeat();
            return beat;
        }
        if (beatCountdown_ > 0 && --beatCountdown_ == 0)
        {
            beatPending_ = true;
            pendingBeatRef_ = lastFrameRef_ - static_cast<int64_t>(hop_);
        }
        if (--framesToPrediction_ <= 0) predictNextBeat();
        return beat;
    }

    /** @brief Parabolic refinement of the winning period, on log period. The
     *  bank is geometric, so the three points are equally spaced in the
     *  variable being interpolated. The refinement reads the EXPLANATORY
     *  score: the metrical level is already settled by the time this runs, and
     *  the tapping preference varies from bin to bin, so interpolating over it
     *  would tilt the parabola and pull the reported period towards the
     *  preferred tempo -- measured at -0.53% at the top of the default range
     *  before this used the preference-free score. */
    [[nodiscard]] double refinePeriod(int i) const noexcept
    {
        if (i <= 0 || i >= bankSize_ - 1) return bankPeriod_[static_cast<size_t>(i)];
        const double a = scoreAt(i - 1, false);
        const double b = scoreAt(i, false);
        const double c = scoreAt(i + 1, false);
        const double den = a - 2.0 * b + c;
        if (!(std::abs(den) > 0.0)) return bankPeriod_[static_cast<size_t>(i)];
        double d = 0.5 * (a - c) / den;
        d = std::clamp(d, -0.5, 0.5);
        return bankPeriod_[static_cast<size_t>(i)]
             * std::pow(2.0, d / static_cast<double>(kBinsPerOctave));
    }

    /**
     * @brief Score of a bank candidate.
     * @param withPreference apply the tapping preference to the level test.
     *
     * The two readings answer different questions and both are needed. WITH
     * the preference is the adjudicated score: which level to report. WITHOUT
     * it is the explanatory score: how much of the envelope that level
     * accounts for, which is what the confidence must be discounted against.
     * Using the adjudicated score for both would let the preference hide the
     * ambiguity it was applied to resolve -- the moment the tracker decides a
     * level, the loser's adjudicated score collapses, and a confidence read
     * off that ratio would report certainty produced by the decision itself
     * rather than by the signal.
     */
    [[nodiscard]] double scoreAt(int i, bool withPreference = true) const noexcept
    {
        const size_t u = static_cast<size_t>(i);
        double s = bankPrior_[u] * bankCoherence_[u];
        for (int k = 0; k < kNumHarmonics; ++k)
        {
            const int sub = i + harmonicOffset_[k];
            if (sub >= bankSize_) continue;
            const double r = withPreference
                ? bankLevelRatio_[static_cast<size_t>(k) * static_cast<size_t>(bankSize_) + u]
                : 1.0;
            s *= (1.0 - std::min(1.0, r * bankCoherence_[static_cast<size_t>(sub)]));
        }
        return s;
    }

    /**
     * @brief How much more plausible a tactus the candidate's subharmonic is
     *        than the candidate itself, never less than one.
     *
     * The metrical-level test asks whether the candidate's own subharmonic is
     * as coherent as the candidate. On a pulse train with weaker events
     * between the beats, that coherence is exactly the depth of the accent --
     * and the same signal read the other way up is a backbeat, where the fast
     * level is the beat and the accent sits below it. The two are the same
     * waveform: a click train at rate R with alternate events attenuated is a
     * beat at R with a backbeat, and a beat at R/2 with eighth notes, at once.
     * No measurement of the envelope can separate them, and the only thing
     * that can is which absolute rate a listener would tap. This ratio is
     * that judgement, applied to the accent depth before it is weighed.
     *
     * It never falls below one, so it can only ever make the test REJECT a
     * candidate more readily -- never protect a candidate that is a plain
     * harmonic of the true pulse, whose subharmonic is fully coherent and
     * whose rejection must stay absolute.
     */
    [[nodiscard]] static double levelPreference(double periodSeconds) noexcept
    {
        const double lg = std::log2(periodSeconds / kPriorCentreSeconds);
        const double w = (lg >= 0.0) ? kTactusSlowOctaves : kTactusFastOctaves;
        const double x = lg / w;
        return std::exp(-0.5 * x * x);
    }

    [[nodiscard]] double levelRatio(double periodFrames, double divisor) const noexcept
    {
        if (!(frameRate_ > 0.0) || !(periodFrames > 0.0)) return 1.0;
        const double p = periodFrames / frameRate_;
        const double here = levelPreference(p);
        if (!(here > 0.0)) return 1.0;
        return std::max(1.0, levelPreference(p * divisor) / here);
    }

    // -- Offline stages ------------------------------------------------------

    /** @brief Runs the shared front end over the whole buffer and keeps the
     *  envelope with each frame's position in the caller's timeline. */
    /** @brief Appends the frame the front end just computed to the offline
     *         envelopes. */
    void appendEnvelopeFrame()
    {
        const typename OnsetDetector<T>::OdfFrame f = onset_.getLastOdfFrame();
        const double v = std::isfinite(f.value) ? static_cast<double>(f.value) : 0.0;
        env_.push_back(std::max(0.0, v));
        envRef_.push_back(f.referenceSample);
        for (int g = 0; g < kRegisters; ++g)
        {
            const double r = static_cast<double>(f.registers[static_cast<size_t>(g)]);
            envReg_[static_cast<size_t>(g)].push_back(std::isfinite(r) ? std::max(0.0, r) : 0.0);
        }
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

        // Register balance. The onset function is a mean over quarter-tone
        // bands, and the upper registers own most of them: a kick lights a
        // handful of bands below 200 Hz while hats, strums and consonants light
        // a hundred. On a dense master the pulse the kick states then barely
        // shows in the mean, and the backbeat and syllable rhythm decide the
        // metrical level instead. So the level is decided on a second,
        // register-balanced envelope: each register conditioned on its own -
        // floor removed, scaled to unit spread - and the four summed, so that
        // a register carries evidence in proportion to how regular it is, not
        // to how many bands it spans. The grid itself is still laid on the
        // plain envelope, where every onset keeps its own weight. Measured on a limited pop master at 135
        // BPM (kick on every beat, snare on two and four, strummed guitar and
        // a syllabic vocal on top), the coherence at half the tempo falls from
        // 0.88 of the coherence at the beat to 0.38.
        envBal_.clear();
        bool haveRegisters = false;
        for (const auto& r : envReg_)
            if (static_cast<int>(r.size()) == n)
                for (const double v : r)
                    if (v > 0.0) { haveRegisters = true; break; }
        if (haveRegisters)
        {
            // A register counts in proportion to how periodic it is: its
            // normalised autocorrelation peak over the searched lags. A
            // register holding one isolated event - a tone switching on - or
            // unpatterned material has no such peak and adds nothing, where
            // plain unit-spread scaling would have amplified it to equal voice.
            int iLo = 0, iHi = 0;
            unpackRange(activeRange_.load(std::memory_order_relaxed), iLo, iHi);
            const int lagLo = std::max(1, static_cast<int>(std::floor(bankPeriod_[static_cast<size_t>(iLo)])));
            const int lagHi = std::min(n / 2, static_cast<int>(std::ceil(bankPeriod_[static_cast<size_t>(iHi)])));
            envBal_.assign(static_cast<size_t>(n), 0.0);
            for (auto& r : envReg_)
            {
                conditionSeries(r, true);
                double zero = 0.0;
                for (const double v : r) zero += v * v;
                zero /= static_cast<double>(n);
                double peak = 0.0;
                if (zero > 0.0)
                    for (int lag = lagLo; lag <= lagHi; ++lag)
                    {
                        checkOfflineWork();
                        double acc = 0.0;
                        for (int i = 0; i + lag < n; ++i)
                            acc += r[static_cast<size_t>(i)] * r[static_cast<size_t>(i + lag)];
                        peak = std::max(peak, acc / static_cast<double>(n - lag) / zero);
                    }
                const double w = std::clamp(peak, 0.0, 1.0);
                for (int i = 0; i < n; ++i) envBal_[static_cast<size_t>(i)] += w * r[static_cast<size_t>(i)];
            }
            conditionSeries(envBal_, false);   // already floor-free: rescale only
        }
        conditionSeries(env_, true);
    }

    /** @brief Floor removal by a centred slow mean (optional), then unit RMS. */
    void conditionSeries(WorkVector<double>& e, bool removeFloor)
    {
        const int n = static_cast<int>(e.size());
        if (n < 1) return;

        const int half = std::max(1, static_cast<int>(std::lround(0.5 * kBaselineSeconds
                                                                  * frameRate_)));
        scratchA_.assign(static_cast<size_t>(n) + 1, 0.0);
        for (int i = 0; i < n; ++i)
            scratchA_[static_cast<size_t>(i) + 1] = scratchA_[static_cast<size_t>(i)]
                                                  + e[static_cast<size_t>(i)];

        double sumSq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            if ((i & 255) == 0) checkOfflineWork();
            const int lo = std::max(0, i - half);
            const int hi = std::min(n, i + half + 1);
            const double mean = (scratchA_[static_cast<size_t>(hi)]
                                 - scratchA_[static_cast<size_t>(lo)])
                              / static_cast<double>(hi - lo);
            const double v = removeFloor ? std::max(0.0, e[static_cast<size_t>(i)] - mean)
                                         : e[static_cast<size_t>(i)];
            e[static_cast<size_t>(i)] = v;
            sumSq += v * v;
        }

        const double rms = std::sqrt(sumSq / static_cast<double>(n));
        if (rms > 0.0)
        {
            const double g = 1.0 / rms;
            for (int i = 0; i < n; ++i) e[static_cast<size_t>(i)] *= g;
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
        return coherence(env_, periodFrames);
    }

    [[nodiscard]] double coherence(const WorkVector<double>& envelope,
                                    double periodFrames) const
    {
        const int n = static_cast<int>(envelope.size());
        if (n < 2 || !(periodFrames > 1.0)) return 0.0;

        const int win = std::max(4, static_cast<int>(std::lround(kCoherenceWindowBeats
                                                                 * periodFrames)));
        if (win > n) return coherenceWindow(envelope, 0, n, periodFrames);

        const int step = std::max(1, win / 2);
        double acc = 0.0;
        int count = 0;
        for (int start = 0; start + win <= n; start += step)
        {
            acc += coherenceWindow(envelope, start, start + win, periodFrames);
            ++count;
        }
        if (count == 0) return coherenceWindow(envelope, 0, n, periodFrames);
        return acc / static_cast<double>(count);
    }

    [[nodiscard]] double coherenceWindow(const WorkVector<double>& envelope,
                                          int from, int to, double periodFrames) const
    {
        const double w = 2.0 * static_cast<double>(pi<double>) / periodFrames;
        double re = 0.0, im = 0.0, mass = 0.0;
        for (int i = from; i < to; ++i)
        {
            if ((i & 4095) == 0) checkOfflineWork();
            const double v = envelope[static_cast<size_t>(i)];
            const double a = w * static_cast<double>(i);
            re += v * std::cos(a);
            im -= v * std::sin(a);
            mass += v;
        }
        if (!(mass > kMassFloor)) return 0.0;
        return std::min(1.0, std::sqrt(re * re + im * im) / mass);
    }

    /** @brief Adjudicated and explanatory scores over the same envelope. */
    [[nodiscard]] std::array<double, 2> offlineCandidateScores(
        const WorkVector<double>& envelope, double period) const
    {
        const double centre = kPriorCentreSeconds * frameRate_;
        const double lg = std::log2(period / centre) / kPriorWidthOctaves;
        const double base = std::exp(-0.5 * lg * lg) * coherence(envelope, period);
        double adjudicated = base, explains = base;
        for (const double multiple : kHarmonicDivisors)
        {
            const double share = coherence(envelope, multiple * period);
            adjudicated *= (1.0 - std::min(1.0, levelRatio(period, multiple) * share));
            explains *= (1.0 - std::min(1.0, share));
        }
        return { adjudicated, explains };
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
        fundamentalFound_ = false;

        const int n = static_cast<int>(env_.size());
        const int lagLo = std::max(1, static_cast<int>(std::floor(
                                          bankPeriod_[static_cast<size_t>(iLo)])));
        const int lagHi = std::min(n / 2, static_cast<int>(std::ceil(
                                              bankPeriod_[static_cast<size_t>(iHi)])));
        if (lagHi <= lagLo) return;

        // Score one guard lag on either side so a genuine maximum exactly at
        // the requested range endpoint can be tested against both neighbors.
        // Previously both endpoints were excluded from candidate generation.
        acf_.assign(static_cast<size_t>(lagHi + 2), 0.0);
        const double centre = kPriorCentreSeconds * frameRate_;
        for (int lag = std::max(1, lagLo - 1); lag <= lagHi + 1; ++lag)
        {
            checkOfflineWork();
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
        for (int lag = lagLo; lag <= lagHi; ++lag)
        {
            const double v = acf_[static_cast<size_t>(lag)];
            if (v > acf_[static_cast<size_t>(lag - 1)]
                && v >= acf_[static_cast<size_t>(lag + 1)] && v > 0.0)
            {
                // A guard lag helps locate the maximum; it must not introduce
                // an out-of-range metrical hypothesis after interpolation.
                const double refined = refineLag(lag);
                if (refined >= bankPeriod_[static_cast<size_t>(iLo)]
                    && refined <= bankPeriod_[static_cast<size_t>(iHi)])
                    candidates_.push_back(lag);
            }
        }
        if (candidates_.empty()) return;

        // A candidate is a lag at which the envelope actually repeats. Local
        // maxima of the correlation's noise floor between its real peaks - a
        // hundredth of their height - are lags at which it does not, and the
        // coherence ranking below cannot tell them apart from real pulses on
        // dense material, where every coherence is small: measured on a
        // limited 92 BPM master, a 141 BPM bump at 0.3% of the tallest peak
        // outranked the 92 BPM peak itself. Only peaks within a fixed ratio
        // of the tallest are ranked.
        {
            double top = 0.0;
            for (const int lag : candidates_) top = std::max(top, acf_[static_cast<size_t>(lag)]);
            size_t w = 0;
            for (const int lag : candidates_)
                if (acf_[static_cast<size_t>(lag)] >= kCandidatePeakFloor * top)
                    candidates_[w++] = lag;
            candidates_.resize(w);
        }

        // Within one reading the correlation is the judge. Candidates closer
        // than the nearest metrical relation (kNeighbourOctaves) are the same
        // reading at two tunings, and the ranking below is not built to tell
        // those apart: it weighs a period against its own multiples, and a
        // lag beside the period, whose multiples fall between the repeats,
        // is weighed against nothing and ranks high for it. One that
        // correlates less than kNeighbourPeakFloor of another within that
        // distance is a shoulder of the other's peak, and is not ranked.
        {
            const WorkVector<int> all(candidates_.begin(), candidates_.end(),
                                      candidates_.get_allocator());
            size_t w = 0;
            for (size_t i = 0; i < all.size(); ++i)
            {
                bool shoulder = false;
                for (size_t j = 0; j < all.size() && !shoulder; ++j)
                    shoulder = j != i
                            && std::abs(std::log2(static_cast<double>(all[i])
                                                  / static_cast<double>(all[j])))
                                   < kNeighbourOctaves
                            && kNeighbourPeakFloor * acf_[static_cast<size_t>(all[j])]
                                   > acf_[static_cast<size_t>(all[i])];
                if (!shoulder) candidates_[w++] = all[i];
            }
            candidates_.resize(w);
        }

        // Rank by the octave discriminator, not by correlation height. Two
        // scores per candidate, for the two questions of scoreAt(): the
        // adjudicated one decides the level, the explanatory one is what the
        // confidence is discounted against.
        candScore_.assign(candidates_.size(), 0.0);
        candExplains_.assign(candidates_.size(), 0.0);
        for (size_t k = 0; k < candidates_.size(); ++k)
        {
            const double tau = refineLag(candidates_[k]);
            const auto scores = offlineCandidateScores(env_, tau);
            candScore_[k] = scores[0];
            candExplains_[k] = scores[1];
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
        fundamentalFound_ = anyFundamental;
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

        // The reported alternative and the ambiguity it implies are two
        // different quantities and are taken from the two different scores.
        // What gets REPORTED is the level a listener could plausibly have
        // tapped instead, so it is the best adjudicated alternative. What
        // DISCOUNTS the confidence is the best explanation of the envelope at
        // any other level, tapping preference set aside -- on a polyrhythm the
        // most competitive explanation is the tatum, which is not a level
        // anyone taps, and a confidence blind to it would call the signal
        // unambiguous because the reported alternative happens to be modest.
        int k2 = -1;
        double bestOther = 0.0;
        for (size_t k = 0; k < candidates_.size(); ++k)
        {
            const double sep = std::abs(std::log2(static_cast<double>(candidates_[k])
                                                  / static_cast<double>(candidates_[k1])));
            if (sep < kCandidateSeparationOctaves) continue;
            if (k2 < 0 || candScore_[k] > candScore_[static_cast<size_t>(k2)])
                k2 = static_cast<int>(k);
            bestOther = std::max(bestOther, candExplains_[k]);
        }
        if (k2 >= 0)
        {
            tau2 = refineLag(candidates_[static_cast<size_t>(k2)]);
            const double top = candExplains_[k1];
            secondaryShare_ = (top > 0.0) ? std::clamp(bestOther / top, 0.0, 1.0) : 0.0;
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

    /** @brief Shared measurements for a proposed metrical period. */
    [[nodiscard]] std::array<double, kLevelRawFeatures> levelFeatures(
        double period, const WorkVector<double>& balanced) const noexcept
    {
        std::array<double, kLevelRawFeatures> features {};
        size_t k = 0;
        features[k++] = std::log2(periodToBpm(period) / 120.0);
        features[k++] = pulseShare(env_, period);
        features[k++] = pulseShare(balanced, period);
        for (const auto& r : envReg_) features[k++] = pulseShare(r, period);
        features[k++] = pulseShare(env_, 2.0 * period);
        for (const auto& r : envReg_) features[k++] = pulseShare(r, 2.0 * period);
        features[k++] = repetition(env_, period);
        features[k++] = repetition(balanced, period);
        features[k++] = pulseShare(env_, 3.0 * period);
        features[k++] = pulseShare(envReg_[0], 3.0 * period);
        features[k++] = pulseShare(balanced, 2.0 * period);
        features[k++] = pulseShare(balanced, 3.0 * period);
        return features;
    }

    /**
     * @brief Selects the binary metrical reading, then considers rational
     *        alternatives supported by a pulse of their own.
     *
     * The binary scorer retains its coefficients and alternate-beat gate.
     * Third/triple and two-against-three periods can then challenge its winner
     * when binary or ternary accents exist. A new reading also needs coherent
     * register-balanced pulse support: autocorrelation alone can favor an
     * interval whose pulses cancel, such as half the triplet rate.
     *
     * Candidates stay within the active period range. A winning rational
     * alternative reports the binary winner as secondary; otherwise the
     * original binary decision and its secondary are preserved.
     */
    void chooseMetricalLevel(int iLo, int iHi, double& tau1, double& tau2)
    {
        const double pMin = bankPeriod_[static_cast<size_t>(iLo)];
        const double pMax = bankPeriod_[static_cast<size_t>(iHi)];
        const WorkVector<double>& balanced = (envBal_.size() == env_.size()) ? envBal_ : env_;
        std::array<std::array<double, kLevelRawFeatures>, kLevelCandidates> raw {};
        for (int c = 0; c < kLevelCandidates; ++c)
            raw[static_cast<size_t>(c)] = levelFeatures(
                tau1 * kLevelMultiples[static_cast<size_t>(c)], balanced);

        const auto& proposed = raw[static_cast<size_t>(kLevelMiddle)];
        if (!(proposed[kLevelShareAtPeriod] > 0.0)) return;
        const double contrastFloor = kLevelContrastFloor * proposed[kLevelShareAtPeriod];
        int best = kLevelMiddle;
        if (proposed[kLevelShareAtDouble] >= contrastFloor)
        {
            best = -1;
            double bestScore = 0.0;
            for (int c = 0; c < kLevelCandidates; ++c)
            {
                const double period = tau1 * kLevelMultiples[static_cast<size_t>(c)];
                if (c != kLevelMiddle && (period < pMin || period > pMax)) continue;
                const auto& f = raw[static_cast<size_t>(c)];
                double score = kLevelBias[static_cast<size_t>(c)];
                for (size_t j = 0; j < kLevelRawFeatures; ++j)
                    score += kLevelWeights[j] * f[j]
                           + kLevelWeights[kLevelRawFeatures + j] * (f[j] - proposed[j]);
                score += kLevelWeights[2 * kLevelRawFeatures] * f[0] * f[0];
                if (best < 0 || score > bestScore) { best = c; bestScore = score; }
            }
        }

        const double binaryPeriod = tau1 * kLevelMultiples[static_cast<size_t>(best)];
        double chosenPeriod = binaryPeriod;
        bool ternaryChosen = false;
        std::array<double, kTernaryMultiples.size()> supportedPeriods {};
        size_t supportedCount = 0;
        if (std::max(proposed[kLevelShareAtDouble], proposed[kLevelShareAtTriple]) >= contrastFloor)
        {
            const auto& reference = raw[static_cast<size_t>(best)];
            double bestGain = 0.0;
            for (size_t c = 0; c < kTernaryMultiples.size(); ++c)
            {
                const double period = tau1 * kTernaryMultiples[c];
                if (period < pMin || period > pMax) continue;
                const auto f = levelFeatures(period, balanced);
                if (f[kLevelBalancedShareAtPeriod] < kMinFundamentalCoherence) continue;
                supportedPeriods[supportedCount++] = period;
                double gain = kTernaryBias[c];
                for (size_t j = 0; j < kLevelRawFeatures; ++j)
                    gain += kTernaryWeights[j] * f[j]
                          + kTernaryWeights[kLevelRawFeatures + j] * (f[j] - reference[j]);
                gain += kTernaryWeights[2 * kLevelRawFeatures] * f[0] * f[0];
                if (gain > bestGain)
                {
                    bestGain = gain;
                    chosenPeriod = period;
                    ternaryChosen = true;
                }
            }
        }
        if (ternaryChosen)
        {
            tau2 = binaryPeriod;
            tau1 = chosenPeriod;
        }
        else if (best != kLevelMiddle)
        {
            tau2 = tau1;
            tau1 = binaryPeriod;
        }
        if (ternaryChosen || best != kLevelMiddle)
            refreshMetricalAmbiguity(balanced, tau1, tau2,
                                    { supportedPeriods.data(), supportedCount });
    }

    /** @brief Rebase ambiguity with explanatory scores after a level change. */
    void refreshMetricalAmbiguity(const WorkVector<double>& envelope,
                                  double chosen, double displaced,
                                  std::span<const double> addedPeriods)
    {
        const double top = offlineCandidateScores(envelope, chosen)[1];
        double bestOther = 0.0;
        for (size_t k = 0; k < candidates_.size(); ++k)
            if (std::abs(std::log2(refineLag(candidates_[k]) / chosen))
                    >= kCandidateSeparationOctaves)
                bestOther = std::max(bestOther, candExplains_[k]);
        const auto consider = [&](double period) {
            if (period > 0.0 && std::abs(std::log2(period / chosen))
                                  >= kCandidateSeparationOctaves)
                bestOther = std::max(bestOther, offlineCandidateScores(envelope, period)[1]);
        };
        consider(displaced);
        for (const double period : addedPeriods) consider(period);
        secondaryShare_ = top > 0.0 ? std::clamp(bestOther / top, 0.0, 1.0)
                                   : (bestOther > 0.0 ? 1.0 : 0.0);
    }

    /**
     * @brief Share of an envelope's mass in phase with a pulse of the given
     *        period, over half-overlapping windows of kLevelWindowBeats.
     *
     * The level model's own measure, deliberately not coherence(): it weights
     * each window by its mass rather than counting windows equally, so quiet
     * passages do not speak as loudly as the groove, and it clips at zero
     * rather than reading the conditioned envelope's dips as anti-phase
     * pulses. The weights were fitted to exactly this measure.
     */
    [[nodiscard]] static double pulseShare(const WorkVector<double>& e,
                                           double periodFrames) noexcept
    {
        const int n = static_cast<int>(e.size());
        const int win = static_cast<int>(kLevelWindowBeats * periodFrames);
        if (win < 4) return 0.0;
        const double w = 2.0 * static_cast<double>(pi<double>) / periodFrames;
        const double dc = std::cos(w), ds = std::sin(w);
        double acc = 0.0, mass = 0.0;
        for (int start = 0; start + win <= n; start += win / 2)
        {
            // The phasor is seeded exactly at each window start and rotated
            // within it; over one window the rotation's rounding is far
            // below anything the features resolve.
            double c = std::cos(w * static_cast<double>(start));
            double s = std::sin(w * static_cast<double>(start));
            double re = 0.0, im = 0.0, m = 0.0;
            for (int i = start; i < start + win; ++i)
            {
                const double v = std::max(0.0, e[static_cast<size_t>(i)]);
                re += v * c;
                im += v * s;
                m += v;
                const double cn = c * dc - s * ds;
                s = s * dc + c * ds;
                c = cn;
            }
            if (m > 0.0) { acc += std::sqrt(re * re + im * im); mass += m; }
        }
        return (mass > 0.0) ? acc / mass : 0.0;
    }

    /** @brief Normalised autocorrelation of an envelope at the nearest whole
     *         lag to the period, unbiased for the overlap it is taken over. */
    [[nodiscard]] static double repetition(const WorkVector<double>& e,
                                           double periodFrames) noexcept
    {
        const int n = static_cast<int>(e.size());
        const int lag = static_cast<int>(std::lround(periodFrames));
        if (lag <= 0 || lag >= n) return 0.0;
        double s0 = 0.0, s = 0.0;
        for (int i = 0; i < n; ++i) s0 += e[static_cast<size_t>(i)] * e[static_cast<size_t>(i)];
        for (int i = 0; i + lag < n; ++i)
            s += e[static_cast<size_t>(i)] * e[static_cast<size_t>(i + lag)];
        return (s0 > 0.0) ? s / s0 * static_cast<double>(n) / static_cast<double>(n - lag)
                          : 0.0;
    }

    /**
     * @brief Lays the delivered grid at the period already decided, in the
     *        caller's timeline.
     *
     * Two passes of the recursion, because holding a grid and following a
     * tempo ask opposite things of the interval cost. A cost light enough to
     * follow a tempo that moves is light enough to take a beat off the grid
     * onto a louder note next to it and bring it back at the following one:
     * the anticipated sixteenth of a samba, the swung eighth of a shuffle.
     * A cost heavy enough to refuse that holds the grid to one period, and
     * loses a tempo that leaves it.
     *
     * The first pass is loose (kFollowTightness) and only has to follow. From
     * its intervals come the period the music keeps overall and the period
     * it moves at around each beat (holdToGrid()), and the second pass is
     * held, at the tightness asked for, to the overall period wherever the
     * two agree, and follows the local one where they part. An interval of
     * the held pass never costs more than kIntervalCapBeats beats of the
     * loose grid, so that a change of phase the music made is still taken.
     *
     * The loose pass does not reach every tempo from the period it is laid
     * around, and the period decided may be one of two tempi the signal
     * keeps: from 96 BPM the pass does not reach 126, and slides over that
     * part of the signal, three beats to four of the music. So where the
     * ranking kept a second period that could be a second tempo
     * (secondTempo()), the loose pass is also laid around the period between
     * the two, and that grid is the one read if it is at the two tempi: a
     * share of its intervals in runs at each, its beats carrying more than
     * those of the first grid (kSecondTempoShare, kSecondTempoRun,
     * kSecondTempoGain).
     *
     * The held grid is the one delivered unless it carries less than
     * kHeldShare of what the loose grid carries: a tempo that swings faster
     * than its period can be read over a few beats looks steady to the first
     * pass's statistics and is not, and there the loose grid is the grid.
     *
     * Where the period has no pulse of its own under it -- the caller
     * restricted the range past the real one -- there is nothing to follow,
     * and one pass at the tightness asked for lays the grid.
     *
     * @return false if the signal is too short for the period.
     */
    bool layGrid(double tau, WorkVector<int64_t>& gridOut)
    {
        gridOut.clear();
        localPeriod_.clear();
        localTightness_.clear();
        intervalCap_ = 0.0;
        alphaScale_ = 1.0;

        WorkVector<int> grid(detail::AccountedAllocator<int>{&allocationAccount_});
        bool have = false;
        if (coherence(tau) >= kMinFundamentalCoherence)
        {
            WorkVector<int> loose(detail::AccountedAllocator<int>{&allocationAccount_});
            const double asked = tightness_.load(std::memory_order_relaxed);
            alphaScale_ = std::min(1.0, kFollowTightness / asked);
            const bool followed = runDp(tau, loose, true) && loose.size() >= 4;
            alphaScale_ = 1.0;
            if (followed)
            {
                // A signal at two tempi, the second out of the loose pass's
                // reach from the first. Laid around the period between them
                // the loose grid follows both.
                double given = tau;
                if (const double second = secondTempo(tau); second > 0.0)
                {
                    const double between = std::sqrt(tau * second);
                    WorkVector<int> both(detail::AccountedAllocator<int>{&allocationAccount_});
                    alphaScale_ = std::min(1.0, kFollowTightness / asked);
                    const bool laid = runDp(between, both, true) && both.size() >= 4;
                    alphaScale_ = 1.0;

                    // Taken if it is a grid at the two tempi: that share of
                    // its intervals in runs at each of the two periods, and
                    // each of its beats carrying that much more than a beat
                    // of the first grid, both on the score it was laid on. A
                    // grid that settled on one period, the second or one
                    // between the two, or that goes from one to the other
                    // every few beats, follows another reading of the tempo,
                    // and which reading is the beat was decided before this.
                    bool taken = false;
                    if (laid)
                        taken = shareInRuns(both, tau) >= kSecondTempoShare
                             && shareInRuns(both, second) >= kSecondTempoShare
                             && carriedBy(both) * static_cast<double>(loose.size())
                                    > kSecondTempoGain * carriedBy(loose)
                                          * static_cast<double>(both.size());
                    if (taken)
                    {
                        loose.swap(both);
                        given = between;
                    }
                    else
                        buildLocalScore(tau);   // back on the score the first chain was laid on
                }

                double overall = overallPeriod(loose, given);
                if (std::abs(std::log2(overall / given)) > kRecentreOctaves)
                {
                    // The loose grid moves at another period than the one it
                    // was given: laid again around its own, it no longer pays
                    // for the distance at the far end of a tempo that swings.
                    WorkVector<int> again(detail::AccountedAllocator<int>{&allocationAccount_});
                    alphaScale_ = std::min(1.0, kFollowTightness / asked);
                    const bool laid = runDp(overall, again, true) && again.size() >= 4;
                    alphaScale_ = 1.0;
                    if (laid)
                    {
                        loose.swap(again);
                        overall = overallPeriod(loose, overall);
                    }
                    else
                        buildLocalScore(given);   // back on the score the first chain was laid on
                }
                intervalCap_ = kIntervalCapBeats * carriedBy(loose) / static_cast<double>(loose.size());
                holdToGrid(loose, overall);
                have = runDp(overall, grid) && grid.size() >= 2;
                localPeriod_.clear();
                localTightness_.clear();
                intervalCap_ = 0.0;

                // Both grids on the score the held one was laid on, the loose
                // one continued past its ends the way the held one was.
                extendGrid(loose);
                if (!have || carriedBy(grid) < kHeldShare * carriedBy(loose))
                {
                    grid.swap(loose);
                    have = grid.size() >= 2;
                }
            }
        }
        if (!have) have = runDp(tau, grid) && grid.size() >= 2;
        if (!have) return false;

        gridOut.reserve(grid.size());
        for (const int f : grid)
            gridOut.push_back(beatSample(f));
        return true;
    }

    /** @brief Onset strength a grid's beats carry, on the score in force. */
    [[nodiscard]] double carriedBy(const WorkVector<int>& beatFrames) const noexcept
    {
        double sum = 0.0;
        for (const int f : beatFrames) sum += local_[static_cast<size_t>(f)];
        return sum;
    }

    /** @brief Mean of the central half of a set of intervals: the loose grid
     *  takes beats off the period in pairs, a short interval and the long
     *  one that brings it back, and the central half leaves both out, while
     *  a mean of what remains resolves the period well below the frame the
     *  intervals are counted in. Sorts @p intervals. */
    [[nodiscard]] static double centralMean(WorkVector<double>& intervals)
    {
        std::sort(intervals.begin(), intervals.end());
        const size_t q = intervals.size() / 4;
        double sum = 0.0;
        for (size_t j = q; j < intervals.size() - q; ++j) sum += intervals[j];
        return sum / static_cast<double>(intervals.size() - 2 * q);
    }

    /** @brief The period a grid keeps overall, within kHoldReachOctaves of
     *  @p around so that it cannot settle on another reading. */
    [[nodiscard]] double overallPeriod(const WorkVector<int>& beats, double around) const
    {
        WorkVector<double> intervals(detail::AccountedAllocator<double>{&allocationAccount_});
        intervals.reserve(beats.size());
        for (size_t k = 1; k < beats.size(); ++k)
            intervals.push_back(static_cast<double>(beats[k] - beats[k - 1]));
        const double reach = std::exp2(kHoldReachOctaves);
        return std::clamp(centralMean(intervals), around / reach, around * reach);
    }

    /** @brief Share of a grid's intervals that sit within kHoldWidth, in log
     *  period, of @p period in runs of kSecondTempoRun or more. */
    [[nodiscard]] static double shareInRuns(const WorkVector<int>& beats, double period) noexcept
    {
        size_t inRuns = 0, run = 0;
        for (size_t k = 1; k < beats.size(); ++k)
        {
            if (std::abs(std::log(static_cast<double>(beats[k] - beats[k - 1]) / period))
                <= kHoldWidth)
            {
                ++run;
                continue;
            }
            if (run >= kSecondTempoRun) inRuns += run;
            run = 0;
        }
        if (run >= kSecondTempoRun) inRuns += run;
        return static_cast<double>(inRuns) / static_cast<double>(beats.size() - 1);
    }

    /**
     * @brief The period of a second tempo the signal may keep beside
     *        @p tau; 0 if the ranking kept none.
     *
     * A second tempo is a ranked candidate between kSecondTempoNear and
     * kSecondTempoFar octaves from @p tau that correlates at least
     * kSecondTempoPeakFloor of what @p tau does; of several, the one that
     * correlates most.
     */
    [[nodiscard]] double secondTempo(double tau) const noexcept
    {
        const auto own = static_cast<size_t>(std::llround(tau));
        if (own >= acf_.size() || !(acf_[own] > 0.0)) return 0.0;
        double most = 0.0, second = 0.0;
        for (const int lag : candidates_)
        {
            const double octaves = std::abs(std::log2(static_cast<double>(lag) / tau));
            if (octaves < kSecondTempoNear || octaves > kSecondTempoFar) continue;
            const double v = acf_[static_cast<size_t>(lag)];
            if (v >= kSecondTempoPeakFloor * acf_[own] && v > most)
            {
                most = v;
                second = static_cast<double>(lag);
            }
        }
        return second;
    }

    /**
     * @brief Reads off the loose grid what the held grid is held to: fills
     *        the per-frame target period and tightness.
     *
     * The period the loose grid moves at around each beat is the central
     * mean of the intervals within kPeriodReachBeats beats, kept within
     * kHoldReachOctaves of the overall one -- of the overall one and not of
     * the period the candidates gave, which need not sit in the middle of a
     * tempo that moves.
     *
     * Where the tempo changes at once those intervals are two tempi, and
     * their central mean is a period that none of them has and that can be
     * the overall one itself: held to it, the grid rounds the change off and
     * sits beside the beats either side of it (20 and 65 ms late on clicks
     * that go from 120 to 100 BPM). There the intervals on one side of the
     * beat spread far less than all of them do (kStepSpreadShare), and the
     * period is read on that side.
     *
     * A frame is judged as the end of an interval, so its period is the one
     * read half a period before it. With `away` the log of that period over
     * the overall one, the grid is held by 1 / (1 + (away / width)^4): the
     * target is the overall period moved towards the local one by what is
     * not held, and the tightness the one asked for where held, that of the
     * loose pass where not.
     *
     * The width is kHoldSpreads times the spread of the intervals the period
     * was read from, between kSteadyHoldWidth and kHoldWidth. Where the
     * loose grid's intervals are spread out, a local period a few hundredths
     * off the overall one is the overall one read badly, and the grid is
     * held. Where they all agree it is a tempo of its own, and a grid held
     * away from it, even by half, sits beside the beat wherever the tempo
     * changes.
     */
    void holdToGrid(const WorkVector<int>& beats, double overall)
    {
        const int n = static_cast<int>(env_.size());
        const size_t count = beats.size() - 1;
        const double reach = std::exp2(kHoldReachOctaves);
        const detail::AccountedAllocator<double> doubles(&allocationAccount_);

        WorkVector<double> window(doubles);
        window.reserve(static_cast<size_t>(2 * kPeriodReachBeats + 1));
        const auto interval = [&beats](size_t k) {
            return static_cast<double>(beats[k + 1] - beats[k]);
        };

        // The period a run of intervals keeps, and how far they sit from it.
        const auto read = [&window, &interval](size_t first, size_t last,
                                               double& period, double& spread) {
            window.clear();
            for (size_t j = first; j <= last; ++j) window.push_back(interval(j));
            period = centralMean(window);
            spread = 0.0;
            for (const double x : window) spread += std::abs(std::log(x / period));
            spread /= static_cast<double>(window.size());
        };

        // The local period of each interval, at the interval's centre, as
        // the log of its distance from the overall one.
        WorkVector<double> away(count, 0.0, doubles), centre(count, 0.0, doubles),
            width(count, kHoldWidth, doubles);
        const size_t each = static_cast<size_t>(kPeriodReachBeats);
        for (size_t k = 0; k < count; ++k)
        {
            const size_t first = (k > each ? k - each : 0);
            const size_t last = std::min(count - 1, k + each);
            double period = 0.0, spread = 0.0;
            read(first, last, period, spread);

            // One side alone, where it is whole and far the steadier.
            double steadiest = kStepSpreadShare * spread, readSpread = spread;
            if (k >= each && last > k)
            {
                double sidePeriod = 0.0, sideSpread = 0.0;
                read(k - each, k, sidePeriod, sideSpread);
                if (sideSpread < steadiest)
                {
                    steadiest = readSpread = sideSpread;
                    period = sidePeriod;
                }
            }
            if (k + each < count && first < k)
            {
                double sidePeriod = 0.0, sideSpread = 0.0;
                read(k, k + each, sidePeriod, sideSpread);
                if (sideSpread < steadiest)
                {
                    steadiest = readSpread = sideSpread;
                    period = sidePeriod;
                }
            }
            width[k] = std::clamp(kHoldSpreads * readSpread, kSteadyHoldWidth, kHoldWidth);
            away[k] = std::log(std::clamp(period, overall / reach, overall * reach) / overall);
            centre[k] = 0.5 * static_cast<double>(beats[k] + beats[k + 1]);
        }

        const double asked = tightness_.load(std::memory_order_relaxed);
        const double follow = std::min(asked, kFollowTightness);
        localPeriod_.assign(static_cast<size_t>(n), overall);
        localTightness_.assign(static_cast<size_t>(n), asked);
        size_t k = 0;
        for (int i = 0; i < n; ++i)
        {
            if ((i & 255) == 0) checkOfflineWork();
            const double at = static_cast<double>(i) - 0.5 * overall;
            while (k + 2 < count && centre[k + 1] <= at) ++k;
            double a = away[k], w = width[k];
            if (at >= centre[count - 1]) { a = away[count - 1]; w = width[count - 1]; }
            else if (at > centre[k] && k + 1 < count)
            {
                const double span = centre[k + 1] - centre[k];
                const double share = span > 0.0 ? std::clamp((at - centre[k]) / span, 0.0, 1.0) : 0.0;
                a += share * (away[k + 1] - a);
                w += share * (width[k + 1] - w);
            }
            const double r = a / w;
            const double held = 1.0 / (1.0 + r * r * r * r);
            localPeriod_[static_cast<size_t>(i)] = overall * std::exp((1.0 - held) * a);
            localTightness_[static_cast<size_t>(i)] = follow + held * (asked - follow);
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

    /**
     * @brief The dynamic program of the file header, plus backtrace, the
     *        trimming of the two ends and their continuation.
     * @param chainOnly Stops after the trimming: the chain of the recursion
     *        itself, for reading its intervals.
     * @return false if the signal is too short for the period asked for.
     *
     * The target period and the tightness are single numbers for the whole
     * signal unless localPeriod_ and localTightness_ have been filled (see
     * holdToGrid()), in which case each frame is judged against its own. An
     * interval costs at most intervalCap_ when that is set.
     *
     * A beat need not have one before it. Every frame further than half a
     * period into the signal has candidates for a previous beat, and a chain
     * forced to take one starts every grid with an interval that is whatever
     * the distance from the first frames to the first note happens to be;
     * a tight grid then pays for that interval by placing its first real beat
     * late (28 ms at 90 BPM, measured on clicks after half a second of
     * silence). So where every previous beat would leave the running total
     * below zero the chain starts afresh: a total that low is the opening of
     * the signal, since afterwards it holds everything the grid has gathered.
     */
    bool runDp(double periodFrames, WorkVector<int>& beatFrames, bool chainOnly = false)
    {
        beatFrames.clear();
        const int n = static_cast<int>(env_.size());
        const bool varying = (localPeriod_.size() == env_.size());
        const bool weighted = varying && (localTightness_.size() == env_.size());
        const int lo = std::max(1, static_cast<int>(std::lround(kSearchLowFactor
                                                                * periodFrames)));
        const int hi = std::max(lo + 1, static_cast<int>(std::lround(kSearchHighFactor
                                                                     * periodFrames)));
        if (n <= hi + 2) return false;

        buildLocalScore(periodFrames);

        const double alpha = tightness_.load(std::memory_order_relaxed) * alphaScale_;
        const double cap = intervalCap_;
        cumScore_.assign(static_cast<size_t>(n), 0.0);
        backlink_.assign(static_cast<size_t>(n), -1);

        for (int i = 0; i < n; ++i)
        {
            if ((i & 255) == 0) checkOfflineWork();
            const double target = varying ? localPeriod_[static_cast<size_t>(i)]
                                          : periodFrames;
            const double weight = weighted ? localTightness_[static_cast<size_t>(i)] : alpha;
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
                double cost = weight * lr * lr;
                if (cap > 0.0 && cost > cap) cost = cap;
                const double s = cumScore_[static_cast<size_t>(j)] - cost;
                if (bestJ < 0 || s > best) { best = s; bestJ = j; }
            }
            if (bestJ >= 0 && best < 0.0) { best = 0.0; bestJ = -1; }
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
        if (chainOnly) return true;

        extendGrid(beatFrames);
        return beatFrames.size() >= 2;
    }

    /** @brief Frames at the head of the signal whose analysis window reaches
     *  back past its first sample (see kEdgeSupportFraction). */
    [[nodiscard]] int edgeFrames() const noexcept
    {
        return static_cast<int>(std::ceil(kEdgeSeconds * frameRate_));
    }

    /**
     * @brief Continues the grid off both ends of what the recursion returned,
     *        as far as the envelope keeps supporting it.
     *
     * The recursion ends where its running total is largest, and a beat a
     * little off the period it is held to can cost it more than the beat
     * carries. So the two ends are stepped out at the grid's own interval,
     * and each candidate is kept if the envelope carries, within
     * kExtendFraction of a period of where it is expected, the share of a
     * beat that keeps a beat at the ends of the grid (see kSupportFraction).
     * Silence past the end of the music ends the extension at once, and so
     * does a note held through where a beat would be.
     */
    void extendGrid(WorkVector<int>& beatFrames) const
    {
        if (beatFrames.size() < 3) return;
        const int n = static_cast<int>(env_.size());

        WorkVector<int> ibi(detail::AccountedAllocator<int>{&allocationAccount_});
        ibi.reserve(beatFrames.size());
        for (size_t k = 1; k < beatFrames.size(); ++k)
            ibi.push_back(beatFrames[k] - beatFrames[k - 1]);
        std::sort(ibi.begin(), ibi.end());
        const int step = ibi[ibi.size() / 2];
        if (step < 2) return;

        double mean = 0.0;
        for (const int f : beatFrames) mean += local_[static_cast<size_t>(f)];
        mean /= static_cast<double>(beatFrames.size());
        const double floorValue = kSupportFraction * mean;
        const double edgeFloor = kEdgeSupportFraction * mean;
        const int edge = edgeFrames();
        const int win = std::max(1, static_cast<int>(std::lround(step * kExtendFraction)));

        const auto supported = [&](int guess, double least) {
            int best = -1;
            double bestValue = least;
            for (int i = std::max(0, guess - win); i <= std::min(n - 1, guess + win); ++i)
                if (local_[static_cast<size_t>(i)] > bestValue)
                { bestValue = local_[static_cast<size_t>(i)]; best = i; }
            return best;
        };

        WorkVector<int> front(detail::AccountedAllocator<int>{&allocationAccount_});
        for (int guess = beatFrames.front() - step; guess >= 0; guess -= step)
        {
            const int best = supported(guess, guess - win <= edge ? edgeFloor : floorValue);
            if (best < 0) break;
            front.push_back(best);
            guess = best;
        }
        std::reverse(front.begin(), front.end());

        WorkVector<int> back(detail::AccountedAllocator<int>{&allocationAccount_});
        for (int guess = beatFrames.back() + step; guess <= n - 1; guess += step)
        {
            const int best = supported(guess, floorValue);
            if (best < 0) break;
            back.push_back(best);
            guess = best;
        }

        if (front.empty() && back.empty()) return;
        WorkVector<int> merged(detail::AccountedAllocator<int>{&allocationAccount_});
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
            if ((i & 255) == 0) checkOfflineWork();
            double acc = 0.0;
            const int kLo = std::max(-half, -i);
            const int kHi = std::min(half, n - 1 - i);
            for (int k = kLo; k <= kHi; ++k)
                acc += env_[static_cast<size_t>(i + k)]
                     * kernel_[static_cast<size_t>(k + half)];
            local_[static_cast<size_t>(i)] = acc;
        }
    }

    /** @brief Drops leading and trailing beats the envelope does not support
     *  (see kSupportFraction and, for the first frames of the signal,
     *  kEdgeSupportFraction). The recursion carries a beat through a passage
     *  with nothing on it rather than leave the chain open, and at the two
     *  ends of the music those beats are nobody's. */
    void trimBeats(WorkVector<int>& beatFrames) const
    {
        if (beatFrames.size() < 3) return;

        double mean = 0.0;
        for (const int f : beatFrames) mean += local_[static_cast<size_t>(f)];
        mean /= static_cast<double>(beatFrames.size());
        const double floorValue = kSupportFraction * mean;
        const double edgeFloor = kEdgeSupportFraction * mean;
        const int edge = edgeFrames();

        size_t first = 0;
        while (first + 2 < beatFrames.size()
               && local_[static_cast<size_t>(beatFrames[first])]
                      < (beatFrames[first] <= edge ? edgeFloor : floorValue))
            ++first;

        size_t last = beatFrames.size() - 1;
        while (last > first + 1
               && local_[static_cast<size_t>(beatFrames[last])] < floorValue)
            --last;

        if (first > 0 || last + 1 < beatFrames.size())
            beatFrames = WorkVector<int>(beatFrames.begin()
                                              + static_cast<std::ptrdiff_t>(first),
                                          beatFrames.begin()
                                              + static_cast<std::ptrdiff_t>(last) + 1,
                                          beatFrames.get_allocator());
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
     *
     * The parabola is fitted to the CONDITIONED envelope and not to the
     * smoothed score the recursion scores against. Which frame a beat belongs
     * to is a question about the neighbourhood, and smoothing helps answer it;
     * where inside that frame the transient fell is a question about the peak
     * itself, and smoothing an asymmetric peak -- an onset function rises fast
     * and decays slowly -- moves its apex late in proportion to the smoothing
     * width. Measured over a full sub-hop phase sweep at 90, 120 and 160 BPM,
     * fitting the smoothed curve put the grid +4.45, +4.23 and +4.00 ms late,
     * the tempo dependence being the giveaway, since the width is a fraction
     * of the period; fitting the conditioned envelope gives +2.58, +2.56 and
     * +2.53 ms, of which +1.55 to +1.60 ms is the front end's own transient
     * lead and not this path's to remove.
     */
    [[nodiscard]] int64_t beatSample(int f) const noexcept
    {
        const int n = static_cast<int>(env_.size());
        const int64_t base = envRef_[static_cast<size_t>(f)];
        if (f <= 0 || f >= n - 1) return std::max<int64_t>(0, base);
        const double a = env_[static_cast<size_t>(f - 1)];
        const double b = env_[static_cast<size_t>(f)];
        const double c = env_[static_cast<size_t>(f + 1)];
        const double den = a - 2.0 * b + c;
        if (!(den < 0.0)) return std::max<int64_t>(0, base);   // not a peak: nothing to interpolate
        const double d = std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
        return std::max<int64_t>(0, base + static_cast<int64_t>(std::llround(d * static_cast<double>(hop_))));
    }

    /**
     * @brief Least-squares samples-per-beat through the delivered grid.
     *
     * Each beat is numbered by how many median intervals lie between it and
     * its predecessor (at least one), not by its position in the list, so a
     * beat the grid skipped or an extra one it inserted moves the fit by one
     * interval's worth of residual instead of tilting the whole line. The
     * tempo reported is therefore the grid's own: measured over 46 synthetic
     * pop masters from 92 to 150 BPM, it never differs from the tempo of the
     * grid's median interval by more than 0.55%, which is the frame
     * quantisation of that median itself.
     *
     * An isolated half-beat phase change needs a fractional index: counting
     * it as a whole beat moved steady 85/100/140 BPM click trains by up to
     * 4.01%. Recognise it only between three stable intervals on either side,
     * within an eighth of a beat of a half-integer, and across at most three
     * transition intervals. This preserves the long-span fit without treating
     * swing, a tempo ramp or arbitrary grid jitter as repeated phase changes.
     * The six phase-change cases now stay within 0.003%; the recorded-music
     * evaluation above includes this fit, with the level weights unchanged.
     */
    [[nodiscard]] static double fitBeatSlope(const WorkVector<int64_t>& beats)
    {
        const size_t n = beats.size();
        if (n < 2) return 0.0;
        WorkVector<double> ibi(detail::AccountedAllocator<double>{beats.get_allocator().account()});
        ibi.reserve(n - 1);
        for (size_t k = 1; k < n; ++k) ibi.push_back(static_cast<double>(beats[k] - beats[k - 1]));
        WorkVector<double> sorted = ibi;
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2), sorted.end());
        const double med = sorted[sorted.size() / 2];
        if (!(med > 0.0)) return 0.0;

        // Reuse the median scratch for interval counts. The fractional
        // transition ends exactly on a half-beat; its internal indices keep
        // their relative timing. No segment of the grid is discarded.
        for (size_t j = 0; j < ibi.size(); ++j)
            sorted[j] = std::max(1.0, std::round(ibi[j] / med));
        const auto stable = [&](size_t j) {
            const double r = ibi[j] / med;
            return r >= 0.75 && std::abs(r - std::round(r)) < 0.125;
        };
        for (size_t j = 3; j + 3 < ibi.size(); ++j)
        {
            if (stable(j) || !stable(j - 1) || !stable(j - 2) || !stable(j - 3))
                continue;
            for (size_t width = 1; width <= 3; ++width)
            {
                const size_t end = j + width;
                if (end + 3 > ibi.size()
                    || !stable(end) || !stable(end + 1) || !stable(end + 2))
                    continue;
                double total = 0.0;
                for (size_t q = j; q < end; ++q) total += ibi[q] / med;
                const double half = std::floor(total) + 0.5;
                if (std::abs(total - half) >= 0.125) continue;
                for (size_t q = j; q < end; ++q)
                    sorted[q] = (ibi[q] / med) * half / total;
                j = end - 1;
                break;
            }
        }

        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, index = 0.0;
        for (size_t k = 0; k < n; ++k)
        {
            if (k > 0) index += sorted[k - 1];
            const double y = static_cast<double>(beats[k]);
            sx += index; sy += y; sxx += index * index; sxy += index * y;
        }
        const double dn = static_cast<double>(n);
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
        linearBaseline_ = 0.0;
        scorePeak_ = 0.0;
        scorePeakLinear_ = 0.0;
        std::fill(beatScore_.begin(), beatScore_.end(), 0.0);
        beatScoreWrite_ = 0;
        beatScoreFrames_ = 0;
        beatPeriod_ = 0.0;
        haveBeatPeriod_ = false;
        resonatorLead_ = 0.0;
        framesToPrediction_ = -1;
        beatCountdown_ = 0;
        beatPending_ = false;
        pendingBeatRef_ = 0;
        heldLevel_ = -1;
        levelChallengeFrames_ = 0;
        std::fill(combRing_.begin(), combRing_.end(), 0.0);
        std::fill(combDelta_.begin(), combDelta_.end(), 0.0);
        combWrite_ = 0;
        combCountdown_ = kCombUpdateFrames;
        combBin_ = -1;
        combFollow_ = -1;
        combFresh_ = false;
        combAlike_ = false;
        std::fill(regRing_.begin(), regRing_.end(), 0.0);
        regBaseline_.fill(0.0);

        tempoAndConfidence_.store(packTempo(0.0f, 0.0f), std::memory_order_relaxed);
        beatLatched_.store(false, std::memory_order_relaxed);
        lastBeatSample_.store(-1, std::memory_order_relaxed);
        beatTempo_.store(0.0f, std::memory_order_relaxed);
    }

    /// Feedback of the cumulative beat score (the paper's alpha).
    static constexpr double kBeatScoreFeedback = 0.9;
    /// Sharpness of the beat score's log-Gaussian interval weight (the
    /// paper's eta).
    static constexpr double kBeatIntervalSharpness = 5.0;
    /// Width, in periods, of the preference for continuing the beat in force.
    static constexpr double kPredictionSpread = 4.0;
    /// Relative score below the strongest at which a rival pulse train makes
    /// the beat score ambiguous.
    static constexpr double kAmbiguousScore = 0.05;
    /// Phase distance, in periods, beyond which a peak of the continued score
    /// is a different pulse train rather than the same one.
    static constexpr double kRivalSeparation = 0.15;
    /// Beats, of its own period, that a different metrical level must keep
    /// outscoring the one in force before the beat clock follows it.
    static constexpr double kLevelHoldBeats = 4.0;
    /// Comb stage (buildComb()): seconds of envelope it correlates, frames
    /// between estimates, multiples of the period it reads, the lag the
    /// Rayleigh prior peaks at, the transition width, the local mean removed
    /// before the correlation, how far the multiples may drift, and when a
    /// coherent subharmonic takes the choice over (foldToFundamental()).
    static constexpr double kCombSeconds = 6.0;
    static constexpr int kCombUpdateFrames = 32;
    static constexpr int kCombMultiples = 4;
    static constexpr double kCombPriorSeconds = 0.40;
    static constexpr double kCombHalfWeight = 0.5; ///< Weight of the half-period lag.
    /// Memory of the running peaks the beat score normalises by.
    static constexpr double kScorePeakSeconds = 10.0;
    static constexpr double kCombTransitionOctaves = 0.19;
    static constexpr double kCombThresholdSeconds = 0.081;
    static constexpr double kCombWidthSeconds = 0.0116;
    static constexpr double kCombFoldFloor = 0.5;
    static constexpr double kCombFoldRatio = 0.8;
    /// sameEventsAtHalfPeriod(): least repetition at twice the period for a
    /// register to count, and the share of it every counted register must
    /// reach at the period.
    static constexpr double kIsochronyFloor = 0.2;
    static constexpr double kIsochronyRatio = 0.8;
    /// Entries of the interval weight's table.
    static constexpr int kBeatTableSize = 2048;
    /// ln 2 plus rounding room: the interval weight's table span on ln(d / tau).
    static constexpr double kBeatTableSpan = 0.6931471805599453 + 0.05;

    /// Below this the exponentially-windowed onset mass is numerically nothing
    /// and the coherence ratio would be a quotient of two rounding errors.
    static constexpr double kMassFloor = 1e-12;

    // -- Members -------------------------------------------------------------

    detail::AllocationAccount allocationAccount_;
    void* offlineCheckpointContext_ = nullptr;
    void (*offlineCheckpoint_)(void*) = nullptr;
    void checkOfflineWork() const
    {
        if (offlineCheckpoint_) offlineCheckpoint_(offlineCheckpointContext_);
    }
    OnsetDetector<T> onset_;

    double sampleRate_ = 44100.0;
    double frameRate_ = 200.0;
    int hop_ = 221;

    // Resonator bank (built at prepare, read on the audio thread).
    int bankSize_ = 0;
    WorkVector<double> bankPeriod_;
    WorkVector<double> bankCosCoef_, bankSinCoef_, bankDecay_, bankPrior_;
    int harmonicOffset_[kNumHarmonics] = { 0, 0, 0 };
    WorkVector<double> zRe_, zIm_, zMass_, bankCoherence_;
    WorkVector<double> bankLevelRatio_;

    // Causal stream state (audio thread only).
    int64_t samplesPushed_ = 0;
    int64_t frameIndex_ = 0;
    int64_t lastFrameRef_ = 0;
    double baseline_ = 0.0;
    double linearBaseline_ = 0.0;  ///< Trailing mean of the linear flux.
    double scorePeak_ = 0.0;       ///< Running peaks of the two conditioned
    double scorePeakLinear_ = 0.0; ///< functions the beat score reads.
    double peakDecay_ = 0.9995;    ///< Per-frame decay of those peaks.
    double baselineCoef_ = 0.005;
    // Causal beat clock (audio thread only; sized at prepare).
    int beatScoreLen_ = 64;
    WorkVector<double> beatScore_, beatScratch_, beatAhead_, beatLogLag_, beatWeightTable_;
    int64_t beatScoreWrite_ = 0;
    int64_t beatScoreFrames_ = 0;
    double beatPeriod_ = 0.0;
    bool haveBeatPeriod_ = false;
    double resonatorLead_ = 0.0;
    int framesToPrediction_ = -1;
    int beatCountdown_ = 0;
    bool beatPending_ = false;
    int64_t pendingBeatRef_ = 0;
    int heldLevel_ = -1;
    int levelChallengeFrames_ = 0;
    // Comb stage (audio thread only; sized at prepare).
    std::unique_ptr<FFTReal<double>> combFft_;
    WorkVector<double> combRing_, combSum_, combTime_, combSpec_, combAcf_;
    WorkVector<double> combDelta_, combNext_, combStep_, combPrior_, combKernel_;
    int combLength_ = 0;
    int combFftSize_ = 0;
    int combSpan_ = 0;
    int64_t combWrite_ = 0;
    int combCountdown_ = kCombUpdateFrames;
    int combBin_ = -1;
    int combFollow_ = -1;
    bool combFresh_ = false;
    bool combAlike_ = false;
    WorkVector<double> regRing_;   ///< Register envelopes, kRegisters rows of combLength_.
    std::array<double, OnsetDetector<T>::kNumRegisters> regBaseline_ {};

    // Offline scratch (analyze() and the offline session only).
    int64_t offlinePushed_ = 0;
    bool offlineOpen_ = false;
    WorkVector<double> env_;
    WorkVector<int64_t> envRef_;
    static constexpr int kRegisters = OnsetDetector<T>::kNumRegisters;
    static_assert(kRegisters == 4, "the level model (kLevelWeights) reads four register envelopes");
    std::array<WorkVector<double>, kRegisters> envReg_;   ///< Per-register envelopes (offline).
    WorkVector<double> envBal_;   ///< Register-balanced envelope: decides the metrical level.
    double alphaScale_ = 1.0;      ///< Tightness multiplier for the grid being built.
    double intervalCap_ = 0.0;     ///< Most an interval may cost in the grid being built (0: no cap).
    WorkVector<double> local_, cumScore_, acf_, scratchA_, kernel_;
    WorkVector<double> localPeriod_;      ///< Target period per frame, while a held grid is laid.
    WorkVector<double> localTightness_;   ///< Tightness per frame, alongside it.
    WorkVector<int> backlink_;
    WorkVector<int> candidates_;
    WorkVector<double> candScore_, candExplains_;
    double secondaryShare_ = 0.0;
    bool fundamentalFound_ = false;   ///< Last ranking had a pulse in range.

    // Published state.
    std::atomic<bool> prepared_ { false };
    std::atomic<std::uint64_t> tempoAndConfidence_ { 0 };
    std::atomic<std::uint32_t> activeRange_ { 0 };
    /// Range in BPM as asked for, kept so prepare() can rebuild activeRange_.
    std::atomic<std::uint64_t> requestedRange_ {
        packTempo(static_cast<float>(kDefaultMinBpm), static_cast<float>(kDefaultMaxBpm)) };
    std::atomic<double> tightness_ { kDefaultTightness };
    std::atomic<bool> beatLatched_ { false };
    std::atomic<std::int64_t> lastBeatSample_ { -1 };
    std::atomic<float> beatTempo_ { 0.0f };
    std::atomic<std::int64_t> latencySamples_ { 770 };
};

} // namespace dspark
