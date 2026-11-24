// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TonalBalance.h
 * @brief An adaptive equalizer driven by an auditory model: it measures the
 *        loudness of the programme in bands of the ear's scale and moves the
 *        programme toward a balanced shape, raising what is missing and
 *        taming what stands out, as fine as setDetail() and with the total
 *        loudness kept.
 *
 * Two readings. The broad one is a model of hearing: every 128 samples at
 * 48 kHz, a transform of 2048 under an asymmetric window whose weight sits
 * on the most recent samples (both scale by octaves with the rate). The
 * power spectrum is smoothed across frequency with a width that follows
 * frequency (SpectralEnvelope), read in 72 bands uniform on the Bark scale
 * from 20 Hz to 20 kHz (AuditoryBands) and smoothed across the bands (the
 * ear's spread). In each band the covariance of the two channels is smoothed
 * in time, longer in the low end, and the band's power is its larger
 * eigenvalue: the level of the dominant component wherever it sits in the
 * stereo image. The cross term of that covariance is complex (one spectrum
 * times the conjugate of the other), so a channel that is the other one
 * delayed reads as one source and leaves no comb in the band powers. The
 * specific loudness of each band (the power law over the threshold in quiet)
 * is smoothed in time and read in decibels of power.
 *
 * Each band counts by how audible it is: nothing while its power is under
 * its threshold in quiet, in full from 10 dB over it. In the bands at the
 * ends, where the threshold climbs, a programme's loudness is almost
 * nothing, and read as a level it would be a hole of tens of decibels; an
 * inaudible band neither shapes the trend nor takes a correction. And a
 * band far under the programme (setFloor() dB under its loudest band, in
 * the model's level) is the floor of a recording or the end of the content:
 * the balance does not give it back.
 *
 * From there the balance: a broad trend across the bands (a smoothing in
 * which each band weighs by its audibility) against a reference shape, a
 * formula, flat loudness per Bark tilted by setTilt() around 1 kHz.
 * setBalance() is the share of the way the trend moves toward it, through a
 * soft knee (setKnee()). setBalanceAsymmetry() gives the bands under the
 * reference (what is lacking, lifted) and those over it (what is in excess,
 * lowered) slopes of their own, and setBalanceBias() moves the point they
 * are told apart at.
 *
 * The fine reading is the detail (setDetail(), 1 to 1/24 of an octave): the
 * power at 1024 points of log frequency, over the whole band, through a
 * window that follows the detail (DetailAnalysis: a third of a second in the
 * low end, 21 ms at the top), smoothed in time. Each point stands against
 * its two sides (mean loudness, half an octave or two details, that
 * leave a detail around the point out): a dip is a point under both, a peak
 * one over both, so a slope and the fall of the content at an end are
 * neither. setFillDips() raises the dips and setTrimPeaks() lowers the peaks,
 * through a knee of their own (setContrastKnee()); what lies over 30 dB
 * under its sides is empty rather than low (the space between the partials
 * of a note) and is let go by 45. A dip is looked for no finer than a sixth
 * of an octave, for the same reason, and that space is not cut with the
 * partials either; the peaks follow the detail down to a 24th.
 *
 * The corrections are limited (setRange()) and faded at the edges of
 * setFrequencyRange(), and then offset so the total loudness of the
 * programme does not change (the correction redistributes), the detail
 * counted in each band of the model by the power it moves there: the offset
 * is solved with the limits and the edges of the range in place, so the sum
 * is exact whenever the bands in range have room to move
 * (getLoudnessChange() reads what is left). setLevelCompensation(true)
 * converts loudness to decibels with each band's own loudness exponent (the
 * low end, whose loudness grows faster with level, moves less in decibels
 * for the same change of loudness).
 *
 * The curve, the broad balance between the bands' centres plus the detail at
 * each point, is played by a DetailFilter (minimum phase, with detail in the low
 * end too), and getDisplayCorrection() is the response it applies.
 * processBlock(buffer, key) analyses an external key instead, with a fade of
 * 50 ms when the key comes and goes.
 *
 * Limits: a sixth of an octave is told apart from 50 Hz, a twelfth from
 * 100 Hz, a 24th from 200 Hz; the finer, the later the low end follows.
 *
 * Latency (getLatency()): 774 samples at 44.1 and 48 kHz (17.6 and 16.1 ms),
 * doubling by octaves with the rate and halving with it to 16 kHz.
 *
 * Threading: prepare() and setState() on the setup thread (allocate);
 * processBlock() and reset() on the audio thread; the setters are lock-free
 * atomic publications, safe from any thread. Non-finite values are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, AuditoryBands.h, DenormalGuard.h,
 * DetailAnalysis.h, DetailFilter.h, DspMath.h, FFT.h, SpectralEnvelope.h,
 * SpectralFilter.h, StateBlob.h, WindowFunctions.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AuditoryBands.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DetailAnalysis.h"
#include "../Core/DetailFilter.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/SpectralEnvelope.h"
#include "../Core/SpectralFilter.h"
#include "../Core/StateBlob.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class TonalBalance
 * @brief Adaptive equalizer toward a balanced loudness shape.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TonalBalance
{
public:
    static constexpr int kBands = 72; ///< Bands of the loudness model.

    /**
     * @brief What a balanced programme is (setReference()). FlatLoudness, the
     *        default: the same loudness in every band of the ear's scale.
     *        EvenPower: the loudness, band by band, of a programme with the
     *        same power in every band at this programme's level, read
     *        through the same model; it already falls where the ear's
     *        threshold climbs and follows the level (a quiet programme is
     *        not asked for the low end only a loud one can show).
     */
    enum class Reference { FlatLoudness, EvenPower };

    /**
     * @brief How the analysis follows the programme in time (setResponse()).
     *        ByBand, the default: two smoothings with the same time per band,
     *        longer in the low end (200 ms at 1 kHz, 400 ms at 100 Hz). Common:
     *        a short integration per band (a few periods of its centre, 8 to
     *        100 ms) and then one response time for every band
     *        (setResponseTime()), so the whole curve moves together.
     */
    enum class Response { ByBand, Common };

    /**
     * @brief How often the two readings run (setQuality(); it applies at the
     *        next prepare()). Realtime, the default: the model every
     *        sixteenth of its transform, the long windows of the detail in
     *        turns. Render: the model twice as often and every window at
     *        every hop. The latency is the same in both.
     */
    enum class Quality { Realtime, Render };
    static constexpr int kMaxChannels = 2; ///< Channels analysed (the others take the same curve).

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        fs_ = spec.sampleRate;
        channels_ = std::min(spec.numChannels, kMaxChannels);
        // The model scales by octaves with the rate (2048 at 44.1 to 48 kHz).
        const int octaves = static_cast<int>(std::lround(std::log2(std::max(fs_, 8000.0) / 48000.0)));
        fft_ = std::clamp(2048 << std::max(0, octaves), 1024, 16384);
        if (octaves < 0) fft_ = std::max(512, 2048 >> -octaves);
        render_ = quality_.load(std::memory_order_relaxed) == Quality::Render
            || (qualityFollowsRender_.load(std::memory_order_relaxed) && offlineRendering_.load(std::memory_order_relaxed));
        qualityInUse_.store(render_ ? Quality::Render : Quality::Realtime, std::memory_order_relaxed);
        const int realtimeHop = std::max(16, fft_ / 16);
        hop_ = render_ ? std::max(8, realtimeHop / 2) : realtimeHop;
        const int fall = fft_ / 8;
        transform_ = std::make_unique<FFTReal<double>>(static_cast<size_t>(fft_));
        window_.resize(static_cast<size_t>(fft_));
        WindowFunctions<double>::asymmetricHann(window_.data(), fft_, fall);
        double sw = 0.0, squares = 0.0;
        for (const double w : window_) { sw += w; squares += w * w; }
        // Power scale: a full-scale sine reads 1 in its bin.
        powerScale_ = 4.0 / (sw * sw);
        // What a band of noise reads over what it holds, in that scale.
        modelNoiseBins_ = static_cast<double>(fft_) * squares / (sw * sw);
        bins_ = fft_ / 2 + 1;
        bands_.prepare(fs_, fft_, kBands, 20.0, std::min(20000.0, 0.45 * fs_));
        bands_.setReferenceSpl(static_cast<double>(referenceSpl_.load(std::memory_order_relaxed)));
        envelope_.prepare(bins_, 1.0 / 6.0);
        focus_.prepare(window_.data(), fft_, fft_, 1.0 / 6.0);
        for (auto* v : { &focusRe_[0], &focusRe_[1], &focusIm_[0], &focusIm_[1] }) v->assign(static_cast<size_t>(bins_), 0.0);
        ring_.assign(static_cast<size_t>(channels_), std::vector<double>(static_cast<size_t>(fft_), 0.0));
        spectra_.assign(static_cast<size_t>(channels_), std::vector<double>(static_cast<size_t>(fft_ + 2), 0.0));
        fed_.assign(static_cast<size_t>(kFeed), T(0));
        frame_.assign(static_cast<size_t>(fft_), 0.0);
        moments_.assign(static_cast<size_t>(kMoments * bins_), 0.0);
        bandMoments_.assign(static_cast<size_t>(kMoments * kBands), 0.0);
        for (int b = 0; b < kBands; ++b) centreOctave_[static_cast<size_t>(b)] = std::log2(bands_.getCentre(b) / 1000.0);
        for (auto* v : { &ll_, &rr_, &lr_, &li_, &loud_, &evenLoud_, &corr_, &tmpA_, &tmpB_, &tmpC_, &tmpD_ }) v->assign(kBands, 0.0);
        for (int b = 0; b < kBands; ++b)
        {
            const double tau = 0.001 * std::clamp(200.0 * std::pow(1000.0 / bands_.getCentre(b), 0.3), 20.0, 2000.0);
            timeCoeff_[static_cast<size_t>(b)] = std::exp(-static_cast<double>(hop_) / (tau * fs_));
            // The short integration of Response::Common: four periods of the centre, 8 to 100 ms.
            const double brief = std::clamp(4.0 / bands_.getCentre(b), 0.008, 0.1);
            briefCoeff_[static_cast<size_t>(b)] = std::exp(-static_cast<double>(hop_) / (brief * fs_));
        }
        filter_.prepare(spec);
        filter_.setLinked(true);
        analysis_.setLinked(true);
        analysis_.setExhaustive(render_);
        analysis_.prepare(AudioSpec { fs_, spec.maxBlockSize, channels_ });
        prepareDetail();
        prepared_ = true;
        reset();
    }

    /** @brief Clears the analysis, the corrections and the filter. */
    void reset() noexcept
    {
        for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
        for (auto* v : { &ll_, &rr_, &lr_, &li_, &loud_, &evenLoud_, &corr_ }) std::fill(v->begin(), v->end(), 0.0);
        ringPos_ = hopCounter_ = 0;
        keyFade_ = 0.0;
        warm_ = false;
        filter_.reset();
        analysis_.reset();
        resetDetail();
        for (auto& g : published_) g.store(T(0), std::memory_order_relaxed);
        loudnessChange_.store(T(0), std::memory_order_relaxed);
        raised_.store(T(0), std::memory_order_relaxed);
        lowered_.store(T(0), std::memory_order_relaxed);
        total_.store(T(-200), std::memory_order_relaxed);
        centroidShift_.store(T(0), std::memory_order_relaxed);
    }

    // -- Controls ------------------------------------------------------------

    /** @brief How far the trend moves toward the reference shape, 0 to 1 (default 0.5). */
    void setBalance(T amount) noexcept { store(balance_, amount, T(0), T(1)); }
    /** @brief How much dips (a detail under both of its sides, a sixth of an octave or wider) are raised, 0 to 1 (default 0.3). */
    void setFillDips(T amount) noexcept { store(fillDips_, amount, T(0), T(1)); }
    /** @brief How much peaks (a detail over both of its sides) are lowered, 0 to 1 (default 0.5). */
    void setTrimPeaks(T amount) noexcept { store(trimPeaks_, amount, T(0), T(1)); }
    /** @brief Tilt of the reference shape in dB per octave around 1 kHz, -6 to +6 (default 0: flat loudness per Bark); it acts through setBalance(). */
    void setTilt(T dbPerOctave) noexcept { store(tilt_, dbPerOctave, T(-6), T(6)); }
    /** @brief Largest correction of a band, 0 to 24 dB (default 6). */
    void setRange(T db) noexcept { store(range_, db, T(0), T(24)); }
    /** @brief Knee of the balance, 0.1 to 12 dB (default 3); the dips and peaks have setContrastKnee(). */
    void setKnee(T db) noexcept { store(knee_, db, T(0.1), T(12)); }
    /**
     * @brief How differently the balance lifts what is lacking and lowers what
     *        is in excess, -1 to 1 (default 0: alike). Toward 1 the bands under
     *        the reference are lifted by up to twice the share setBalance()
     *        gives and those over it lowered by less, down to nothing; toward
     *        -1 the other way. The loudness is kept as always, so what changes
     *        is the shape: with a positive asymmetry a hole in the balance is
     *        filled more than a hump is cut. State key "balanceAsymmetry".
     */
    void setBalanceAsymmetry(T amount) noexcept { store(balanceAsymmetry_, amount, T(-1), T(1)); }
    /**
     * @brief Moves the point the balance tells lacking from excess at, -6 to
     *        +6 dB (default 0: the programme's own mean). Up, more of the
     *        bands count as lacking; down, more as in excess. With an
     *        asymmetry (setBalanceAsymmetry()) it decides how much of the
     *        programme is lifted and how much lowered; with the balance alike
     *        on both sides the offset that keeps the loudness takes it back
     *        but for what the knee bends. Past the range (setRange()) the
     *        corrections only meet their limits. State key "balanceBias".
     */
    void setBalanceBias(T db) noexcept { store(balanceBias_, db, T(-6), T(6)); }
    /**
     * @brief Knee of the dips and peaks, 0.1 to 12 dB (default 3): smaller
     *        and the ripple of a noisy programme is followed, larger and
     *        only what stands well out is touched.
     */
    void setContrastKnee(T db) noexcept { store(contrastKnee_, db, T(0.1), T(12)); }
    /**
     * @brief The detail: how finely the peaks and the dips (these no finer
     *        than a sixth, so that the space between partials is not raised)
     *        are read, played and drawn over the whole band, an octave to 1/24
     *        (default 1/4). It moves while playing, at the same latency; the
     *        balance and the loudness go by the 72 bands of the ear whatever
     *        it is.
     *        The finer, the more the trim takes single partials, and what
     *        keeps the loudness then lifts the space between them with the rest.
     */
    void setDetail(T octaves) noexcept
    {
        store(detail_, octaves, static_cast<T>(DetailAnalysis<T>::kFinest), static_cast<T>(DetailAnalysis<T>::kWidest));
    }
    /**
     * @brief Rounds the approach to the range, 0 to 6 dB (0, the default, is
     *        the hard limit): a correction that grows (a dip that deepens)
     *        bends into the range through a second knee and settles on it,
     *        with no corner.
     */
    void setRangeKnee(T db) noexcept { store(rangeKnee_, db, T(0), T(6)); }
    /**
     * @brief How far under its loudest band (in the model's level) a band may
     *        lie and still be given back by the balance, 10 to 120 dB
     *        (default 30). What lies further down is the floor of a recording
     *        or the end of the content: the balance leaves it where it is,
     *        and it takes no part in keeping the loudness. The last 10 dB
     *        over the floor fade in. A dip between two sides that sound is
     *        filled whatever its depth.
     */
    void setFloor(T db) noexcept { store(floor_, db, T(10), T(120)); }
    /** @brief Frequencies the corrections act on, with edges one Bark wide (default 40 Hz to 16 kHz: the ends of the band, where the threshold in quiet climbs, are left alone). */
    void setFrequencyRange(T loHz, T hiHz) noexcept
    {
        if (!std::isfinite(loHz) || !std::isfinite(hiHz)) return;
        const T lo = std::clamp(std::min(loHz, hiHz), T(10), T(24000));
        loHz_.store(lo, std::memory_order_relaxed);
        hiHz_.store(std::clamp(std::max(loHz, hiHz), lo, T(24000)), std::memory_order_relaxed);
    }
    /** @brief The shape the programme is compared against (see Reference; out-of-range values read as FlatLoudness). */
    void setReference(Reference r) noexcept
    {
        reference_.store(r == Reference::EvenPower ? Reference::EvenPower : Reference::FlatLoudness, std::memory_order_relaxed);
    }
    /** @brief Realtime or Render (see Quality; applies at the next prepare()). */
    void setQuality(Quality q) noexcept { quality_.store(q == Quality::Render ? Quality::Render : Quality::Realtime, std::memory_order_relaxed); }
    /**
     * @brief With it on, a prepare() made while the host renders offline
     *        (setOfflineRendering(true)) takes the Render quality whatever
     *        setQuality() says (off by default).
     */
    void setQualityFollowsRender(bool on) noexcept { qualityFollowsRender_.store(on, std::memory_order_relaxed); }
    /** @brief Tells the processor whether the host renders offline (read by the next prepare()). */
    void setOfflineRendering(bool on) noexcept { offlineRendering_.store(on, std::memory_order_relaxed); }
    /** @brief How the analysis follows the programme in time (see Response; out-of-range values read as ByBand). */
    void setResponse(Response r) noexcept
    {
        response_.store(r == Response::Common ? Response::Common : Response::ByBand, std::memory_order_relaxed);
    }
    /** @brief The response time of Response::Common, 10 to 2000 ms (default 60). */
    void setResponseTime(T ms) noexcept { store(responseMs_, ms, T(10), T(2000)); }
    /**
     * @brief Compensates a listening level this many dB lower than the one
     *        the programme is balanced at, -30 to +30 (default 0). The ear
     *        loses the two ends first as the level falls (their loudness
     *        grows faster with level), so each band is raised by the level
     *        times one minus the ratio of the mid exponent to its own: with
     *        20 dB, 4.7 dB at 50 Hz and 3.3 dB at 12 kHz against 1.2 dB at
     *        1 kHz, before the offset that keeps the total loudness (the
     *        volume does not change). Negative values compensate a louder
     *        listening. Limited by setRange() like every correction.
     */
    void setListeningCompensation(T db) noexcept { store(listening_, db, T(-30), T(30)); }
    /** @brief Loudness to decibels with each band's exponent (default off: one exponent). */
    void setLevelCompensation(bool on) noexcept { levelComp_.store(on, std::memory_order_relaxed); }
    /**
     * @brief Loudness to decibels with the slope of the loudness curve at
     *        each band's own level (default off): what a band lacks or has in
     *        excess is read with that slope, and so is the offset that keeps
     *        the loudness. Well over the threshold in
     *        quiet that slope is the band's exponent; near it a band's
     *        loudness grows faster with its power (the slope is twice the
     *        exponent 10 dB over the threshold, 1.45 times 20 dB over), and
     *        a fixed exponent overstates what the band lacks: the ends of a
     *        quiet programme are asked for less.
     */
    void setLocalSlope(bool on) noexcept { localSlope_.store(on, std::memory_order_relaxed); }
    /**
     * @brief Reads each band through a stretch of the analysis window of its
     *        own, shorter as frequency rises (SpectralFocus; default off: the
     *        whole window at every frequency). The low end keeps its
     *        resolution in frequency and the top reads a change of the
     *        programme from a stretch of a couple of milliseconds, which
     *        shows with a short response (Response::Common at 10 ms: a turn
     *        from dark to bright is followed at 8 kHz in 20 ms in place of
     *        28). On a steady programme the corrections are the same (within
     *        0.03 dB).
     */
    void setTimeFocus(bool on) noexcept { timeFocus_.store(on, std::memory_order_relaxed); }
    /** @brief The SPL a full-scale sine stands for, 60 to 130 dB (default 100). */
    void setReferenceLevel(T dbSpl) noexcept { store(referenceSpl_, dbSpl, T(60), T(130)); refDirty_.store(true, std::memory_order_release); }

    [[nodiscard]] T getBalance() const noexcept { return balance_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getFillDips() const noexcept { return fillDips_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTrimPeaks() const noexcept { return trimPeaks_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTilt() const noexcept { return tilt_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRange() const noexcept { return range_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getKnee() const noexcept { return knee_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBalanceAsymmetry() const noexcept { return balanceAsymmetry_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBalanceBias() const noexcept { return balanceBias_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getContrastKnee() const noexcept { return contrastKnee_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDetail() const noexcept { return detail_.load(std::memory_order_relaxed); }
    /** @return How many bands getBandCentre() and getCorrection() have: those of the model. */
    [[nodiscard]] static constexpr int getNumBands() noexcept { return kBands; }
    [[nodiscard]] T getRangeKnee() const noexcept { return rangeKnee_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getFloor() const noexcept { return floor_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLowFrequency() const noexcept { return loHz_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHighFrequency() const noexcept { return hiHz_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getLevelCompensation() const noexcept { return levelComp_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getLocalSlope() const noexcept { return localSlope_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getTimeFocus() const noexcept { return timeFocus_.load(std::memory_order_relaxed); }
    [[nodiscard]] Reference getReference() const noexcept { return reference_.load(std::memory_order_relaxed); }
    [[nodiscard]] Response getResponse() const noexcept { return response_.load(std::memory_order_relaxed); }
    [[nodiscard]] Quality getQuality() const noexcept { return quality_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getQualityFollowsRender() const noexcept { return qualityFollowsRender_.load(std::memory_order_relaxed); }
    /** @return The quality the last prepare() took. */
    [[nodiscard]] Quality getQualityInUse() const noexcept { return qualityInUse_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getResponseTime() const noexcept { return responseMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getListeningCompensation() const noexcept { return listening_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getReferenceLevel() const noexcept { return referenceSpl_.load(std::memory_order_relaxed); }

    /** @return The latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return filter_.getLatency(); }
    /** @return The centre of a band of the model, in hertz. */
    [[nodiscard]] double getBandCentre(int band) const noexcept { return bands_.getCentre(band); }
    /**
     * @return The correction of a band of the model in dB, the detail in it
     *         counted by the power it moves there (a readout for any thread).
     *         getDisplayCorrection() has the curve point by point.
     */
    [[nodiscard]] T getCorrection(int band) const noexcept
    {
        return published_[static_cast<size_t>(std::clamp(band, 0, kBands - 1))].load(std::memory_order_relaxed);
    }
    /** @brief Points of the display curve, evenly in log frequency from 20 Hz to 20 kHz. */
    static constexpr int kDisplayPoints = SpectralReadout<T>::kPoints;
    /** @return The frequency of a display point, in hertz. */
    [[nodiscard]] static double getDisplayFrequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }
    /** @return The correction the filter plays at a display point, in dB (any thread). */
    [[nodiscard]] T getDisplayCorrection(int point) const noexcept { return filter_.playedValue(0, point); }
    /**
     * @return The change of the programme's total loudness the corrections
     *         make, in dB of the loudness ratio: 0 while it is conserved,
     *         away from 0 only when the limits leave the bands in range no
     *         room to make up for it (a readout for any thread).
     */
    [[nodiscard]] T getLoudnessChange() const noexcept { return loudnessChange_.load(std::memory_order_relaxed); }
    /**
     * @return The share of the programme's loudness that the bands raised
     *         have gained (0 or more; 0.1 is a tenth of the total). A readout
     *         for any thread, like the three below.
     */
    [[nodiscard]] T getLoudnessRaised() const noexcept { return raised_.load(std::memory_order_relaxed); }
    /** @return The share of the programme's loudness that the bands lowered have lost (0 or less). */
    [[nodiscard]] T getLoudnessLowered() const noexcept { return lowered_.load(std::memory_order_relaxed); }
    /** @return The programme's total loudness as the model reads it, in dB (10 log10 of the sum of the bands' loudness; -200 in silence). */
    [[nodiscard]] T getLoudness() const noexcept { return total_.load(std::memory_order_relaxed); }
    /** @return How far the corrections move the centre of the loudness along the ear's scale, in Bark (positive: brighter). */
    [[nodiscard]] T getCentroidShift() const noexcept { return centroidShift_.load(std::memory_order_relaxed); }
    /** @return The response the filter plays at a frequency in dB, as last handed to the display (any thread). */
    [[nodiscard]] double getResponse(double hz) const noexcept
    {
        if (!(hz > 0.0)) return 0.0;
        const double at = std::clamp((kPoints - 1) * std::log(hz / 20.0) / std::log(1000.0), 0.0, static_cast<double>(kPoints - 1));
        const int under = std::min(static_cast<int>(at), kPoints - 2);
        const double a = static_cast<double>(filter_.playedValue(0, under)), b = static_cast<double>(filter_.playedValue(0, under + 1));
        return a + (at - under) * (b - a);
    }

    // -- Processing ----------------------------------------------------------

    /** @brief Analyses and equalizes the programme in place. */
    void processBlock(AudioBufferView<T> buffer) noexcept { run(buffer, nullptr); }

    /** @brief Equalizes the programme by the analysis of a key (same channels or one). */
    void processBlock(AudioBufferView<T> buffer, AudioBufferView<T> key) noexcept
    {
        run(buffer, key.getNumSamples() >= buffer.getNumSamples() && key.getNumChannels() > 0 ? &key : nullptr);
    }

    // -- State ---------------------------------------------------------------

    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TBAL"), 2);
        w.write("balance", static_cast<float>(getBalance()));
        w.write("fillDips", static_cast<float>(getFillDips()));
        w.write("trimPeaks", static_cast<float>(getTrimPeaks()));
        w.write("tilt", static_cast<float>(getTilt()));
        w.write("range", static_cast<float>(getRange()));
        w.write("knee", static_cast<float>(getKnee()));
        w.write("balanceAsymmetry", static_cast<float>(getBalanceAsymmetry()));
        w.write("balanceBias", static_cast<float>(getBalanceBias()));
        w.write("contrastKnee", static_cast<float>(getContrastKnee()));
        w.write("detailOctaves", static_cast<float>(getDetail()));
        w.write("rangeKnee", static_cast<float>(getRangeKnee()));
        w.write("floor", static_cast<float>(getFloor()));
        w.write("loHz", static_cast<float>(getLowFrequency()));
        w.write("hiHz", static_cast<float>(getHighFrequency()));
        w.write("levelComp", getLevelCompensation());
        w.write("localSlope", getLocalSlope());
        w.write("timeFocus", getTimeFocus());
        w.write("reference", static_cast<int32_t>(getReference()));
        w.write("response", static_cast<int32_t>(getResponse()));
        w.write("quality", static_cast<int32_t>(getQuality()));
        w.write("qualityFollowsRender", getQualityFollowsRender());
        w.write("responseMs", static_cast<float>(getResponseTime()));
        w.write("listening", static_cast<float>(getListeningCompensation()));
        w.write("refSpl", static_cast<float>(getReferenceLevel()));
        return w.blob();
    }

    /**
     * @brief Restores the settings. A blob of the first version gives the
     *        default detail, or a third, a sixth or a twelfth of an octave
     *        when it saved one of three steps.
     */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TBAL")) return false;
        setBalance(static_cast<T>(r.read("balance", 0.5f)));
        setFillDips(static_cast<T>(r.read("fillDips", 0.3f)));
        setTrimPeaks(static_cast<T>(r.read("trimPeaks", 0.5f)));
        setTilt(static_cast<T>(r.read("tilt", 0.0f)));
        setRange(static_cast<T>(r.read("range", 6.0f)));
        setKnee(static_cast<T>(r.read("knee", 3.0f)));
        setBalanceAsymmetry(static_cast<T>(r.read("balanceAsymmetry", 0.0f)));
        setBalanceBias(static_cast<T>(r.read("balanceBias", 0.0f)));
        setContrastKnee(static_cast<T>(r.read("contrastKnee", 3.0f)));
        if (r.processorVersion() < 2)
        {
            const int step = r.read("detail", -1);
            setDetail(step == 0 ? T(1.0 / 3.0) : (step == 1 ? T(1.0 / 6.0) : (step == 2 ? T(1.0 / 12.0) : T(1.0 / 4.0))));
        }
        else setDetail(static_cast<T>(r.read("detailOctaves", 0.25f)));
        setRangeKnee(static_cast<T>(r.read("rangeKnee", 0.0f)));
        setFloor(static_cast<T>(r.read("floor", 30.0f)));
        setFrequencyRange(static_cast<T>(r.read("loHz", 40.0f)), static_cast<T>(r.read("hiHz", 16000.0f)));
        setLevelCompensation(r.read("levelComp", false));
        setLocalSlope(r.read("localSlope", false));
        setTimeFocus(r.read("timeFocus", false));
        setReference(r.read("reference", 0) == 1 ? Reference::EvenPower : Reference::FlatLoudness);
        setResponse(r.read("response", 0) == 1 ? Response::Common : Response::ByBand);
        setQuality(r.read("quality", 0) == 1 ? Quality::Render : Quality::Realtime);
        setQualityFollowsRender(r.read("qualityFollowsRender", false));
        setResponseTime(static_cast<T>(r.read("responseMs", 60.0f)));
        setListeningCompensation(static_cast<T>(r.read("listening", 0.0f)));
        setReferenceLevel(static_cast<T>(r.read("refSpl", 100.0f)));
        return true;
    }

private:
    static void store(std::atomic<T>& a, T v, T lo, T hi) noexcept
    {
        if (std::isfinite(v)) a.store(std::clamp(v, lo, hi), std::memory_order_relaxed);
    }

    void run(AudioBufferView<T>& buffer, const AudioBufferView<T>* key) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        if (nCh <= 0) return;
        const bool referenceMoved = refDirty_.exchange(false, std::memory_order_acquire);
        if (referenceMoved) bands_.setReferenceSpl(static_cast<double>(referenceSpl_.load(std::memory_order_relaxed)));
        if (const T detail = detail_.load(std::memory_order_relaxed); referenceMoved || detail != lastDetail_)
        {
            lastDetail_ = detail;
            refreshDetail();
        }
        const double fadeTo = key != nullptr ? 1.0 : 0.0, fadeStep = 1.0 / (0.05 * fs_);
        filter_.processBlock(
            buffer,
            [&](int offset, int count) noexcept { feed(buffer, key, nCh, offset, count, fadeTo, fadeStep); },
            [&](int, T* curveDb) noexcept {
                analysis_.update(nCh);
                design(curveDb);
            });
        if (!filter_.isDisplayDue(shownCurve_)) return;
        shownCurve_ = filter_.getCurveCount();
        filter_.publishPlayed(nCh);
    }

    /**
     * @brief What is about to be taken of the programme (or of the key,
     *        faded) into both readings, the model running at each of its hops.
     */
    void feed(AudioBufferView<T>& buffer, const AudioBufferView<T>* key, int nCh, int offset, int count,
              double fadeTo, double fadeStep) noexcept
    {
        // What is not a number neither is read nor enters the filter.
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            T* x = buffer.getChannel(ch) + offset;
            for (int k = 0; k < count; ++k)
                if (!std::isfinite(x[k])) x[k] = T(0);
        }
        int done = 0;
        while (done < count)
        {
            const int chunk = std::min({ count - done, hop_ - hopCounter_, kFeed });
            double fade = keyFade_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* x = buffer.getChannel(ch) + offset + done;
                const T* k = key != nullptr ? key->getChannel(std::min(ch, key->getNumChannels() - 1)) + offset + done : nullptr;
                T* fed = fed_.data();
                auto& ring = ring_[static_cast<size_t>(ch)];
                fade = keyFade_;
                if (fade == 0.0 && fadeTo == 0.0)
                {
                    // The programme alone, already finite: as it comes.
                    const int head = std::min(chunk, fft_ - ringPos_);
                    double* r = ring.data();
                    for (int i = 0; i < head; ++i) r[ringPos_ + i] = static_cast<double>(x[i]);
                    for (int i = head; i < chunk; ++i) r[i - head] = static_cast<double>(x[i]);
                    analysis_.feed(ch, x, chunk);
                    continue;
                }
                for (int i = 0; i < chunk; ++i)
                {
                    fade = fadeTo > fade ? std::min(fadeTo, fade + fadeStep) : std::max(fadeTo, fade - fadeStep);
                    double a = static_cast<double>(x[i]);
                    if (fade > 0.0 && k != nullptr) a += fade * (static_cast<double>(k[i]) - a);
                    if (!std::isfinite(a)) a = 0.0;
                    fed[i] = static_cast<T>(a);
                    ring[static_cast<size_t>((ringPos_ + i) & (fft_ - 1))] = a;
                }
                analysis_.feed(ch, fed, chunk);
            }
            keyFade_ = fade;
            ringPos_ = (ringPos_ + chunk) & (fft_ - 1);
            hopCounter_ += chunk;
            if (hopCounter_ >= hop_)
            {
                hopCounter_ = 0;
                analyse(nCh);
            }
            done += chunk;
        }
    }

    /**
     * @brief The moments of the bins into those of the bands, in tmpA_ and on.
     * @tparam Lanes The moments a bin has: kMoments, or one for a single channel.
     * @param smoothed Whether through the envelope first (the focus has smoothed already).
     */
    template <int Lanes>
    void readBands(bool smoothed) noexcept
    {
        if (smoothed) envelope_.template computeInterleaved<Lanes>(moments_.data());
        bands_.template bandPowersInterleaved<Lanes>(moments_.data(), bandMoments_.data());
        std::vector<double>* const into[kMoments] = { &tmpA_, &tmpB_, &tmpC_, &tmpD_ };
        for (int j = 0; j < Lanes; ++j)
        {
            double* out = into[j]->data();
            for (int b = 0; b < kBands; ++b) out[b] = bandMoments_[static_cast<size_t>(Lanes * b + j)];
            AuditoryBands<double>::smoothAcross(out, kBands, 0.25);
        }
    }

    /** @brief One hop of the model: the broad balance and the loudness. */
    void analyse(int nCh) noexcept
    {
        for (int ch = 0; ch < nCh; ++ch)
        {
            const double* r = ring_[static_cast<size_t>(ch)].data();
            const double* w = window_.data();
            double* frame = frame_.data();
            const int older = fft_ - ringPos_;
            for (int k = 0; k < older; ++k) frame[k] = r[ringPos_ + k] * w[k];
            for (int k = 0; k < ringPos_; ++k) frame[older + k] = r[k] * w[older + k];
            transform_->forward(frame_.data(), spectra_[static_cast<size_t>(ch)].data());
        }
        // Band powers of the two channels and of their cross term, each from
        // a spectrum smoothed across frequency: the four moments of each bin
        // side by side, read together.
        const double* s0 = spectra_[0].data();
        const double* s1 = spectra_[static_cast<size_t>(nCh > 1 ? 1 : 0)].data();
        const bool focused = timeFocus_.load(std::memory_order_relaxed);
        double* m = moments_.data();
        if (focused)
        {
            // Through the focus: each channel's complex spectrum smoothed
            // across frequency (every bin read through a stretch of the window
            // of its own, shorter as frequency rises), then the products.
            const size_t other = nCh > 1 ? 1 : 0;
            focus_.smooth(s0, focusRe_[0].data(), focusIm_[0].data());
            if (nCh > 1) focus_.smooth(s1, focusRe_[1].data(), focusIm_[1].data());
            const double* ar = focusRe_[0].data();
            const double* ai = focusIm_[0].data();
            const double* br = focusRe_[other].data();
            const double* bi = focusIm_[other].data();
            for (int k = 0; k < bins_; ++k)
            {
                const double scale = powerScale_ * focus_.getGain(k);
                if (nCh > 1)
                {
                    m[kMoments * k] = scale * (ar[k] * ar[k] + ai[k] * ai[k]);
                    m[kMoments * k + 1] = scale * (br[k] * br[k] + bi[k] * bi[k]);
                    m[kMoments * k + 2] = scale * (ar[k] * br[k] + ai[k] * bi[k]);
                    m[kMoments * k + 3] = scale * (ai[k] * br[k] - ar[k] * bi[k]);
                }
                else
                {
                    m[k] = scale * (ar[k] * ar[k] + ai[k] * ai[k]);
                }
            }
        }
        else if (nCh > 1)
        {
            for (int k = 0; k < bins_; ++k)
            {
                const double lr = s0[2 * k], li = s0[2 * k + 1], rr = s1[2 * k], ri = s1[2 * k + 1];
                m[kMoments * k] = powerScale_ * (lr * lr + li * li);
                m[kMoments * k + 1] = powerScale_ * (rr * rr + ri * ri);
                m[kMoments * k + 2] = powerScale_ * (lr * rr + li * ri);
                m[kMoments * k + 3] = powerScale_ * (li * rr - lr * ri);
            }
        }
        else
        {
            for (int k = 0; k < bins_; ++k) m[k] = powerScale_ * (s0[2 * k] * s0[2 * k] + s0[2 * k + 1] * s0[2 * k + 1]);
        }
        if (nCh > 1) readBands<kMoments>(!focused);
        else readBands<1>(!focused);
        // The two smoothings in time: by band (both the band's own time,
        // longer in the low end) or a short integration per band and then a
        // response common to all.
        const bool common = response_.load(std::memory_order_relaxed) == Response::Common;
        const double commonCoeff = std::exp(-static_cast<double>(hop_)
            / (0.001 * static_cast<double>(responseMs_.load(std::memory_order_relaxed)) * fs_));
        // The covariance in time; the dominant power.
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double c = warm_ ? (common ? briefCoeff_[i] : timeCoeff_[i]) : 0.0;
            ll_[i] = tmpA_[i] + c * (ll_[i] - tmpA_[i]);
            if (nCh > 1)
            {
                rr_[i] = tmpB_[i] + c * (rr_[i] - tmpB_[i]);
                lr_[i] = tmpC_[i] + c * (lr_[i] - tmpC_[i]);
                li_[i] = tmpD_[i] + c * (li_[i] - tmpD_[i]);
                tmpA_[i] = AuditoryScale::dominantPower(ll_[i], rr_[i], lr_[i], li_[i]);
            }
            else
            {
                tmpA_[i] = ll_[i];
            }
        }
        bands_.specificLoudness(tmpA_.data(), tmpB_.data());
        const bool comp = levelComp_.load(std::memory_order_relaxed);
        constexpr double kMidExponent = 0.23;
        // The even programme: the same power in every band (the mean of this
        // programme's), and its loudness through the same model and the same
        // smoothing in time.
        double meanPower = 0.0;
        for (int b = 0; b < kBands; ++b) meanPower += tmpA_[static_cast<size_t>(b)];
        meanPower /= kBands;
        for (int b = 0; b < kBands; ++b) tmpD_[static_cast<size_t>(b)] = meanPower;
        bands_.specificLoudness(tmpD_.data(), tmpC_.data());
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double c = warm_ ? (common ? commonCoeff : timeCoeff_[i]) : 0.0;
            loud_[i] = tmpB_[i] + c * (loud_[i] - tmpB_[i]);
            evenLoud_[i] = tmpC_[i] + c * (evenLoud_[i] - tmpC_[i]);
        }
        warm_ = true;
        const bool evenPower = reference_.load(std::memory_order_relaxed) == Reference::EvenPower;
        const bool localSlope = localSlope_.load(std::memory_order_relaxed);
        // Loudness in decibels of power, the trend, the two deviations.
        const double tilt = static_cast<double>(tilt_.load(std::memory_order_relaxed));
        const double lo = static_cast<double>(loHz_.load(std::memory_order_relaxed));
        const double hi = static_cast<double>(hiHz_.load(std::memory_order_relaxed));
        const double zLo = AuditoryScale::hzToBark(lo), zHi = AuditoryScale::hzToBark(hi);
        // How audible each band is: its power over its threshold in quiet,
        // nothing under it, all of it from kAudibleSpan dB over. A band
        // under the threshold has a loudness of almost nothing, and its
        // level would read as a hole of tens of decibels: it neither shapes
        // the trend nor takes a correction (filling it would raise content
        // nobody hears, the ends of the band first).
        std::array<double, kBands> level {}, trend {}, weight {}, exponent {}, steep {}, audible {}, weighed {}, present {}, inRange {};
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            exponent[i] = comp ? bands_.getExponent(b) : kMidExponent;
            steep[i] = exponent[i];
            if (localSlope)
            {
                // d ln N / d ln P of N = (P + th)^a - th^a at the power this
                // band's loudness stands for: a far over the threshold, 1 under it.
                const double a = bands_.getExponent(b), th = bands_.getThreshold(b);
                const double whole = loud_[i] + bands_.getThresholdLoudness(b);   // (P + th)^a
                const double power = std::pow(whole, 1.0 / a) - th;          // P
                steep[i] = loud_[i] > 1e-30 && power > 0.0 ? std::clamp(a * power * whole / ((power + th) * loud_[i]), exponent[i], 1.0) : 1.0;
            }
            level[i] = 10.0 * std::log10(loud_[i] + 1e-12) / exponent[i];
            // Against the even programme, the level is how far this one
            // stands from it in each band.
            if (evenPower) level[i] -= 10.0 * std::log10(evenLoud_[i] + 1e-12) / exponent[i];
            const double over = 10.0 * std::log10((tmpA_[i] + 1e-30) / bands_.getThreshold(b));
            audible[i] = std::clamp(over / kAudibleSpan, 0.0, 1.0);
            trend[i] = audible[i] * level[i];
            weighed[i] = audible[i];
            const double z = bands_.getBark(b);
            inRange[i] = std::clamp(z - zLo + 0.5, 0.0, 1.0) * std::clamp(zHi - z + 0.5, 0.0, 1.0);
            weight[i] = inRange[i] * audible[i];
        }
        // How far under the programme each band lies: one setFloor() dB under
        // the loudest audible band (in the model's level) is no part of the
        // balance to give back (the floor of a recording, what lies past the
        // end of the content), and it takes neither the balance nor the
        // offset that keeps the loudness; it counts in full from kMaskedSpan
        // dB over that. A dip between two sides that sound is still a dip.
        {
            const double floorDb = static_cast<double>(floor_.load(std::memory_order_relaxed));
            double loudest = -1e9;
            for (int b = 0; b < kBands; ++b)
                if (audible[static_cast<size_t>(b)] > 0.0) loudest = std::max(loudest, level[static_cast<size_t>(b)]);
            for (int b = 0; b < kBands; ++b)
                present[static_cast<size_t>(b)] = std::clamp((level[static_cast<size_t>(b)] - loudest + floorDb) / kMaskedSpan, 0.0, 1.0);
        }
        // The trend across the bands, each weighed by how audible it is (a
        // normalized smoothing: the audible neighbours of a band decide it).
        AuditoryBands<double>::smoothAcross(trend.data(), kBands, 0.9);
        AuditoryBands<double>::smoothAcross(weighed.data(), kBands, 0.9);
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            trend[i] = weighed[i] > 1e-9 ? trend[i] / weighed[i] : level[i];
        }
        double meanDev = 0.0, wsum = 0.0;
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double ref = tilt * centreOctave_[i];
            tmpC_[i] = trend[i] - ref;
            meanDev += weight[i] * tmpC_[i];
            wsum += weight[i];
        }
        meanDev = wsum > 0.0 ? meanDev / wsum : 0.0;
        const double balance = static_cast<double>(balance_.load(std::memory_order_relaxed));
        // What is lacking (under the reference) and what is in excess, each
        // with its share of the balance; the point between them moved by the bias.
        const double asymmetry = static_cast<double>(balanceAsymmetry_.load(std::memory_order_relaxed));
        const double bias = static_cast<double>(balanceBias_.load(std::memory_order_relaxed));
        const double liftShare = balance * (1.0 + asymmetry), cutShare = balance * (1.0 - asymmetry);
        const double range = static_cast<double>(range_.load(std::memory_order_relaxed));
        const double knee = static_cast<double>(knee_.load(std::memory_order_relaxed));
        const auto soft = [knee](double d) { return std::copysign(std::sqrt(d * d + knee * knee) - knee, d); };
        const double rangeKnee = static_cast<double>(rangeKnee_.load(std::memory_order_relaxed));
        const double listening = static_cast<double>(listening_.load(std::memory_order_relaxed));
        const double midExponent = AuditoryScale::loudnessExponent(3300.0);
        // The limit of the range: hard, or the smaller of the correction and
        // the range through a hyperbola (set to pass through 0 and to settle
        // on the range).
        const double atZero = 0.5 * (range - std::sqrt(range * range + rangeKnee * rangeKnee));
        const auto limited = [range, rangeKnee, atZero](double c) {
            if (!(rangeKnee > 0.0)) return std::clamp(c, -range, range);
            const double a = std::abs(c), past = a - range;
            const double rounded = 0.5 * (a + range - std::sqrt(past * past + rangeKnee * rangeKnee));
            return std::copysign((rounded - atZero) * range / (range - atZero), c);
        };
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            // With the local slope, what a band lacks or has in excess is read with
            // the slope of its own loudness curve (never more than the exponent says).
            const double ease = exponent[i] / steep[i];
            const double global = ease * (tmpC_[i] - meanDev - bias);
            // The balance by how present the band is (limited first, so a band
            // half way down to the floor takes half of it), then what the
            // detail moves in the band.
            double c = limited(-(global < 0.0 ? liftShare : cutShare) * soft(global));
            if (present[i] < 1.0) c *= present[i];
            c = std::clamp(c * weight[i] + detailMean_[i], -range, range);
            // The listening level: a fixed contour, whatever the programme.
            if (listening != 0.0)
                c = std::clamp(c + listening * (1.0 - midExponent / bands_.getExponent(b)) * inRange[i], -range, range);
            corr_[i] = c;
        }
        // The total loudness stays: every band in range takes the same
        // loudness ratio (by its weight), found with the limits in place: a
        // band at its limit or out of the range gives nothing, so the others
        // make up for it. The loudness after an offset rises with it; a few
        // steps on its logarithm (exact in one when no band is held) with a
        // bracket find where it meets the loudness before.
        constexpr double kPerDb = 0.23025850929940457;   // ln(10) / 10
        std::array<double, kBands> part {};
        double before = 0.0;
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            part[i] = ((b == 0 || b == kBands - 1) ? 0.5 : 1.0) * loud_[i];
            before += part[i];
        }
        const auto loudnessAt = [&](double offset, double& slope) {
            double sum = 0.0;
            slope = 0.0;
            for (int b = 0; b < kBands; ++b)
            {
                const auto i = static_cast<size_t>(b);
                const double raw = corr_[i] + offset * weight[i] * present[i] / steep[i];
                const double l = part[i] * std::exp(kPerDb * steep[i] * std::clamp(raw, -range, range));
                sum += l;
                if (raw > -range && raw < range) slope += l * kPerDb * weight[i] * present[i];
            }
            return sum;
        };
        double offset = 0.0, change = 0.0;
        if (before > 1e-20)
        {
            double under = -kOffsetLimit, over = kOffsetLimit;   // the offset lies between them
            bool measured = false;   // whether `change` was read at the offset kept
            for (int step = 0; step < kOffsetSteps; ++step)
            {
                double slope = 0.0;
                const double after = loudnessAt(offset, slope);
                change = std::log(std::max(after, 1e-300) / before);
                measured = true;
                if (std::abs(change) < 1e-9) break;
                if (change > 0.0) over = offset; else under = offset;
                // Where every band that could move is at its limit there is
                // no slope: the offset is halfway back in the bracket.
                const double next = slope > 1e-30 * before ? offset - change * after / slope : over + 1.0;
                offset = next > under && next < over ? next : 0.5 * (under + over);
                measured = false;
            }
            if (!measured)
            {
                double slope = 0.0;
                change = std::log(std::max(loudnessAt(offset, slope), 1e-300) / before);
            }
            change /= kPerDb;
        }
        loudnessChange_.store(static_cast<T>(change), std::memory_order_relaxed);
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            corr_[i] = std::clamp(corr_[i] + offset * weight[i] * present[i] / steep[i], -range, range);
            // The broad curve: the band's, less what was the detail's.
            base_[i] = corr_[i] - detailMean_[i];
            published_[i].store(static_cast<T>(corr_[i]), std::memory_order_relaxed);
        }
        // What the corrections do to the loudness: the share the raised bands
        // gain, the share the lowered ones lose, the total, and how far its
        // centre moves along the ear's scale.
        {
            double raised = 0.0, lowered = 0.0, after = 0.0, centreBefore = 0.0, centreAfter = 0.0;
            for (int b = 0; b < kBands; ++b)
            {
                const auto i = static_cast<size_t>(b);
                const double l = part[i] * std::exp(kPerDb * steep[i] * corr_[i]);
                const double d = l - part[i];
                if (d > 0.0) raised += d; else lowered += d;
                after += l;
                centreBefore += part[i] * bands_.getBark(b);
                centreAfter += l * bands_.getBark(b);
            }
            const bool sounding = before > 1e-20 && after > 1e-20;
            raised_.store(static_cast<T>(sounding ? raised / before : 0.0), std::memory_order_relaxed);
            lowered_.store(static_cast<T>(sounding ? lowered / before : 0.0), std::memory_order_relaxed);
            total_.store(static_cast<T>(sounding ? 10.0 * std::log10(before) : -200.0), std::memory_order_relaxed);
            centroidShift_.store(static_cast<T>(sounding ? centreAfter / after - centreBefore / before : 0.0), std::memory_order_relaxed);
        }
    }

    // -- The detail: the dips and the peaks over the whole band -----------------

    static constexpr int kPoints = SpectralReadout<T>::kPoints;

    /** @brief The grid's maps onto the bands of the model, and the buffers. */
    void prepareDetail()
    {
        const auto size = static_cast<size_t>(kPoints);
        for (auto* v : { &power_, &smoothPower_, &seen_, &level_, &gapLevel_, &heard_, &asked_, &detailGain_,
                         &lowerSide_, &upperSide_, &quarter_, &octave_, &threshold_, &slowCoeff_ })
            v->assign(size, T(0));
        sum_.assign(size + 1, 0.0);
        place_.resize(size);
        for (auto* v : { &heldSum_, &movedSum_ }) v->assign(size + 1, 0.0);
        for (auto* v : { &pointShare_, &seenShare_ }) v->assign(size, 0.0);
        for (auto* v : { &seenFrom_, &seenTo_ }) v->assign(size, 0);
        hopSeconds_ = static_cast<double>(filter_.getHopSize()) / fs_;
        top_ = 0;
        for (int i = 0; i < kPoints; ++i)
        {
            const auto u = static_cast<size_t>(i);
            const double hz = DetailFilter<T, DetailLatency::Block>::frequency(i);
            if (hz < 0.45 * fs_) top_ = i + 1;
            octave_[u] = static_cast<T>(std::log2(hz / 1000.0));
            const double at = std::clamp((AuditoryScale::hzToBark(hz) - bands_.getBark(0)) / bands_.getSpacing(), 0.0, kBands - 1.0);
            const int under = std::min(static_cast<int>(at), kBands - 2);
            const double t = at - under;
            Place& p = place_[u];
            p.band = under;
            p.part = static_cast<T>(t);
            // Catmull-Rom between the centres: the broad curve has no corners.
            p.weight[0] = static_cast<T>(0.5 * t * ((2.0 - t) * t - 1.0));
            p.weight[1] = static_cast<T>(0.5 * (t * t * (3.0 * t - 5.0) + 2.0));
            p.weight[2] = static_cast<T>(0.5 * t * ((4.0 - 3.0 * t) * t + 1.0));
            p.weight[3] = static_cast<T>(0.5 * t * t * (t - 1.0));
            // Few bins at the bottom: the low end takes longer to settle.
            slowCoeff_[u] = static_cast<T>(std::exp(-hopSeconds_ / std::clamp(1.5 * std::sqrt(100.0 / hz), kShortestSmoothing, 2.0)));
        }
        detailGlide_ = static_cast<T>(std::exp(-hopSeconds_ / kDetailGlideSeconds));
        lastDetail_ = detail_.load(std::memory_order_relaxed);
        refreshDetail();
    }

    /** @brief What follows the detail and the reference level: the reading, the sides, the thresholds. */
    void refreshDetail() noexcept
    {
        const double width = static_cast<double>(lastDetail_);
        analysis_.setWidth(width);
        const double perOctave = (kPoints - 1) * std::numbers::ln2 / std::log(1000.0);
        aside_ = std::max(2, static_cast<int>(std::lround(0.5 * width * perOctave)));
        // The rule of the space between partials comes in from a sixth of an
        // octave down to a twelfth, so that a sweep of the detail has no step.
        gapShare_ = static_cast<T>(std::clamp(std::log2(kFinestDip / width), 0.0, 1.0));
        gapHalf_ = static_cast<int>(std::lround(0.5 * kFinestDip * perOctave));
        acrossHalf_ = std::max(1, static_cast<int>(std::lround(width * perOctave / 8.0)));
        sidePole_ = std::exp(-1.0 / (std::max(kSideOctaves, 2.0 * width) * perOctave));
        const double bandShare = 0.5 * (std::exp2(0.5 * width) - std::exp2(-0.5 * width)) + 0.5 * (std::pow(1000.0, 1.0 / (kPoints - 1)) - 1.0);
        for (int i = 0; i < kPoints; ++i)
        {
            const auto u = static_cast<size_t>(i);
            const double hz = DetailFilter<T, DetailLatency::Block>::frequency(i);
            const Place& p = place_[u];
            // A point holds the power of its width (of the window's, where that
            // is wider); the threshold is that of a band of the model around it.
            const double held = std::max(bandShare * hz, 1.5 / std::max(analysis_.getWindowSeconds(i), 1e-3));
            const double band = AuditoryScale::criticalBandwidth(hz) * bands_.getSpacing() * modelNoiseBins_;
            const double a = 10.0 * std::log10(bands_.getThreshold(p.band) + 1e-30), b = 10.0 * std::log10(bands_.getThreshold(p.band + 1) + 1e-30);
            threshold_[u] = static_cast<T>(a + static_cast<double>(p.part) * (b - a) - 10.0 * std::log10(band / held));
            // A point stands for the hertz to the next: that share of what it holds.
            pointShare_[u] = hz / held;
            // And a band of the model sees a stretch around it: its envelope's
            // sixth of an octave, or the lobe of its window where that is wider.
            const double reach = std::max(hz * 0.5 * (std::exp2(1.0 / 12.0) - std::exp2(-1.0 / 12.0)), 0.5 * modelNoiseBins_ * fs_ / fft_);
            const double last = kPoints - 1.0, perLog = last / std::log(1000.0);
            const int from = static_cast<int>(std::lround(perLog * std::log(std::max(20.0, hz - reach) / 20.0)));
            const int to = static_cast<int>(std::lround(std::min(last, perLog * std::log((hz + reach) / 20.0)))) + 1;
            seenFrom_[u] = std::clamp(from, 0, i);
            seenTo_[u] = std::clamp(to, i + 1, kPoints);
            seenShare_[u] = hz / (2.0 * reach);
        }
    }

    void resetDetail() noexcept
    {
        for (auto* v : { &smoothPower_, &seen_, &detailGain_ }) std::fill(v->begin(), v->end(), T(0));
        base_.fill(0.0);
        detailMean_.fill(0.0);
        shownCurve_ = 0;
    }

    /** @brief The curve of a hop: the broad balance between the bands' centres, and the detail. */
    void design(T* curve) noexcept
    {
        readDetail();
        const T range = range_.load(std::memory_order_relaxed);
        for (int i = 0; i < kPoints; ++i)
        {
            const Place& p = place_[static_cast<size_t>(i)];
            const auto under = static_cast<size_t>(p.band);
            const double before = base_[under > 0 ? under - 1 : 0], after = base_[std::min(under + 2, static_cast<size_t>(kBands - 1))];
            const T broad = static_cast<T>(static_cast<double>(p.weight[0]) * before + static_cast<double>(p.weight[1]) * base_[under]
                                           + static_cast<double>(p.weight[2]) * base_[under + 1] + static_cast<double>(p.weight[3]) * after);
            curve[i] = std::clamp(broad + detailGain_[static_cast<size_t>(i)], -range, range);
        }
    }

    /** @brief The dips and the peaks of a hop, and what they move in each band of the model. */
    void readDetail() noexcept
    {
        analysis_.read(0, power_.data());
        const bool common = response_.load(std::memory_order_relaxed) == Response::Common;
        const T shared = common ? static_cast<T>(std::exp(-hopSeconds_
            / std::max(kShortestCommon, 0.001 * static_cast<double>(responseMs_.load(std::memory_order_relaxed))))) : T(0);
        for (int i = 0; i < top_; ++i)
        {
            const auto u = static_cast<size_t>(i);
            const T keep = common ? shared : slowCoeff_[u];
            const T p = std::isfinite(power_[u]) ? power_[u] : T(0);
            smoothPower_[u] = p + keep * (smoothPower_[u] - p);
            seen_[u] = T(1) + keep * (seen_[u] - T(1));   // of the smoothing filled: no rise out of the start
            power_[u] = seen_[u] > T(0) ? smoothPower_[u] / seen_[u] : T(0);
            level_[u] = T(4.3429448190325183) * fastLog(power_[u] + T(1e-30));
            heard_[u] = std::clamp((level_[u] - threshold_[u]) / static_cast<T>(kAudibleSpan), T(0), T(1));
        }
        const T fill = fillDips_.load(std::memory_order_relaxed), trim = trimPeaks_.load(std::memory_order_relaxed);
        std::fill(asked_.begin(), asked_.end(), T(0));
        if (fill > T(0) || trim > T(0)) readContrast(fill, trim);

        std::array<double, kBands> moved {}, held {};
        bool moving = false;
        for (int i = 0; i < kPoints; ++i)
        {
            const auto u = static_cast<size_t>(i);
            T& g = detailGain_[u];
            g = asked_[u] + detailGlide_ * (g - asked_[u]);
            if (asked_[u] == T(0) && std::abs(g) < T(1e-6)) g = T(0);
            moving = moving || g != T(0);
        }
        detailMean_.fill(0.0);
        if (!moving) return;
        // By the power it moves there, as the model reads it: a band of the
        // model sees what is around it through its window and its envelope,
        // so a gap beside a partial is as loud as the partial makes it, and
        // what raises the gap adds next to nothing.
        double power = 0.0, change = 0.0;
        for (int i = 0; i < top_; ++i)
        {
            const auto u = static_cast<size_t>(i);
            heldSum_[u] = power;
            movedSum_[u] = change;
            const double p = static_cast<double>(power_[u]) * pointShare_[u];
            power += p;
            change += p * static_cast<double>(std::exp(T(0.23025850929940457) * detailGain_[u]) - T(1));
        }
        heldSum_[static_cast<size_t>(top_)] = power;
        movedSum_[static_cast<size_t>(top_)] = change;
        for (int i = 0; i < top_; ++i)
        {
            const auto u = static_cast<size_t>(i);
            const Place& p = place_[u];
            const double part = static_cast<double>(p.part);
            const auto first = static_cast<size_t>(std::min(seenFrom_[u], top_)), end = static_cast<size_t>(std::min(seenTo_[u], top_));
            const double share = seenShare_[u];
            const double around = share * (heldSum_[end] - heldSum_[first]), moves = share * (movedSum_[end] - movedSum_[first]);
            const auto under = static_cast<size_t>(p.band);
            held[under] += (1.0 - part) * around;
            held[under + 1] += part * around;
            moved[under] += (1.0 - part) * moves;
            moved[under + 1] += part * moves;
        }
        AuditoryBands<double>::smoothAcross(held.data(), kBands, kReadAcross);
        AuditoryBands<double>::smoothAcross(moved.data(), kBands, kReadAcross);
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            if (held[i] > 0.0) detailMean_[i] = 10.0 * std::log10(std::max(1.0 + moved[i] / held[i], 1e-12));
        }
    }

    /** @brief Each point against its two sides, in asked_. */
    void readContrast(T fill, T trim) noexcept
    {
        const T* db = level_.data();
        const T* power = power_.data();
        const T* heard = heard_.data();
        const double pole = sidePole_;
        // Audible points alone, and none within the detail of the point itself.
        // A side is the mean of its loudness (the power to a quarter, as the
        // ear has it): the emptiness past a group of lines would pull a mean
        // of decibels under the gaps inside the group, and in a mean of
        // amplitudes a line as wide as the window would hide the one beside it.
        const auto sideDb = [](double sum, double weight) {
            return T(17.371779276130073) * fastLog(static_cast<T>(sum / weight) + T(1e-8));
        };
        T* quarter = quarter_.data();
        for (int i = 0; i < top_; ++i) quarter[i] = heard[i] * std::sqrt(std::sqrt(power[i]));
        double num = 0.0, den = 0.0;
        for (int i = 0; i < top_; ++i)
        {
            if (const int j = i - aside_; j >= 0)
            {
                num = pole * num + static_cast<double>(quarter[j]);
                den = pole * den + static_cast<double>(heard[j]);
            }
            lowerSide_[static_cast<size_t>(i)] = den > 1e-9 ? sideDb(num, den) : db[i];
        }
        num = 0.0; den = 0.0;
        for (int i = top_ - 1; i >= 0; --i)
        {
            if (const int j = i + aside_; j < top_)
            {
                num = pole * num + static_cast<double>(quarter[j]);
                den = pole * den + static_cast<double>(heard[j]);
            }
            upperSide_[static_cast<size_t>(i)] = den > 1e-9 ? sideDb(num, den) : db[i];
        }

        const T* dips = db;
        if (gapShare_ > T(0))
        {
            // A dip is looked for no finer than a sixth of an octave: what is
            // narrower between two partials is the space between them.
            double total = 0.0;
            for (int i = 0; i < top_; ++i)
            {
                sum_[static_cast<size_t>(i)] = total;
                total += static_cast<double>(power[i]);
            }
            sum_[static_cast<size_t>(top_)] = total;
            for (int i = 0; i < top_; ++i)
            {
                const int first = std::max(0, i - gapHalf_), end = std::min(top_, i + gapHalf_ + 1);
                const double around = (sum_[static_cast<size_t>(end)] - sum_[static_cast<size_t>(first)]) / (end - first);
                const T over = T(4.3429448190325183) * fastLog(static_cast<T>(around) + T(1e-30)) - db[i];
                gapLevel_[static_cast<size_t>(i)] = db[i] + gapShare_ * std::max(T(0), over);
            }
            dips = gapLevel_.data();
        }
        const T knee = contrastKnee_.load(std::memory_order_relaxed);
        const auto bend = [knee](T d) { return std::copysign(std::sqrt(d * d + knee * knee) - knee, d); };
        const T range = range_.load(std::memory_order_relaxed), rangeKnee = rangeKnee_.load(std::memory_order_relaxed);
        const T atZero = T(0.5) * (range - std::sqrt(range * range + rangeKnee * rangeKnee));
        const T from = std::log2(loHz_.load(std::memory_order_relaxed) / T(1000));
        const T to = std::log2(hiHz_.load(std::memory_order_relaxed) / T(1000));
        for (int i = 0; i < top_; ++i)
        {
            const auto u = static_cast<size_t>(i);
            if (heard[i] == T(0)) continue;
            // The edges of the range are a third of an octave wide here (one
            // Bark would reach from 40 Hz to 90 Hz).
            const T inside = std::clamp(T(3) * (octave_[u] - from) + T(0.5), T(0), T(1))
                           * std::clamp(T(3) * (to - octave_[u]) + T(0.5), T(0), T(1));
            if (inside == T(0)) continue;
            const T under = std::min(T(0), dips[i] - std::min(lowerSide_[u], upperSide_[u]));
            // Nor is that space cut with the partials beside it.
            const T over = std::max(T(0), db[i] - std::max(lowerSide_[u], upperSide_[u]) - (dips[i] - db[i]));
            const T low = std::clamp((static_cast<T>(kEmpty) + under) / static_cast<T>(kEmptyFade), T(0), T(1));
            T c = -trim * bend(over) - fill * low * bend(std::max(under, static_cast<T>(-kDipLimit)));
            if (!(rangeKnee > T(0))) c = std::clamp(c, -range, range);
            else
            {
                const T a = std::abs(c), past = a - range;
                const T rounded = T(0.5) * (a + range - std::sqrt(past * past + rangeKnee * rangeKnee));
                c = std::copysign((rounded - atZero) * range / (range - atZero), c);
            }
            asked_[u] = inside * heard[i] * c;
        }
        // No finer than the detail: over an eighth of it on either side, a point
        // at least (a step of one point would ring between two of them).
        double sum = 0.0;
        for (int i = 0; i < kPoints; ++i)
        {
            sum_[static_cast<size_t>(i)] = sum;
            sum += static_cast<double>(asked_[static_cast<size_t>(i)]);
        }
        sum_[static_cast<size_t>(kPoints)] = sum;
        for (int i = 0; i < kPoints; ++i)
        {
            const int first = std::max(0, i - acrossHalf_), end = std::min(kPoints, i + acrossHalf_ + 1);
            asked_[static_cast<size_t>(i)] = static_cast<T>((sum_[static_cast<size_t>(end)] - sum_[static_cast<size_t>(first)]) / (end - first));
        }
    }

    /// Where a point lies between the centres of the model's bands.
    struct Place
    {
        T weight[4] {};
        T part = T(0);
        int band = 0;
    };

    static constexpr int kFeed = 256;                  ///< Most samples fed to the readings at once.
    static constexpr double kReadAcross = 0.1;     ///< What a band shares with the next when the detail's moves are counted.
    static constexpr double kSideOctaves = 0.5;        ///< Reach of the two sides a point is read against (two details, where that is more).
    static constexpr double kFinestDip = 1.0 / 6.0;    ///< A dip is looked for no finer, in octaves.
    static constexpr double kDipLimit = 18.0;          ///< A dip counts no deeper.
    static constexpr double kEmpty = 45.0;             ///< This far under its sides a point is empty rather than low.
    static constexpr double kEmptyFade = 15.0;         ///< And is let go over this many dB before that.
    static constexpr double kShortestSmoothing = 0.4;  ///< The detail's smoothing at the top, in seconds.
    static constexpr double kShortestCommon = 0.1;     ///< And its shortest with Response::Common.
    static constexpr double kDetailGlideSeconds = 0.1; ///< Glide of what the detail asks.
    static constexpr double kAudibleSpan = 10.0;  ///< dB over the threshold in quiet from which a band counts in full.
    static constexpr double kMaskedSpan = 10.0;   ///< dB over the floor from which a band counts in full for the balance.
    static constexpr double kOffsetLimit = 60.0;  ///< Largest loudness offset looked for, in dB.
    static constexpr int kOffsetSteps = 24;       ///< Most steps of the search for the loudness offset (a few, as a rule).

    double fs_ = 48000.0, powerScale_ = 1.0, modelNoiseBins_ = 1.5, keyFade_ = 0.0;
    int channels_ = 0, fft_ = 2048, hop_ = 128, bins_ = 1025;
    int ringPos_ = 0, hopCounter_ = 0;
    bool prepared_ = false, warm_ = false;

    std::atomic<T> balance_ { T(0.5) }, fillDips_ { T(0.3) }, trimPeaks_ { T(0.5) }, tilt_ { T(0) };
    std::atomic<T> balanceAsymmetry_ { T(0) }, balanceBias_ { T(0) };   ///< setBalanceAsymmetry(), setBalanceBias().
    std::atomic<T> range_ { T(6) }, knee_ { T(3) }, loHz_ { T(40) }, hiHz_ { T(16000) }, referenceSpl_ { T(100) };
    std::atomic<bool> levelComp_ { false }, refDirty_ { false }, localSlope_ { false };
    std::atomic<Reference> reference_ { Reference::FlatLoudness };
    std::atomic<Response> response_ { Response::ByBand };
    std::atomic<Quality> quality_ { Quality::Realtime }, qualityInUse_ { Quality::Realtime };
    std::atomic<bool> qualityFollowsRender_ { false }, offlineRendering_ { false };
    bool render_ = false;   ///< The quality the last prepare() took (audio thread).
    std::atomic<T> responseMs_ { T(60) }, listening_ { T(0) };
    std::array<std::atomic<T>, kBands> published_ {};
    std::atomic<T> loudnessChange_ { T(0) }, raised_ { T(0) }, lowered_ { T(0) }, total_ { T(-200) }, centroidShift_ { T(0) };
    std::atomic<T> rangeKnee_ { T(0) }, floor_ { T(30) };
    std::atomic<T> contrastKnee_ { T(3) }, detail_ { T(1.0 / 4.0) };

    std::unique_ptr<FFTReal<double>> transform_;
    AuditoryBands<double> bands_;
    SpectralEnvelope<double> envelope_;
    SpectralFocus<double> focus_;
    std::array<std::vector<double>, 2> focusRe_, focusIm_;
    std::atomic<bool> timeFocus_ { false };
    static constexpr int kMoments = 4;   ///< Of a bin of two channels: both powers and their cross term.
    std::vector<double> window_, frame_, moments_, bandMoments_;
    std::array<double, kBands> centreOctave_ {}; ///< The centres in octaves from 1 kHz.
    std::vector<std::vector<double>> ring_, spectra_;
    std::vector<double> ll_, rr_, lr_, li_, loud_, evenLoud_, corr_, tmpA_, tmpB_, tmpC_, tmpD_;
    std::array<double, kBands> timeCoeff_ {}, briefCoeff_ {};

    // The detail, and the filter that plays both.
    DetailFilter<T, DetailLatency::Block> filter_;
    DetailAnalysis<T> analysis_;
    std::array<double, kBands> base_ {};       ///< The broad curve at the model's centres.
    std::array<double, kBands> detailMean_ {}; ///< What the detail moves in each band, by power.
    std::vector<double> heldSum_, movedSum_;   ///< The running sums of what each point holds and moves.
    std::vector<double> pointShare_, seenShare_; ///< A point's share of what it holds, and of the stretch the model reads.
    std::vector<int> seenFrom_, seenTo_;       ///< That stretch, in points.
    std::vector<Place> place_;
    std::vector<T> fed_, power_, smoothPower_, seen_, level_, gapLevel_, heard_, asked_, detailGain_;
    std::vector<T> lowerSide_, upperSide_, quarter_, octave_, threshold_, slowCoeff_;
    std::vector<double> sum_;
    double hopSeconds_ = 256.0 / 48000.0, sidePole_ = 0.98;
    T lastDetail_ = T(1.0 / 4.0), detailGlide_ = T(0);
    T gapShare_ = T(0);        ///< How much the space between partials is left alone.
    int top_ = 0, aside_ = 8, acrossHalf_ = 1, gapHalf_ = 0, shownCurve_ = 0;
};

} // namespace dspark
