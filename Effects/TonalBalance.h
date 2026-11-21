// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TonalBalance.h
 * @brief An adaptive equalizer driven by an auditory model: it measures the
 *        loudness of the programme in bands of the ear's scale and moves a
 *        bank of filters toward a balanced shape, raising what is missing
 *        and taming what stands out, while the total loudness stays.
 *
 * The analysis looks at the programme every hop (128 samples at 48 kHz)
 * through a transform of 2048 with an asymmetric window whose weight sits
 * on the most recent samples (low latency); the analysis scales by octaves
 * with the sample rate, so it sees the same span at any rate. The power
 * spectrum is smoothed across frequency with a width that follows frequency
 * (SpectralEnvelope), read in 72 bands uniform on the Bark scale from 20 Hz
 * to 20 kHz (AuditoryBands) and smoothed across the bands (the ear's
 * spread). In each band the covariance of the two channels is smoothed in
 * time, longer in the low end, and the band's power is its larger
 * eigenvalue: the level of the dominant component wherever it sits in the
 * stereo image. The cross term of that covariance is complex (one spectrum
 * times the conjugate of the other), so a channel that is the other one
 * delayed reads as one source and leaves no comb in the band powers. The
 * specific loudness of each band (the power law over the threshold in
 * quiet) is smoothed in time and read in decibels of power.
 *
 * Each band counts by how audible it is: nothing while its power is under
 * its threshold in quiet, in full from 10 dB over it. In the bands at the
 * ends, where the threshold climbs, a programme's loudness is almost
 * nothing, and read as a level it would be a hole of tens of decibels; an
 * inaudible band neither shapes the trend nor takes a correction. And a
 * band far under the programme (setFloor() dB under its loudest band, in
 * the model's level) is the floor of a recording or the end of the content:
 * the balance does not give it back, though a dip between two sides that
 * sound is still filled.
 *
 * From there: a broad trend across the bands (a smoothing in which each band
 * weighs by its audibility), and two deviations per band, the trend against
 * a reference shape (a tilt in dB per octave; the global balance) and the
 * band against its two sides (the local contrast: a dip is a band under the
 * audible bands both below and above it, a peak a band over both, so the
 * fall of the content at an end of the band is neither). The reference is
 * a formula, flat loudness per Bark tilted by setTilt() around 1 kHz, and
 * the tilt acts through setBalance() (the share of the way the trend moves
 * toward it). Two
 * soft-knee curves (hyperbolas, with a different slope on each side) turn a
 * deviation into a correction: setBalance() moves the trend toward the
 * reference, setFillDips() raises dips and setTrimPeaks() lowers peaks. The
 * corrections are limited
 * (setRange()), faded at the edges of setFrequencyRange() over one Bark, and
 * then offset so the total loudness of the programme does not change (the
 * correction redistributes): the offset is solved with the limits and the
 * edges of the range in place, so the sum is exact whenever the bands in
 * range have room to move (getLoudnessChange() reads what is left).
 * setLevelCompensation(true) converts loudness to decibels with each band's
 * own loudness exponent (the low end, whose loudness grows faster with
 * level, moves less in decibels for the same change of loudness).
 *
 * The gains drive a GraphicBank (a bell per band, a shelf at each end,
 * their gains solved by least squares so the bank's response is the one
 * asked for) whose sections glide over each hop; the bank refines its
 * solution one hop in eight and carries that correction over the others, so
 * its response does not step with a period of eight hops. The programme is
 * delayed to line up with the analysis. processBlock(buffer, key) analyses
 * an external key instead, with a fade of 50 ms when the key comes and goes.
 *
 * Latency: the analysis delay, about 16 ms at 48 kHz, plus 7 samples of the
 * bank in mono (getLatency()).
 *
 * Threading: prepare() and setState() on the setup thread (allocate);
 * processBlock() and reset() on the audio thread; the setters are lock-free
 * atomic publications, safe from any thread. Non-finite values are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, AuditoryBands.h, DenormalGuard.h,
 * DspMath.h, FFT.h, GraphicBank.h, SpectralEnvelope.h, StateBlob.h,
 * WindowFunctions.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AuditoryBands.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/GraphicBank.h"
#include "../Core/SpectralEnvelope.h"
#include "../Core/StateBlob.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
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
    static constexpr int kBands = 72;      ///< Bands of the analysis and of the bank.

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
     * @brief How fine the analysis and the bank run (setQuality(); it
     *        applies at the next prepare()). Realtime, the default: a hop of
     *        a sixteenth of the transform, the bank's gains refined one hop
     *        in eight and carried over the others. Render: half that hop,
     *        and the bank refined three times at every hop (its response
     *        lands closer to the corrections asked), at about three times
     *        the cost. The latency is the same in both.
     */
    enum class Quality { Realtime, Render };
    static constexpr int kMaxChannels = 2; ///< Channels processed (more pass through).

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        fs_ = spec.sampleRate;
        channels_ = std::min(spec.numChannels, kMaxChannels);
        // The analysis scales by octaves with the rate (2048 at 44.1 to 48 kHz).
        const int octaves = static_cast<int>(std::lround(std::log2(std::max(fs_, 8000.0) / 48000.0)));
        fft_ = std::clamp(2048 << std::max(0, octaves), 1024, 16384);
        if (octaves < 0) fft_ = std::max(512, 2048 >> -octaves);
        // The render quality halves the hop; the programme waits as long in both.
        render_ = quality_.load(std::memory_order_relaxed) == Quality::Render
            || (qualityFollowsRender_.load(std::memory_order_relaxed) && offlineRendering_.load(std::memory_order_relaxed));
        qualityInUse_.store(render_ ? Quality::Render : Quality::Realtime, std::memory_order_relaxed);
        const int realtimeHop = std::max(16, fft_ / 16);
        hop_ = render_ ? std::max(8, realtimeHop / 2) : realtimeHop;
        const int fall = fft_ / 8;
        transform_ = std::make_unique<FFTReal<double>>(static_cast<size_t>(fft_));
        window_.resize(static_cast<size_t>(fft_));
        WindowFunctions<double>::asymmetricHann(window_.data(), fft_, fall);
        double sw = 0.0, swi = 0.0;
        for (int i = 0; i < fft_; ++i) { sw += window_[static_cast<size_t>(i)]; swi += window_[static_cast<size_t>(i)] * i; }
        // Power scale: a full-scale sine reads 1 in its bin.
        powerScale_ = 4.0 / (sw * sw);
        // The programme waits for the analysis: from the window's centre of
        // mass to its end, plus half a hop (the gains glide over a hop).
        analysisDelay_ = std::max(1, static_cast<int>(std::lround(static_cast<double>(fft_ - 1) - swi / sw + 0.5 * realtimeHop)));
        bins_ = fft_ / 2 + 1;
        bands_.prepare(fs_, fft_, kBands, 20.0, std::min(20000.0, 0.45 * fs_));
        bands_.setReferenceSpl(static_cast<double>(referenceSpl_.load(std::memory_order_relaxed)));
        envelope_.prepare(bins_, 1.0 / 6.0);
        focus_.prepare(window_.data(), fft_, fft_, 1.0 / 6.0);
        for (auto* v : { &focusRe_[0], &focusRe_[1], &focusIm_[0], &focusIm_[1] }) v->assign(static_cast<size_t>(bins_), 0.0);
        std::array<double, kBands> centres {};
        for (int b = 0; b < kBands; ++b) centres[static_cast<size_t>(b)] = bands_.getCentre(b);
        bank_.prepare(fs_, centres.data(), kBands, channels_);
        ring_.assign(static_cast<size_t>(channels_), std::vector<double>(static_cast<size_t>(fft_), 0.0));
        spectra_.assign(static_cast<size_t>(channels_), std::vector<double>(static_cast<size_t>(fft_ + 2), 0.0));
        delay_.assign(static_cast<size_t>(channels_), std::vector<T>(static_cast<size_t>(analysisDelay_), T(0)));
        frame_.assign(static_cast<size_t>(fft_), 0.0);
        binPower_.assign(static_cast<size_t>(bins_), 0.0);
        binSmooth_.assign(static_cast<size_t>(bins_), 0.0);
        for (auto* v : { &ll_, &rr_, &lr_, &li_, &loud_, &evenLoud_, &corr_, &tmpA_, &tmpB_, &tmpC_, &tmpD_ }) v->assign(kBands, 0.0);
        for (int b = 0; b < kBands; ++b)
        {
            const double tau = 0.001 * std::clamp(200.0 * std::pow(1000.0 / bands_.getCentre(b), 0.3), 20.0, 2000.0);
            timeCoeff_[static_cast<size_t>(b)] = std::exp(-static_cast<double>(hop_) / (tau * fs_));
            // The short integration of Response::Common: four periods of the centre, 8 to 100 ms.
            const double brief = std::clamp(4.0 / bands_.getCentre(b), 0.008, 0.1);
            briefCoeff_[static_cast<size_t>(b)] = std::exp(-static_cast<double>(hop_) / (brief * fs_));
        }
        prepared_ = true;
        reset();
    }

    /** @brief Clears the analysis, the corrections, the bank and the delay. */
    void reset() noexcept
    {
        for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
        for (auto& d : delay_) std::fill(d.begin(), d.end(), T(0));
        for (auto* v : { &ll_, &rr_, &lr_, &li_, &loud_, &evenLoud_, &corr_ }) std::fill(v->begin(), v->end(), 0.0);
        ringPos_ = delayPos_ = hopCounter_ = hops_ = 0;
        keyFade_ = 0.0;
        warm_ = false;
        bank_.reset();
        std::array<T, kBands> zero {};
        bank_.setGains(zero.data(), 0, 0);
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
    /** @brief How much dips (bands under both of their sides) are raised, 0 to 1 (default 0.3). */
    void setFillDips(T amount) noexcept { store(fillDips_, amount, T(0), T(1)); }
    /** @brief How much peaks (bands over both of their sides) are lowered, 0 to 1 (default 0.5). */
    void setTrimPeaks(T amount) noexcept { store(trimPeaks_, amount, T(0), T(1)); }
    /** @brief Tilt of the reference shape in dB per octave around 1 kHz, -6 to +6 (default 0: flat loudness per Bark); it acts through setBalance(). */
    void setTilt(T dbPerOctave) noexcept { store(tilt_, dbPerOctave, T(-6), T(6)); }
    /** @brief Largest correction of a band, 0 to 24 dB (default 6). */
    void setRange(T db) noexcept { store(range_, db, T(0), T(24)); }
    /** @brief Knee of the correction curves, 0.1 to 12 dB (default 3). */
    void setKnee(T db) noexcept { store(knee_, db, T(0.1), T(12)); }
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

    /** @return The latency in samples (the analysis delay plus the bank's). */
    [[nodiscard]] int getLatency() const noexcept { return analysisDelay_ + bank_.getLatency(); }
    /** @return A band's centre in hertz. */
    [[nodiscard]] double getBandCentre(int band) const noexcept { return bands_.getCentre(band); }
    /** @return The correction of a band in dB (a readout for any thread). */
    [[nodiscard]] T getCorrection(int band) const noexcept
    {
        return published_[static_cast<size_t>(std::clamp(band, 0, kBands - 1))].load(std::memory_order_relaxed);
    }
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
    /** @return The bank's response at a frequency in dB (audio thread or after processing). */
    [[nodiscard]] double getResponse(double hz) const noexcept { return bank_.responseDb(hz); }

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
        StateWriter w(stateId("TBAL"), 1);
        w.write("balance", static_cast<float>(getBalance()));
        w.write("fillDips", static_cast<float>(getFillDips()));
        w.write("trimPeaks", static_cast<float>(getTrimPeaks()));
        w.write("tilt", static_cast<float>(getTilt()));
        w.write("range", static_cast<float>(getRange()));
        w.write("knee", static_cast<float>(getKnee()));
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
        if (refDirty_.exchange(false, std::memory_order_acquire))
            bands_.setReferenceSpl(static_cast<double>(referenceSpl_.load(std::memory_order_relaxed)));
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        const int nS = buffer.getNumSamples();
        if (nCh <= 0) return;
        const double fadeStep = 1.0 / (0.05 * fs_);
        const double fadeTo = key != nullptr ? 1.0 : 0.0;
        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, hop_ - hopCounter_);
            // Feed the analysis (programme or key, faded) and delay the programme.
            for (int k = 0; k < chunk; ++k)
            {
                keyFade_ = fadeTo > keyFade_ ? std::min(fadeTo, keyFade_ + fadeStep) : std::max(fadeTo, keyFade_ - fadeStep);
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* data = buffer.getChannel(ch);
                    double a = static_cast<double>(data[i + k]);
                    if (keyFade_ > 0.0 && key != nullptr)
                    {
                        const int kc = std::min(ch, key->getNumChannels() - 1);
                        a += keyFade_ * (static_cast<double>(key->getChannel(kc)[i + k]) - a);
                    }
                    ring_[static_cast<size_t>(ch)][static_cast<size_t>((ringPos_ + k) & (fft_ - 1))] = std::isfinite(a) ? a : 0.0;
                    auto& d = delay_[static_cast<size_t>(ch)];
                    const T x = data[i + k];
                    data[i + k] = d[static_cast<size_t>(delayPos_)];
                    d[static_cast<size_t>(delayPos_)] = std::isfinite(x) ? x : T(0);
                }
                delayPos_ = delayPos_ + 1 == static_cast<int>(delay_[0].size()) ? 0 : delayPos_ + 1;
            }
            ringPos_ = (ringPos_ + chunk) & (fft_ - 1);
            // The bank on the delayed chunk.
            {
                T* ptr[kMaxChannels] {};
                for (int ch = 0; ch < nCh; ++ch) ptr[ch] = buffer.getChannel(ch) + i;
                bank_.processBlock(AudioBufferView<T>(ptr, nCh, chunk));
            }
            hopCounter_ += chunk;
            if (hopCounter_ >= hop_)
            {
                hopCounter_ = 0;
                analyse(nCh);
            }
            i += chunk;
        }
    }

    /** @brief One hop of the analysis, and the new gains of the bank. */
    void analyse(int nCh) noexcept
    {
        for (int ch = 0; ch < nCh; ++ch)
        {
            const auto& r = ring_[static_cast<size_t>(ch)];
            for (int k = 0; k < fft_; ++k)
                frame_[static_cast<size_t>(k)] = r[static_cast<size_t>((ringPos_ + k) & (fft_ - 1))] * window_[static_cast<size_t>(k)];
            transform_->forward(frame_.data(), spectra_[static_cast<size_t>(ch)].data());
        }
        // Band powers of the two channels and of their cross term, each from
        // a spectrum smoothed across frequency.
        const double* s0 = spectra_[0].data();
        const double* s1 = spectra_[static_cast<size_t>(nCh > 1 ? 1 : 0)].data();
        if (timeFocus_.load(std::memory_order_relaxed))
        {
            // Through the focus: each channel's complex spectrum smoothed
            // across frequency (every bin read through a stretch of the window
            // of its own, shorter as frequency rises), then the products.
            const auto focused = [&](auto&& binValue, std::vector<double>& out) {
                for (int k = 0; k < bins_; ++k) binSmooth_[static_cast<size_t>(k)] = powerScale_ * focus_.getGain(k) * binValue(static_cast<size_t>(k));
                bands_.bandPowers(binSmooth_.data(), out.data());
                AuditoryBands<double>::smoothAcross(out.data(), kBands, 0.25);
            };
            const size_t other = nCh > 1 ? 1 : 0;
            focus_.smooth(s0, focusRe_[0].data(), focusIm_[0].data());
            if (nCh > 1) focus_.smooth(s1, focusRe_[1].data(), focusIm_[1].data());
            const auto& ar = focusRe_[0];
            const auto& ai = focusIm_[0];
            const auto& br = focusRe_[other];
            const auto& bi = focusIm_[other];
            focused([&](size_t k) { return ar[k] * ar[k] + ai[k] * ai[k]; }, tmpA_);
            if (nCh > 1)
            {
                focused([&](size_t k) { return br[k] * br[k] + bi[k] * bi[k]; }, tmpB_);
                focused([&](size_t k) { return ar[k] * br[k] + ai[k] * bi[k]; }, tmpC_);
                focused([&](size_t k) { return ai[k] * br[k] - ar[k] * bi[k]; }, tmpD_);
            }
        }
        else
        {
            const auto bandsOf = [&](auto&& binValue, std::vector<double>& out) {
                for (int k = 0; k < bins_; ++k) binPower_[static_cast<size_t>(k)] = binValue(k);
                envelope_.compute(binPower_.data(), binSmooth_.data());
                bands_.bandPowers(binSmooth_.data(), out.data());
                AuditoryBands<double>::smoothAcross(out.data(), kBands, 0.25);
            };
            bandsOf([&](int k) { return powerScale_ * (s0[2 * k] * s0[2 * k] + s0[2 * k + 1] * s0[2 * k + 1]); }, tmpA_);
            if (nCh > 1)
            {
                bandsOf([&](int k) { return powerScale_ * (s1[2 * k] * s1[2 * k] + s1[2 * k + 1] * s1[2 * k + 1]); }, tmpB_);
                bandsOf([&](int k) { return powerScale_ * (s0[2 * k] * s1[2 * k] + s0[2 * k + 1] * s1[2 * k + 1]); }, tmpC_);
                bandsOf([&](int k) { return powerScale_ * (s0[2 * k + 1] * s1[2 * k] - s0[2 * k] * s1[2 * k + 1]); }, tmpD_);
            }
        }
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
                const double whole = loud_[i] + std::pow(th, a);              // (P + th)^a
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
            const double ref = tilt * std::log2(bands_.getCentre(b) / 1000.0);
            tmpC_[i] = trend[i] - ref;
            meanDev += weight[i] * tmpC_[i];
            wsum += weight[i];
        }
        meanDev = wsum > 0.0 ? meanDev / wsum : 0.0;
        // The two sides of each band: what the audible bands below it say,
        // and what those above it say (one-pole averages that leave the band
        // itself out). A dip is a band under both sides and a peak a band
        // over both; where the content falls away at an end of the band,
        // nothing louder lies beyond, so the fall is neither. A side with no
        // audible band says nothing (it reads as the band itself).
        std::array<double, kBands> below {}, above {};
        {
            double num = 0.0, den = 0.0;
            for (int b = 0; b < kBands; ++b)
            {
                const auto i = static_cast<size_t>(b);
                below[i] = den > 1e-9 ? num / den : level[i];
                num = kSidePole * num + audible[i] * level[i];
                den = kSidePole * den + audible[i];
            }
            num = 0.0; den = 0.0;
            for (int b = kBands - 1; b >= 0; --b)
            {
                const auto i = static_cast<size_t>(b);
                above[i] = den > 1e-9 ? num / den : level[i];
                num = kSidePole * num + audible[i] * level[i];
                den = kSidePole * den + audible[i];
            }
        }
        const double balance = static_cast<double>(balance_.load(std::memory_order_relaxed));
        const double fillDips = static_cast<double>(fillDips_.load(std::memory_order_relaxed));
        const double trimPeaks = static_cast<double>(trimPeaks_.load(std::memory_order_relaxed));
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
            const double global = ease * (tmpC_[i] - meanDev);
            const double dip = ease * std::min(0.0, level[i] - std::min(below[i], above[i]));
            const double peak = ease * std::max(0.0, level[i] - std::max(below[i], above[i]));
            // The balance by how present the band is (limited first, so a band
            // half way down to the floor takes half of it), then the local
            // contrast.
            double c = -balance * soft(global);
            if (present[i] < 1.0) c = limited(c) * present[i];
            c = limited(c - trimPeaks * soft(peak) - fillDips * soft(dip)) * weight[i];
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
                if (std::abs(change) < 1e-9 || !(slope > 1e-30 * before)) break;
                if (change > 0.0) over = offset; else under = offset;
                const double next = offset - change * after / slope;
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
        std::array<T, kBands> gains {};
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            corr_[i] = std::clamp(corr_[i] + offset * weight[i] * present[i] / steep[i], -range, range);
            gains[i] = static_cast<T>(corr_[i]);
            published_[i].store(gains[i], std::memory_order_relaxed);
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
        // The bank refines one hop in eight and carries that correction over
        // the others, so its response does not step with a period of eight hops.
        bank_.setGains(gains.data(), hop_, render_ ? 3 : ((hops_++ & 7) == 0 ? 1 : GraphicBank<T, 8>::kCarryRefinement));
    }

    static constexpr double kAudibleSpan = 10.0;  ///< dB over the threshold in quiet from which a band counts in full.
    static constexpr double kMaskedSpan = 10.0;   ///< dB over the floor from which a band counts in full for the balance.
    static constexpr double kSidePole = 0.9;      ///< Pole of the one-sided averages a dip or a peak is read against.
    static constexpr double kOffsetLimit = 60.0;  ///< Largest loudness offset looked for, in dB.
    static constexpr int kOffsetSteps = 8;        ///< Steps of the search for the loudness offset.

    double fs_ = 48000.0, powerScale_ = 1.0, keyFade_ = 0.0;
    int channels_ = 0, fft_ = 2048, hop_ = 128, bins_ = 1025, analysisDelay_ = 0;
    int ringPos_ = 0, delayPos_ = 0, hopCounter_ = 0, hops_ = 0;
    bool prepared_ = false, warm_ = false;

    std::atomic<T> balance_ { T(0.5) }, fillDips_ { T(0.3) }, trimPeaks_ { T(0.5) }, tilt_ { T(0) };
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

    std::unique_ptr<FFTReal<double>> transform_;
    AuditoryBands<double> bands_;
    SpectralEnvelope<double> envelope_;
    SpectralFocus<double> focus_;
    std::array<std::vector<double>, 2> focusRe_, focusIm_;
    std::atomic<bool> timeFocus_ { false };
    GraphicBank<T, 8> bank_;
    std::vector<double> window_, frame_, binPower_, binSmooth_;
    std::vector<std::vector<double>> ring_, spectra_;
    std::vector<std::vector<T>> delay_;
    std::vector<double> ll_, rr_, lr_, li_, loud_, evenLoud_, corr_, tmpA_, tmpB_, tmpC_, tmpD_;
    std::array<double, kBands> timeCoeff_ {}, briefCoeff_ {};
};

} // namespace dspark
