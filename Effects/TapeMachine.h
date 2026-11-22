// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TapeMachine.h
 * @brief Physical tape machine: JA hysteresis, EQ standards, losses, transport.
 *
 * A complete record/playback chain built around the Jiles-Atherton
 * hysteresis core (Core/Hysteresis.h), modelling the five things that make
 * tape sound like tape:
 *
 * 1. **Record/playback equalization** (NAB or CCIR, speed-dependent time
 *    constants): the signal is HF-emphasized (and LF-boosted for NAB) before
 *    hitting the "tape", and de-emphasized on playback with the exact
 *    digital inverse. The round trip is flat by construction - but the
 *    hysteresis sees the emphasized signal, so highs saturate first at slow
 *    speeds, exactly as on hardware.
 * 2. **Magnetic hysteresis with REAL AC bias** at 4x oversampling by DEFAULT
 *    (configurable, and switchable off, see setOversampling): an
 *    ultrasonic carrier (a quarter of the internal rate, exact table) is
 *    summed with the signal into a push-pull pair of JA instances per
 *    channel (+carrier / -carrier, output averaged, like a centre-tapped
 *    record head). The carrier erases the loop's branch memory exactly as
 *    hardware bias does, the push-pull cancels the carrier, its odd
 *    harmonics and the remanence DC, and the even folds die in the
 *    downsampler. `bias` maps to carrier amplitude: under-bias drops below
 *    the erase threshold (real grit and level instability) and the top end
 *    rises, over-bias adds the self-erasure roll-off of short wavelengths
 *    from the optimum on. Gain is calibrated
 *    per setting through the same biased chain, so drive changes
 *    saturation, not level.
 * 3. **Playback loss effects** with the physical formulas (Kadis): spacing
 *    loss 54.6*d/lambda dB, gap loss sinc(pi*g/lambda), thickness loss
 *    (1-e^(-4*pi*delta/lambda))/(4*pi*delta/lambda), with lambda =
 *    speed/frequency - rendered into a linear-phase FIR per speed (63 taps
 *    at 48 kHz, scaled with the rate so the response is the same at every
 *    rate), plus the head-bump resonance, at the speed over the length of
 *    the head's contact with the tape (22.5, 45, 90 and 180 Hz at 3.75, 7.5,
 *    15 and 30 ips with the half-track head).
 * 4. **Wow & flutter**: one shared transport modulation (slow drift random
 *    walk + 0.55 Hz wow + 8.3/23 Hz flutter + scrape band) driving a
 *    fractional delay - identical on all channels, like a real capstan. The
 *    hiss generator uses its own RNG stream, so enabling noise never changes
 *    the transport's realisation.
 * 5. **Tape hiss** (optional, default off).
 * 6. **Physical band edges**: the line input is AC coupled (two one-pole
 *    high-passes at 4 Hz), so DC and the slowest drift never lean the
 *    tape's loop to one side; a 2nd-order 24 Hz high-pass models the
 *    AC-coupled playback amplifier (kills subsonics and any residual DC,
 *    like the hardware's LF roll-off), and the loss FIR always applies the
 *    hard reproduce-gap cutoff above ~21.5 kHz (no real head reads shorter
 *    wavelengths; it also removes the last even-order bias intermodulation
 *    sidebands near the base-rate Nyquist).
 *
 * The machine around the tape, all off or at the classic setting by
 * default (each one is described at its setter):
 * - setTransport(Transport::Speed): the tape speed as a state with inertia,
 *   its deviation from the capstan and the reels at rates that follow it,
 *   and the delay as the head distance over the speed.
 * - setNoise() puts the hiss on the tape, before the reproduce head and its
 *   de-emphasis, so it has the colour of the speed and the standard;
 *   setHum() adds mains hum after the tape; setChannelBleed() leaks each channel
 *   of a pair into the other, rising with frequency.
 * - setEvenHarmonics() adds an even term before the tape; setOutputIron() adds
 *   an output transformer as a parallel branch (low frequencies saturate it).
 * - setHeadModel(HeadModel::Contour): the reproduce head's low end as a
 *   chain of resonances; setElectronics(Electronics::Circuit): the record and
 *   reproduce networks from their analog transfer functions at the internal
 *   rate; setAlignment(): the reproduce HF and LF trims.
 * - setFormulation(), setTrackFormat(), setCalibrationLevel() and
 *   setMonitor(): the tape, the track width, the operating level and the
 *   monitored head, as on a machine being aligned.
 *
 * The dry path of the mix control is delay-compensated to getLatency() and,
 * between the two ends of the mix, takes the phase (not the magnitude) of
 * the machine's high-passes, so a partial mix keeps its bottom. The
 * mix ramps at no more than full scale per 20 ms, like the other saturators:
 * the wet stream is aligned to the dry, but at high drive the timbral
 * difference a step would jump across is not small.
 * Channels beyond the prepared count pass through untouched.
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored; parameter changes are
 * published with a release store and consumed at the next block). prepare()
 * is setup-thread only (allocates; invalid specs are ignored and an
 * unprepared instance passes audio through). reset() belongs to the stream
 * owner. getState()/setState() are setup/UI threads. A parameter change
 * re-calibrates the drive makeup on a scratch push-pull pair, spread over
 * the following blocks (one internal sample of calibration per internal
 * sample of audio, about 24 ms of audio, allocation-free) instead of running
 * it all inside one callback. The calibrated (drive, makeup, bias) set is
 * adopted when its calibration completes and reached through a ~30 ms
 * geometric ramp, so automation neither overruns the callback nor clicks.
 * Blocks longer than the prepared maximum are processed in pieces.
 *
 * Cost: the push-pull biased core runs two JA solvers at the active
 * oversampling factor (4x default), ~11% of one core at 48 kHz stereo on a
 * desktop CPU at 4x - the price of physical AC bias (history-independent LF
 * response, exact even-harmonic cancellation, ultrasonic residue below
 * -75 dB). A SIMD lane-parallel JA (2 channels x 2 polarities) is the noted
 * future optimisation if a lighter budget is ever needed.
 *
 * **Oversampling transparency.** setOversampling(int) (setup thread only;
 * it reallocates and re-calibrates like prepare()) selects the internal
 * oversampling factor in {1,2,4,8,16}; the DEFAULT is 4. The AC-bias
 * carrier is always a quarter of the internal rate, so the factor sets how
 * far the carrier sits above the audio band: at 4x/48k it is ultrasonic
 * (48 kHz) and its folds die in the downsampler; CPU scales ~linearly with
 * the factor. factor = 1 means OFF - no internal resampling and zero added
 * oversampler latency, but then the carrier lands at a quarter of the base
 * rate (12 kHz at 48k), i.e. IN-BAND, so 1x is only sensible when the host
 * runs a high sample rate or the surrounding chain is already oversampled
 * (the AC-bias model needs ultrasonic headroom). getLatency() always
 * reflects the active factor (oversampler group delay + loss FIR + transport
 * delay), so hosts get correct PDC.
 *
 * **The carrier and the gain.** Two sign levels are a dither with two
 * peaks, and the gain of such a loop rises with the level before it
 * saturates: 2.8 dB at 1 kHz and 3.2 dB at 10 kHz at the nominal bias, up to
 * 9.9 dB at full bias. setBiasShape(BiasShape::FourLevel) takes that away
 * at twice the internal rate and about twice the cost; see it.
 *
 * Dependencies: Core/Hysteresis.h, Core/Oversampling.h, Core/Biquad.h,
 * Core/DryWetMixer.h,
 * Core/FilterDesign.h, Core/NormalizedLadder.h, Core/TapeTransport.h, Core/SimdOps.h,
 * Core/AudioSpec.h, Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h,
 * Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/FilterDesign.h"
#include "../Core/Hysteresis.h"
#include "../Core/NormalizedLadder.h"
#include "../Core/Oversampling.h"
#include "../Core/SimdOps.h"
#include "../Core/StateBlob.h"
#include "../Core/TapeTransport.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class TapeMachine
 * @brief Reel-to-reel tape emulation with physical hysteresis and transport.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TapeMachine
{
public:
    /** @brief Playback equalization standard. */
    enum class Standard { NAB, CCIR };

    /** @brief Tape speed (3.75 ips comes last, so stored states keep their meaning). */
    enum class Speed { IPS_7_5, IPS_15, IPS_30, IPS_3_75 };

    /**
     * @brief How the transport moves the tape (setTransport()).
     *
     * Modulated, the default, reads the tape through a delay moved by a sum
     * of drift, wow, flutter and scrape. Speed keeps the speed of the tape
     * as a state: the reproduce pitch is the speed of the moment, whose
     * deviation is a smoothed noise, the capstan's rotation (with a third of
     * its second harmonic) and the reel's rotation, at rates that follow the
     * speed (5 Hz and 0.6 Hz at 15 ips) and deeper at slower speeds (by the
     * square root of the ratio); the delay is held at the head distance over
     * the speed, and a change of setSpeed() glides the speed with inertia
     * (0.45 s to speed up, 0.7 s to slow down) instead of jumping.
     */
    enum class Transport { Modulated, Speed };

    /** @brief Mains frequency of the hum (setHum()). */
    enum class Mains { Hz50, Hz60 };

    /**
     * @brief Tape formulation (setFormulation()).
     *
     * Classic is the default material. HighOutput saturates 3 dB later,
     * takes 20 % more coercive force from the bias and hisses 1.5 dB less;
     * Vintage saturates 2 dB earlier with a more reversible loop, takes 20 %
     * less coercive force and hisses 3 dB more. The bias control keeps its
     * meaning: 0.5 is the optimum of the material in use.
     */
    enum class Formulation { Classic, HighOutput, Vintage };

    /**
     * @brief Track format (setTrackFormat()): the signal grows with the
     *        track width and the hiss with its square root, so Half
     *        (2.0 mm, the default) keeps the hiss as set, Full (6.3 mm)
     *        lowers it by 5.0 dB and Quarter (1.0 mm) raises it by 3.0 dB.
     *        Each format has its head, and the head bump follows the length
     *        of its contact with the tape (the speed over that length): at
     *        15 ips it sits at 90 Hz with Half, 72 Hz with Full and 112 Hz
     *        with Quarter.
     */
    enum class TrackFormat { Half, Full, Quarter };

    /**
     * @brief What the output monitors (setMonitor()). PlayHead, the default, is
     *        the reproduce head; RecordHead reads through the record head, whose
     *        wider gap (6 um against 3) and spacing lose more of the top end;
     *        Input is the input itself, delayed like the tape path. Changes
     *        fade over 20 ms.
     */
    enum class Monitor { PlayHead, RecordHead, Source };

    /**
     * @brief The reproduce head's low end (setHeadModel()). Bump, the
     *        default, is the single head-bump resonance; Contour is a chain
     *        of four resonances, the bump and three ripples above it at 1.45,
     *        1.95 and 2.5 times its frequency with alternating signs and
     *        falling size, each designed on its own analog prototype (in
     *        place at any rate).
     */
    enum class HeadModel { Bump, Contour };

    /**
     * @brief Record and reproduce electronics (setElectronics()).
     *
     * Ideal, the default, runs the record emphasis at the session rate and
     * its inverse after the tape. Circuit runs both networks from their
     * analog transfer functions at the internal rate, around the tape, as
     * normalized ladders: the record network (the emphasis and a
     * second-order record amplifier limit at 45 kHz) and the reproduce
     * network (the de-emphasis with the trims of setAlignment() and a
     * first-order amplifier pole at 50 kHz; with the 24 Hz output coupling, a
     * fifth-order reproduce chain). The round trip then carries the
     * amplifiers' gentle top-end roll-off. The networks it brings in start
     * from rest: a setup choice.
     */
    enum class Electronics { Ideal, Circuit };

    // -- Lifecycle ---------------------------------------------------------------

    /** @brief Allocates the whole chain. Invalid specs (non-positive or
     *  non-finite rate, block size or channel count) are ignored: the previous
     *  state is kept and an unprepared instance stays pass-through.
     *
     *  A prepared machine asked for another block size alone (same rate and
     *  channels) keeps its memory: the tape, the filters and what is still
     *  inside its delays go on as if nothing had been asked, since
     *  processBlock() splits any block to the size it was built for. Any
     *  other call starts from rest. */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        if (prepared_.load(std::memory_order_relaxed) && spec.sampleRate == spec_.sampleRate
            && spec.numChannels == spec_.numChannels && spec.maxBlockSize != spec_.maxBlockSize)
        {
            spec_.maxBlockSize = spec.maxBlockSize;
            return;
        }
        prepared_.store(false, std::memory_order_relaxed);
        spec_ = spec;
        sampleRate_ = spec.sampleRate;
        mixMaxStep_ = static_cast<T>(1.0 / std::max(1.0, sampleRate_ * 0.02));
        numChannels_ = spec.numChannels;
        maxBlock_ = std::max(spec.maxBlockSize, 1);
        if (osAutomatic_)
            osRequested_ = Oversampling<T>::factorForRate(spec.sampleRate, kAutomaticInternalRate);
        osFactor_ = factorInUse(spec.sampleRate);

        // The hysteresis core runs at the active oversampling factor (4x
        // default) so the quarter-internal-rate AC bias carrier and its
        // sidebands stay clear of the audio band. factor = 1 = OFF: no
        // resampling and zero added latency; the carrier is then in-band.
        const double osRate = sampleRate_ * static_cast<double>(osFactor_);
        if (osFactor_ > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(
                osFactor_, Oversampling<T>::Quality::High,
                osMode_ == OversamplingMode::MinimumPhase ? Oversampling<T>::Mode::PolyphaseIir
                                                          : Oversampling<T>::Mode::LinearPhaseFir);
            oversampler_->prepare(spec);
        }
        else
        {
            oversampler_.reset();
        }

        // Push-pull pair per channel: +carrier and -carrier instances, output
        // averaged. Odd-in-bias terms (the carrier, its 3rd harmonic fold and
        // the loop's remanence DC) cancel exactly, as in the centre-tapped
        // record-head circuits of real machines.
        hysteresis_.assign(static_cast<size_t>(numChannels_), {});
        hysteresisN_.assign(static_cast<size_t>(numChannels_), {});
        for (auto& h : hysteresis_)
        {
            h.prepare(osRate);
            h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
        }
        for (auto& h : hysteresisN_)
        {
            h.prepare(osRate);
            h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
        }

        recordHF_.assign(static_cast<size_t>(numChannels_), {});
        recordLF_.assign(static_cast<size_t>(numChannels_), {});
        playHF_.assign(static_cast<size_t>(numChannels_), {});
        playLF_.assign(static_cast<size_t>(numChannels_), {});
        headBump_.assign(static_cast<size_t>(numChannels_), {});
        overBiasLp_.assign(static_cast<size_t>(numChannels_), 0.0);
        inputAc_.assign(static_cast<size_t>(numChannels_), {});
        inputAcPole_ = std::exp(-2.0 * std::numbers::pi * kInputCouplingHz / sampleRate_);
        recTrim_.assign(static_cast<size_t>(numChannels_), {});
        playTrim_.assign(static_cast<size_t>(numChannels_), {});
        ironLp_.assign(static_cast<size_t>(numChannels_), 0.0);
        ironA_ = 1.0 - std::exp(-2.0 * std::numbers::pi * kIronCornerHz / sampleRate_);
        resting_ = false;
        wakeHold_ = 0;
        const size_t channels = static_cast<size_t>(numChannels_);
        hissHF_.assign(channels, {});
        hissLF_.assign(channels, {});
        xtHp_.assign(channels, {});
        trPre_.assign(channels, {});
        trPost_.assign(channels, {});
        trMean_.assign(channels, 0.0);
        asymMean_.assign(channels, 0.0);
        wet_.assign(channels, 0.0);
        headContour_.assign(channels, {});
        recNet_.assign(channels, {});
        playNet_.assign(channels, {});
        {
            // Channel bleed: a first-order high-pass at 10 kHz (its corner kept
            // in place), scaled to unity at 1 kHz. Transformer branch: a
            // one-pole low-pass at 60 Hz before the curve (flux follows
            // voltage over frequency) and its inverse, capped at 4 kHz, after.
            const double wc = 2.0 * sampleRate_ * std::tan(std::numbers::pi * std::min(10000.0, 0.45 * sampleRate_) / sampleRate_);
            const ShelfSection hp = firstOrder(1.0, 0.0, 1.0, wc, sampleRate_);
            for (auto& s : xtHp_) s = hp;
            xtNorm_ = 1.0 / std::max(1e-9, shelfMagnitude(hp, 2.0 * std::numbers::pi * 1000.0 / sampleRate_));
            const double w1 = 2.0 * std::numbers::pi * kTransformerCornerHz;
            const double w2 = 2.0 * std::numbers::pi * std::min(4000.0, 0.45 * sampleRate_);
            const ShelfSection pre = firstOrder(0.0, w1, 1.0, w1, sampleRate_);
            const ShelfSection post = firstOrder(1.0 / w1, 1.0, 1.0 / w2, 1.0, sampleRate_);
            for (auto& s : trPre_) s = pre;
            for (auto& s : trPost_) s = post;
        }

        // AC-coupled playback amplifier: fixed 2nd-order 24 Hz high-pass
        // (real machines roll off there; also blocks remanence DC).
        outHp_.assign(static_cast<size_t>(numChannels_), {});
        {
            const auto hc = BiquadCoeffs::makeHighPass(sampleRate_, kOutputCouplingHz, 0.707);
            for (auto& h : outHp_)
                h = PeakSection { hc.b0, hc.b1, hc.b2, hc.a1, hc.a2, 0.0, 0.0 };
        }

        // Loss FIR: 63 taps at 48 kHz, scaled with the rate so the design
        // grid keeps the same frequency spacing (a fixed 63 taps at 192 kHz
        // smeared the reproduce-gap cut: -3.67 dB at 18 kHz instead of -1.57).
        firLen_ = std::max(kFirLenAt48k,
                           64 * static_cast<int>(std::lround(sampleRate_ / 48000.0)) - 1);
        firCenter_ = (firLen_ - 1) / 2;
        firHalf_ = 1;
        while (firHalf_ < firLen_ + 1) firHalf_ <<= 1;
        firState_.assign(static_cast<size_t>(numChannels_),
                         std::vector<T>(static_cast<size_t>(2 * firHalf_), T(0)));
        firPos_ = 0;
        firTaps_.assign(static_cast<size_t>(firLen_), T(0));
        {
            const int grid = firLen_ + 1;
            cosTable_.resize(static_cast<size_t>(grid));
            for (int m = 0; m < grid; ++m)
                cosTable_[static_cast<size_t>(m)] =
                    std::cos(2.0 * std::numbers::pi * m / static_cast<double>(grid));
            designMags_.assign(static_cast<size_t>(grid / 2 + 1), 0.0);
            designTaps_.assign(static_cast<size_t>(firLen_), 0.0);
        }
        overBiasX1_.assign(static_cast<size_t>(numChannels_), 0.0);

        // Transport delay scales with the rate so the wow excursion (sized in
        // samples at 48k, scaled by fs/48k) always fits under the centre.
        delayCenter_ = std::max(96, static_cast<int>(96.0 * sampleRate_ / 48000.0));
        int dsz = 1;
        // Room for the speed transport too: its delay is the head distance
        // over the speed, from a quarter to four times the centre.
        while (dsz < 5 * delayCenter_ + 16) dsz <<= 1;
        delayMask_ = dsz - 1;
        delayRing_.assign(static_cast<size_t>(numChannels_),
                          std::vector<T>(static_cast<size_t>(dsz), T(0)));
        delayPos_ = 0;

        // Per-sample modulation smoothing constants (precomputed once).
        const double dt = 1.0 / sampleRate_;
        driftA_ = std::exp(-2.0 * std::numbers::pi * 0.10 * dt);
        scrapeA1_ = std::exp(-2.0 * std::numbers::pi * 90.0 * dt);
        scrapeA2_ = std::exp(-2.0 * std::numbers::pi * 40.0 * dt);
        // The speed transport: the head distance is the delay centre at the
        // nominal speed.
        tapeTransport_.prepare(sampleRate_, static_cast<double>(delayCenter_), speedIps_ > 0.0 ? speedIps_ : 15.0);
        tapeTransport_.setLimits(3.0, static_cast<double>(delayMask_ - 4));
        asymMeanA_ = std::exp(-2.0 * std::numbers::pi * 10.0 * dt / static_cast<double>(osFactor_));
        trMeanA_ = std::exp(-2.0 * std::numbers::pi * 5.0 * dt);

        latency_ = (oversampler_ ? oversampler_->getLatency() : 0) + firCenter_ + delayCenter_;
        // Behind minimum-phase filters the dry copy crosses the same filters
        // (in phase at every frequency) and the ring adds the rest of the delay.
        dryThroughFilters_ = oversampler_ && osMode_ == OversamplingMode::MinimumPhase;
        if (dryThroughFilters_) dryPath_.prepare(spec, *oversampler_);
        dryScratch_.resize(numChannels_, maxBlock_);
        dryDelay_ = latency_ - (dryThroughFilters_ ? oversampler_->getLatency() : 0);
        drySize_ = 1;
        while (drySize_ < latency_ + maxBlock_ + 1) drySize_ <<= 1;
        dryRing_.assign(static_cast<size_t>(numChannels_),
                        std::vector<T>(static_cast<size_t>(drySize_), T(0)));
        dryPos_ = 0;
        for (auto* lead : { &dryInputLead_, &dryOutputLead_, &dryIronLead_ }) lead->prepare(sampleRate_, numChannels_);
        dryInputLead_.setCorner(kInputCouplingHz);
        dryOutputLead_.setCorner(detail::DryPhase::kButterworthShare * kOutputCouplingHz);

        // The first calibration runs here, on the setup thread; later
        // parameter changes calibrate incrementally inside processBlock().
        dirty_.store(false, std::memory_order_relaxed);
        formApplied_ = -1;
        circuitActive_ = false;
        speedTransportActive_ = false;
        speedIps_ = 0.0;
        recomputeLinear();
        calActive_ = false;
        calRestart_ = false;
        startCalibration();
        stepCalibration(calN_);
        hSm_ = hScale_;
        mSm_ = mScale_;
        bSm_ = biasAmp_;
        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears all signal state (keeps parameters). RT-safe. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        clearMachine();
        for (auto& d : dryRing_) std::fill(d.begin(), d.end(), T(0));
        dryPath_.reset();
        resting_ = false;
        wakeHold_ = 0;
        tapeTransport_.reset();
        lastMod_ = 0.0;
        humPhase_ = 0.0;
        aimRamps(true);
        firPos_ = 0;
        delayPos_ = 0;
        dryPos_ = 0;
        currentMix_ = mix_.load(std::memory_order_relaxed); // no fade-in on start
        aimDryPhase(currentMix_);
        for (auto* lead : { &dryInputLead_, &dryOutputLead_, &dryIronLead_ })
        {
            lead->reset();
            lead->land();
        }
        biasPhase_ = 0;
        modPhaseWow_ = 0.0;
        modPhaseFlut_ = 0.0;
        modPhaseFlut2_ = 0.0;
        driftState_ = 0.0;
        scrapeLp1_ = scrapeLp2_ = 0.0;
        rng_ = 0x1357feedu;
        rngNoise_ = 0x2468beefu;
        hSm_ = hScale_;   // no fade-in on start; a running calibration continues
        mSm_ = mScale_;
        bSm_ = biasAmp_;
        if (oversampler_) oversampler_->reset();
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Input drive in dB [-12, +24]. Level-compensated: more drive means
     *  more saturation at roughly constant loudness. Non-finite values are ignored. */
    void setDrive(T driveDb) noexcept
    {
        if (!std::isfinite(driveDb)) return;
        driveDb_.store(std::clamp(driveDb, T(-12), T(24)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Bias setting [0, 1]; 0.5 is nominal calibration (carrier at
     *  3x the JA field constant: full branch-memory erasure, clean odd
     *  saturation). Below nominal the carrier drops under the erase
     *  threshold: growing distortion and the level instability of a real
     *  under-biased machine, and a top end that rises as the bias falls (the
     *  short wavelengths record more strongly than the long ones: a shelf of
     *  up to 4.5 dB, around 6 kHz at 15 ips and in proportion to the speed).
     *  Above nominal, self-erasure progressively rolls off the top octave,
     *  from the optimum on and without a step. Non-finite values are ignored. */
    void setBias(T bias) noexcept
    {
        if (!std::isfinite(bias)) return;
        bias_.store(std::clamp(bias, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Configures internal oversampling of the biased hysteresis core
     *  (visible, tunable and switchable off). SETUP THREAD ONLY - it reallocates
     *  the polyphase filters and re-prepares/re-calibrates the whole chain
     *  exactly like prepare(); never call it concurrently with processBlock().
     *
     * @param factor Power-of-two multiplier in {1,2,4,8,16}. DEFAULT 4. 1 = OFF
     *  (no internal resampling, zero added oversampler latency; the AC-bias
     *  carrier then lands in-band at a quarter of the base rate, so 1x is only
     *  sensible under host/chain oversampling). CPU scales ~linearly with the
     *  factor. Invalid/non-power-of-two values are ignored. getLatency()
     *  reflects the new factor after this call.
     */
    void setOversampling(int factor)
    {
        if (factor < 1 || factor > 16 || (factor & (factor - 1)) != 0) return;
        if (factor == osRequested_ && !osAutomatic_) return;
        osAutomatic_ = false;
        osRequested_ = factor;
        osFactor_ = factorInUse(sampleRate_);
        if (prepared_.load(std::memory_order_relaxed))
            prepare(spec_);
    }

    /**
     * @brief Lets the host rate choose the factor (SETUP THREAD ONLY, as
     *  setOversampling()).
     *
     * With it on, every prepare() takes the smallest factor that brings the
     * core to an internal rate of at least 176.4 kHz: 4x at 44.1 and 48 kHz
     * (the default factor), 2x at 88.2 and 96 kHz, none from 176.4 kHz up. The
     * bias carrier and the hysteresis then run at the same rate whatever the
     * session rate, and a high-rate session stops paying for a factor it does
     * not need. Off by default; an explicit setOversampling() turns it off.
     *
     * @param on True to choose the factor from the rate.
     */
    void setAutomaticOversampling(bool on)
    {
        if (on == osAutomatic_) return;
        osAutomatic_ = on;
        if (on && prepared_.load(std::memory_order_relaxed))
            prepare(spec_);
    }

    /** @brief Whether the factor follows the host rate (setAutomaticOversampling()). */
    [[nodiscard]] bool isOversamplingAutomatic() const noexcept { return osAutomatic_; }

    /** @brief The levels of the bias carrier (see setBiasShape()). */
    enum class BiasShape
    {
        TwoLevel, ///< Every sample at plus or minus the amplitude (default).
        FourLevel ///< An inner and an outer level of each sign: the gain only falls with the level.
    };

    /**
     * @brief Chooses the levels of the bias carrier (SETUP THREAD ONLY, as
     *  setOversampling(): an active instance is prepared again).
     *
     * TwoLevel, the default, is the sound the machine has had: its gain
     * rises with the level before it saturates (3.2 dB at the nominal bias,
     * up to 9.9 dB at full bias).
     *
     * With FourLevel the gain holds and then falls, at any bias, as a tape
     * compresses. It is another sound, quieter when driven: 1 kHz at 0 dBFS
     * leaves 1.9 dB lower. The four levels need twice the headroom over the
     * audio band, so the core then runs at no less than 352.8 kHz whatever
     * setOversampling() asked: 8x at 44.1 and 48 kHz, about twice the cost
     * and a latency of its own (getOversamplingFactor(), getLatency()).
     *
     * @param shape TwoLevel (default) or FourLevel.
     */
    void setBiasShape(BiasShape shape)
    {
        const bool four = shape == BiasShape::FourLevel;
        if (four == fourLevel_) return;
        fourLevel_ = four;
        osFactor_ = factorInUse(sampleRate_);
        if (prepared_.load(std::memory_order_relaxed))
            prepare(spec_);
    }

    /** @brief The levels of the bias carrier (setBiasShape()). */
    [[nodiscard]] BiasShape getBiasShape() const noexcept { return fourLevel_ ? BiasShape::FourLevel : BiasShape::TwoLevel; }

    /** @brief How the oversampler filters (see setOversamplingMode()). */
    enum class OversamplingMode
    {
        LinearPhase,  ///< Linear-phase half-bands: the latency of the factor, the dry path a delay (default).
        MinimumPhase  ///< Elliptic allpass half-bands: a few samples of latency, the dry path through the same filters.
    };

    /**
     * @brief Chooses the oversampler's filters (SETUP THREAD ONLY, as
     *  setOversampling(): an active instance is prepared again).
     *
     * MinimumPhase runs polyphase allpass half-bands: the oversampler's
     * share of the latency falls to a few samples, and the dry copy of the
     * mix crosses the same filters, so a partial mix adds in phase.
     * getLatency() reflects the change.
     *
     * @param mode LinearPhase (default) or MinimumPhase.
     */
    void setOversamplingMode(OversamplingMode mode)
    {
        mode = static_cast<OversamplingMode>(std::clamp(static_cast<int>(mode), 0, 1));
        if (mode == osMode_) return;
        osMode_ = mode;
        if (prepared_.load(std::memory_order_relaxed))
            prepare(spec_);
    }

    /** @brief The oversampler's filters (setOversamplingMode()). */
    [[nodiscard]] OversamplingMode getOversamplingMode() const noexcept { return osMode_; }

    /** @brief Active oversampling factor (1 = off, 4 = default; see setBiasShape()). */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return osFactor_; }

    /** @brief Tape speed (changes EQ time constants, losses and head bump).
     *  Out-of-range values are clamped. */
    void setSpeed(Speed s) noexcept
    {
        const int v = std::clamp(static_cast<int>(s),
                                 static_cast<int>(Speed::IPS_7_5),
                                 static_cast<int>(Speed::IPS_3_75));
        speed_.store(v, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Equalization standard (NAB adds the LF time constant).
     *  Out-of-range values are clamped. */
    void setStandard(Standard s) noexcept
    {
        const int v = std::clamp(static_cast<int>(s),
                                 static_cast<int>(Standard::NAB),
                                 static_cast<int>(Standard::CCIR));
        standard_.store(v, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Playback loss intensity [0, 1] (0 bypasses the loss FIR).
     *  Non-finite values are ignored. */
    void setLossEffects(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        loss_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Head-bump resonance intensity [0, 1] (~2.5 dB at full).
     *  Non-finite values are ignored. */
    void setHeadBump(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        headBumpAmt_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Wow & flutter depth [0, 1] (~0.25% peak pitch deviation at 1).
     *  Non-finite values are ignored. */
    void setWowFlutter(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        wowFlutter_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Tape hiss RMS level in dBFS at the output (e.g. -55 for
     *  audible vintage hiss); values <= -120 disable it (default). The hiss
     *  is laid on the tape, before the reproduce head and the de-emphasis,
     *  so its colour follows the speed and the standard; the level is that
     *  of the output for the half-track format, the classic formulation and
     *  no calibration offset (see setTrackFormat(), setFormulation() and
     *  setCalibrationLevel()), and glides over 50 ms. Non-finite values are
     *  ignored. */
    void setNoise(T dbfs) noexcept
    {
        if (!std::isfinite(dbfs)) return;
        noiseDb_.store(std::clamp(dbfs, T(-200), T(-20)), std::memory_order_relaxed);
    }

    /** @brief Dry/wet mix [0, 1]; dry is latency-compensated. Non-finite
     *  values are ignored. */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Transport model (see Transport). Out-of-range values are clamped. */
    void setTransport(Transport t) noexcept
    {
        transport_.store(std::clamp(static_cast<int>(t), 0, 1), std::memory_order_relaxed);
    }

    /**
     * @brief Mains hum after the tape: the mains frequency and its second
     *        harmonic 6 dB down, in phase on every channel, its level gliding
     *        over 50 ms. Non-finite levels are ignored.
     * @param dbfs  RMS level in dBFS; -120 and below turn it off (the default).
     * @param mains Mains frequency (50 Hz by default).
     */
    void setHum(T dbfs, Mains mains = Mains::Hz50) noexcept
    {
        if (!std::isfinite(dbfs)) return;
        humDb_.store(std::clamp(dbfs, T(-200), T(-20)), std::memory_order_relaxed);
        mains_.store(std::clamp(static_cast<int>(mains), 0, 1), std::memory_order_relaxed);
    }

    /**
     * @brief Bleed between the channels of each pair (0 and 1, 2 and 3,
     *        and so on): each takes the other through a first-order
     *        high-pass at 10 kHz (rising 6 dB per octave below it, as between
     *        the tracks of a head), its level gliding over 50 ms.
     *        Non-finite values are ignored.
     * @param dB Level at 1 kHz; -120 and below turn it off (the default).
     */
    void setChannelBleed(T dB) noexcept
    {
        if (std::isfinite(dB)) channelBleedDb_.store(std::clamp(dB, T(-200), T(-20)), std::memory_order_relaxed);
    }

    /**
     * @brief Even harmonics of the record path, 0 to 1 (0, the default, is
     *        off): a share of the square of the recorded signal (less its
     *        mean) is added before the tape, so the second harmonic grows
     *        with the level; it glides over 20 ms. Non-finite values are
     *        ignored.
     */
    void setEvenHarmonics(T amount) noexcept
    {
        if (std::isfinite(amount)) evenHarmonics_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Output iron, 0 to 1 (0, the default, is off): a parallel
     *        branch that takes the reproduced signal through a one-pole
     *        low-pass at 60 Hz (the flux of a transformer follows the voltage
     *        over the frequency), a curve with an even and an odd term, and
     *        the inverse filter capped at 4 kHz, and adds what the curve
     *        changed: the low end saturates first. The iron also loses its
     *        low end in a linear way: by the same share the output crosses a
     *        first-order high-pass at 20 Hz (3 dB down there at the full
     *        amount, under 0.3 dB at 100 Hz), which takes with it whatever DC
     *        the even term leaves. The branch always runs and its share
     *        glides over 20 ms. Non-finite values are ignored.
     */
    void setOutputIron(T amount) noexcept
    {
        if (std::isfinite(amount)) outputIron_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Reproduce head model (see HeadModel). Out-of-range values are clamped. */
    void setHeadModel(HeadModel m) noexcept
    {
        headModel_.store(std::clamp(static_cast<int>(m), 0, 1), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Record and reproduce electronics (see Electronics). Out-of-range values are clamped. */
    void setElectronics(Electronics e) noexcept
    {
        electronics_.store(std::clamp(static_cast<int>(e), 0, 1), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Reproduce alignment trims, as the HF and LF reproduce
     *        adjustments of a machine (both 0 by default). Non-finite values
     *        are ignored.
     * @param hfDb Trim of the de-emphasis at high frequencies, -6 to +6 dB.
     * @param lfDb Trim of the low end (the NAB LF network, or a shelf at
     *             50 Hz with CCIR), -6 to +6 dB.
     */
    void setAlignment(T hfDb, T lfDb) noexcept
    {
        if (!std::isfinite(hfDb) || !std::isfinite(lfDb)) return;
        alignHf_.store(std::clamp(hfDb, T(-6), T(6)), std::memory_order_relaxed);
        alignLf_.store(std::clamp(lfDb, T(-6), T(6)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Trim of the record emphasis, -6 to +6 dB (0 by default): a high
     *        shelf around 4 kHz on the record side and its exact inverse on
     *        the reproduce side, as a record equaliser aligned against its
     *        reproduce twin.
     *
     * The round trip stays flat at low level. What changes is how hard the
     * top end reaches the tape: with more emphasis a loud high tone distorts
     * more and the tape's hiss up there falls by as much (5.4 dB between 8
     * and 16 kHz at +6 dB); with less, the top end stays cleaner and the
     * hiss rises. Non-finite values are ignored.
     */
    void setRecordEmphasis(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        recordEmphasis_.store(std::clamp(dB, T(-6), T(6)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }
    /** @brief The trim of the record emphasis in dB (setRecordEmphasis()). */
    [[nodiscard]] T getRecordEmphasis() const noexcept { return recordEmphasis_.load(std::memory_order_relaxed); }

    /** @brief Tape formulation (see Formulation). Out-of-range values are clamped. */
    void setFormulation(Formulation f) noexcept
    {
        formulation_.store(std::clamp(static_cast<int>(f), 0, 2), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Track format (see TrackFormat). Out-of-range values are clamped. */
    void setTrackFormat(TrackFormat f) noexcept
    {
        trackFormat_.store(std::clamp(static_cast<int>(f), 0, 2), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Operating level above the reference flux, 0 to 9 dB (0 by
     *        default): the tape is driven that much harder for the same input
     *        while the output level is kept, and the hiss falls by as much
     *        against the signal (a machine aligned to an elevated level). A
     *        1 kHz tone at -12 dBFS carries 0.16 % THD at 0 dB, 0.31 % at
     *        3 dB and 0.60 % at 6 dB: at 6 dB the machine distorts at the
     *        reference level as tape machines are specified to (0.4 to 0.8 %),
     *        the distortion rising about fourfold per 6 dB of level (third
     *        order). Non-finite values are ignored.
     */
    void setCalibrationLevel(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        calLevel_.store(std::clamp(dB, T(0), T(9)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Monitored head or input (see Monitor). Out-of-range values are clamped. */
    void setMonitor(Monitor m) noexcept
    {
        monitor_.store(std::clamp(static_cast<int>(m), 0, 2), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    [[nodiscard]] Transport getTransport() const noexcept { return static_cast<Transport>(transport_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getHum() const noexcept { return humDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mains getMains() const noexcept { return static_cast<Mains>(mains_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getChannelBleed() const noexcept { return channelBleedDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getEvenHarmonics() const noexcept { return evenHarmonics_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getOutputIron() const noexcept { return outputIron_.load(std::memory_order_relaxed); }
    [[nodiscard]] HeadModel getHeadModel() const noexcept { return static_cast<HeadModel>(headModel_.load(std::memory_order_relaxed)); }
    [[nodiscard]] Electronics getElectronics() const noexcept { return static_cast<Electronics>(electronics_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getAlignmentHf() const noexcept { return alignHf_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getAlignmentLf() const noexcept { return alignLf_.load(std::memory_order_relaxed); }
    [[nodiscard]] Formulation getFormulation() const noexcept { return static_cast<Formulation>(formulation_.load(std::memory_order_relaxed)); }
    [[nodiscard]] TrackFormat getTrackFormat() const noexcept { return static_cast<TrackFormat>(trackFormat_.load(std::memory_order_relaxed)); }
    [[nodiscard]] T getCalibrationLevel() const noexcept { return calLevel_.load(std::memory_order_relaxed); }
    [[nodiscard]] Monitor getMonitor() const noexcept { return static_cast<Monitor>(monitor_.load(std::memory_order_relaxed)); }

    [[nodiscard]] T getDrive() const noexcept { return driveDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBias() const noexcept { return bias_.load(std::memory_order_relaxed); }
    [[nodiscard]] Speed getSpeed() const noexcept
    {
        return static_cast<Speed>(speed_.load(std::memory_order_relaxed));
    }
    [[nodiscard]] Standard getStandard() const noexcept
    {
        return static_cast<Standard>(standard_.load(std::memory_order_relaxed));
    }
    [[nodiscard]] T getLossEffects() const noexcept { return loss_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHeadBump() const noexcept { return headBumpAmt_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWowFlutter() const noexcept { return wowFlutter_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getNoise() const noexcept { return noiseDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Total latency in samples (active oversampler + loss FIR +
     *  transport delay); reflects the current factor for host PDC. */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

    /** @brief Compatibility alias of getLatency(), in prepared-rate samples. */
    [[nodiscard]] int getLatencySamples() const noexcept { return getLatency(); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TAPE"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("drive", static_cast<float>(driveDb_.load(std::memory_order_relaxed)));
        w.write("bias", static_cast<float>(bias_.load(std::memory_order_relaxed)));
        w.write("speed", speed_.load(std::memory_order_relaxed));
        w.write("standard", standard_.load(std::memory_order_relaxed));
        w.write("loss", static_cast<float>(loss_.load(std::memory_order_relaxed)));
        w.write("headBump", static_cast<float>(headBumpAmt_.load(std::memory_order_relaxed)));
        w.write("wowFlutter", static_cast<float>(wowFlutter_.load(std::memory_order_relaxed)));
        w.write("noise", static_cast<float>(noiseDb_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("oversampling", osRequested_);
        w.write("biasShape", fourLevel_ ? 1 : 0);
        w.write("oversamplingAuto", osAutomatic_);
        w.write("oversamplingMode", static_cast<int32_t>(osMode_));
        w.write("transport", transport_.load(std::memory_order_relaxed));
        w.write("humDb", static_cast<float>(humDb_.load(std::memory_order_relaxed)));
        w.write("mains", mains_.load(std::memory_order_relaxed));
        w.write("channelBleed", static_cast<float>(channelBleedDb_.load(std::memory_order_relaxed)));
        w.write("evenHarmonics", static_cast<float>(evenHarmonics_.load(std::memory_order_relaxed)));
        w.write("outputIron", static_cast<float>(outputIron_.load(std::memory_order_relaxed)));
        w.write("headModel", headModel_.load(std::memory_order_relaxed));
        w.write("electronics", electronics_.load(std::memory_order_relaxed));
        w.write("alignHf", static_cast<float>(alignHf_.load(std::memory_order_relaxed)));
        w.write("alignLf", static_cast<float>(alignLf_.load(std::memory_order_relaxed)));
        w.write("recordEmphasis", static_cast<float>(recordEmphasis_.load(std::memory_order_relaxed)));
        w.write("formulation", formulation_.load(std::memory_order_relaxed));
        w.write("trackFormat", trackFormat_.load(std::memory_order_relaxed));
        w.write("calibrationLevel", static_cast<float>(calLevel_.load(std::memory_order_relaxed)));
        w.write("monitor", monitor_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TAPE")) return false;
        setDrive(static_cast<T>(r.read("drive", 0.0f)));
        setBias(static_cast<T>(r.read("bias", 0.5f)));
        setSpeed(static_cast<Speed>(r.read("speed", 1)));
        setStandard(static_cast<Standard>(r.read("standard", 0)));
        setLossEffects(static_cast<T>(r.read("loss", 0.5f)));
        setHeadBump(static_cast<T>(r.read("headBump", 0.5f)));
        setWowFlutter(static_cast<T>(r.read("wowFlutter", 0.15f)));
        setNoise(static_cast<T>(r.read("noise", -200.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setTransport(static_cast<Transport>(r.read("transport", 0)));
        setHum(static_cast<T>(r.read("humDb", -200.0f)), static_cast<Mains>(r.read("mains", 0)));
        setChannelBleed(static_cast<T>(r.read("channelBleed", -200.0f)));
        setEvenHarmonics(static_cast<T>(r.read("evenHarmonics", 0.0f)));
        setOutputIron(static_cast<T>(r.read("outputIron", 0.0f)));
        setHeadModel(static_cast<HeadModel>(r.read("headModel", 0)));
        setElectronics(static_cast<Electronics>(r.read("electronics", 0)));
        setAlignment(static_cast<T>(r.read("alignHf", 0.0f)), static_cast<T>(r.read("alignLf", 0.0f)));
        setRecordEmphasis(static_cast<T>(r.read("recordEmphasis", 0.0f)));
        setFormulation(static_cast<Formulation>(r.read("formulation", 0)));
        setTrackFormat(static_cast<TrackFormat>(r.read("trackFormat", 0)));
        setCalibrationLevel(static_cast<T>(r.read("calibrationLevel", 0.0f)));
        setMonitor(static_cast<Monitor>(r.read("monitor", 0)));
        // Default 4 = the historical fixed factor (older blobs restore 4x).
        setOversampling(r.read("oversampling", 4));
        setAutomaticOversampling(r.read("oversamplingAuto", false));
        setBiasShape(r.read("biasShape", 0) == 1 ? BiasShape::FourLevel : BiasShape::TwoLevel);
        setOversamplingMode(static_cast<OversamplingMode>(r.read("oversamplingMode", 0)));
        return true;
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Processes a block in-place. Pass-through until prepare() succeeds. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        if (nCh == 0 || nS == 0) return;

        // Front-door non-finite guard: a single NaN/Inf input sample would
        // poison the recursive record/playback EQ, push-pull JA hysteresis,
        // loss-FIR ring, transport delay and over-bias/DC state PERMANENTLY -
        // the JA core mutes to exact silence forever (only reset() clears it,
        // not clean input). Replace bad samples with silence before they reach
        // any state, so a transient upstream glitch cannot mute the channel for
        // the rest of the stream.
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                if (!std::isfinite(d[i])) d[i] = T(0);
        }

        // Acquire pairs with the setters' release stores so the recompute
        // always sees the values published before the flag. Linear stages
        // change at once; the drive makeup calibrates over the next blocks.
        if (dirty_.load(std::memory_order_relaxed)
            && dirty_.exchange(false, std::memory_order_acquire))
        {
            recomputeLinear();
            if (calActive_) calRestart_ = true;
            else startCalibration();
        }

        for (int off = 0; off < nS; off += maxBlock_)
            processChunk(buffer.getSubView(off, std::min(maxBlock_, nS - off)), nCh);
    }

private:
    void processChunk(AudioBufferView<T> buffer, int nCh) noexcept
    {
        const int nS = buffer.getNumSamples();
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;
        const double wfDepth = static_cast<double>(wowFlutter_.load(std::memory_order_relaxed));
        const bool speedTransport = transport_.load(std::memory_order_relaxed) == 1;
        if (speedTransport != speedTransportActive_)
        {
            // Into the speed model from where the modulated delay stands; back
            // to the modulated model at its own position.
            if (speedTransport)
            {
                tapeTransport_.setSpeedNow(speedIps_);
                tapeTransport_.syncDelay(static_cast<double>(delayCenter_) + lastMod_);
            }
            speedTransportActive_ = speedTransport;
        }
        const bool circuit = circuitActive_;
        aimRamps(false);
        if (hissOn() && hissNormDirty_) computeHissNorm();
        const bool hissActive = hiss_.active();
        const bool humActive = hum_.active();
        const bool xtActive = xt_.active();
        const bool trActive = tr_.active();
        const bool monitorMoving = monitorRamp_.now != 1.0 || monitorRamp_.target != 1.0;
        aimDryPhase(mixTarget);
        const double humStep = (mains_.load(std::memory_order_relaxed) == 1 ? 60.0 : 50.0) / sampleRate_;

        // 1. Dry snapshot for the latency-compensated mix (through the
        //    oversampler's filters when they are minimum phase).
        if (dryThroughFilters_)
        {
            for (int ch = 0; ch < nCh; ++ch)
                std::copy_n(buffer.getChannel(ch), nS, dryScratch_.getChannel(ch));
            auto dryView = dryScratch_.toView().getSubView(0, nS);
            dryPath_.process(dryView);
        }
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* in = dryThroughFilters_ ? dryScratch_.getChannel(ch) : buffer.getChannel(ch);
            auto& dry = dryRing_[static_cast<size_t>(ch)];
            int dp = dryPos_;
            for (int i = 0; i < nS; ++i)
            {
                dry[static_cast<size_t>(dp)] = in[i];
                dp = (dp + 1) & (drySize_ - 1);
            }
        }

        // At rest: with the mix settled on the dry side, or the source being
        // monitored, nothing of the machine is heard, so it does not run. The
        // dry ring keeps its place and a pending calibration goes on.
        if ((mixTarget == T(0) && mixStart == T(0)) || (monitorRamp_.now == 0.0 && monitorRamp_.target == 0.0))
        {
            for (int i = 0; i < nS; ++i)
            {
                const double inTurn = dryInputLead_.advance(), outTurn = dryOutputLead_.advance(), ironTurn = dryIronLead_.advance();
                const int at = (dryPos_ + i - dryDelay_) & (drySize_ - 1);
                for (int ch = 0; ch < nCh; ++ch)
                    buffer.getChannel(ch)[i] = turnDry(ch, dryRing_[static_cast<size_t>(ch)][static_cast<size_t>(at)], inTurn, outTurn, ironTurn);
            }
            dryPos_ = (dryPos_ + nS) & (drySize_ - 1);
            if (calActive_) stepCalibration(nS * osFactor_);
            resting_ = true;
            return;
        }
        if (resting_)
        {
            // Back from rest: the machine's memory is of another time. It
            // starts from rest and runs unheard until its own latency has
            // passed; then the mix (or the monitor) fades it in.
            clearMachine();
            hSm_ = hScale_;
            mSm_ = mScale_;
            bSm_ = biasAmp_;
            wakeHold_ = latency_ + 1;
            resting_ = false;
        }
        const int warm = std::min(wakeHold_, nS);
        wakeHold_ -= warm;

        // 1b. The line input is AC coupled, as a machine's is: DC and the
        // slowest drift never reach the tape, where they would lean the loop
        // to one side (even harmonics that no later stage removes). Two
        // one-pole high-passes in series: critically damped, no overshoot; in
        // double, the difference taken first so that a constant cancels
        // exactly. Then the record side of the emphasis trim.
        {
            const double p = inputAcPole_, g = 0.5 * (1.0 + p);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = buffer.getChannel(ch);
                auto& s = inputAc_[static_cast<size_t>(ch)];
                auto& trim = recTrim_[static_cast<size_t>(ch)];
                for (int i = 0; i < nS; ++i)
                {
                    const double x = static_cast<double>(d[i]);
                    const double y1 = g * (x - s[0]) + p * s[1];
                    s[0] = x;
                    s[1] = y1;
                    const double y2 = g * (y1 - s[2]) + p * s[3];
                    s[2] = y1;
                    s[3] = y2;
                    d[i] = static_cast<T>(trim.process(y2));
                }
            }
        }

        // 2. Record EQ (emphasis) at base rate (the circuit runs its record
        // network at the internal rate instead).
        if (!circuit)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = buffer.getChannel(ch);
                auto& hf = recordHF_[static_cast<size_t>(ch)];
                auto& lf = recordLF_[static_cast<size_t>(ch)];
                for (int i = 0; i < nS; ++i)
                    d[i] = static_cast<T>(hf.process(lf.process(static_cast<double>(d[i]))));
            }
        }

        // 3. Push-pull AC-bias hysteresis at 4x. One shared carrier phase for
        // all channels (a machine has a single bias oscillator); each channel
        // runs a +carrier and a -carrier JA instance and averages them, which
        // cancels every odd-in-bias term exactly (carrier, its folded 3rd
        // harmonic, remanence DC) like a centre-tapped record head.
        {
            const bool osOn = (oversampler_ != nullptr);
            auto osView = osOn ? oversampler_->upsample(buffer) : buffer;
            const int osN = osView.getNumSamples();
            const int phaseStart = biasPhase_;
            // Geometric ramps of the calibrated set (drive, makeup, bias):
            // the pair spans orders of magnitude, so power-law interpolation
            // keeps it on its calibration curve (as in TransformerModel).
            double hRat = 1.0, mRat = 1.0, bRat = 1.0;
            double hEnd = hSm_, mEnd = mSm_, bEnd = bSm_;
            if (osN > 0 && (hSm_ != hScale_ || mSm_ != mScale_ || bSm_ != biasAmp_))
            {
                const double kSm = 1.0 - std::exp(-static_cast<double>(osN)
                    / (kGainRampSeconds * sampleRate_ * static_cast<double>(osFactor_)));
                const auto glide = [kSm](double from, double to) noexcept
                {
                    const double e = from * std::pow(to / from, kSm);
                    return std::abs(std::log(e / to)) < 1e-9 ? to : e;
                };
                hEnd = glide(hSm_, hScale_);
                mEnd = glide(mSm_, mScale_);
                bEnd = glide(bSm_, biasAmp_);
                const double inv = 1.0 / static_cast<double>(osN);
                hRat = std::pow(hEnd / hSm_, inv);
                mRat = std::pow(mEnd / mSm_, inv);
                bRat = std::pow(bEnd / bSm_, inv);
            }
            const bool ramping = hRat != 1.0 || mRat != 1.0 || bRat != 1.0;
            const bool asymActive = asym_.active();
            LinearRamp asymAfter = asym_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = osView.getChannel(ch);
                auto& hp = hysteresis_[static_cast<size_t>(ch)];
                auto& hn = hysteresisN_[static_cast<size_t>(ch)];
                auto& rec = recNet_[static_cast<size_t>(ch)];
                auto& play = playNet_[static_cast<size_t>(ch)];
                double& mean = asymMean_[static_cast<size_t>(ch)];
                LinearRamp asym = asym_;
                double inScale = hSm_, mS = mSm_, B = bSm_;
                int phase = phaseStart;
                for (int i = 0; i < osN; ++i)
                {
                    if (ramping)
                    {
                        inScale *= hRat;
                        mS *= mRat;
                        B *= bRat;
                    }
                    double xs = static_cast<double>(d[i]);
                    if (circuit) xs = rec[1].processSample(rec[0].processSample(xs, 0), 0);
                    if (asymActive)
                    {
                        // Even term less its running mean: no DC reaches the tape.
                        const double square = xs * xs;
                        mean = asymMeanA_ * mean + (1.0 - asymMeanA_) * square;
                        xs += asym.next() * kAsymmetryShare * (square - mean);
                    }
                    const double x = inScale * xs;
                    const double c = carrier(B, biasNominal_, phase);
                    phase = (phase + 1) & 7;
                    const double mp = static_cast<double>(hp.processSample(static_cast<T>(x + c)));
                    const double mn = static_cast<double>(hn.processSample(static_cast<T>(x - c)));
                    double out = (mS * 0.5) * (mp + mn);
                    if (circuit) out = play[1].processSample(play[0].processSample(out, 0), 0);
                    d[i] = static_cast<T>(out);
                }
                asymAfter = asym;
            }
            if (nCh > 0) asym_ = asymAfter;
            if (circuit)
                for (int ch = 0; ch < nCh; ++ch)
                    for (int s = 0; s < 2; ++s)
                    {
                        recNet_[static_cast<size_t>(ch)][static_cast<size_t>(s)].advanceRamp(osN);
                        playNet_[static_cast<size_t>(ch)][static_cast<size_t>(s)].advanceRamp(osN);
                    }
            hSm_ = hEnd;
            mSm_ = mEnd;
            bSm_ = bEnd;
            biasPhase_ = (phaseStart + osN) & 7;
            // Advance a pending makeup calibration by as many internal
            // samples as this block has: bounded work per callback.
            if (calActive_) stepCalibration(osN);
            if (osOn) oversampler_->downsample(buffer);
        }

        // The over-bias low-pass starts from the signal it receives, not from
        // rest: switching it on with a zero state stepped the output.
        const bool overBiasStart = overBiasOn_ && !overBiasActive_;
        overBiasActive_ = overBiasOn_;

        // 4-6. Per frame: hiss onto the tape, loss FIR and head, transport,
        // playback EQ, over-bias, output coupling and transformer per channel;
        // then the channel bleed, hum, monitor and mix across the channels.
        for (int i = 0; i < nS; ++i)
        {
            // One shared transport per frame (all channels move together,
            // like tape past a single capstan).
            double readPos;
            if (speedTransport)
            {
                readPos = tapeTransport_.next(wfDepth);
            }
            else
            {
                lastMod_ = nextTransportMod(wfDepth);
                readPos = static_cast<double>(delayCenter_) + lastMod_;
            }
            const auto readInt = static_cast<int>(std::floor(readPos));
            const T frac = static_cast<T>(readPos - readInt);
            const double hissAmp = hissActive ? hiss_.next() * hissNorm_ * kUniformToUnitRms : 0.0;
            const double trShare = trActive ? tr_.next() : 0.0;

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = buffer.getChannel(ch);
                T in = d[i];

                // Hiss is magnetic noise on the tape: it is read through the
                // head losses and the reproduce de-emphasis (own RNG stream:
                // enabling it must not change the transport's realisation).
                if (hissActive)
                {
                    rngNoise_ = rngNoise_ * 1664525u + 1013904223u;
                    double n1 = (static_cast<double>(rngNoise_ >> 8) / 8388608.0 - 1.0) * hissAmp;
                    if (circuit)
                        n1 = hissLF_[static_cast<size_t>(ch)].process(hissHF_[static_cast<size_t>(ch)].process(n1));
                    in += static_cast<T>(n1);
                }

                // Loss FIR (linear phase) on a per-channel mirrored ring.
                auto& fir = firState_[static_cast<size_t>(ch)];
                fir[static_cast<size_t>(firPos_)] = in;
                fir[static_cast<size_t>(firPos_ + firHalf_)] = in; // mirrored
                const T* win = &fir[static_cast<size_t>(firPos_ + firHalf_ - (firLen_ - 1))];
                T y = simd::dotProduct(firTaps_.data(), win, firLen_);

                // Head bump resonance, or the contour chain.
                if (contourActive_)
                    for (auto& s : headContour_[static_cast<size_t>(ch)]) y = s.process(y);
                else
                    y = headBump_[static_cast<size_t>(ch)].process(y);

                // Transport fractional delay: interpolate at t - readInt -
                // frac, between p1 = x(t-readInt) and p2 = x(t-readInt-1).
                auto& dl = delayRing_[static_cast<size_t>(ch)];
                dl[static_cast<size_t>(delayPos_)] = y;
                const int base = delayPos_ - readInt;
                const int dm = delayMask_;
                const T p0 = dl[static_cast<size_t>((base + 1) & dm)];
                const T p1 = dl[static_cast<size_t>(base & dm)];
                const T p2 = dl[static_cast<size_t>((base - 1) & dm)];
                const T p3 = dl[static_cast<size_t>((base - 2) & dm)];
                T w;
                if (speedTransport)
                {
                    // Four-point Lagrange (nodes -1, 0, 1, 2).
                    const T f = frac;
                    const T fm1 = f - T(1), fm2 = f - T(2), fp1 = f + T(1);
                    w = -f * fm1 * fm2 * T(1.0 / 6.0) * p0 + fp1 * fm1 * fm2 * T(0.5) * p1
                        - fp1 * f * fm2 * T(0.5) * p2 + fp1 * f * fm1 * T(1.0 / 6.0) * p3;
                }
                else
                {
                    // Catmull-Rom.
                    w = p1 + T(0.5) * frac * (p2 - p0
                        + frac * (T(2) * p0 - T(5) * p1 + T(4) * p2 - p3
                        + frac * (T(3) * (p1 - p2) + p3 - p0)));
                }

                // Playback EQ (exact inverse de-emphasis; the circuit ran its
                // reproduce network before the downsampler).
                if (!circuit)
                    w = static_cast<T>(playLF_[static_cast<size_t>(ch)].process(
                            playHF_[static_cast<size_t>(ch)].process(static_cast<double>(w))));
                // The reproduce side of the emphasis trim (the tape's hiss crosses it too).
                w = static_cast<T>(playTrim_[static_cast<size_t>(ch)].process(static_cast<double>(w)));

                // Over-bias self-erasure: wider recording zone partially
                // erases short wavelengths (first-order low-pass engaged above
                // nominal bias only; identity at or below nominal). Its zero
                // matches the analog magnitude at Nyquist, so the roll-off is
                // the same at every rate (the plain one-pole fell 4 dB short
                // at 18 kHz at 48 kHz).
                if (overBiasOn_)
                {
                    auto& lp = overBiasLp_[static_cast<size_t>(ch)];
                    auto& x1 = overBiasX1_[static_cast<size_t>(ch)];
                    const double xin = static_cast<double>(w);
                    if (overBiasStart && i == 0) { lp = xin; x1 = xin; }
                    lp = obB0_ * xin + obB1_ * x1 + obP_ * lp;
                    x1 = xin;
                    w = static_cast<T>(lp);
                }

                // AC-coupled playback amplifier (2nd-order 24 Hz high-pass:
                // subsonic/DC roll-off of the real hardware).
                w = outHp_[static_cast<size_t>(ch)].process(w);

                // Output transformer branch: always running, added by its share.
                {
                    const double wd = static_cast<double>(w);
                    const double u = trPre_[static_cast<size_t>(ch)].process(wd);
                    double& m = trMean_[static_cast<size_t>(ch)];
                    m = trMeanA_ * m + (1.0 - trMeanA_) * u * u;
                    const double bent = trPost_[static_cast<size_t>(ch)].process(
                        kTransformerEven * (u * u - m) - kTransformerOdd * u * u * u);
                    // The iron's low end: a low-pass that always runs, taken
                    // off by the same share (a first-order high-pass at the
                    // full amount). What DC the even term leaves goes with it.
                    double& low = ironLp_[static_cast<size_t>(ch)];
                    const double sum = trShare > 0.0 ? wd + trShare * bent : wd;
                    low += ironA_ * (sum - low);
                    if (trShare > 0.0) w = static_cast<T>(sum - trShare * low);
                }
                wet_[static_cast<size_t>(ch)] = static_cast<double>(w);
            }

            if (xtActive)
            {
                const double g = xt_.next();
                for (int ch = 0; ch + 1 < nCh; ch += 2)
                {
                    const double a = wet_[static_cast<size_t>(ch)], b = wet_[static_cast<size_t>(ch + 1)];
                    const double fromA = xtHp_[static_cast<size_t>(ch)].process(a);
                    const double fromB = xtHp_[static_cast<size_t>(ch + 1)].process(b);
                    wet_[static_cast<size_t>(ch)] = a + g * fromB;
                    wet_[static_cast<size_t>(ch + 1)] = b + g * fromA;
                }
            }
            if (humActive)
            {
                const double amp = hum_.next();
                humPhase_ += humStep;
                if (humPhase_ >= 1.0) humPhase_ -= 1.0;
                const double ph = 2.0 * std::numbers::pi * humPhase_;
                const double h = amp * kHumNorm * (std::sin(ph) + 0.5 * std::sin(2.0 * ph));
                for (int ch = 0; ch < nCh; ++ch) wet_[static_cast<size_t>(ch)] += h;
            }

            // Latency-compensated monitor and mix.
            // While the machine warms up after a rest the dry side alone is
            // heard, and the mix and the monitor wait for it.
            const double monitor = i < warm ? 0.0 : (monitorMoving ? monitorRamp_.next() : 1.0);
            const T mixVal = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(std::max(0, i + 1 - warm)));
            const double inTurn = dryInputLead_.advance(), outTurn = dryOutputLead_.advance(), ironTurn = dryIronLead_.advance();
            const bool wetAlone = mixVal == T(1) && monitor == 1.0;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = buffer.getChannel(ch);
                const auto& dry = dryRing_[static_cast<size_t>(ch)];
                const int dryIdx = (dryPos_ + i - dryDelay_) & (drySize_ - 1);
                const T rawDry = dry[static_cast<size_t>(dryIdx)];
                const T turned = turnDry(ch, rawDry, inTurn, outTurn, ironTurn);
                const T drySample = wetAlone ? rawDry : turned;
                T w = static_cast<T>(wet_[static_cast<size_t>(ch)]);
                if (monitor != 1.0)
                    w = static_cast<T>(static_cast<double>(drySample)
                        + (static_cast<double>(w) - static_cast<double>(drySample)) * monitor);
                d[i] = drySample + (w - drySample) * mixVal;
            }

            firPos_ = (firPos_ + 1) & (firHalf_ - 1);
            delayPos_ = (delayPos_ + 1) & delayMask_;
        }
        dryPos_ = (dryPos_ + nS) & (drySize_ - 1);
        currentMix_ = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(nS - warm));
    }

    /**
     * @brief Where the phase of the dry side goes: that of the high-passes
     *        while the machine is mixed in, none where the dry side is the output.
     */
    void aimDryPhase(T mixTarget) noexcept
    {
        const bool turn = mixTarget > T(0) && monitor_.load(std::memory_order_relaxed) != 2;
        const double iron = static_cast<double>(outputIron_.load(std::memory_order_relaxed));
        dryInputLead_.aim(turn);
        dryOutputLead_.aim(turn);
        // The iron takes off a share of a one-pole low-pass: an all-pass at
        // half that share of its corner leads by as much.
        dryIronLead_.setCorner(0.5 * kIronCornerHz * std::max(iron, 0.05));
        dryIronLead_.aim(turn && iron >= 0.05);
    }

    /** @brief A sample of the dry side through the phase of the high-passes. */
    [[nodiscard]] T turnDry(int channel, T x, double inTurn, double outTurn, double ironTurn) noexcept
    {
        return dryIronLead_.processSample(channel,
            dryOutputLead_.processSample(channel, dryInputLead_.processSample(channel, x, inTurn), outTurn), ironTurn);
    }

    /** @brief The factor in use: the one asked for, or the least the four levels need. */
    [[nodiscard]] int factorInUse(double rate) const noexcept
    {
        return fourLevel_ ? std::max(osRequested_, Oversampling<T>::factorForRate(rate, kFourLevelInternalRate)) : osRequested_;
    }

    /**
     * @brief The carrier at one of its eight phases.
     * @param amplitude The amplitude of the bias in use.
     * @param nominal   The optimum of the material: over it the inner level stays.
     * @param phase     The phase, 0 to 7.
     */
    [[nodiscard]] double carrier(double amplitude, double nominal, int phase) const noexcept
    {
        const double sign = kBiasTable[static_cast<size_t>(phase)];
        if (!fourLevel_) return amplitude * sign;
        return sign * ((phase & 1) ? kOuterLevel * amplitude : kInnerLevel * std::min(amplitude, nominal));
    }

    /** @brief A linear ramp towards a target, at a step fixed when it is aimed. */
    struct LinearRamp
    {
        double now = 0.0, target = 0.0, step = 0.0;

        void aim(double to, double samples) noexcept
        {
            if (to == target) return;
            target = to;
            step = std::abs(to - now) / std::max(1.0, samples);
        }
        [[nodiscard]] double next() noexcept
        {
            if (now != target)
                now = (now < target) ? std::min(target, now + step) : std::max(target, now - step);
            return now;
        }
        [[nodiscard]] bool active() const noexcept { return now != 0.0 || target != 0.0; }
    };

    [[nodiscard]] bool hissOn() const noexcept { return noiseDb_.load(std::memory_order_relaxed) > T(-120); }

    /** @brief Aims the level ramps at the settings (and lands them with `land`). */
    void aimRamps(bool land) noexcept
    {
        const double fs = sampleRate_;
        // Hiss: the setting, less what the track width, the formulation and
        // the operating level take off it against the signal.
        double hiss = 0.0;
        if (hissOn())
        {
            static constexpr double kTrackWidthMm[3] = { 2.0, 6.3, 1.0 };
            const int track = std::clamp(trackFormat_.load(std::memory_order_relaxed), 0, 2);
            const double offset = 10.0 * std::log10(kTrackWidthMm[0] / kTrackWidthMm[static_cast<size_t>(track)])
                + formulationOf(formulation_.load(std::memory_order_relaxed)).hissDb
                - static_cast<double>(calLevel_.load(std::memory_order_relaxed));
            hiss = std::pow(10.0, (static_cast<double>(noiseDb_.load(std::memory_order_relaxed)) + offset) / 20.0);
        }
        const T humDb = humDb_.load(std::memory_order_relaxed);
        const double hum = humDb > T(-120) ? std::pow(10.0, static_cast<double>(humDb) / 20.0) : 0.0;
        const T xtDb = channelBleedDb_.load(std::memory_order_relaxed);
        const double xt = xtDb > T(-120) ? std::pow(10.0, static_cast<double>(xtDb) / 20.0) * xtNorm_ : 0.0;
        if (xt > 0.0 && !xt_.active())
            for (auto& s : xtHp_) s.clear();
        const double monitor = monitor_.load(std::memory_order_relaxed) == 2 ? 0.0 : 1.0;
        hiss_.aim(hiss, 0.05 * fs);
        hum_.aim(hum, 0.05 * fs);
        xt_.aim(xt, 0.05 * fs);
        tr_.aim(static_cast<double>(outputIron_.load(std::memory_order_relaxed)), 0.02 * fs);
        asym_.aim(static_cast<double>(evenHarmonics_.load(std::memory_order_relaxed)), 0.02 * fs * static_cast<double>(osFactor_));
        monitorRamp_.aim(monitor, 0.02 * fs);
        if (land)
        {
            hiss_.now = hiss_.target;
            hum_.now = hum_.target;
            xt_.now = xt_.target;
            tr_.now = tr_.target;
            asym_.now = asym_.target;
            monitorRamp_.now = monitorRamp_.target;
        }
    }

    /**
     * @brief The hiss scale that makes white noise on the tape come out at
     *        unit RMS: one over the RMS gain of the reproduce chain (loss FIR,
     *        head, de-emphasis, over-bias and output coupling) over a grid of
     *        frequencies.
     */
    void computeHissNorm() noexcept
    {
        hissNormDirty_ = false;
        if (numChannels_ <= 0) return;
        constexpr int kGrid = 256;
        double power = 0.0;
        for (int k = 0; k < kGrid; ++k)
        {
            const double w = std::numbers::pi * (static_cast<double>(k) + 0.5) / kGrid;
            const double cw = std::cos(w), sw = std::sin(w);
            double re = 0.0, im = 0.0, rr = 1.0, ri = 0.0;
            for (int n = 0; n < firLen_; ++n)
            {
                re += designTaps_[static_cast<size_t>(n)] * rr;
                im += designTaps_[static_cast<size_t>(n)] * ri;
                const double nr = rr * cw + ri * sw;
                ri = ri * cw - rr * sw;
                rr = nr;
            }
            double g2 = (re * re + im * im) * firNorm_ * firNorm_;
            if (contourActive_)
                for (const auto& s : headContour_[0]) g2 *= peakMagnitude(s, w) * peakMagnitude(s, w);
            else
                g2 *= peakMagnitude(headBump_[0], w) * peakMagnitude(headBump_[0], w);
            const double play = shelfMagnitude(playHF_[0], w) * shelfMagnitude(playLF_[0], w);
            g2 *= play * play;
            if (overBiasOn_)
            {
                const double nr = obB0_ + obB1_ * cw, ni = -obB1_ * sw;
                const double dr = 1.0 - obP_ * cw, di = obP_ * sw;
                g2 *= (nr * nr + ni * ni) / (dr * dr + di * di);
            }
            g2 *= peakMagnitude(outHp_[0], w) * peakMagnitude(outHp_[0], w);
            power += g2;
        }
        power /= kGrid;
        hissNorm_ = power > 1e-20 ? 1.0 / std::sqrt(power) : 1.0;
    }

    // -- Building blocks -----------------------------------------------------------
    // -- Building blocks -----------------------------------------------------------

    /** @brief One-pole/one-zero section with its exact inverse (double state). */
    struct ShelfSection
    {
        double b0 = 1.0, b1 = 0.0, a1 = 0.0;
        double x1 = 0.0, y1 = 0.0;

        void clear() noexcept { x1 = 0.0; y1 = 0.0; }
        [[nodiscard]] double process(double x) noexcept
        {
            const double y = b0 * x + b1 * x1 - a1 * y1;
            x1 = x;
            y1 = y;
            return y;
        }
    };

    /** @brief Minimal biquad in double (head bump, output high-pass). */
    struct PeakSection
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;

        void clear() noexcept { z1 = 0.0; z2 = 0.0; }
        [[nodiscard]] T process(T x) noexcept
        {
            const double in = static_cast<double>(x);
            const double y = b0 * in + z1;
            z1 = b1 * in - a1 * y + z2;
            z2 = b2 * in - a2 * y;
            return static_cast<T>(y);
        }
    };

    /** @brief HF emphasis (kHi*s*t + 1)/(s*t + 1) and LF shelf (s*t + kLo)/(s*t + 1)
     *  via bilinear transform; `inverse` swaps numerator and denominator. */
    static ShelfSection makeHFShelf(double t, double kHi, double fs, bool inverse) noexcept
    {
        const double kt = 2.0 * fs * t;
        double n0 = kHi * kt + 1.0, n1 = 1.0 - kHi * kt;
        double d0 = kt + 1.0,       d1 = 1.0 - kt;
        if (inverse) { std::swap(n0, d0); std::swap(n1, d1); }
        return { n0 / d0, n1 / d0, d1 / d0, 0.0, 0.0 };
    }

    static ShelfSection makeLFShelf(double t, double kLo, double fs, bool inverse) noexcept
    {
        const double kt = 2.0 * fs * t;
        double n0 = kt + kLo, n1 = kLo - kt;
        double d0 = kt + 1.0, d1 = 1.0 - kt;
        if (inverse) { std::swap(n0, d0); std::swap(n1, d1); }
        return { n0 / d0, n1 / d0, d1 / d0, 0.0, 0.0 };
    }

    /** @brief Bilinear transform of (n1 s + n0) / (d1 s + d0). */
    static ShelfSection firstOrder(double n1, double n0, double d1, double d0, double fs) noexcept
    {
        const double k = 2.0 * fs;
        const double a0 = d1 * k + d0;
        return { (n1 * k + n0) / a0, (n0 - n1 * k) / a0, (d0 - d1 * k) / a0, 0.0, 0.0 };
    }

    /** @brief Magnitude of a first-order section at w (radians per sample). */
    [[nodiscard]] static double shelfMagnitude(const ShelfSection& s, double w) noexcept
    {
        const double c = std::cos(w), sn = std::sin(w);
        const double nr = s.b0 + s.b1 * c, ni = -s.b1 * sn;
        const double dr = 1.0 + s.a1 * c, di = -s.a1 * sn;
        return std::sqrt((nr * nr + ni * ni) / std::max(1e-300, dr * dr + di * di));
    }

    /** @brief Magnitude of a biquad section at w (radians per sample). */
    [[nodiscard]] static double peakMagnitude(const PeakSection& s, double w) noexcept
    {
        const double c1 = std::cos(w), s1 = std::sin(w), c2 = std::cos(2.0 * w), s2 = std::sin(2.0 * w);
        const double nr = s.b0 + s.b1 * c1 + s.b2 * c2, ni = -(s.b1 * s1 + s.b2 * s2);
        const double dr = 1.0 + s.a1 * c1 + s.a2 * c2, di = -(s.a1 * s1 + s.a2 * s2);
        return std::sqrt((nr * nr + ni * ni) / std::max(1e-300, dr * dr + di * di));
    }

    /** @brief JA parameters and hiss offset of a formulation. */
    struct FormulationParams
    {
        double ms, a, alpha, k, c, hissDb;
    };

    [[nodiscard]] static FormulationParams formulationOf(int f) noexcept
    {
        switch (std::clamp(f, 0, 2))
        {
        case 1:  return { 3.5e5 * 1.4125375446227544, 2.2e4 * 1.4125375446227544, 1.6e-3, 2.7e4 * 1.2, 0.17, -1.5 };
        case 2:  return { 3.5e5 * 0.7943282347242815, 2.2e4 * 0.7943282347242815, 1.6e-3, 2.7e4 * 0.8, 0.25, 3.0 };
        default: return { 3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17, 0.0 };
        }
    }

    /** @brief Rebuilds the linear stages (EQ, losses, head bump, over-bias)
     *  and records the calibration request for drive, bias and emphasis. */
    /** @brief Clears the memory of the machine's side (all but the dry ring). */
    void clearMachine() noexcept
    {
        for (auto& h : hysteresis_) h.reset();
        for (auto& h : hysteresisN_) h.reset();
        for (auto& f : firState_) std::fill(f.begin(), f.end(), T(0));
        for (auto& d : delayRing_) std::fill(d.begin(), d.end(), T(0));
        for (auto& s : inputAc_) s = {};
        for (auto& s : recTrim_) s.clear();
        for (auto& s : playTrim_) s.clear();
        for (auto& v : ironLp_) v = 0.0;
        for (auto& s : recordHF_) s.clear();
        for (auto& s : recordLF_) s.clear();
        for (auto& s : playHF_) s.clear();
        for (auto& s : playLF_) s.clear();
        for (auto& v : overBiasLp_) v = 0.0;
        for (auto& v : overBiasX1_) v = 0.0;
        // Clear STATE only: wiping coefficients here used to leave the head
        // bump at identity until the next parameter change re-ran recompute.
        for (auto& b : headBump_) b.clear();
        for (auto& hp : outHp_) hp.clear();
        for (auto& s : hissHF_) s.clear();
        for (auto& s : hissLF_) s.clear();
        for (auto& s : xtHp_) s.clear();
        for (auto& s : trPre_) s.clear();
        for (auto& s : trPost_) s.clear();
        for (auto& v : trMean_) v = 0.0;
        for (auto& v : asymMean_) v = 0.0;
        for (auto& v : wet_) v = 0.0;
        for (auto& chain : headContour_) for (auto& s : chain) s.clear();
        for (auto& n : recNet_) for (auto& s : n) s.reset();
        for (auto& n : playNet_) for (auto& s : n) s.reset();
        if (oversampler_) oversampler_->reset();
    }

    /** @brief The speed in inches per second. */
    [[nodiscard]] static double speedInIps(Speed speed) noexcept
    {
        switch (speed)
        {
        case Speed::IPS_3_75: return 3.75;
        case Speed::IPS_7_5:  return 7.5;
        case Speed::IPS_30:   return 30.0;
        case Speed::IPS_15:   break;
        }
        return 15.0;
    }

    void recomputeLinear() noexcept
    {
        const double driveDbV = static_cast<double>(driveDb_.load(std::memory_order_relaxed));
        const double drive = std::pow(10.0, driveDbV / 20.0);
        const double bias = static_cast<double>(bias_.load(std::memory_order_relaxed));
        const auto speed = static_cast<Speed>(speed_.load(std::memory_order_relaxed));
        const double ips = speedInIps(speed);
        const auto standard = static_cast<Standard>(standard_.load(std::memory_order_relaxed));
        const double lossAmt = static_cast<double>(loss_.load(std::memory_order_relaxed));
        const double bumpAmt = static_cast<double>(headBumpAmt_.load(std::memory_order_relaxed));

        // --- bias -> carrier amplitude and over-bias erasure -------------------
        // Real AC bias: the control maps to the push-pull carrier amplitude in
        // units of the JA 'a' parameter. Nominal (0.5) sits at B = 3a: the
        // carrier sweeps well past the coercivity every cycle, erasing the
        // loop's branch memory exactly like hardware bias (measured: LF gain
        // history-independent to < 0.1 dB). Under-bias drops B below the
        // erase threshold: the loop keeps partial branch memory - the REAL
        // grit and level instability of an under-biased machine. Over-bias
        // adds the self-erasure of short wavelengths (wider recording zone)
        // as a one-pole roll-off.
        // Formulation: the JA parameters of the material, set on the running
        // pairs when it changes; the bias scales with its coercive force so
        // that 0.5 stays the optimum of the material.
        const int form = formulation_.load(std::memory_order_relaxed);
        const FormulationParams fp = formulationOf(form);
        if (form != formApplied_)
        {
            for (auto& h : hysteresis_) h.setParameters(fp.ms, fp.a, fp.alpha, fp.k, fp.c);
            for (auto& h : hysteresisN_) h.setParameters(fp.ms, fp.a, fp.alpha, fp.k, fp.c);
            formApplied_ = form;
            hissNormDirty_ = true;
        }
        calReq_.form = fp;
        const double biasB = std::min(6.0, 3.0 * std::pow(9.0, bias - 0.5));
        calReq_.biasAmp = biasB * 2.2e4 * (fp.k / 2.7e4);
        biasNominal_ = 3.0 * 2.2e4 * (fp.k / 2.7e4);
        calReq_.biasNominal = biasNominal_;
        if (biasB > 3.0)
        {
            // Gentle self-erasure (corner gliding 22 kHz -> ~12.8 kHz at full
            // over-bias): first-order low-pass with its pole at the analog
            // corner and a zero that matches the analog gain at Nyquist.
            // Between the optimum (B = 3) and B = 3.5 the low-pass at its
            // widest comes in by degrees, so the top end darkens from the
            // optimum on without a step.
            const double fc = 22000.0 * (3.5 / std::max(biasB, 3.5));
            const double pole = std::exp(-2.0 * std::numbers::pi * fc / sampleRate_);
            const double r = 0.5 * sampleRate_ / fc;
            const double nyq = 1.0 / std::sqrt(1.0 + r * r);
            const double b0 = 0.5 * ((1.0 - pole) + nyq * (1.0 + pole));
            const double b1 = 0.5 * ((1.0 - pole) - nyq * (1.0 + pole));
            const double share = std::min(1.0, (biasB - 3.0) / 0.5);
            obP_ = pole;
            obB0_ = (1.0 - share) + share * b0;
            obB1_ = share * b1 - (1.0 - share) * pole;
            overBiasOn_ = true;
        }
        else if (bias < 0.5)
        {
            // Under the optimum the short wavelengths record more strongly
            // than the long ones: a high shelf that grows as the bias falls,
            // around a frequency that follows the speed (the wavelength).
            const double lift = std::pow(10.0, kUnderBiasLiftDb * (1.0 - 2.0 * bias) / 20.0);
            const double centre = std::clamp(kUnderBiasCentreHz * ips / 15.0, 1000.0, 0.3 * sampleRate_);
            const ShelfSection s = makeHFShelf(1.0 / (2.0 * std::numbers::pi * centre * std::sqrt(lift)), lift, sampleRate_, false);
            obP_ = -s.a1;
            obB0_ = s.b0;
            obB1_ = s.b1;
            overBiasOn_ = true;
        }
        else
        {
            overBiasOn_ = false;
        }

        // --- EQ time constants per standard and speed --------------------------
        double t2 = 50e-6;                 // HF time constant
        bool useLF = false;                // NAB LF constant 3180 us
        switch (standard)
        {
        case Standard::NAB:
            t2 = (speed == Speed::IPS_30) ? 17.5e-6 : (speed == Speed::IPS_3_75) ? 90e-6 : 50e-6;
            useLF = speed != Speed::IPS_30;
            break;
        case Standard::CCIR:
            // 3.75 ips carries 90 us and the low-frequency constant in both standards.
            t2 = (speed == Speed::IPS_3_75) ? 90e-6
               : (speed == Speed::IPS_7_5) ? 70e-6
               : (speed == Speed::IPS_15) ? 35e-6 : 17.5e-6;
            useLF = speed == Speed::IPS_3_75;
            break;
        }
        constexpr double kHiCap = 4.0;     // +12 dB emphasis cap (practical alignment)
        constexpr double kLoBoost = 2.0;   // +6 dB NAB LF record boost
        const double tLo = 3180e-6;
        const double hfTrim = static_cast<double>(alignHf_.load(std::memory_order_relaxed));
        const double lfTrim = static_cast<double>(alignLf_.load(std::memory_order_relaxed));
        const double gHf = std::pow(10.0, hfTrim / 20.0), gLf = std::pow(10.0, lfTrim / 20.0);

        // The emphasis trim: a shelf on the record side and its exact inverse
        // on the reproduce side (both the identity, to the bit, at 0 dB).
        double trimAt1k = 1.0;
        {
            const double gRec = std::pow(10.0, static_cast<double>(recordEmphasis_.load(std::memory_order_relaxed)) / 20.0);
            const double tTrim = 1.0 / (2.0 * std::numbers::pi * kEmphasisTrimHz * std::sqrt(gRec));
            const ShelfSection up = makeHFShelf(tTrim, gRec, sampleRate_, false);
            const ShelfSection down = makeHFShelf(tTrim, gRec, sampleRate_, true);
            for (auto& s : recTrim_)
            {
                const double x1 = s.x1, y1 = s.y1;
                s = up;
                s.x1 = x1;
                s.y1 = y1;
            }
            for (auto& s : playTrim_)
            {
                const double x1 = s.x1, y1 = s.y1;
                s = down;
                s.x1 = x1;
                s.y1 = y1;
            }
            trimAt1k = shelfMagnitude(up, 2.0 * std::numbers::pi * 1000.0 / sampleRate_);
        }

        // --- drive scaling with operating-level makeup -------------------------
        // 0 dBFS at nominal drive maps to H = 1.2a: moderate saturation. Tape
        // gain is level-dependent, so unity can only be defined at one
        // operating point: calibrate empirically at -12 dBFS programme level
        // by running a short 1 kHz burst through the same push-pull biased
        // chain on two scratch instances (a few thousand model samples, only
        // on parameter changes). Quieter material reads slightly low, hotter
        // material blooms into compression - like tape.
        // The operating level drives the tape harder; the makeup measures the
        // whole gain, so the output level is kept.
        const double calGain = std::pow(10.0, static_cast<double>(calLevel_.load(std::memory_order_relaxed)) / 20.0);
        calReq_.hScale = drive * 1.2 * 2.2e4 * calGain;
        {
            // -12 dBFS reference, seen through the record emphasis: at 1 kHz
            // the HF emphasis already lifts the level (about +3.7 dB NAB-15),
            // so calibrating with the raw amplitude would leave the whole
            // path low by the differential compression. Align at the level
            // the loop actually sees, like a real machine's record-level
            // alignment.
            const double w1 = 2.0 * std::numbers::pi * 1000.0 * t2;
            const double emph1k = std::sqrt((1.0 + kHiCap * kHiCap * w1 * w1)
                                            / (1.0 + w1 * w1));
            calReq_.calAmp = 0.25 * emph1k * trimAt1k;
            calReq_.drive = drive;
        }

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            auto& rhf = recordHF_[static_cast<size_t>(ch)];
            auto& rlf = recordLF_[static_cast<size_t>(ch)];
            auto& phf = playHF_[static_cast<size_t>(ch)];
            auto& plf = playLF_[static_cast<size_t>(ch)];

            const double sx1 = rhf.x1, sy1 = rhf.y1;
            rhf = makeHFShelf(t2, kHiCap, sampleRate_, false);
            rhf.x1 = sx1; rhf.y1 = sy1;

            const double lx1 = rlf.x1, ly1 = rlf.y1;
            rlf = useLF ? makeLFShelf(tLo, kLoBoost, sampleRate_, false) : ShelfSection {};
            rlf.x1 = lx1; rlf.y1 = ly1;

            const double px1 = phf.x1, py1 = phf.y1;
            phf = makeHFShelf(t2, kHiCap / gHf, sampleRate_, true);
            phf.x1 = px1; phf.y1 = py1;

            const double qx1 = plf.x1, qy1 = plf.y1;
            plf = useLF ? makeLFShelf(tLo, kLoBoost / gLf, sampleRate_, true)
                : (lfTrim != 0.0 ? makeLFShelf(tLo, 1.0 / gLf, sampleRate_, true) : ShelfSection {});
            plf.x1 = qx1; plf.y1 = qy1;

            // The hiss's own de-emphasis, for the circuit (whose reproduce
            // network runs before the tape's noise is laid on).
            auto& hhf = hissHF_[static_cast<size_t>(ch)];
            auto& hlf = hissLF_[static_cast<size_t>(ch)];
            const double hx1 = hhf.x1, hy1 = hhf.y1, lx = hlf.x1, ly = hlf.y1;
            hhf = phf; hhf.x1 = hx1; hhf.y1 = hy1;
            hlf = plf; hlf.x1 = lx; hlf.y1 = ly;
        }

        // Circuit electronics: the record and reproduce networks from their
        // analog transfer functions, at the internal rate, as ladders.
        const bool circuitWanted = electronics_.load(std::memory_order_relaxed) == 1;
        if (circuitWanted)
        {
            const double osRate = sampleRate_ * static_cast<double>(osFactor_);
            const double kLoR = useLF ? kLoBoost : 1.0;
            const double kHiP = kHiCap / gHf, kLoP = kLoR / gLf;
            const AnalogSection rec1 = AnalogSection::make(kHiCap * t2 * tLo, kHiCap * t2 * kLoR + tLo, kLoR,
                                                           t2 * tLo, t2 + tLo, 1.0);
            const AnalogSection play1 = AnalogSection::make(t2 * tLo, t2 + tLo, 1.0,
                                                            kHiP * t2 * tLo, kHiP * t2 * kLoP + tLo, kLoP);
            const double recLimit = std::min(45000.0, 0.45 * osRate), playPole = std::min(50000.0, 0.45 * osRate);
            const double corner = 1.0 / (2.0 * std::numbers::pi * t2);
            const BiquadCoeffs r1 = FilterDesign::bilinear(rec1, osRate, corner);
            const BiquadCoeffs r2 = FilterDesign::bilinear(AnalogSection::lowPass(recLimit), osRate, recLimit);
            const BiquadCoeffs q1 = FilterDesign::bilinear(play1, osRate, corner);
            const BiquadCoeffs q2 = FilterDesign::bilinear(AnalogSection::firstOrderLowPass(playPole), osRate, playPole);
            const int ramp = static_cast<int>(0.05 * osRate);
            for (int ch = 0; ch < numChannels_; ++ch)
            {
                auto& rec = recNet_[static_cast<size_t>(ch)];
                auto& play = playNet_[static_cast<size_t>(ch)];
                if (!circuitActive_)
                {
                    rec[0].setCoeffs(r1); rec[1].setCoeffs(r2); play[0].setCoeffs(q1); play[1].setCoeffs(q2);
                    for (auto& s : rec) s.reset();
                    for (auto& s : play) s.reset();
                }
                else
                {
                    rec[0].rampCoeffs(r1, ramp); rec[1].rampCoeffs(r2, ramp);
                    play[0].rampCoeffs(q1, ramp); play[1].rampCoeffs(q2, ramp);
                }
            }
            if (!circuitActive_)
                for (auto& s : hissHF_) s.clear();
        }
        else if (circuitActive_)
        {
            // Back to the ideal shelves: they start from rest.
            for (auto& s : recordHF_) s.clear();
            for (auto& s : recordLF_) s.clear();
            for (auto& s : playHF_) s.clear();
            for (auto& s : playLF_) s.clear();
        }
        circuitActive_ = circuitWanted;

        // --- loss-effect FIR (63 taps, linear phase) ---------------------------
        // A new speed under the speed transport: the tape keeps moving at the
        // old one and glides to the new one with inertia.
        if (speedTransportActive_ && speedIps_ > 0.0) tapeTransport_.setSpeed(ips);
        else tapeTransport_.setSpeedNow(ips);
        speedIps_ = ips;
        const double v = ips * 0.0254;            // m/s
        // The record head (record-head monitoring) has the wider gap and spacing.
        const bool sync = monitor_.load(std::memory_order_relaxed) == 1;
        const double gap = sync ? 6.0e-6 : 3.0e-6;           // playback head gap (m)
        const double spacing = sync ? 1.0e-6 : 0.5e-6;       // head-tape spacing (m)
        constexpr double thickness = 1.0e-6;      // effective coating depth (m)

        const int grid = firLen_ + 1;             // design grid (64 points at 48 kHz)
        double* mags = designMags_.data();
        for (int kBin = 0; kBin <= grid / 2; ++kBin)
        {
            const double f = kBin * sampleRate_ / grid;
            double mag = 1.0;
            if (f > 1.0)
            {
                const double lambda = v / f;
                const double spacingLoss = std::pow(10.0, -54.6 * (spacing / lambda) / 20.0);
                const double gx = std::numbers::pi * gap / lambda;
                const double gapLoss = (gx < 1e-9) ? 1.0
                    : std::abs(std::sin(gx) / gx);
                const double tx = 4.0 * std::numbers::pi * thickness / lambda;
                const double thickLoss = (tx < 1e-9) ? 1.0 : (1.0 - std::exp(-tx)) / tx;
                mag = spacingLoss * gapLoss * thickLoss;
            }
            mags[kBin] = (1.0 - lossAmt) + lossAmt * mag;
            // Physical reproduce-gap cutoff: no real head reads anything
            // above ~21.5 kHz. Always active (independent of the loss
            // amount); it also removes the last even-order bias
            // intermodulation sidebands just below the base-rate Nyquist.
            // Raised-cosine transition 19.5k -> 21.5k avoids the passband
            // Gibbs ripple of a hard step on the design grid.
            if (f >= 21500.0)
                mags[kBin] = 0.0;
            else if (f > 19500.0)
                mags[kBin] *= 0.5 + 0.5 * std::cos(std::numbers::pi * (f - 19500.0) / 2000.0);
        }
        double* taps = designTaps_.data();
        for (int n = 0; n < firLen_; ++n)
        {
            // Frequency sampling with the cosines taken from a table built in
            // prepare(): the design stays cheap at any FIR length.
            const int offset = n - firCenter_;
            double acc = mags[0];
            for (int kBin = 1; kBin < grid / 2; ++kBin)
            {
                const int idx = ((kBin * offset) % grid + grid) % grid;
                acc += 2.0 * mags[kBin] * cosTable_[static_cast<size_t>(idx)];
            }
            acc += mags[grid / 2] * ((offset & 1) ? -1.0 : 1.0);
            const double hann = 0.5 - 0.5 * cosTable_[static_cast<size_t>((n + 1) % grid)];
            taps[n] = acc * hann / grid;
        }
        // Normalize to exact unity at the 1 kHz calibration frequency so the
        // windowing/gap-cut of the design never shifts the calibrated level.
        {
            const double w1k = 2.0 * std::numbers::pi * 1000.0 / sampleRate_;
            double re = 0.0, im = 0.0;
            for (int n = 0; n < firLen_; ++n)
            {
                re += taps[n] * std::cos(w1k * n);
                im -= taps[n] * std::sin(w1k * n);
            }
            const double g = std::sqrt(re * re + im * im);
            const double norm = (g > 1e-9) ? 1.0 / g : 1.0;
            for (int n = 0; n < firLen_; ++n) taps[n] *= norm;
            firNorm_ = 1.0;
        }
        for (int n = 0; n < firLen_; ++n)
            firTaps_[static_cast<size_t>(firLen_ - 1 - n)] =
                static_cast<T>(taps[n]);              // reversed for dotProduct

        // --- head bump ----------------------------------------------------------
        // The bump sits where the recorded wavelength meets the length of the
        // head's contact with the tape: its frequency is the speed over that
        // length, and each track format has its head (the wider the track,
        // the longer the contact).
        static constexpr double kContactRatio[3] = { 1.0, 0.8, 1.25 };   // half, full and quarter track
        const double bumpHz = kBumpHzPerIps * ips
            * kContactRatio[static_cast<size_t>(std::clamp(trackFormat_.load(std::memory_order_relaxed), 0, 2))];
        const double bumpDb = 2.5 * bumpAmt;
        const auto bc = BiquadCoeffs::makePeak(sampleRate_, bumpHz, 0.9, bumpDb);
        for (auto& b : headBump_)
        {
            const double pz1 = b.z1, pz2 = b.z2;
            b = PeakSection { bc.b0, bc.b1, bc.b2, bc.a1, bc.a2, 0.0, 0.0 };
            b.z1 = pz1;
            b.z2 = pz2;
        }

        // Head contour: the bump and three ripples above it, each a resonance
        // designed on its analog prototype.
        contourActive_ = headModel_.load(std::memory_order_relaxed) == 1;
        if (contourActive_)
        {
            static constexpr double kRatio[kContourSections] = { 1.0, 1.45, 1.95, 2.5 };
            static constexpr double kShare[kContourSections] = { 1.0, -0.5, 0.28, -0.15 };
            static constexpr double kQ[kContourSections] = { 0.9, 2.0, 2.5, 3.0 };
            for (int k = 0; k < kContourSections; ++k)
            {
                const size_t kk = static_cast<size_t>(k);
                const BiquadCoeffs c = FilterDesign::digital(
                    AnalogSection::peak(bumpHz * kRatio[kk], kQ[kk], kContourDb * bumpAmt * kShare[kk]), sampleRate_);
                for (auto& chain : headContour_)
                {
                    auto& s = chain[kk];
                    const double z1 = s.z1, z2 = s.z2;
                    s = PeakSection { c.b0, c.b1, c.b2, c.a1, c.a2, z1, z2 };
                }
            }
        }
        hissNormDirty_ = true;
    }

    /** @brief Starts the drive-makeup calibration for the latest request.
     *  0 dBFS at nominal drive maps to H = 1.2a. Tape gain is
     *  level-dependent, so unity can only be defined at one operating point:
     *  the makeup is measured at -12 dBFS programme level by running a 1 kHz
     *  burst through the same push-pull biased chain on two scratch
     *  instances. Quieter material reads slightly low, hotter material blooms
     *  into compression - like tape. */
    void startCalibration() noexcept
    {
        calRun_ = calReq_;
        const double fs4 = sampleRate_ * static_cast<double>(osFactor_);
        // 16 ms settle + 8 ms measured: the biased loop's mean state needs a
        // few hundred carrier cycles to reach its steady branch.
        calN_ = static_cast<int>(0.024 * fs4);
        calFrom_ = (calN_ * 2) / 3;
        calW_ = 2.0 * std::numbers::pi * 1000.0 / fs4;
        const FormulationParams& fp = calRun_.form;
        calib_.prepare(fs4);
        calib_.setParameters(fp.ms, fp.a, fp.alpha, fp.k, fp.c);
        calib2_.prepare(fs4);
        calib2_.setParameters(fp.ms, fp.a, fp.alpha, fp.k, fp.c);
        calI_ = 0;
        calRe_ = calIm_ = 0.0;
        calMeas_ = 0;
        calActive_ = true;
    }

    /** @brief Runs up to `budget` calibration samples; adopts the result
     *  (drive, makeup and bias together) when the calibration completes. */
    void stepCalibration(int budget) noexcept
    {
        const int stop = std::min(calN_, calI_ + std::max(budget, 0));
        for (; calI_ < stop; ++calI_)
        {
            const int i = calI_;
            const double s = std::sin(calW_ * i);
            const double x = calRun_.hScale * calRun_.calAmp * s;
            const double c = carrier(calRun_.biasAmp, calRun_.biasNominal, i & 7);
            const double m = 0.5
                * (static_cast<double>(calib_.processSample(static_cast<T>(x + c)))
                 + static_cast<double>(calib2_.processSample(static_cast<T>(x - c))));
            // Measure the FUNDAMENTAL of the averaged pair (Goertzel-style
            // correlation), not the raw RMS: the raw output still carries the
            // even-order carrier residue that the downsampler removes in the
            // real path, and it would inflate the measurement.
            if (i >= calFrom_)
            {
                calRe_ += m * std::cos(calW_ * i);
                calIm_ += m * s;
                ++calMeas_;
            }
        }
        if (calI_ < calN_) return;
        const double fund = 2.0 * std::sqrt(calRe_ * calRe_ + calIm_ * calIm_)
                          / std::max(1, calMeas_);
        // Partial loudness link: a calibration that pins the reference level
        // EXACTLY kills the drive knob (inaudible below 0 dB where tape stays
        // clean, a pure attenuator above as compression eats level). A
        // +0.25 dB/dB residual slope keeps it alive: backing off cleans AND
        // drops slightly, pushing holds level while the tape density grows.
        mScale_ = ((fund > 0.0) ? calRun_.calAmp / fund : 1.0) * std::pow(calRun_.drive, 0.25);
        hScale_ = calRun_.hScale;
        biasAmp_ = calRun_.biasAmp;
        calActive_ = false;
        if (calRestart_)
        {
            calRestart_ = false;
            startCalibration();
        }
    }

    /** @brief Shared transport modulation in samples: drift + wow + flutter + scrape. */
    [[nodiscard]] double nextTransportMod(double depth) noexcept
    {
        // Always advance the transport state so engaging the control later
        // does not jump phases; only the output is scaled by depth.
        const double dt = 1.0 / sampleRate_;
        modPhaseWow_ += 0.55 * dt;
        if (modPhaseWow_ >= 1.0) modPhaseWow_ -= 1.0;
        modPhaseFlut_ += 8.3 * dt;
        if (modPhaseFlut_ >= 1.0) modPhaseFlut_ -= 1.0;
        modPhaseFlut2_ += 23.0 * dt;
        if (modPhaseFlut2_ >= 1.0) modPhaseFlut2_ -= 1.0;

        rng_ = rng_ * 1664525u + 1013904223u;
        const double n = static_cast<double>(rng_ >> 8) / 8388608.0 - 1.0;

        // Slow drift: heavily low-passed random walk, clamped to +/-18 samples.
        driftState_ = driftA_ * driftState_ + (1.0 - driftA_) * n * 600.0;
        const double drift = std::clamp(driftState_, -18.0, 18.0);

        // Scrape band (~40-90 Hz): difference of two one-poles on noise.
        scrapeLp1_ = scrapeA1_ * scrapeLp1_ + (1.0 - scrapeA1_) * n;
        scrapeLp2_ = scrapeA2_ * scrapeLp2_ + (1.0 - scrapeA2_) * n;
        const double scrape = (scrapeLp1_ - scrapeLp2_) * 0.6;

        if (depth <= 0.0)
            return 0.0;

        const double twoPi = 2.0 * std::numbers::pi;
        // Component amplitudes in delay samples at 48k, scaled to the actual
        // rate (the delay centre scales identically). The 0.4 factor calibrates
        // full depth to ~0.6 % peak-to-peak measured pitch deviation (a worn
        // machine); the 0.15 default lands near healthy-transport spec.
        const double rateScale = sampleRate_ / 48000.0;
        const double wow   = 27.8 * std::sin(twoPi * modPhaseWow_);
        const double flut  = 0.55 * std::sin(twoPi * modPhaseFlut_);
        const double flut2 = 0.066 * std::sin(twoPi * modPhaseFlut2_);

        return 0.4 * depth * rateScale * (drift + wow + flut + flut2 + scrape);
    }

    // -- Members --------------------------------------------------------------------
    static constexpr int kFirLenAt48k = 63;
    static constexpr double kGainRampSeconds = 0.03;
    static constexpr double kUniformToUnitRms = 1.7320508075688772;  ///< sqrt(3)
    static constexpr int kContourSections = 4;
    static constexpr double kContourDb = 3.0;              ///< Peak of the contour's bump at full head bump.
    static constexpr double kAsymmetryShare = 0.15;        ///< Even term at full asymmetry.
    static constexpr double kTransformerCornerHz = 60.0;
    static constexpr double kInputCouplingHz = 4.0;      ///< Corner of the AC-coupled input.
    static constexpr double kOutputCouplingHz = 24.0;    ///< Corner of the AC-coupled playback amplifier.
    static constexpr double kIronCornerHz = 20.0;        ///< Corner of the iron's low end at the full amount.
    static constexpr double kUnderBiasLiftDb = 4.5;      ///< Top-end lift with no bias at all.
    static constexpr double kUnderBiasCentreHz = 6000.0; ///< Its centre at 15 ips.
    static constexpr double kEmphasisTrimHz = 4000.0;    ///< Centre of the record emphasis trim.
    static constexpr double kBumpHzPerIps = 6.0;         ///< Head bump of the half-track head: 90 Hz at 15 ips.
    static constexpr double kTransformerEven = 0.03;       ///< Terms of the transformer's curve.
    static constexpr double kTransformerOdd = 0.2;
    static constexpr double kHumNorm = 1.2649110640673518; ///< 1 / RMS of sin + sin(2x) / 2.

    AudioSpec spec_ {};              ///< Last valid spec, for setOversampling re-prepare.
    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    int maxBlock_ = 0;
    std::atomic<bool> prepared_ { false };
    int latency_ = 0;
    int drySize_ = 1;
    int osRequested_ = 4;            ///< The factor asked for (setOversampling(), the automatic one).
    bool fourLevel_ = false;         ///< The levels of the bias carrier (setBiasShape(); setup thread).
    double biasNominal_ = 3.0 * 2.2e4; ///< The bias at the optimum of the material.
    static constexpr double kFourLevelInternalRate = 352800.0;
    static constexpr double kInnerLevel = 0.2, kOuterLevel = 0.6; ///< Of the bias amplitude, inner level up to the optimum.
    int osFactor_ = 4;               ///< Oversampling factor (setup thread; 1 = off, 4 default).
    bool osAutomatic_ = false;       ///< The factor follows the host rate (setAutomaticOversampling()).
    OversamplingMode osMode_ = OversamplingMode::LinearPhase; ///< setOversamplingMode() (setup thread).
    OversamplingDryPath<T> dryPath_; ///< The dry copy through minimum-phase filters.
    AudioBuffer<T> dryScratch_;      ///< The dry block on its way through dryPath_.
    detail::DryPhase dryInputLead_, dryOutputLead_, dryIronLead_; ///< The phase of the high-passes on the dry side.
    bool dryThroughFilters_ = false; ///< The dry copy crosses the filters (minimum phase).
    int dryDelay_ = 0;               ///< Delay of the dry ring: what the filters do not bring.
    static constexpr double kAutomaticInternalRate = 176400.0; ///< Rate the automatic factor reaches.

    std::unique_ptr<Oversampling<T>> oversampler_;
    std::vector<Hysteresis<T>> hysteresis_;   ///< +carrier instances (push-pull pair A).
    std::vector<Hysteresis<T>> hysteresisN_;  ///< -carrier instances (push-pull pair B).
    Hysteresis<T> calib_;    ///< Scratch +carrier instance for makeup calibration.
    Hysteresis<T> calib2_;   ///< Scratch -carrier instance for makeup calibration.

    /// AC bias carrier at a quarter of the internal rate, sampled 45 degrees
    /// off its zero crossings: sqrt(2) * sin(pi * k / 2 + pi / 4), so every
    /// sample sits at the carrier amplitude; two periods per 8-phase cycle,
    /// shared by all channels like a machine's single bias oscillator. At a
    /// quarter of the rate every harmonic the hysteresis draws from the
    /// carrier folds back onto 0, the carrier itself or the internal Nyquist
    /// frequency, so its products with the programme itself stay a quarter of
    /// the internal rate, less the audio band, away from 0 - above what the
    /// downsampler keeps. Products of the programme's own HARMONICS with twice
    /// the carrier are not covered by that argument: with bright material and
    /// drive they reach the band (measured at the default 4x, 48 kHz: a
    /// 19.25 kHz tone at -6 dBFS leaves a 255 Hz product (5f - 2fc) at
    /// -46 dBc at drive 0 and -6 dBc at +12 dB; 8x moves it to -98 dBc). At 0.375 of the rate its third harmonic folded
    /// onto the base-rate Nyquist frequency and mirrored the programme about
    /// a quarter of the base rate: a 10.1 kHz tone at -30 dBFS came out with
    /// a 13.9 kHz image 9.9 dB down, and 80 dB down now.
    static constexpr double kBiasTable[8] = {
        1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0
    };

    std::vector<ShelfSection> recordHF_, recordLF_, playHF_, playLF_;
    std::vector<PeakSection> headBump_;
    std::vector<PeakSection> outHp_;        ///< AC-coupling 24 Hz high-pass.
    std::vector<double> overBiasLp_;        ///< Over-bias self-erasure LP output state.
    std::vector<double> overBiasX1_;        ///< Over-bias LP input state.
    std::vector<std::array<double, 4>> inputAc_;   ///< Input coupling: two one-pole high-passes (input and output of each).
    double inputAcPole_ = 0.0;                     ///< Their pole.
    std::vector<ShelfSection> recTrim_, playTrim_; ///< The emphasis trim: a shelf and its inverse.
    std::vector<double> ironLp_;                   ///< The iron's low end: the low-pass taken off.
    double ironA_ = 0.0;                           ///< Its coefficient.
    bool resting_ = false;                         ///< The machine did not run in the last chunk (audio thread).
    int wakeHold_ = 0;                             ///< Frames left of the warm-up after a rest.

    std::vector<T> firTaps_;
    std::vector<std::vector<T>> firState_;
    std::vector<double> cosTable_, designMags_, designTaps_;  ///< FIR design scratch (prepare).
    int firLen_ = kFirLenAt48k;
    int firCenter_ = (kFirLenAt48k - 1) / 2;
    int firHalf_ = 64;                      ///< Half of the double-write mirrored ring.
    int firPos_ = 0;

    std::vector<std::vector<T>> delayRing_;
    int delayPos_ = 0;
    int delayCenter_ = 96;
    int delayMask_ = 255;
    double driftA_ = 0.0, scrapeA1_ = 0.0, scrapeA2_ = 0.0;

    std::vector<std::vector<T>> dryRing_;
    int dryPos_ = 0;

    double hScale_ = 1.0, mScale_ = 1.0;   ///< Calibrated targets.
    double biasAmp_ = 3.0 * 2.2e4;   ///< Carrier amplitude target in H units.
    double hSm_ = 1.0, mSm_ = 1.0, bSm_ = 3.0 * 2.2e4;   ///< Ramped values (audio thread).
    bool overBiasOn_ = false;
    bool overBiasActive_ = false;   ///< overBiasOn_ as of the previous block (audio thread).
    double obB0_ = 1.0, obB1_ = 0.0, obP_ = 0.0;   ///< Over-bias LP coefficients.

    struct CalibrationRequest
    {
        double hScale = 1.0, biasAmp = 3.0 * 2.2e4, biasNominal = 3.0 * 2.2e4, calAmp = 0.25, drive = 1.0;
        FormulationParams form { 3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17, 0.0 };
    };
    CalibrationRequest calReq_, calRun_;
    bool calActive_ = false, calRestart_ = false;
    int calN_ = 0, calFrom_ = 0, calI_ = 0, calMeas_ = 0;
    double calW_ = 0.0, calRe_ = 0.0, calIm_ = 0.0;
    int biasPhase_ = 0;              ///< Shared carrier phase (audio thread).

    double modPhaseWow_ = 0.0, modPhaseFlut_ = 0.0, modPhaseFlut2_ = 0.0;
    double driftState_ = 0.0, scrapeLp1_ = 0.0, scrapeLp2_ = 0.0;
    uint32_t rng_ = 0x1357feedu;       ///< Transport modulation stream.
    uint32_t rngNoise_ = 0x2468beefu;  ///< Hiss stream (independent of transport).

    std::atomic<T> driveDb_ { T(0) };
    std::atomic<T> bias_ { T(0.5) };
    std::atomic<int> speed_ { static_cast<int>(Speed::IPS_15) };
    std::atomic<int> standard_ { static_cast<int>(Standard::NAB) };
    std::atomic<T> loss_ { T(0.5) };
    std::atomic<T> headBumpAmt_ { T(0.5) };
    std::atomic<T> wowFlutter_ { T(0.15) };
    std::atomic<T> noiseDb_ { T(-200) };
    std::atomic<T> mix_ { T(1) };
    T currentMix_ = T(1);            ///< Audio-thread mix ramp state (exact landing).
    T mixMaxStep_ = T(1.0 / 960.0);  ///< Mix ramp rate: full scale per 20 ms.
    std::atomic<bool> dirty_ { true };

    // The machine around the tape (settings).
    std::atomic<int> transport_ { 0 }, mains_ { 0 }, headModel_ { 0 }, electronics_ { 0 };
    std::atomic<int> formulation_ { 0 }, trackFormat_ { 0 }, monitor_ { 0 };
    std::atomic<T> humDb_ { T(-200) }, channelBleedDb_ { T(-200) }, evenHarmonics_ { T(0) }, outputIron_ { T(0) };
    std::atomic<T> alignHf_ { T(0) }, alignLf_ { T(0) }, calLevel_ { T(0) }, recordEmphasis_ { T(0) };

    // ... and its state (audio thread).
    bool speedTransportActive_ = false, circuitActive_ = false, contourActive_ = false, hissNormDirty_ = true;
    int formApplied_ = -1;
    double speedIps_ = 0.0, lastMod_ = 0.0;
    TapeTransport<T> tapeTransport_;
    double humPhase_ = 0.0, hissNorm_ = 1.0, firNorm_ = 1.0, xtNorm_ = 1.0;
    double asymMeanA_ = 0.0, trMeanA_ = 0.0;
    LinearRamp hiss_, hum_, xt_, tr_, asym_, monitorRamp_ { 1.0, 1.0, 0.0 };
    std::vector<double> wet_, asymMean_, trMean_;
    std::vector<ShelfSection> hissHF_, hissLF_, xtHp_, trPre_, trPost_;
    std::vector<std::array<PeakSection, kContourSections>> headContour_;
    std::vector<std::array<NormalizedLadder<double, 1>, 2>> recNet_, playNet_;
};

} // namespace dspark
