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
 * 2. **Magnetic hysteresis with REAL AC bias** at 4x oversampling: an
 *    ultrasonic carrier (0.375 * internal rate, exact 8-phase table) is
 *    summed with the signal into a push-pull pair of JA instances per
 *    channel (+carrier / -carrier, output averaged, like a centre-tapped
 *    record head). The carrier erases the loop's branch memory exactly as
 *    hardware bias does, the push-pull cancels the carrier, its odd
 *    harmonics and the remanence DC, and the even folds die in the
 *    downsampler. `bias` maps to carrier amplitude: under-bias drops below
 *    the erase threshold (real grit and level instability), over-bias adds
 *    the self-erasure roll-off of short wavelengths. Gain is calibrated
 *    per setting through the same biased chain, so drive changes
 *    saturation, not level.
 * 3. **Playback loss effects** with the physical formulas (Kadis): spacing
 *    loss 54.6*d/lambda dB, gap loss sinc(pi*g/lambda), thickness loss
 *    (1-e^(-4*pi*delta/lambda))/(4*pi*delta/lambda), with lambda =
 *    speed/frequency - rendered into a 63-tap linear-phase FIR per speed,
 *    plus the speed-dependent head-bump resonance (45/90/180 Hz at
 *    7.5/15/30 ips).
 * 4. **Wow & flutter**: one shared transport modulation (slow drift random
 *    walk + 0.55 Hz wow + 8.3/23 Hz flutter + scrape band) driving a
 *    fractional delay - identical on all channels, like a real capstan. The
 *    hiss generator uses its own RNG stream, so enabling noise never changes
 *    the transport's realisation.
 * 5. **Tape hiss** (optional, default off).
 * 6. **Physical band edges**: a 2nd-order 24 Hz high-pass models the
 *    AC-coupled playback amplifier (kills subsonics and any residual DC,
 *    like the hardware's LF roll-off), and the loss FIR always applies the
 *    hard reproduce-gap cutoff above ~21.5 kHz (no real head reads shorter
 *    wavelengths; it also removes the last even-order bias intermodulation
 *    sidebands near the base-rate Nyquist).
 *
 * The dry path of the mix control is delay-compensated to getLatency(). The
 * mix is applied per block without smoothing: the wet stream is correlated
 * with and aligned to the dry, so a mix step moves the output by the (small)
 * timbral difference only - measured below the steady-state sample delta.
 * Channels beyond the prepared count pass through untouched.
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored; parameter changes are
 * published with a release store and consumed at the next block). prepare()
 * is setup-thread only (allocates; invalid specs are ignored and an
 * unprepared instance passes audio through). reset() belongs to the stream
 * owner. getState()/setState() are setup/UI threads. The recompute on a
 * parameter change runs a short scratch calibration on the audio thread
 * (allocation-free, a few thousand samples of model time).
 *
 * Cost: the push-pull biased core runs two JA solvers at 4x, ~11% of one
 * core at 48 kHz stereo on a desktop CPU - the price of physical AC bias
 * (history-independent LF response, exact even-harmonic cancellation,
 * ultrasonic residue below -75 dB). A SIMD lane-parallel JA (2 channels x
 * 2 polarities) is the noted future optimisation if a lighter budget is
 * ever needed.
 *
 * Dependencies: Core/Hysteresis.h, Core/Oversampling.h, Core/Biquad.h,
 * Core/SimdOps.h, Core/AudioSpec.h, Core/AudioBuffer.h, Core/DspMath.h,
 * Core/DenormalGuard.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Hysteresis.h"
#include "../Core/Oversampling.h"
#include "../Core/SimdOps.h"
#include "../Core/StateBlob.h"

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

    /** @brief Tape speed. */
    enum class Speed { IPS_7_5, IPS_15, IPS_30 };

    // -- Lifecycle ---------------------------------------------------------------

    /** @brief Allocates the whole chain. Invalid specs (non-positive or
     *  non-finite rate, block size or channel count) are ignored: the previous
     *  state is kept and an unprepared instance stays pass-through. */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        maxBlock_ = std::max(spec.maxBlockSize, 1);

        // The hysteresis core runs at 4x so the 0.375 * internal-rate AC bias
        // carrier and its sidebands stay clear of the audio band.
        oversampler_ = std::make_unique<Oversampling<T>>(4, Oversampling<T>::Quality::High);
        oversampler_->prepare(spec);

        // Push-pull pair per channel: +carrier and -carrier instances, output
        // averaged. Odd-in-bias terms (the carrier, its 3rd harmonic fold and
        // the loop's remanence DC) cancel exactly, as in the centre-tapped
        // record-head circuits of real machines.
        hysteresis_.assign(static_cast<size_t>(numChannels_), {});
        hysteresisN_.assign(static_cast<size_t>(numChannels_), {});
        for (auto& h : hysteresis_)
        {
            h.prepare(sampleRate_ * 4.0);
            h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
        }
        for (auto& h : hysteresisN_)
        {
            h.prepare(sampleRate_ * 4.0);
            h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
        }

        recordHF_.assign(static_cast<size_t>(numChannels_), {});
        recordLF_.assign(static_cast<size_t>(numChannels_), {});
        playHF_.assign(static_cast<size_t>(numChannels_), {});
        playLF_.assign(static_cast<size_t>(numChannels_), {});
        headBump_.assign(static_cast<size_t>(numChannels_), {});
        overBiasLp_.assign(static_cast<size_t>(numChannels_), 0.0);

        // AC-coupled playback amplifier: fixed 2nd-order 24 Hz high-pass
        // (real machines roll off there; also blocks remanence DC).
        outHp_.assign(static_cast<size_t>(numChannels_), {});
        {
            const auto hc = BiquadCoeffs<double>::makeHighPass(sampleRate_, 24.0, 0.707);
            for (auto& h : outHp_)
                h = PeakSection { hc.b0, hc.b1, hc.b2, hc.a1, hc.a2, 0.0, 0.0 };
        }

        firState_.assign(static_cast<size_t>(numChannels_),
                         std::vector<T>(static_cast<size_t>(kFirRing), T(0)));
        firPos_ = 0;
        firTaps_.assign(static_cast<size_t>(kFirLen), T(0));

        // Transport delay scales with the rate so the wow excursion (sized in
        // samples at 48k, scaled by fs/48k) always fits under the centre.
        delayCenter_ = std::max(96, static_cast<int>(96.0 * sampleRate_ / 48000.0));
        int dsz = 1;
        while (dsz < 2 * delayCenter_ + 8) dsz <<= 1;
        delayMask_ = dsz - 1;
        delayRing_.assign(static_cast<size_t>(numChannels_),
                          std::vector<T>(static_cast<size_t>(dsz), T(0)));
        delayPos_ = 0;

        // Per-sample modulation smoothing constants (precomputed once).
        const double dt = 1.0 / sampleRate_;
        driftA_ = std::exp(-2.0 * std::numbers::pi * 0.10 * dt);
        scrapeA1_ = std::exp(-2.0 * std::numbers::pi * 90.0 * dt);
        scrapeA2_ = std::exp(-2.0 * std::numbers::pi * 40.0 * dt);

        latency_ = oversampler_->getLatency() + kFirCenter + delayCenter_;
        drySize_ = 1;
        while (drySize_ < latency_ + maxBlock_ + 1) drySize_ <<= 1;
        dryRing_.assign(static_cast<size_t>(numChannels_),
                        std::vector<T>(static_cast<size_t>(drySize_), T(0)));
        dryPos_ = 0;

        prepared_.store(true, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
        reset();
    }

    /** @brief Clears all signal state (keeps parameters). RT-safe. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        for (auto& h : hysteresis_) h.reset();
        for (auto& h : hysteresisN_) h.reset();
        for (auto& f : firState_) std::fill(f.begin(), f.end(), T(0));
        for (auto& d : delayRing_) std::fill(d.begin(), d.end(), T(0));
        for (auto& d : dryRing_) std::fill(d.begin(), d.end(), T(0));
        for (auto& s : recordHF_) s.clear();
        for (auto& s : recordLF_) s.clear();
        for (auto& s : playHF_) s.clear();
        for (auto& s : playLF_) s.clear();
        for (auto& v : overBiasLp_) v = 0.0;
        // Clear STATE only: wiping coefficients here used to leave the head
        // bump at identity until the next parameter change re-ran recompute.
        for (auto& b : headBump_) b.clear();
        for (auto& h : outHp_) h.clear();
        firPos_ = 0;
        delayPos_ = 0;
        dryPos_ = 0;
        biasPhase_ = 0;
        modPhaseWow_ = 0.0;
        modPhaseFlut_ = 0.0;
        modPhaseFlut2_ = 0.0;
        driftState_ = 0.0;
        scrapeLp1_ = scrapeLp2_ = 0.0;
        rng_ = 0x1357feedu;
        rngNoise_ = 0x2468beefu;
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
     *  under-biased machine. Above nominal, self-erasure progressively rolls
     *  off the top octave. Non-finite values are ignored. */
    void setBias(T bias) noexcept
    {
        if (!std::isfinite(bias)) return;
        bias_.store(std::clamp(bias, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Tape speed (changes EQ time constants, losses and head bump).
     *  Out-of-range values are clamped. */
    void setSpeed(Speed s) noexcept
    {
        const int v = std::clamp(static_cast<int>(s),
                                 static_cast<int>(Speed::IPS_7_5),
                                 static_cast<int>(Speed::IPS_30));
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

    /** @brief Tape hiss level in dBFS (e.g. -55 for audible vintage hiss);
     *  values <= -120 disable it (default). Non-finite values are ignored. */
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

    /** @brief Total latency in samples (oversampler + loss FIR + transport delay). */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

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

        // Acquire pairs with the setters' release stores so the recompute
        // always sees the values published before the flag.
        if (dirty_.load(std::memory_order_relaxed)
            && dirty_.exchange(false, std::memory_order_acquire))
            recompute();

        const T mixVal = mix_.load(std::memory_order_relaxed);
        const double noiseAmp = std::pow(10.0, static_cast<double>(
            noiseDb_.load(std::memory_order_relaxed)) / 20.0);
        const bool noiseOn = noiseDb_.load(std::memory_order_relaxed) > T(-120);
        const double wfDepth = static_cast<double>(wowFlutter_.load(std::memory_order_relaxed));

        // 1. Dry snapshot for the latency-compensated mix.
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* in = buffer.getChannel(ch);
            auto& dry = dryRing_[static_cast<size_t>(ch)];
            int dp = dryPos_;
            for (int i = 0; i < nS; ++i)
            {
                dry[static_cast<size_t>(dp)] = in[i];
                dp = (dp + 1) & (drySize_ - 1);
            }
        }

        // 2. Record EQ (emphasis) at base rate.
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            auto& hf = recordHF_[static_cast<size_t>(ch)];
            auto& lf = recordLF_[static_cast<size_t>(ch)];
            for (int i = 0; i < nS; ++i)
                d[i] = static_cast<T>(hf.process(lf.process(static_cast<double>(d[i]))));
        }

        // 3. Push-pull AC-bias hysteresis at 4x. One shared carrier phase for
        // all channels (a machine has a single bias oscillator); each channel
        // runs a +carrier and a -carrier JA instance and averages them, which
        // cancels every odd-in-bias term exactly (carrier, its folded 3rd
        // harmonic, remanence DC) like a centre-tapped record head.
        {
            auto osView = oversampler_->upsample(buffer);
            const int osN = osView.getNumSamples();
            const int phaseStart = biasPhase_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = osView.getChannel(ch);
                auto& hp = hysteresis_[static_cast<size_t>(ch)];
                auto& hn = hysteresisN_[static_cast<size_t>(ch)];
                const double inScale = hScale_;
                const double outScale = mScale_ * 0.5;
                const double B = biasAmp_;
                int phase = phaseStart;
                for (int i = 0; i < osN; ++i)
                {
                    const double x = inScale * static_cast<double>(d[i]);
                    const double c = B * kBiasTable[static_cast<size_t>(phase)];
                    phase = (phase + 1) & 7;
                    const double mp = static_cast<double>(hp.processSample(static_cast<T>(x + c)));
                    const double mn = static_cast<double>(hn.processSample(static_cast<T>(x - c)));
                    d[i] = static_cast<T>(outScale * (mp + mn));
                }
            }
            biasPhase_ = (phaseStart + osN) & 7;
            oversampler_->downsample(buffer);
        }

        // 4-6. Loss FIR + head bump, transport modulation, playback EQ, noise, mix.
        for (int i = 0; i < nS; ++i)
        {
            // One shared transport modulation per frame (all channels move
            // together, like tape past a single capstan).
            const double mod = nextTransportMod(wfDepth);
            const double readPos = static_cast<double>(delayCenter_) + mod;
            const auto readInt = static_cast<int>(std::floor(readPos));
            const T frac = static_cast<T>(readPos - readInt);

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = buffer.getChannel(ch);

                // Loss FIR (linear phase, 63 taps) on a per-channel ring.
                auto& fir = firState_[static_cast<size_t>(ch)];
                fir[static_cast<size_t>(firPos_)] = d[i];
                fir[static_cast<size_t>(firPos_ + kFirRing / 2)] = d[i]; // mirrored
                const T* win = &fir[static_cast<size_t>(firPos_ + kFirRing / 2 - (kFirLen - 1))];
                T y = simd::dotProduct(firTaps_.data(), win, kFirLen);

                // Head bump resonance.
                y = headBump_[static_cast<size_t>(ch)].process(y);

                // Transport (wow & flutter) fractional delay: interpolate at
                // t - readInt - frac, i.e. between p1 = x(t-readInt) and
                // p2 = x(t-readInt-1), with Catmull-Rom neighbours around them.
                auto& dl = delayRing_[static_cast<size_t>(ch)];
                dl[static_cast<size_t>(delayPos_)] = y;
                const int base = delayPos_ - readInt;
                const int dm = delayMask_;
                const T p0 = dl[static_cast<size_t>((base + 1) & dm)];
                const T p1 = dl[static_cast<size_t>(base & dm)];
                const T p2 = dl[static_cast<size_t>((base - 1) & dm)];
                const T p3 = dl[static_cast<size_t>((base - 2) & dm)];
                T w = p1 + T(0.5) * frac * (p2 - p0
                      + frac * (T(2) * p0 - T(5) * p1 + T(4) * p2 - p3
                      + frac * (T(3) * (p1 - p2) + p3 - p0)));

                // Playback EQ (exact inverse de-emphasis).
                w = static_cast<T>(playLF_[static_cast<size_t>(ch)].process(
                        playHF_[static_cast<size_t>(ch)].process(static_cast<double>(w))));

                // Over-bias self-erasure: wider recording zone partially
                // erases short wavelengths (one-pole LP engaged above nominal
                // bias only; identity at or below nominal).
                if (overBiasA_ > 0.0)
                {
                    auto& lp = overBiasLp_[static_cast<size_t>(ch)];
                    lp += overBiasA_ * (static_cast<double>(w) - lp);
                    w = static_cast<T>(lp);
                }

                // AC-coupled playback amplifier (2nd-order 24 Hz high-pass:
                // subsonic/DC roll-off of the real hardware).
                w = outHp_[static_cast<size_t>(ch)].process(w);

                // Hiss (own RNG stream: enabling it must not change the
                // transport modulation's realisation).
                if (noiseOn)
                {
                    rngNoise_ = rngNoise_ * 1664525u + 1013904223u;
                    const double n1 = static_cast<double>(rngNoise_ >> 8) / 8388608.0 - 1.0;
                    w += static_cast<T>(noiseAmp * n1 * 0.35);
                }

                // Latency-compensated mix.
                const auto& dry = dryRing_[static_cast<size_t>(ch)];
                const int dryIdx = (dryPos_ + i - latency_) & (drySize_ - 1);
                const T drySample = dry[static_cast<size_t>(dryIdx)];
                d[i] = drySample + (w - drySample) * mixVal;
            }

            firPos_ = (firPos_ + 1) & (kFirRing / 2 - 1);
            delayPos_ = (delayPos_ + 1) & delayMask_;
        }
        dryPos_ = (dryPos_ + nS) & (drySize_ - 1);
    }

private:
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

    /** @brief Rebuilds everything affected by drive/bias/speed/standard/loss. */
    void recompute() noexcept
    {
        const double driveDbV = static_cast<double>(driveDb_.load(std::memory_order_relaxed));
        const double drive = std::pow(10.0, driveDbV / 20.0);
        const double bias = static_cast<double>(bias_.load(std::memory_order_relaxed));
        const auto speed = static_cast<Speed>(speed_.load(std::memory_order_relaxed));
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
        const double biasB = std::min(6.0, 3.0 * std::pow(9.0, bias - 0.5));
        biasAmp_ = biasB * 2.2e4;
        if (biasB > 3.5)
        {
            // Gentle self-erasure: ~-1.5 dB at 10 kHz per +1 B/a over nominal
            // (one-pole corner gliding 22 kHz -> ~12.8 kHz at full over-bias).
            const double fc = 22000.0 * (3.5 / biasB);
            overBiasA_ = 1.0 - std::exp(-2.0 * std::numbers::pi * fc / sampleRate_);
        }
        else
        {
            overBiasA_ = 0.0;
        }

        // --- EQ time constants per standard and speed --------------------------
        double t2 = 50e-6;                 // HF time constant
        bool useLF = false;                // NAB LF constant 3180 us
        switch (standard)
        {
        case Standard::NAB:
            t2 = (speed == Speed::IPS_30) ? 17.5e-6 : 50e-6;
            useLF = speed != Speed::IPS_30;
            break;
        case Standard::CCIR:
            t2 = (speed == Speed::IPS_7_5) ? 70e-6
               : (speed == Speed::IPS_15) ? 35e-6 : 17.5e-6;
            useLF = false;
            break;
        }
        constexpr double kHiCap = 4.0;     // +12 dB emphasis cap (practical alignment)
        constexpr double kLoBoost = 2.0;   // +6 dB NAB LF record boost
        const double tLo = 3180e-6;

        // --- drive scaling with operating-level makeup -------------------------
        // 0 dBFS at nominal drive maps to H = 1.2a: moderate saturation. Tape
        // gain is level-dependent, so unity can only be defined at one
        // operating point: calibrate empirically at -12 dBFS programme level
        // by running a short 1 kHz burst through the same push-pull biased
        // chain on two scratch instances (a few thousand model samples, only
        // on parameter changes). Quieter material reads slightly low, hotter
        // material blooms into compression - like tape.
        hScale_ = drive * 1.2 * 2.2e4;
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
            const double kCalAmp = 0.25 * emph1k;
            const double fs4 = sampleRate_ * 4.0;
            // 16 ms settle + 8 ms measured: the biased loop's mean state
            // needs a few hundred carrier cycles to reach its steady branch.
            const int calN = static_cast<int>(0.024 * fs4);
            const int calFrom = (calN * 2) / 3;
            calib_.prepare(fs4);
            calib_.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
            calib2_.prepare(fs4);
            calib2_.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
            // Measure the FUNDAMENTAL of the averaged pair (Goertzel-style
            // correlation), not the raw RMS: the raw output still carries the
            // even-order carrier residue that the downsampler removes in the
            // real path, and it would inflate the measurement.
            const double wCal = 2.0 * std::numbers::pi * 1000.0 / fs4;
            double outRe = 0.0, outIm = 0.0;
            int meas = 0;
            for (int i = 0; i < calN; ++i)
            {
                const double s = std::sin(wCal * i);
                const double x = hScale_ * kCalAmp * s;
                const double c = biasAmp_ * kBiasTable[static_cast<size_t>(i & 7)];
                const double m = 0.5
                    * (static_cast<double>(calib_.processSample(static_cast<T>(x + c)))
                     + static_cast<double>(calib2_.processSample(static_cast<T>(x - c))));
                if (i >= calFrom)
                {
                    outRe += m * std::cos(wCal * i);
                    outIm += m * std::sin(wCal * i);
                    ++meas;
                }
            }
            const double fund = 2.0 * std::sqrt(outRe * outRe + outIm * outIm)
                              / std::max(1, meas);
            mScale_ = (fund > 0.0) ? kCalAmp / fund : 1.0;

            // Partial loudness link: the per-drive calibration above pins the
            // reference level EXACTLY, which kills the drive knob (inaudible
            // below 0 dB where tape stays clean, a pure attenuator above as
            // compression eats level). A +0.25 dB/dB residual slope keeps it
            // alive: backing off cleans AND drops slightly, pushing holds
            // level while the tape density grows.
            mScale_ *= std::pow(drive, 0.25);
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
            phf = makeHFShelf(t2, kHiCap, sampleRate_, true);
            phf.x1 = px1; phf.y1 = py1;

            const double qx1 = plf.x1, qy1 = plf.y1;
            plf = useLF ? makeLFShelf(tLo, kLoBoost, sampleRate_, true) : ShelfSection {};
            plf.x1 = qx1; plf.y1 = qy1;
        }

        // --- loss-effect FIR (63 taps, linear phase) ---------------------------
        const double ips = (speed == Speed::IPS_7_5) ? 7.5
                         : (speed == Speed::IPS_15) ? 15.0 : 30.0;
        const double v = ips * 0.0254;            // m/s
        constexpr double gap = 3.0e-6;            // playback head gap (m)
        constexpr double spacing = 0.5e-6;        // head-tape spacing (m)
        constexpr double thickness = 1.0e-6;      // effective coating depth (m)

        constexpr int kGrid = kFirLen + 1;        // 64-point design grid
        double mags[kGrid / 2 + 1];
        for (int kBin = 0; kBin <= kGrid / 2; ++kBin)
        {
            const double f = kBin * sampleRate_ / kGrid;
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
            // Gibbs ripple of a hard step on the 64-point design grid.
            if (f >= 21500.0)
                mags[kBin] = 0.0;
            else if (f > 19500.0)
                mags[kBin] *= 0.5 + 0.5 * std::cos(std::numbers::pi * (f - 19500.0) / 2000.0);
        }
        double taps[kFirLen];
        for (int n = 0; n < kFirLen; ++n)
        {
            double acc = mags[0];
            for (int kBin = 1; kBin < kGrid / 2; ++kBin)
                acc += 2.0 * mags[kBin]
                     * std::cos(2.0 * std::numbers::pi * kBin * (n - kFirCenter)
                                / static_cast<double>(kGrid));
            acc += mags[kGrid / 2] * std::cos(std::numbers::pi * (n - kFirCenter));
            const double hann = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * (n + 1)
                                                     / (kFirLen + 1));
            taps[n] = acc * hann / kGrid;
        }
        // Normalize to exact unity at the 1 kHz calibration frequency so the
        // windowing/gap-cut of the design never shifts the calibrated level.
        {
            const double w1k = 2.0 * std::numbers::pi * 1000.0 / sampleRate_;
            double re = 0.0, im = 0.0;
            for (int n = 0; n < kFirLen; ++n)
            {
                re += taps[n] * std::cos(w1k * n);
                im -= taps[n] * std::sin(w1k * n);
            }
            const double g = std::sqrt(re * re + im * im);
            const double norm = (g > 1e-9) ? 1.0 / g : 1.0;
            for (int n = 0; n < kFirLen; ++n) taps[n] *= norm;
        }
        for (int n = 0; n < kFirLen; ++n)
            firTaps_[static_cast<size_t>(kFirLen - 1 - n)] =
                static_cast<T>(taps[n]);              // reversed for dotProduct

        // --- head bump ----------------------------------------------------------
        const double bumpHz = (speed == Speed::IPS_7_5) ? 45.0
                            : (speed == Speed::IPS_15) ? 90.0 : 180.0;
        const double bumpDb = 2.5 * bumpAmt;
        const auto bc = BiquadCoeffs<double>::makePeak(sampleRate_, bumpHz, 0.9, bumpDb);
        for (auto& b : headBump_)
        {
            const double pz1 = b.z1, pz2 = b.z2;
            b = PeakSection { bc.b0, bc.b1, bc.b2, bc.a1, bc.a2, 0.0, 0.0 };
            b.z1 = pz1;
            b.z2 = pz2;
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
    static constexpr int kFirLen = 63;
    static constexpr int kFirCenter = 31;
    static constexpr int kFirRing = 128;        // double-write mirrored ring

    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    int maxBlock_ = 0;
    std::atomic<bool> prepared_ { false };
    int latency_ = 0;
    int drySize_ = 1;

    std::unique_ptr<Oversampling<T>> oversampler_;
    std::vector<Hysteresis<T>> hysteresis_;   ///< +carrier instances (push-pull pair A).
    std::vector<Hysteresis<T>> hysteresisN_;  ///< -carrier instances (push-pull pair B).
    Hysteresis<T> calib_;    ///< Scratch +carrier instance for makeup calibration.
    Hysteresis<T> calib2_;   ///< Scratch -carrier instance for makeup calibration.

    /// AC bias carrier at 0.375 * internal rate: sin(2*pi*3k/8), one exact
    /// 8-phase period (3 carrier cycles), shared by all channels like a
    /// machine's single bias oscillator.
    static constexpr double kBiasTable[8] = {
        0.0,  0.70710678118654752, -1.0,  0.70710678118654752,
        0.0, -0.70710678118654752,  1.0, -0.70710678118654752
    };

    std::vector<ShelfSection> recordHF_, recordLF_, playHF_, playLF_;
    std::vector<PeakSection> headBump_;
    std::vector<PeakSection> outHp_;        ///< AC-coupling 24 Hz high-pass.
    std::vector<double> overBiasLp_;        ///< Over-bias self-erasure LP state.

    std::vector<T> firTaps_;
    std::vector<std::vector<T>> firState_;
    int firPos_ = 0;

    std::vector<std::vector<T>> delayRing_;
    int delayPos_ = 0;
    int delayCenter_ = 96;
    int delayMask_ = 255;
    double driftA_ = 0.0, scrapeA1_ = 0.0, scrapeA2_ = 0.0;

    std::vector<std::vector<T>> dryRing_;
    int dryPos_ = 0;

    double hScale_ = 1.0, mScale_ = 1.0;
    double biasAmp_ = 3.0 * 2.2e4;   ///< Carrier amplitude in H units.
    double overBiasA_ = 0.0;         ///< Over-bias LP coefficient (0 = off).
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
    std::atomic<bool> dirty_ { true };
};

} // namespace dspark
