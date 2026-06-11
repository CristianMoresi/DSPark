// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file TubePreamp.h
 * @brief Tube preamp: Koren 12AX7 stages, WDF tone circuit, supply sag.
 *
 * Circuit-level preamp modelling, not a waveshaper:
 *
 * - **Triode stages** (1 or 2) solved per sample with Newton-Raphson on the
 *   plate current. The tube follows Koren's improved SPICE model (Koren
 *   1996; published 12AX7 parameters MU=100, EX=1.4, KG1=1060, KP=600,
 *   KVB=300) in a classic common-cathode stage: 300 V supply, 100 kΩ plate
 *   load, 1.5 kΩ cathode resistor with its 22 µF bypass capacitor
 *   integrated trapezoidally — the capacitor state is folded into the
 *   Newton equation, so each sample solves the true implicit system. Grid
 *   conduction is approximated by a soft clamp toward +0.7 V (full blocking
 *   distortion needs the input-coupling state and is left for a later pass).
 * - **Supply sag**: the effective B+ droops with smoothed plate current
 *   (one-pole, ~70 ms) times a sag resistance — drive into the stage and
 *   the headroom breathes back, the classic touch response.
 * - **Tone stack**: the full Fender '59 Bassman FMV treble/bass/middle
 *   network, solved exactly as a 12-port WDF R-type adaptor
 *   (wdf::ToneStackFMV — verified sample-exact against the symbolic
 *   transfer function of Yeh & Smith, DAFx-06). The stack sits between the
 *   stages and is driven by the first stage's real ~38 kΩ output impedance,
 *   so it loads the tube exactly like the hardware. Controls interact
 *   non-orthogonally — that is the circuit, not a bug.
 * - Output level: the circuit's program response at the reference tone
 *   setting is measured once in prepare() (settled channel, pink-weighted
 *   multitone). A 3-section EQ designed from that measurement flattens the
 *   FMV stack's fixed ~10 dB mid-scoop envelope — neutral knobs sound
 *   neutral, the tone controls act relative to flat, and the triode's
 *   harmonic character is untouched. Loudness divides out the measured
 *   reference and drive^0.75 (partial link: the knob keeps a +0.25 dB/dB
 *   residual slope so backing off is audible and high drive holds level
 *   while density grows). Use setOutput() for static trim; gain changes
 *   are ramped (~30 ms) so slider drags stay click-free.
 *
 * The nonlinear core runs at 2x oversampling. THD signature verified in the
 * suite: single-stage distortion is 2nd-harmonic dominant (asymmetric
 * triode), DC operating point matches an independent high-precision solve
 * of the same circuit equations (the check SPICE would perform).
 *
 * Dependencies: WDF.h, Oversampling.h, AudioSpec.h, AudioBuffer.h,
 * DspMath.h, DenormalGuard.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Oversampling.h"
#include "../Core/StateBlob.h"
#include "../Core/WDF.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class TubePreamp
 * @brief One/two 12AX7 stages with sag and a WDF tone circuit.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TubePreamp
{
public:
    // -- Lifecycle ---------------------------------------------------------------

    /** @brief Allocates the chain (one circuit instance per channel). */
    void prepare(const AudioSpec& spec)
    {
        if (spec.sampleRate <= 0.0 || spec.numChannels < 1) return;
        sampleRate_ = spec.sampleRate;
        fs2_ = 2.0 * sampleRate_;
        numChannels_ = spec.numChannels;
        maxBlock_ = std::max(spec.maxBlockSize, 1);

        oversampler_ = std::make_unique<Oversampling<T>>(2, Oversampling<T>::Quality::High);
        oversampler_->prepare(spec);

        channels_.clear();
        channels_.resize(static_cast<size_t>(numChannels_));
        for (auto& ch : channels_)
            ch = std::make_unique<ChannelState>(fs2_);

        latency_ = oversampler_->getLatency();
        drySize_ = 1;
        while (drySize_ < latency_ + maxBlock_ + 1) drySize_ <<= 1;
        dryRing_.assign(static_cast<size_t>(numChannels_),
                        std::vector<T>(static_cast<size_t>(drySize_), T(0)));
        dryPos_ = 0;

        calibrateReference();

        prepared_ = true;
        dirty_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Re-settles every stage at its DC operating point. RT-safe. */
    void reset() noexcept
    {
        if (!prepared_) return;
        const double sagR = static_cast<double>(sag_.load(std::memory_order_relaxed)) * 40e3;
        for (auto& ch : channels_)
            ch->reset(sagR);
        for (auto& d : dryRing_)
            std::fill(d.begin(), d.end(), T(0));
        dryPos_ = 0;
        if (oversampler_) oversampler_->reset();
        // Seed the anti-zipper ramps at their targets: no fade-in on start.
        hScaleSm_ = -1.0;
        outGainSm_ = -1.0;
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Input drive in dB [-12, +36]; level-compensated. */
    void setDrive(T db) noexcept
    {
        driveDb_.store(std::clamp(db, T(-12), T(36)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Treble control of the FMV stack [0, 1]. */
    void setTreble(T treble) noexcept
    {
        treble_.store(std::clamp(treble, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Bass control of the FMV stack [0, 1] (log-taper, like the original). */
    void setBass(T bass) noexcept
    {
        bass_.store(std::clamp(bass, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Middle control of the FMV stack [0, 1]. */
    void setMiddle(T middle) noexcept
    {
        middle_.store(std::clamp(middle, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Supply sag depth [0, 1] (0 = stiff supply). */
    void setSag(T sag) noexcept
    {
        sag_.store(std::clamp(sag, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Number of triode stages (1 = clean/edge, 2 = high gain). */
    void setStages(int stages) noexcept
    {
        stages_.store(std::clamp(stages, 1, 2), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Static output trim in dB [-24, +12]. */
    void setOutput(T db) noexcept
    {
        outputDb_.store(std::clamp(db, T(-24), T(12)), std::memory_order_relaxed);
    }

    /** @brief Dry/wet mix [0, 1]; dry is latency-compensated. */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Latency in samples (the 2x oversampler only). */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

    /** @brief Effective B+ supply voltage of channel 0 (sag meter readout). */
    [[nodiscard]] T getSupplyVoltage() const noexcept
    {
        return supplyNow_.load(std::memory_order_relaxed);
    }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TUBE"), 1);
        w.write("drive", driveDb_.load(std::memory_order_relaxed));
        w.write("treble", treble_.load(std::memory_order_relaxed));
        w.write("bass", bass_.load(std::memory_order_relaxed));
        w.write("middle", middle_.load(std::memory_order_relaxed));
        w.write("sag", sag_.load(std::memory_order_relaxed));
        w.write("stages", stages_.load(std::memory_order_relaxed));
        w.write("output", outputDb_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TUBE")) return false;
        setDrive(static_cast<T>(r.read("drive", 0.0f)));
        setTreble(static_cast<T>(r.read("treble", 0.5f)));
        setBass(static_cast<T>(r.read("bass", 0.5f)));
        setMiddle(static_cast<T>(r.read("middle", 0.5f)));
        setSag(static_cast<T>(r.read("sag", 0.3f)));
        setStages(r.read("stages", 1));
        setOutput(static_cast<T>(r.read("output", 0.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        return true;
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Processes a block in-place. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        if (nCh == 0 || nS == 0) return;

        if (dirty_.exchange(false, std::memory_order_relaxed))
            recompute();

        const T mixVal = mix_.load(std::memory_order_relaxed);
        const double outGain = mScale_
            * std::pow(10.0, static_cast<double>(outputDb_.load(std::memory_order_relaxed)) / 20.0);

        // Dry snapshot.
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

        // Nonlinear circuit at 2x.
        {
            auto osView = oversampler_->upsample(buffer);
            const int osN = osView.getNumSamples();

            // Anti-zipper: GEOMETRIC in-block ramps toward the current
            // targets (~30 ms across blocks), shared by all channels. A flat
            // per-block step clicked while dragging drive; and since
            // (hScale, outGain) is a compensation pair spanning orders of
            // magnitude (outGain ~ 1/drive), linear interpolation transits
            // through over-gained states — power-law interpolation keeps
            // the pair on the calibration curve.
            if (outGainSm_ <= 0.0) outGainSm_ = outGain;
            if (hScaleSm_ <= 0.0)  hScaleSm_ = hScale_;
            const double kSm = 1.0 - std::exp(-static_cast<double>(osN) / (0.030 * fs2_));
            const double outEnd = outGainSm_ * std::pow(outGain / outGainSm_, kSm);
            const double hEnd   = hScaleSm_ * std::pow(hScale_ / hScaleSm_, kSm);
            const double outRat = std::pow(outEnd / outGainSm_, 1.0 / static_cast<double>(osN));
            const double hRat   = std::pow(hEnd / hScaleSm_, 1.0 / static_cast<double>(osN));

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = osView.getChannel(ch);
                auto& state = *channels_[static_cast<size_t>(ch)];
                double g = outGainSm_, h = hScaleSm_;
                for (int i = 0; i < osN; ++i)
                {
                    g *= outRat;
                    h *= hRat;
                    d[i] = static_cast<T>(g
                        * state.processSample(h * static_cast<double>(d[i]),
                                              numStagesActive_, sagR_));
                }
            }
            outGainSm_ = outEnd;
            hScaleSm_ = hEnd;
            oversampler_->downsample(buffer);
            supplyNow_.store(static_cast<T>(kBplus - sagR_ * channels_[0]->ipLP),
                             std::memory_order_relaxed);
        }

        // Latency-compensated mix.
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            const auto& dry = dryRing_[static_cast<size_t>(ch)];
            for (int i = 0; i < nS; ++i)
            {
                const int idx = (dryPos_ + i - latency_) & (drySize_ - 1);
                const T drySample = dry[static_cast<size_t>(idx)];
                d[i] = drySample + (d[i] - drySample) * mixVal;
            }
        }
        dryPos_ = (dryPos_ + nS) & (drySize_ - 1);
    }

private:
    // -- Circuit constants (classic 12AX7 common-cathode stage) -------------------
    static constexpr double kMu = 100.0, kEx = 1.4, kKg1 = 1060.0;
    static constexpr double kKp = 600.0, kKvb = 300.0;
    static constexpr double kBplus = 300.0;
    static constexpr double kRL = 100e3;
    static constexpr double kRk = 1.5e3;
    static constexpr double kCk = 22e-6;
    static constexpr double kInterstage = 0.12;   ///< Divider into stage 2.

    /** @brief Koren plate current and its partial derivatives. */
    static void koren(double vpk, double vgk, double& ip,
                      double& dIpdVpk, double& dIpdVgk) noexcept
    {
        vpk = std::max(vpk, 0.0);
        const double s = std::sqrt(kKvb + vpk * vpk);
        const double u = kKp * (1.0 / kMu + vgk / s);

        double sp = 0.0, sig = 0.0;                  // softplus(u), logistic(u)
        if (u > 30.0)       { sp = u; sig = 1.0; }
        else if (u < -30.0) { sp = std::exp(u); sig = sp; }
        else
        {
            const double eu = std::exp(u);
            sp = std::log1p(eu);
            sig = eu / (1.0 + eu);
        }

        const double e1 = (vpk / kKp) * sp;
        if (e1 <= 0.0)
        {
            ip = 0.0;
            dIpdVpk = 0.0;
            dIpdVgk = 0.0;
            return;
        }
        const double e1ex1 = std::pow(e1, kEx - 1.0);
        ip = 2.0 * e1ex1 * e1 / kKg1;
        const double dIpdE1 = 2.0 * kEx * e1ex1 / kKg1;

        const double dUdVpk = -kKp * vgk * vpk / (s * s * s);
        const double dE1dVpk = sp / kKp + (vpk / kKp) * sig * dUdVpk;
        const double dE1dVgk = vpk * sig / s;
        dIpdVpk = dIpdE1 * dE1dVpk;
        dIpdVgk = dIpdE1 * dE1dVgk;
    }

    /** @brief One common-cathode stage with trapezoidal cathode bypass. */
    struct TriodeStage
    {
        double fs2 = 96000.0;
        double ip = 8e-4;          ///< Plate current state / NR seed.
        double vk = 1.2;           ///< Cathode voltage (bypass cap state).
        double fPrev = 0.0;        ///< Previous dVk/dt for the trapezoid.
        double vpDC = 200.0;       ///< Plate voltage at the operating point.

        void settleDC(double bplus) noexcept
        {
            // Static operating point: Vk = Ip*Rk (capacitor fully charged).
            double i = 8e-4;
            for (int it = 0; it < 60; ++it)
            {
                const double vkS = i * kRk;
                const double vpk = bplus - i * kRL - vkS;
                double ipK = 0.0, dVpk = 0.0, dVgk = 0.0;
                koren(vpk, -vkS, ipK, dVpk, dVgk);
                const double f = i - ipK;
                const double fp = 1.0 - (dVpk * (-(kRL + kRk)) + dVgk * (-kRk));
                const double di = f / fp;
                i -= di;
                i = std::clamp(i, 0.0, bplus / (kRL + kRk));
                if (std::abs(di) < 1e-15) break;
            }
            ip = i;
            vk = i * kRk;
            fPrev = 0.0;
            vpDC = bplus - i * kRL;
        }

        /** @brief Processes one grid-volt sample, returns AC plate voltage. */
        [[nodiscard]] double processSample(double vg, double bplusEff) noexcept
        {
            // Soft grid-conduction clamp toward +0.7 V.
            if (vg > 0.0)
                vg = 0.7 * std::tanh(vg / 0.7);

            // Trapezoidal cathode bypass: Ck dVk/dt = Ip - Vk/Rk, with fPrev
            // holding the previous NET CURRENT (Ip - Vk/Rk), so the update is
            //   Vk_n = Vk_{n-1} + (T/2Ck)(I_n + I_{n-1}) = kA + kB * Ip_n.
            const double h2c = 0.5 / (fs2 * kCk);
            const double denom = 1.0 + h2c / kRk;
            const double kA = (vk + h2c * fPrev) / denom;
            const double kB = h2c / denom;

            const double iMax = bplusEff / kRL + 1e-3;
            double i = std::clamp(ip, 0.0, iMax);

            for (int it = 0; it < 8; ++it)
            {
                const double vkN = kA + kB * i;
                const double vpk = bplusEff - i * kRL - vkN;
                const double vgk = vg - vkN;
                double ipK = 0.0, dVpk = 0.0, dVgk = 0.0;
                koren(vpk, vgk, ipK, dVpk, dVgk);

                const double f = i - ipK;
                const double fp = 1.0 - (dVpk * (-kRL - kB) + dVgk * (-kB));
                const double di = f / std::max(fp, 1e-6);
                i -= di;
                i = std::clamp(i, 0.0, iMax);
                if (std::abs(di) < 1e-12) break;
            }

            const double vkN = kA + kB * i;
            fPrev = i - vkN / kRk;      // net capacitor current for the trapezoid
            vk = vkN;
            ip = i;
            return (bplusEff - i * kRL) - vpDC;     // AC component
        }
    };

    /** @brief Full per-channel circuit: two stages + FMV tone stack + sag. */
    struct ChannelState
    {
        explicit ChannelState(double fs2In)
            : fmv(38e3, 1e6)   // driven by the stage's real output impedance
        {
            stage1.fs2 = fs2In;
            stage2.fs2 = fs2In;
            fs2 = fs2In;
            fmv.prepare(fs2In);
        }

        /** Settles the DC operating point CONSISTENTLY with the sagged
         *  supply: fixed point on B+ = kBplus - sagR*(Ip1+Ip2). Settling at
         *  the stiff kBplus while processSample() immediately applies the
         *  sag drop produced a ~19 V supply step at sag 0.3 — an audible
         *  activation thump, and worse: it sat inside the old 10 ms
         *  calibration window, inflating outSq and burying the whole wet
         *  path ~20 dB under unity. */
        void reset(double sagR) noexcept
        {
            double bp = kBplus;
            for (int it = 0; it < 8; ++it)
            {
                stage1.settleDC(bp);
                stage2.settleDC(bp);
                const double bpNew = kBplus - sagR * (stage1.ip + stage2.ip);
                if (std::abs(bpNew - bp) < 1e-9) break;
                bp = bpNew;
            }
            ipLP = stage1.ip + stage2.ip;
            outHpX = outHpY = 0.0;
            fmv.reset();
            for (auto& set : flatten)
                for (auto& f : set)
                    f.reset();
        }

        /** Installs the reference-flattening EQ for one stage count. */
        void setFlattenCoeffs(int stageCount,
                              const std::array<BiquadCoeffs<double>, 3>& c) noexcept
        {
            auto& set = flatten[static_cast<size_t>(stageCount - 1)];
            for (int k = 0; k < 3; ++k)
                set[static_cast<size_t>(k)].setCoeffs(c[static_cast<size_t>(k)]);
        }

        void setToneControls(double t, double b, double m) noexcept
        {
            fmv.setControls(t, b, m);   // rebuilds the R-type scattering
        }

        [[nodiscard]] double processSample(double vgIn, int numStages, double sagR) noexcept
        {
            // Supply sag: B+ droops with smoothed total plate current.
            const double sagAlpha = 1.0 - std::exp(-1.0 / (0.07 * fs2));
            const double iTotal = stage1.ip + (numStages > 1 ? stage2.ip : 0.0);
            ipLP += sagAlpha * (iTotal - ipLP);
            const double bplusEff = kBplus - sagR * ipLP;

            // Stage 1 -> FMV tone stack -> (stage 2) -> output high-pass.
            double v = stage1.processSample(vgIn, bplusEff);
            v = fmv.processSample(v);
            if (numStages > 1)
                v = stage2.processSample(v * kInterstage, bplusEff);

            // Output coupling high-pass (~8 Hz, removes residual sag drift).
            const double a = 1.0 - 2.0 * std::numbers::pi * 8.0 / fs2;
            const double y = a * (outHpY + v - outHpX);
            outHpX = v;
            outHpY = y;
            // Restore absolute polarity for single-stage use.
            double out = (numStages > 1) ? y : -y;

            // Reference-flattening EQ: undoes the FMV stack's fixed envelope
            // at the neutral tone setting (designed in calibrateReference
            // from the measured response), so neutral knobs sound neutral
            // and the tone controls act RELATIVE to flat. The triode's
            // harmonic character is untouched (this stage is linear).
            auto& fl = flatten[numStages > 1 ? 1 : 0];
            out = fl[0].processSample(out, 0);
            out = fl[1].processSample(out, 0);
            out = fl[2].processSample(out, 0);
            return out;
        }

        double fs2 = 96000.0;
        TriodeStage stage1, stage2;
        double ipLP = 1.6e-3;
        double outHpX = 0.0, outHpY = 0.0;
        std::array<std::array<Biquad<double, 1>, 3>, 2> flatten;   ///< Per stage count.

        wdf::ToneStackFMV<double> fmv;   ///< Exact Bassman stack (R-type WDF).
    };

    /** @brief Applies parameter changes: stack controls, sag R, calibration. */
    void recompute() noexcept
    {
        const double drive = std::pow(10.0, static_cast<double>(
            driveDb_.load(std::memory_order_relaxed)) / 20.0);
        const double t = static_cast<double>(treble_.load(std::memory_order_relaxed));
        const double b = static_cast<double>(bass_.load(std::memory_order_relaxed));
        const double m = static_cast<double>(middle_.load(std::memory_order_relaxed));
        const double sag = static_cast<double>(sag_.load(std::memory_order_relaxed));
        numStagesActive_ = stages_.load(std::memory_order_relaxed);

        hScale_ = drive;                       // 0 dBFS -> 1 V grid at drive 0
        // Note on physics: a class-A preamp draws near-constant average
        // current, so supply sag shifts the operating point rather than
        // pumping like a push-pull power amp. The audible touch response of
        // this model comes from the cathode-bypass bias shift (modelled in
        // TriodeStage); the sag control changes voicing, not loudness.
        sagR_ = sag * 40e3;
        for (auto& ch : channels_)
            ch->setToneControls(t, b, m);

        // Loudness: divide out the circuit's program gain at the REFERENCE
        // tone setting (measured once in prepare on a settled channel) and
        // MOST of the drive factor. The link is partial (drive^0.75, i.e. a
        // residual +0.25 dB/dB slope): an exact 1/drive link leaves the knob
        // audibly dead below 0 dB (the circuit is still clean there) and
        // turns it into a pure attenuator above (compression eats level
        // faster than the link returns it). With the residual slope, -12 dB
        // drive sits ~3 dB lower and clean, high drive holds level while the
        // density grows. The tone knobs stay fully audible: their deviation
        // from the 0.5/0.5/0.5 reference is part of the tone, not the level.
        // Cheap by construction (no scratch processing on the audio thread).
        mScale_ = 1.0 / std::max(std::pow(drive, 0.75)
                                 * gRef_[static_cast<size_t>(numStagesActive_ - 1)], 1e-9);
    }

    /**
     * Measures the per-stage-count reference response of a settled channel
     * at the reference tone (0.5/0.5/0.5), reference sag voicing and drive
     * 0 dB, using a pink-weighted multitone (one tone per octave,
     * 100..6400 Hz) at program level (~-15 dBFS RMS). From the per-tone
     * gains it designs a 3-section flattening EQ (low shelf / mid peak /
     * high shelf) that removes the FMV stack's fixed envelope — the raw
     * stack at neutral knobs is a ~10 dB mid scoop, far too much colour
     * for an insert effect — and then computes the loudness reference
     * gRef WITH the flattener in place. A single 1 kHz tone is the wrong
     * probe for either job (the old 1 kHz/10 ms calibration also measured
     * INSIDE the supply-sag settling step, burying the wet path ~20 dB).
     */
    void calibrateReference() noexcept
    {
        constexpr double kRefTones[] = { 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0 };
        constexpr double kSagRef = 0.3 * 40e3;
        const int settle = static_cast<int>(0.060 * fs2_);
        const int meas   = static_cast<int>(0.050 * fs2_);
        constexpr double kAmpPerTone = 0.095 / 2.6457513;

        for (int st = 1; st <= 2; ++st)
        {
            ChannelState cal(fs2_);   // fresh: flattener is passthrough here
            cal.setToneControls(0.5, 0.5, 0.5);
            cal.reset(kSagRef);

            // Per-tone Goertzel over the measured tail.
            std::array<double, 7> gs1 {}, gs2 {}, gc {};
            for (int k = 0; k < 7; ++k)
                gc[static_cast<size_t>(k)] =
                    2.0 * std::cos(2.0 * std::numbers::pi * kRefTones[k] / fs2_);
            for (int i = 0; i < settle + meas; ++i)
            {
                double x = 0.0;
                for (int k = 0; k < 7; ++k)
                    x += std::sin(2.0 * std::numbers::pi * kRefTones[k] * i / fs2_ + k * 1.7);
                x *= kAmpPerTone;
                const double y = cal.processSample(x, st, kSagRef);
                if (i >= settle)
                    for (int k = 0; k < 7; ++k)
                    {
                        auto& s1 = gs1[static_cast<size_t>(k)];
                        auto& s2 = gs2[static_cast<size_t>(k)];
                        const double s0 = y + gc[static_cast<size_t>(k)] * s1 - s2;
                        s2 = s1; s1 = s0;
                    }
            }
            std::array<double, 7> gainDb {};
            for (int k = 0; k < 7; ++k)
            {
                const double c = gc[static_cast<size_t>(k)] * 0.5;
                const double s1 = gs1[static_cast<size_t>(k)], s2 = gs2[static_cast<size_t>(k)];
                const double mag = std::sqrt(std::max(s1 * s1 + s2 * s2 - 2.0 * c * s1 * s2, 0.0))
                                 * 2.0 / meas;
                gainDb[static_cast<size_t>(k)] =
                    20.0 * std::log10(std::max(mag / kAmpPerTone, 1e-9));
            }

            // Flattening EQ from the band means, relative to the overall mean.
            double mean = 0.0;
            for (double g : gainDb) mean += g / 7.0;
            const double lo = 0.5 * (gainDb[0] + gainDb[1]) - mean;
            const double mid = (gainDb[2] + gainDb[3] + gainDb[4]) / 3.0 - mean;
            const double hi = 0.5 * (gainDb[5] + gainDb[6]) - mean;

            std::array<BiquadCoeffs<double>, 3> fc = {
                BiquadCoeffs<double>::makeLowShelf(fs2_, 180.0, -lo),
                BiquadCoeffs<double>::makePeak(fs2_, 800.0, 0.55, -mid),
                BiquadCoeffs<double>::makeHighShelf(fs2_, 4500.0, -hi)
            };
            for (auto& ch : channels_)
                ch->setFlattenCoeffs(st, fc);

            // Loudness reference WITH the flattener folded in analytically.
            double outSq = 0.0;
            for (int k = 0; k < 7; ++k)
            {
                double corr = 1.0;
                for (const auto& c : fc)
                    corr *= c.getMagnitude(kRefTones[k], fs2_);
                const double g = std::pow(10.0, gainDb[static_cast<size_t>(k)] / 20.0) * corr;
                outSq += g * g;
            }
            gRef_[static_cast<size_t>(st - 1)] = std::sqrt(std::max(outSq / 7.0, 1e-18));
        }
    }

    // -- Members --------------------------------------------------------------------
    double sampleRate_ = 48000.0;
    double fs2_ = 96000.0;
    int numChannels_ = 0;
    int maxBlock_ = 0;
    bool prepared_ = false;
    int latency_ = 0;
    int drySize_ = 1;

    std::unique_ptr<Oversampling<T>> oversampler_;
    std::vector<std::unique_ptr<ChannelState>> channels_;

    std::vector<std::vector<T>> dryRing_;
    int dryPos_ = 0;

    double hScale_ = 1.0;
    double mScale_ = 1.0;
    double sagR_ = 0.0;
    int numStagesActive_ = 1;
    std::array<double, 2> gRef_ { 1.0, 1.0 };   ///< Program gain per stage count (prepare-time).
    double hScaleSm_ = -1.0;                    ///< Anti-zipper ramp states (-1 = seed on
    double outGainSm_ = -1.0;                   ///<  first block after prepare/reset).

    std::atomic<T> driveDb_ { T(0) };
    std::atomic<T> treble_ { T(0.5) };
    std::atomic<T> bass_ { T(0.5) };
    std::atomic<T> middle_ { T(0.5) };
    std::atomic<T> sag_ { T(0.3) };
    std::atomic<int> stages_ { 1 };
    std::atomic<T> outputDb_ { T(0) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<T> supplyNow_ { T(300) };
    std::atomic<bool> dirty_ { true };
};

} // namespace dspark
