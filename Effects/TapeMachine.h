// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

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
 *    digital inverse. The round trip is flat by construction — but the
 *    hysteresis sees the emphasized signal, so highs saturate first at slow
 *    speeds, exactly as on hardware.
 * 2. **Magnetic hysteresis** at 2x oversampling, with `bias` mapping to the
 *    JA loop parameters: low bias widens the loop (under-bias grit), high
 *    bias narrows it toward the reversible curve. Small-signal gain is
 *    compensated analytically from the JA susceptibility, so drive changes
 *    saturation, not level.
 * 3. **Playback loss effects** with the physical formulas (Kadis): spacing
 *    loss 54.6·d/λ dB, gap loss sinc(π·g/λ), thickness loss
 *    (1−e^(−4πδ/λ))/(4πδ/λ), with λ = speed/frequency — rendered into a
 *    63-tap linear-phase FIR per speed, plus the speed-dependent head-bump
 *    resonance (45/90/180 Hz at 7.5/15/30 ips).
 * 4. **Wow & flutter**: one shared transport modulation (slow drift random
 *    walk + 0.55 Hz wow + 8.3/23 Hz flutter + scrape band) driving a
 *    fractional delay — identical on all channels, like a real capstan.
 * 5. **Tape hiss** (optional, default off).
 *
 * The dry path of the mix control is delay-compensated to getLatency().
 *
 * Dependencies: Hysteresis.h, Oversampling.h, Biquad.h, SimdOps.h,
 * AudioSpec.h, AudioBuffer.h, DspMath.h, DenormalGuard.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Hysteresis.h"
#include "../Core/Oversampling.h"
#include "../Core/SimdOps.h"

#include <algorithm>
#include <atomic>
#include <cmath>
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

    /** @brief Allocates the whole chain. */
    void prepare(const AudioSpec& spec)
    {
        if (spec.sampleRate <= 0.0 || spec.numChannels < 1) return;
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        maxBlock_ = std::max(spec.maxBlockSize, 1);

        oversampler_ = std::make_unique<Oversampling<T>>(2, Oversampling<T>::Quality::High);
        oversampler_->prepare(spec);

        hysteresis_.assign(static_cast<size_t>(numChannels_), {});
        for (auto& h : hysteresis_)
            h.prepare(sampleRate_ * 2.0);

        recordHF_.assign(static_cast<size_t>(numChannels_), {});
        recordLF_.assign(static_cast<size_t>(numChannels_), {});
        playHF_.assign(static_cast<size_t>(numChannels_), {});
        playLF_.assign(static_cast<size_t>(numChannels_), {});
        headBump_.assign(static_cast<size_t>(numChannels_), {});

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

        prepared_ = true;
        dirty_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears all signal state (keeps parameters). RT-safe. */
    void reset() noexcept
    {
        if (!prepared_) return;
        for (auto& h : hysteresis_) h.reset();
        for (auto& f : firState_) std::fill(f.begin(), f.end(), T(0));
        for (auto& d : delayRing_) std::fill(d.begin(), d.end(), T(0));
        for (auto& d : dryRing_) std::fill(d.begin(), d.end(), T(0));
        for (auto& s : recordHF_) s.clear();
        for (auto& s : recordLF_) s.clear();
        for (auto& s : playHF_) s.clear();
        for (auto& s : playLF_) s.clear();
        for (auto& b : headBump_) b = {};
        firPos_ = 0;
        delayPos_ = 0;
        dryPos_ = 0;
        modPhaseWow_ = 0.0;
        modPhaseFlut_ = 0.0;
        modPhaseFlut2_ = 0.0;
        driftState_ = 0.0;
        scrapeLp1_ = scrapeLp2_ = 0.0;
        rng_ = 0x1357feedu;
        if (oversampler_) oversampler_->reset();
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Input drive in dB [-12, +24]. Level-compensated: more drive means
     *  more saturation at roughly constant loudness. */
    void setDrive(T driveDb) noexcept
    {
        driveDb_.store(std::clamp(driveDb, T(-12), T(24)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Bias setting [0, 1]; 0.5 is nominal calibration. Low bias widens
     *  the hysteresis loop (gritty), high bias cleans up toward reversible. */
    void setBias(T bias) noexcept
    {
        bias_.store(std::clamp(bias, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Tape speed (changes EQ time constants, losses and head bump). */
    void setSpeed(Speed s) noexcept
    {
        speed_.store(static_cast<int>(s), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Equalization standard (NAB adds the LF time constant). */
    void setStandard(Standard s) noexcept
    {
        standard_.store(static_cast<int>(s), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Playback loss intensity [0, 1] (0 bypasses the loss FIR). */
    void setLossEffects(T amount) noexcept
    {
        loss_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Head-bump resonance intensity [0, 1] (~2.5 dB at full). */
    void setHeadBump(T amount) noexcept
    {
        headBumpAmt_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Wow & flutter depth [0, 1] (~0.25% peak pitch deviation at 1). */
    void setWowFlutter(T amount) noexcept
    {
        wowFlutter_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Tape hiss level in dBFS (e.g. -55 for audible vintage hiss);
     *  values <= -120 disable it (default). */
    void setNoise(T dbfs) noexcept
    {
        noiseDb_.store(std::clamp(dbfs, T(-200), T(-20)), std::memory_order_relaxed);
    }

    /** @brief Dry/wet mix [0, 1]; dry is latency-compensated. */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Total latency in samples (oversampler + loss FIR + transport delay). */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

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

        // 3. Hysteresis at 2x.
        {
            auto osView = oversampler_->upsample(buffer);
            const int osN = osView.getNumSamples();
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = osView.getChannel(ch);
                auto& hyst = hysteresis_[static_cast<size_t>(ch)];
                const double inScale = hScale_;
                const double outScale = mScale_;
                for (int i = 0; i < osN; ++i)
                    d[i] = static_cast<T>(outScale * static_cast<double>(
                        hyst.processSample(static_cast<T>(inScale * static_cast<double>(d[i])))));
            }
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

                // Hiss.
                if (noiseOn)
                {
                    rng_ = rng_ * 1664525u + 1013904223u;
                    const double n1 = static_cast<double>(rng_ >> 8) / 8388608.0 - 1.0;
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

    /** @brief Minimal peaking biquad in double (head bump). */
    struct PeakSection
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;

        [[nodiscard]] T process(T x) noexcept
        {
            const double in = static_cast<double>(x);
            const double y = b0 * in + z1;
            z1 = b1 * in - a1 * y + z2;
            z2 = b2 * in - a2 * y;
            return static_cast<T>(y);
        }
    };

    /** @brief HF emphasis (kHi·s·t + 1)/(s·t + 1) and LF shelf (s·t + kLo)/(s·t + 1)
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
        const double drive = std::pow(10.0, static_cast<double>(
            driveDb_.load(std::memory_order_relaxed)) / 20.0);
        const double bias = static_cast<double>(bias_.load(std::memory_order_relaxed));
        const auto speed = static_cast<Speed>(speed_.load(std::memory_order_relaxed));
        const auto standard = static_cast<Standard>(standard_.load(std::memory_order_relaxed));
        const double lossAmt = static_cast<double>(loss_.load(std::memory_order_relaxed));
        const double bumpAmt = static_cast<double>(headBumpAmt_.load(std::memory_order_relaxed));

        // --- bias -> JA loop parameters ----------------------------------------
        // AC bias linearizes recording toward the anhysteretic curve; in JA
        // terms that is a high reversible fraction c. Nominal bias (0.5) sits
        // at c = 0.65: mild loop, soft Langevin compression into saturation.
        // Under-bias drops c toward the raw S-shaped virgin curve (expansion,
        // grit, level instability); over-bias approaches the clean
        // anhysteretic. k scales the residual loop losses the same way.
        const double kEff = 2.7e4 * (1.3 - 0.6 * bias);
        const double cEff = std::clamp(0.35 + 0.6 * bias, 0.05, 0.95);
        for (auto& h : hysteresis_)
            h.setParameters(3.5e5, 2.2e4, 1.6e-3, kEff, cEff);

        // --- drive scaling with operating-level makeup -------------------------
        // 0 dBFS at nominal drive maps to H = 1.2a: moderate saturation. Tape
        // gain is inherently level-dependent (the virgin JA curve is expansive
        // into its knee), so unity can only be defined at one operating point:
        // we calibrate empirically at -12 dBFS programme level by running a
        // short 1 kHz burst through a scratch hysteresis instance (~8 ms of
        // model time, only on parameter changes). Quieter material reads
        // slightly low, hotter material blooms into compression — like tape.
        hScale_ = drive * 1.2 * 2.2e4;
        {
            constexpr double kCalAmp = 0.25;            // -12 dBFS reference
            const double fs2 = sampleRate_ * 2.0;
            const int calN = static_cast<int>(0.008 * fs2);   // 8 ms = 8 cycles
            calib_.prepare(fs2);
            calib_.setParameters(3.5e5, 2.2e4, 1.6e-3, kEff, cEff);
            double inSq = 0.0, outSq = 0.0;
            for (int i = 0; i < calN; ++i)
            {
                const double x = kCalAmp * std::sin(2.0 * std::numbers::pi * 1000.0 * i / fs2);
                const double m = static_cast<double>(
                    calib_.processSample(static_cast<T>(hScale_ * x)));
                if (i >= calN / 2)   // measure once the loop has settled
                {
                    inSq += x * x;
                    outSq += m * m;
                }
            }
            mScale_ = (outSq > 0.0) ? std::sqrt(inSq / outSq) : 1.0;
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
        }
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
            firTaps_[static_cast<size_t>(kFirLen - 1 - n)] =
                static_cast<T>(acc * hann / kGrid);   // reversed for dotProduct
        }

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
    bool prepared_ = false;
    int latency_ = 0;
    int drySize_ = 1;

    std::unique_ptr<Oversampling<T>> oversampler_;
    std::vector<Hysteresis<T>> hysteresis_;
    Hysteresis<T> calib_;   ///< Scratch instance for makeup calibration.

    std::vector<ShelfSection> recordHF_, recordLF_, playHF_, playLF_;
    std::vector<PeakSection> headBump_;

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

    double modPhaseWow_ = 0.0, modPhaseFlut_ = 0.0, modPhaseFlut2_ = 0.0;
    double driftState_ = 0.0, scrapeLp1_ = 0.0, scrapeLp2_ = 0.0;
    uint32_t rng_ = 0x1357feedu;

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
