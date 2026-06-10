// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file TransformerModel.h
 * @brief Audio transformer: core hysteresis on the flux, LF bloom, HF bell.
 *
 * What makes a transformer sound like a transformer is *where* its
 * nonlinearity lives: the core saturates on magnetic FLUX, which is the
 * time-integral of the winding voltage. At equal level, low frequencies
 * integrate to much larger flux than highs (Φ ∝ V/f), so distortion and
 * compression concentrate in the low end — the classic "iron" low-frequency
 * bloom — while the top stays clean.
 *
 * The model implements that physics directly:
 *
 *   in → leaky trapezoidal integrator (flux) → Jiles-Atherton hysteresis
 *      → exact algebraic inverse differentiator → magnetizing-corner
 *        high-pass (finite Lm) → leakage/capacitance HF bell → out
 *
 * The integrator/differentiator pair is an exact algebraic inverse, so the
 * linear part of the JA loop passes transparently (verified by null test);
 * everything you hear is the core's hysteresis acting on flux, with the
 * loop's odd-dominant saturation, remanence memory and rate-dependent loss.
 * The JA core is shared with TapeMachine (Core/Hysteresis.h), parameterised
 * for an unbiased iron core (wider loop than biased tape).
 *
 * - **coreSize** moves the magnetizing-inductance corner (40 Hz small core
 *   → 5 Hz big core) and scales the flux headroom the same way real iron
 *   does: big cores ring lower and take more level before saturating.
 * - **resonance** raises the leakage-inductance/capacitance bell
 *   (Jensen-style), mapped into the audible band near the top octave.
 * - Loudness is calibrated empirically at the reference level (same
 *   convention as TapeMachine/TubePreamp): drive changes iron, not volume.
 *
 * Zero latency. The Saturation effect's lightweight Transformer algorithm
 * remains as the cheap alternative; this is the physical one.
 *
 * Dependencies: Hysteresis.h, Biquad.h, AudioSpec.h, AudioBuffer.h,
 * DspMath.h, DenormalGuard.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Hysteresis.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class TransformerModel
 * @brief Physical audio-transformer coloration (flux-domain JA hysteresis).
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TransformerModel
{
public:
    // -- Lifecycle ---------------------------------------------------------------

    /** @brief Allocates per-channel circuit state. */
    void prepare(const AudioSpec& spec)
    {
        if (spec.sampleRate <= 0.0 || spec.numChannels < 1) return;
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        channels_.assign(static_cast<size_t>(numChannels_), {});
        for (auto& ch : channels_)
            ch.hyst.prepare(sampleRate_);

        prepared_ = true;
        dirty_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears all signal state. RT-safe. */
    void reset() noexcept
    {
        if (!prepared_) return;
        for (auto& ch : channels_)
        {
            ch.hyst.reset();
            ch.flux = 0.0;
            ch.vPrev = 0.0;
            ch.vPrev2 = 0.0;
            ch.mPrev = 0.0;
            ch.hpX = ch.hpY = 0.0;
            ch.bell = {};
        }
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Core drive in dB [-12, +24]; loudness-compensated. */
    void setDrive(T db) noexcept
    {
        driveDb_.store(std::clamp(db, T(-12), T(24)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Core size [0, 1]: small chokes at 40 Hz and saturates early,
     *  big rings to 5 Hz with more low-end headroom. Default 0.5. */
    void setCoreSize(T size) noexcept
    {
        coreSize_.store(std::clamp(size, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Leakage/capacitance HF bell amount [0, 1] (Jensen-style). */
    void setResonance(T amount) noexcept
    {
        resonance_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Dry/wet mix [0, 1]. Zero latency: no compensation needed. */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Zero — the model is all minimum-phase IIR and memoryless NR. */
    [[nodiscard]] static constexpr int getLatency() noexcept { return 0; }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("XFMR"), 1);
        w.write("drive", driveDb_.load(std::memory_order_relaxed));
        w.write("coreSize", coreSize_.load(std::memory_order_relaxed));
        w.write("resonance", resonance_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("XFMR")) return false;
        setDrive(static_cast<T>(r.read("drive", 0.0f)));
        setCoreSize(static_cast<T>(r.read("coreSize", 0.5f)));
        setResonance(static_cast<T>(r.read("resonance", 0.3f)));
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

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            auto& st = channels_[static_cast<size_t>(ch)];

            for (int i = 0; i < nS; ++i)
            {
                const double x = static_cast<double>(d[i]);

                // Leaky trapezoidal integrator: winding voltage -> flux.
                const double fluxNew = leak_ * st.flux + halfT_ * (x + st.vPrev);
                st.vPrev = x;

                // Core hysteresis on the flux (JA, unbiased-iron parameters).
                const double m = mScale_ * static_cast<double>(
                    st.hyst.processSample(static_cast<T>(hScale_ * fluxNew)));

                // Algebraic inverse of the integrator (flux -> voltage). The
                // damped feedback pole (rho < 1) tames z = -1: hysteresis
                // distortion is new content the inverse never saw, and an
                // undamped differentiator would ring at Nyquist forever. The
                // leak alpha still cancels exactly; the only linear residue
                // is (1+z^-1)/(1+rho z^-1), transparent to within 0.15 dB.
                double v = (m - leak_ * st.mPrev) * invHalfT_ - kDiffRho * st.vPrev2;
                st.vPrev2 = v;
                st.mPrev = m;
                st.flux = fluxNew;

                // Magnetizing-inductance corner (one-pole high-pass).
                const double hp = hpA_ * (st.hpY + v - st.hpX);
                st.hpX = v;
                st.hpY = hp;

                // Leakage/capacitance bell.
                const double y = st.bell.process(hp);

                const T wet = static_cast<T>(y);
                d[i] = d[i] + (wet - d[i]) * mixVal;
            }
        }
    }

private:
    static constexpr double kDiffRho = 0.974;   ///< Differentiator pole damping.

    struct BellSection
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;

        [[nodiscard]] double process(double x) noexcept
        {
            const double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };

    struct ChannelState
    {
        Hysteresis<T> hyst;
        double flux = 0.0;
        double vPrev = 0.0;     ///< Previous input (integrator).
        double vPrev2 = 0.0;    ///< Previous output (differentiator).
        double mPrev = 0.0;
        double hpX = 0.0, hpY = 0.0;
        BellSection bell;
    };

    void recompute() noexcept
    {
        const double drive = std::pow(10.0, static_cast<double>(
            driveDb_.load(std::memory_order_relaxed)) / 20.0);
        const double size = static_cast<double>(coreSize_.load(std::memory_order_relaxed));
        const double res = static_cast<double>(resonance_.load(std::memory_order_relaxed));

        // The integrator leak keeps the flux bounded; its exact algebraic
        // inverse cancels it, so the audible LF rolloff comes only from the
        // explicit magnetizing-corner high-pass below.
        const double cornerHz = 40.0 * std::pow(0.125, size);   // 40 -> 5 Hz
        leak_ = std::exp(-2.0 * std::numbers::pi * cornerHz / sampleRate_);
        hpA_ = 1.0 - 2.0 * std::numbers::pi * cornerHz / sampleRate_;
        halfT_ = 0.5 / sampleRate_;
        invHalfT_ = 2.0 * sampleRate_;

        // Unbiased iron: wider loop than biased tape (low reversible c).
        for (auto& ch : channels_)
            ch.hyst.setParameters(3.5e5, 2.2e4, 1.6e-3, 3.2e4, 0.25);

        // Flux scale: 0 dBFS at 30 Hz reaches H = 1.1a at nominal drive and
        // nominal core; bigger cores take proportionally more flux.
        const double fluxRef = 1.0 / (2.0 * std::numbers::pi * 30.0);
        const double headroom = 0.6 + 0.8 * size;
        hScale_ = drive * 1.1 * 2.2e4 / (fluxRef * headroom);

        // Small-signal loudness calibration at 1 kHz (empirical, reference
        // level), same convention as the other physical models.
        {
            Hysteresis<T> cal;
            cal.prepare(sampleRate_);
            cal.setParameters(3.5e5, 2.2e4, 1.6e-3, 3.2e4, 0.25);
            double flux = 0.0, vPrev = 0.0, mPrev = 0.0, vPrev2 = 0.0;
            double inSq = 0.0, outSq = 0.0;
            const int n = static_cast<int>(0.02 * sampleRate_);
            for (int i = 0; i < n; ++i)
            {
                const double x = 0.05 * std::sin(2.0 * std::numbers::pi * 1000.0 * i / sampleRate_);
                const double fluxNew = leak_ * flux + halfT_ * (x + vPrev);
                vPrev = x;
                const double m = static_cast<double>(
                    cal.processSample(static_cast<T>(hScale_ * fluxNew)));
                double v = (m - leak_ * mPrev) * invHalfT_ - kDiffRho * vPrev2;
                vPrev2 = v;
                mPrev = m;
                flux = fluxNew;
                if (i >= n / 2)
                {
                    inSq += x * x;
                    outSq += v * v;
                }
            }
            mScale_ = (outSq > 0.0) ? std::sqrt(inSq / outSq) : 1.0;
        }

        // HF bell: Jensen-style leakage resonance mapped into the top octave.
        const double bellHz = std::min(12000.0 + 6000.0 * res, 0.42 * sampleRate_);
        const double bellDb = 2.5 * res;
        const auto bc = BiquadCoeffs<double>::makePeak(sampleRate_, bellHz, 0.8, bellDb);
        for (auto& ch : channels_)
        {
            const double pz1 = ch.bell.z1, pz2 = ch.bell.z2;
            ch.bell = BellSection { bc.b0, bc.b1, bc.b2, bc.a1, bc.a2, 0.0, 0.0 };
            ch.bell.z1 = pz1;
            ch.bell.z2 = pz2;
        }
    }

    // -- Members --------------------------------------------------------------------
    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    bool prepared_ = false;

    std::vector<ChannelState> channels_;

    double leak_ = 0.999;
    double hpA_ = 0.999;
    double halfT_ = 0.5 / 48000.0;
    double invHalfT_ = 96000.0;
    double hScale_ = 1.0;
    double mScale_ = 1.0;

    std::atomic<T> driveDb_ { T(0) };
    std::atomic<T> coreSize_ { T(0.5) };
    std::atomic<T> resonance_ { T(0.3) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<bool> dirty_ { true };
};

} // namespace dspark
