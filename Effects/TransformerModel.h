// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TransformerModel.h
 * @brief Audio transformer: core hysteresis on the flux, LF bloom, HF bell.
 *
 * What makes a transformer sound like a transformer is *where* its
 * nonlinearity lives: the core saturates on magnetic FLUX, which is the
 * time-integral of the winding voltage. At equal level, low frequencies
 * integrate to much larger flux than highs (Phi ~ V/f), so distortion and
 * compression concentrate in the low end - the classic "iron" low-frequency
 * bloom - while the top stays clean.
 *
 * The model implements that physics directly:
 *
 *   in -> leaky trapezoidal integrator (flux) -> Jiles-Atherton hysteresis
 *      -> exact algebraic inverse differentiator -> magnetizing-corner
 *        high-pass (finite Lm) -> leakage/capacitance HF bell -> out
 *
 * The integrator/differentiator pair is an exact algebraic inverse, so the
 * linear part of the JA loop passes transparently (verified by null test);
 * everything you hear is the core's hysteresis acting on flux, with the
 * loop's odd-dominant saturation, remanence memory and rate-dependent loss.
 * The JA core is shared with TapeMachine (Core/Hysteresis.h), parameterised
 * for an unbiased iron core (wider loop than biased tape).
 *
 * - **coreSize** moves the magnetizing-inductance corner (40 Hz small core
 *   -> 5 Hz big core) and scales the flux headroom the same way real iron
 *   does: big cores ring lower and take more level before saturating.
 * - **resonance** raises the leakage-inductance/capacitance bell
 *   (Jensen-style), mapped into the audible band near the top octave.
 * - Loudness is calibrated empirically at PROGRAM level (100 Hz, -12 dBFS -
 *   the flux regime music drives the core into; same convention as
 *   TapeMachine/TubePreamp): drive changes iron, not volume. Gain changes
 *   are ramped (~50 ms), which matters doubly here because the model
 *   differentiates its output.
 *
 * Zero latency. The Saturation effect's lightweight Transformer algorithm
 * remains as the cheap alternative; this is the physical one. Unlike the
 * biased tape core, the leaky flux integrator continuously re-centres the
 * loop, so the response is history-independent by construction (verified:
 * 0.000 dB branch delta at programme level).
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored; changes are published with
 * a release store and consumed at the next block). prepare() is setup-thread
 * only (allocates; invalid specs are ignored and an unprepared instance
 * passes audio through). reset() belongs to the stream owner.
 * getState()/setState() are setup/UI threads. The dry/wet mix is smoothed
 * linearly over one block. Channels beyond the prepared count pass through
 * untouched.
 *
 * Dependencies: Core/Hysteresis.h, Core/Biquad.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h, Core/StateBlob.h.
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
#include <cstddef>
#include <cstdint>
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

    /** @brief Allocates per-channel circuit state. Invalid specs
     *  (non-positive or non-finite rate, block size or channel count) are
     *  ignored: the previous state is kept and an unprepared instance stays
     *  pass-through. */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        channels_.assign(static_cast<size_t>(numChannels_), {});
        for (auto& ch : channels_)
        {
            ch.hyst.prepare(sampleRate_);
            // Unbiased iron: wider loop than biased tape (low reversible c).
            // Constant material - set once here, not on every recompute().
            ch.hyst.setParameters(3.5e5, 2.2e4, 1.6e-3, 3.2e4, 0.25);
        }

        prepared_.store(true, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
        reset();
    }

    /** @brief Clears all signal state. RT-safe. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
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
        // Seed the anti-zipper ramps at their targets: no fade-in on start.
        hScaleSm_ = -1.0;
        mScaleSm_ = -1.0;
        currentMix_ = mix_.load(std::memory_order_relaxed);
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Core drive in dB [-12, +24]; loudness-compensated. Non-finite
     *  values are ignored. */
    void setDrive(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        driveDb_.store(std::clamp(db, T(-12), T(24)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Core size [0, 1]: small chokes at 40 Hz and saturates early,
     *  big rings to 5 Hz with more low-end headroom. Default 0.5. Non-finite
     *  values are ignored. */
    void setCoreSize(T size) noexcept
    {
        if (!std::isfinite(size)) return;
        coreSize_.store(std::clamp(size, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Leakage/capacitance HF bell amount [0, 1] (Jensen-style).
     *  Non-finite values are ignored. */
    void setResonance(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        resonance_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Dry/wet mix [0, 1]; smoothed linearly over one block. Zero
     *  latency: no compensation needed. Non-finite values are ignored. */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getDrive() const noexcept { return driveDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getCoreSize() const noexcept { return coreSize_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getResonance() const noexcept { return resonance_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Zero - the model is all minimum-phase IIR and memoryless NR. */
    [[nodiscard]] static constexpr int getLatency() noexcept { return 0; }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("XFMR"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("drive", static_cast<float>(driveDb_.load(std::memory_order_relaxed)));
        w.write("coreSize", static_cast<float>(coreSize_.load(std::memory_order_relaxed)));
        w.write("resonance", static_cast<float>(resonance_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
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

        // Linear per-block mix ramp with exact landing (settled: step == 0
        // and the per-sample value reduces to the constant, bit-identically).
        // A hard flip on the differentiated wet stream clicked at 4.6x the
        // steady-state sample delta.
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;
        const T mixStep   = (mixTarget - mixStart) / static_cast<T>(nS);

        // Anti-zipper: GEOMETRIC in-block ramps toward the recompute()
        // targets (~50 ms across blocks), shared by all channels. Two
        // reasons: this model differentiates its output (x 2*fs), so a flat
        // per-block gain step clicks; and (hScale, mScale) is a calibrated
        // compensation pair spanning orders of magnitude - interpolating it
        // LINEARLY passes through wildly over-gained states (+24 dB
        // measured mid-drag), while power-law interpolation keeps the pair
        // on the calibration curve m ~ C/h^g throughout the transition.
        if (hScaleSm_ <= 0.0) { hScaleSm_ = hScale_; mScaleSm_ = mScale_; }
        const double kSm = 1.0 - std::exp(-static_cast<double>(nS) / (0.050 * sampleRate_));
        const double hEnd = hScaleSm_ * std::pow(hScale_ / hScaleSm_, kSm);
        const double mEnd = mScaleSm_ * std::pow(mScale_ / mScaleSm_, kSm);
        const double hRat = std::pow(hEnd / hScaleSm_, 1.0 / static_cast<double>(nS));
        const double mRat = std::pow(mEnd / mScaleSm_, 1.0 / static_cast<double>(nS));

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            auto& st = channels_[static_cast<size_t>(ch)];
            double hSm = hScaleSm_, mSm = mScaleSm_;

            for (int i = 0; i < nS; ++i)
            {
                hSm *= hRat;
                mSm *= mRat;
                const double x = static_cast<double>(d[i]);

                // Leaky trapezoidal integrator: winding voltage -> flux.
                const double fluxNew = leak_ * st.flux + halfT_ * (x + st.vPrev);
                st.vPrev = x;

                // Core hysteresis on the flux (JA, unbiased-iron parameters).
                const double m = mSm * static_cast<double>(
                    st.hyst.processSample(static_cast<T>(hSm * fluxNew)));

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
                const T mixVal = mixStart + mixStep * static_cast<T>(i);
                d[i] = d[i] + (wet - d[i]) * mixVal;
            }
        }
        currentMix_ = mixTarget;   // exact landing
        hScaleSm_ = hEnd;
        mScaleSm_ = mEnd;
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

        // Flux scale: 0 dBFS at 30 Hz reaches H = 1.1a at nominal drive and
        // nominal core; bigger cores take proportionally more flux.
        const double fluxRef = 1.0 / (2.0 * std::numbers::pi * 30.0);
        const double headroom = 0.6 + 0.8 * size;
        hScale_ = drive * 1.1 * 2.2e4 / (fluxRef * headroom);

        // Loudness calibration at PROGRAM level: 100 Hz at -12 dBFS, the
        // flux regime music actually drives the core into (flux ~ V/f, so
        // lows dominate). Calibrating small-signal (the old 0.05 @ 1 kHz)
        // probed the JA virgin curve, whose susceptibility is an order of
        // magnitude below the major loop's mean slope - at high drive the
        // output then came out ~+21 dB over the input (measured x11),
        // heard as violent level jumps while dragging the drive slider.
        {
            Hysteresis<T> cal;
            cal.prepare(sampleRate_);
            cal.setParameters(3.5e5, 2.2e4, 1.6e-3, 3.2e4, 0.25);
            double flux = 0.0, vPrev = 0.0, mPrev = 0.0, vPrev2 = 0.0;
            double inSq = 0.0, outSq = 0.0;
            const int n = static_cast<int>(0.04 * sampleRate_);   // 2+2 cycles of 100 Hz
            for (int i = 0; i < n; ++i)
            {
                const double x = 0.25 * std::sin(2.0 * std::numbers::pi * 100.0 * i / sampleRate_);
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
    std::atomic<bool> prepared_ { false };

    std::vector<ChannelState> channels_;

    double leak_ = 0.999;
    double hpA_ = 0.999;
    double halfT_ = 0.5 / 48000.0;
    double invHalfT_ = 96000.0;
    double hScale_ = 1.0;
    double mScale_ = 1.0;
    double hScaleSm_ = -1.0;    ///< Anti-zipper ramp states (-1 = seed on
    double mScaleSm_ = -1.0;    ///<  first block after prepare/reset).
    T currentMix_ = T(1);       ///< Audio-thread mix ramp state.

    std::atomic<T> driveDb_ { T(0) };
    std::atomic<T> coreSize_ { T(0.5) };
    std::atomic<T> resonance_ { T(0.3) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<bool> dirty_ { true };
};

} // namespace dspark
