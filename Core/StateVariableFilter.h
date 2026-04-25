// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file StateVariableFilter.h
 * @brief Topology-Preserving Transform (TPT) State Variable Filter.
 *
 * A 2nd-order multimode filter based on the Zavalishin/SVF topology.
 * Produces lowpass, highpass, bandpass, notch, allpass, peak, low-shelf,
 * and high-shelf outputs — all from the same structure, often simultaneously.
 *
 * This is the fundamental building block for modular filter design.
 * Unlike cascaded biquads, the SVF:
 * - Is modulation-friendly (cutoff can change every sample without artifacts)
 * - Provides simultaneous multi-output (LP + HP + BP in one processSample call)
 * - Uses zero-delay feedback (no unit-delay in the loop)
 * - Is numerically stable at all frequencies
 *
 * Three levels of API complexity:
 *
 * - **Level 1:** `svf.setCutoff(1000); svf.setMode(LP); svf.processBlock(buf);`
 * - **Level 2:** `svf.setResonance(0.8f); auto [lp,hp,bp] = svf.processMulti(x, ch);`
 * - **Level 3:** Inherit, access g_/R_/ic1eq_/ic2eq_ for custom topologies.
 *
 * References:
 * - Zavalishin, "The Art of VA Filter Design" (2018), ch. 3-4
 * - Valimaki & Smith, "Principles of Digital Audio" (2012)
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   // Level 1 — simple usage:
 *   dspark::StateVariableFilter<float> svf;
 *   svf.prepare(spec);
 *   svf.setCutoff(2000.0f);
 *   svf.setResonance(0.5f);
 *   svf.setMode(dspark::StateVariableFilter<float>::Mode::LowPass);
 *   svf.processBlock(buffer);
 *
 *   // Level 2 — multi-output (all 3 outputs at once):
 *   auto [lp, hp, bp] = svf.processMultiOutput(sample, channel);
 *
 *   // Level 3 — per-sample modulation:
 *   for (int i = 0; i < numSamples; ++i) {
 *       svf.setCutoff(lfo.getNextSample() * 2000 + 500);  // No artifacts
 *       output[i] = svf.processSample(input[i], 0);
 *   }
 * @endcode
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace dspark {

/**
 * @class StateVariableFilter
 * @brief TPT State Variable Filter with simultaneous multi-output.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class StateVariableFilter
{
public:
    /** @brief Filter output mode for processBlock/processSample. */
    enum class Mode
    {
        LowPass,    ///< 2nd-order lowpass (12 dB/oct).
        HighPass,   ///< 2nd-order highpass (12 dB/oct).
        BandPass,   ///< Bandpass (constant skirt gain).
        Notch,      ///< Band-reject (notch).
        AllPass,    ///< Allpass (phase shift, unity gain).
        Bell,       ///< Parametric bell (boost/cut at frequency).
        LowShelf,   ///< Low shelf (boost/cut below frequency).
        HighShelf   ///< High shelf (boost/cut above frequency).
    };

    /**
     * @brief Result struct for simultaneous multi-output processing.
     *
     * A single processSample call produces all three core outputs.
     * Use structured bindings: `auto [lp, hp, bp] = svf.processMultiOutput(x, ch);`
     */
    struct MultiOutput
    {
        T lowpass;
        T highpass;
        T bandpass;
    };

    ~StateVariableFilter() = default;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the filter.
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place using the selected Mode.
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            for (int ch = 0; ch < nCh; ++ch)
                buffer.getChannel(ch)[i] = processSample(buffer.getChannel(ch)[i], ch);
        }
    }

    /**
     * @brief Processes a single sample (selected Mode output).
     *
     * Modulation-friendly: you can call setCutoff() before every sample
     * without artifacts, unlike biquad filters.
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Filtered output (according to current Mode).
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        auto [lp, hp, bp] = processCore(input, channel);
        return selectOutput(input, lp, hp, bp);
    }

    /**
     * @brief Processes a single sample returning LP, HP, BP simultaneously.
     *
     * This is the Level 2 API for DSP engineers who need multiple outputs
     * from the same filter structure (e.g., crossover design, multiband split).
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return {lowpass, highpass, bandpass} outputs.
     */
    [[nodiscard]] MultiOutput processMultiOutput(T input, int channel) noexcept
    {
        return processCore(input, channel);
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        for (auto& s : state_)
        {
            s.ic1eq = T(0);
            s.ic2eq = T(0);
        }
    }

    // -- Parameters (Level 1) ---------------------------------------------------

    /**
     * @brief Sets the cutoff/center frequency.
     *
     * Can be called per-sample without artifacts (unlike biquads).
     *
     * @param hz Frequency in Hz (20 to Nyquist).
     */
    void setCutoff(T hz) noexcept
    {
        // M7d0: before prepare() is called spec_.sampleRate is 0. In that
        // case `clamp(hz, 20, 0)` would hit `lo > hi`, which is UB per
        // cppreference. Store the requested value verbatim and let
        // updateCoefficients() clamp once the sample rate is known.
        if (spec_.sampleRate > 0.0)
            cutoff_ = std::clamp(hz, T(20), static_cast<T>(spec_.sampleRate) * T(0.499));
        else
            cutoff_ = std::max(hz, T(0));
        updateCoefficients();
    }

    /**
     * @brief Sets the resonance (Q).
     *
     * @param resonance 0.0 = no resonance (Butterworth), 1.0 = self-oscillation.
     */
    void setResonance(T resonance) noexcept
    {
        resonance_ = std::clamp(resonance, T(0), T(1));
        // Map 0-1 to Q: 0.5 (wide) to 50 (near self-osc)
        // R = 1/(2*Q), Q_min=0.5 → R=1, Q_max~50 → R≈0.01
        T Q = T(0.5) + resonance_ * T(49.5);
        R_ = T(1) / (T(2) * Q);
        updateBellR();
    }

    /**
     * @brief Sets the Q factor directly.
     *
     * For DSP engineers who think in Q rather than resonance 0-1.
     *
     * @param q Quality factor (0.5 = Butterworth, 0.707 = standard, 10+ = narrow).
     */
    void setQ(T q) noexcept
    {
        q = std::max(q, T(0.01));
        R_ = T(1) / (T(2) * q);
        resonance_ = std::clamp((q - T(0.5)) / T(49.5), T(0), T(1));
        updateBellR();
    }

    /** @brief Sets the output mode. */
    void setMode(Mode mode) noexcept { mode_ = mode; }

    /**
     * @brief Sets gain for Bell/Shelf modes (in dB).
     * @param dB Gain in decibels.
     */
    void setGain(T dB) noexcept
    {
        gainDb_ = dB;
        A_ = std::pow(T(10), std::abs(dB) / T(40)); // sqrt(linear gain)
        updateBellR();
    }

    // -- Getters ----------------------------------------------------------------

    [[nodiscard]] T getCutoff() const noexcept { return cutoff_; }
    [[nodiscard]] T getResonance() const noexcept { return resonance_; }
    [[nodiscard]] Mode getMode() const noexcept { return mode_; }

protected:
    static constexpr int kMaxChannels = 16;

    struct ChannelState
    {
        T ic1eq = T(0); ///< First integrator state.
        T ic2eq = T(0); ///< Second integrator state.
    };

    std::array<ChannelState, kMaxChannels> state_ {};
    AudioSpec spec_ {};
    T cutoff_    = T(1000);
    T resonance_ = T(0);
    T gainDb_    = T(0);
    T g_  = T(0);    // tan(pi * fc / fs) — TPT coefficient
    T R_  = T(1);    // 1/(2*Q) — damping
    T Rbell_ = T(1); // Mode-specific R for Bell (Zavalishin gain-dependent damping)
    T A_  = T(1);    // sqrt(gain) for shelving/bell
    Mode mode_ = Mode::LowPass;

private:
    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate > 0)
        {
            // M7d0: apply the Nyquist clamp here so that setCutoff values
            // stored before prepare() get normalised once we know the SR.
            cutoff_ = std::clamp(cutoff_, T(20),
                                 static_cast<T>(spec_.sampleRate) * T(0.499));
            g_ = static_cast<T>(std::tan(pi<double> * static_cast<double>(cutoff_)
                                         / spec_.sampleRate));
        }
        updateBellR();
    }

    /** @brief Recomputes Rbell_ from R_, A_, gainDb_ for Bell mode. */
    void updateBellR() noexcept
    {
        if (A_ > T(0))
        {
            // Zavalishin Bell EQ topology: modify damping by gain factor A
            // Boost: decrease R (increase Q) => narrower peak compensates for gain spread
            // Cut:   increase R (decrease Q) => wider notch compensates for gain spread
            if (gainDb_ >= T(0))
                Rbell_ = R_ / A_;
            else
                Rbell_ = R_ * A_;
        }
        else
        {
            Rbell_ = R_;
        }
    }

    /**
     * @brief Core TPT SVF processing — produces all 3 outputs.
     *
     * Implements the Zavalishin TPT SVF topology:
     *   v1 = (input - 2R*ic1eq - ic2eq) * a1
     *   v2 = ic1eq + g*v1
     *   ic1eq = 2*v2 - ic1eq   (trapezoidal update)
     *   ic2eq = 2*v3 - ic2eq
     *
     * where a1 = 1/(1 + 2R*g + g*g)
     */
    [[nodiscard]] MultiOutput processCore(T input, int channel) noexcept
    {
        auto& s = state_[channel];

        // For Bell mode, use gain-adjusted damping (Zavalishin ch. 4)
        T Reff = (mode_ == Mode::Bell) ? Rbell_ : R_;

        T a1 = T(1) / (T(1) + T(2) * Reff * g_ + g_ * g_);
        T a2 = g_ * a1;
        T a3 = g_ * a2;

        T v3 = input - s.ic2eq;
        T v1 = a1 * s.ic1eq + a2 * v3;
        T v2 = s.ic2eq + a2 * s.ic1eq + a3 * v3;

        s.ic1eq = T(2) * v1 - s.ic1eq;
        s.ic2eq = T(2) * v2 - s.ic2eq;

        return { v2, input - T(2) * Reff * v1 - v2, v1 };
    }

    [[nodiscard]] T selectOutput(T input, T lp, T hp, T bp) const noexcept
    {
        switch (mode_)
        {
            case Mode::LowPass:  return lp;
            case Mode::HighPass: return hp;
            case Mode::BandPass: return bp;
            case Mode::Notch:    return lp + hp;       // LP + HP = Notch
            case Mode::AllPass:  return lp + hp - T(2) * R_ * bp;  // HP - 2R*BP + LP
            case Mode::Bell:
            {
                // Bell EQ (Zavalishin ch. 4): bandwidth-correct topology.
                // processCore already used Rbell_ for the SVF damping,
                // so the BP bandwidth is correct. Apply gain via mixing:
                //   boost: output = input + (A^2 - 1) * 2*Rbell * BP
                //   cut:   output = input + (1 - 1/A^2) * 2*Rbell * BP
                T k = T(2) * Rbell_;
                return (gainDb_ >= T(0))
                    ? input + (A_ * A_ - T(1)) * k * bp
                    : input + (T(1) - T(1) / (A_ * A_)) * k * bp;
            }
            case Mode::LowShelf:
            {
                // Low shelf (Zavalishin ch. 4): sqrt(A) in the BP term
                // corrects the transition slope.
                // Boost: A * LP + sqrt(A) * 2R * BP + HP
                // Cut:   (1/A) * LP + (1/sqrt(A)) * 2R * BP + HP
                T sqrtA = std::sqrt(A_);
                T twoR  = T(2) * R_;
                if (gainDb_ >= T(0))
                    return A_ * A_ * lp + sqrtA * twoR * bp + hp;
                else
                    return lp / (A_ * A_) + twoR / sqrtA * bp + hp;
            }
            case Mode::HighShelf:
            {
                // High shelf (Zavalishin ch. 4): sqrt(A) in the BP term
                // corrects the transition slope.
                // Boost: LP + sqrt(A) * 2R * BP + A * HP
                // Cut:   LP + (1/sqrt(A)) * 2R * BP + (1/A) * HP
                T sqrtA = std::sqrt(A_);
                T twoR  = T(2) * R_;
                if (gainDb_ >= T(0))
                    return lp + sqrtA * twoR * bp + A_ * A_ * hp;
                else
                    return lp + twoR / sqrtA * bp + hp / (A_ * A_);
            }
        }
        return lp;
    }
};

} // namespace dspark
