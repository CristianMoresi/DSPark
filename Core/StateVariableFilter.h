// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file StateVariableFilter.h
 * @brief Topology-Preserving Transform (TPT) State Variable Filter.
 *
 * A 2nd-order multimode filter based on the Zavalishin/SVF topology.
 * Produces lowpass, highpass, bandpass, notch, allpass, bell, low-shelf,
 * and high-shelf outputs - all from the same structure, often simultaneously.
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
 * - **Level 2:** `auto [lp,hp,bp] = svf.processMultiOutput(x, ch);`
 * - **Level 3:** Inherit, access g_/R_/ic1eq_/ic2eq_ for custom topologies.
 *
 * References:
 * - Zavalishin, "The Art of VA Filter Design" (2018), ch. 3-4
 * - Simper, "Solving the continuous SVF equations using trapezoidal
 *   integration" (cytomic technical paper, 2013)
 *
 * Threading: owner-managed. prepare/reset are setup-time; all setters and
 * process calls belong to the owning audio thread (see the class note).
 * processBlock installs a DenormalGuard; per-sample callers are expected to
 * guard their own callback (framework convention) - resonant tails decay
 * through the denormal range otherwise.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   // Level 1 - simple usage:
 *   dspark::StateVariableFilter<float> svf;
 *   svf.prepare(spec);
 *   svf.setCutoff(2000.0f);
 *   svf.setResonance(0.5f);
 *   svf.setMode(dspark::StateVariableFilter<float>::Mode::LowPass);
 *   svf.processBlock(buffer);
 *
 *   // Level 2 - multi-output (all 3 outputs at once):
 *   auto [lp, hp, bp] = svf.processMultiOutput(sample, channel);
 *
 *   // Level 3 - per-sample modulation:
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

namespace dspark {

/**
 * @class StateVariableFilter
 * @brief TPT State Variable Filter with simultaneous multi-output.
 *
 * @note **Threading contract:** this is a modulation-oriented filter (its whole
 * point is artifact-free per-sample cutoff changes from the audio
 * thread). Parameter setters are therefore intended
 * to be called from the **audio thread** (or externally synchronised), NOT
 * concurrently with process() from a different thread. Internal coefficients are
 * intentionally non-atomic to keep the per-sample modulation path branch-free.
 * For lock-free cross-thread cutoff control use LadderFilter (atomic) or wrap
 * this filter with your own parameter queue.
 *
 * @note Channels beyond kMaxChannels (16) pass through unprocessed in
 * processBlock(); out-of-range channel indices on the per-sample calls are
 * rejected safely (input passed through / zero side outputs).
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
     *
     * Invalid specs (sample rate not > 0, NaN included) are ignored and the
     * previous state is kept. Clears the integrator state.
     *
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!(spec.sampleRate > 0.0)) return;
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

        // Channel-outer / sample-inner: keeps one channel's state and buffer hot
        // in cache for the whole inner loop (cache-friendly, matches LadderFilter).
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                data[i] = processOne(data[i], ch);
        }
    }

    /**
     * @brief Processes a single sample (selected Mode output).
     *
     * Modulation-friendly: you can call setCutoff() before every sample
     * without artifacts, unlike biquad filters.
     *
     * @note Unlike processBlock(), no DenormalGuard is installed here (the
     *       per-sample FP-environment writes would cost more than the filter
     *       itself); per-sample callers should place one DenormalGuard
     *       around their processing loop.
     *
     * @param input   Input sample.
     * @param channel Channel index in [0, kMaxChannels).
     * @return Filtered output (returns @p input unchanged if @p channel is
     *         out of range).
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (channel < 0 || channel >= kMaxChannels) return input;
        return processOne(input, channel);
    }

    /**
     * @brief Processes a single sample returning LP, HP, BP simultaneously.
     *
     * This is the Level 2 API for DSP engineers who need multiple outputs
     * from the same filter structure (e.g., crossover design, multiband split).
     *
     * @param input   Input sample.
     * @param channel Channel index in [0, kMaxChannels). Out of range returns
     *                {input, 0, 0} (an LP/HP split still reconstructs the
     *                input) without touching any state.
     * @return {lowpass, highpass, bandpass} outputs.
     */
    [[nodiscard]] MultiOutput processMultiOutput(T input, int channel) noexcept
    {
        if (channel < 0 || channel >= kMaxChannels) return { input, T(0), T(0) };
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
     * Non-finite values (NaN/Inf) are ignored - they would poison the
     * coefficients permanently otherwise. Before prepare() the sample rate
     * is unknown, so the raw request is stored and clamped to
     * [20, sampleRate * 0.499] once prepare() runs.
     *
     * @param hz Frequency in Hz.
     */
    void setCutoff(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        cutoff_ = std::max(hz, T(0));
        updateCoefficients();
    }

    /**
     * @brief Sets the resonance amount.
     *
     * @param resonance 0.0 maps to Q = 0.5 (critically damped, no peak) and
     *                  1.0 to Q = 50 (a strong resonant peak, near but not at
     *                  self-oscillation). Non-finite values are ignored.
     */
    void setResonance(T resonance) noexcept
    {
        if (!std::isfinite(resonance)) return;  // NaN would poison R_
        resonance_ = std::clamp(resonance, T(0), T(1));
        // Map 0-1 to Q: 0.5 (wide) to 50 (near self-osc). R = 1/(2*Q).
        const T Q = T(0.5) + resonance_ * T(49.5);
        R_ = T(1) / (T(2) * Q);
        updateBellR();
        updateDerivedCoeffs();
    }

    /**
     * @brief Sets the Q factor directly.
     *
     * For DSP engineers who think in Q rather than resonance 0-1.
     *
     * @param q Quality factor, floored at 0.01 (0.5 = critically damped,
     *          0.707 = Butterworth, 10+ = narrow). Non-finite values are
     *          ignored.
     */
    void setQ(T q) noexcept
    {
        if (!std::isfinite(q)) return;  // NaN would poison R_
        q = std::max(q, T(0.01));
        R_ = T(1) / (T(2) * q);
        resonance_ = std::clamp((q - T(0.5)) / T(49.5), T(0), T(1));
        updateBellR();
        updateDerivedCoeffs();
    }

    /** @brief Sets the output mode. */
    void setMode(Mode mode) noexcept
    {
        mode_ = mode;
        updateDerivedCoeffs();
    }

    /**
     * @brief Sets gain for Bell/Shelf modes (in dB).
     * @param dB Gain in decibels. Non-finite values are ignored.
     */
    void setGain(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;  // NaN/Inf would poison A_
        gainDb_ = dB;
        A_ = std::pow(T(10), std::abs(dB) / T(40)); // sqrt(linear gain)
        updateBellR();
        updateDerivedCoeffs();
    }

    // -- Getters ----------------------------------------------------------------

    [[nodiscard]] T getCutoff() const noexcept { return cutoff_; }
    [[nodiscard]] T getResonance() const noexcept { return resonance_; }
    [[nodiscard]] T getQ() const noexcept { return T(1) / (T(2) * R_); }
    [[nodiscard]] T getGain() const noexcept { return gainDb_; }
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
    T g_  = T(0);    // tan(pi * fc / fs) - TPT coefficient
    T R_  = T(1);    // 1/(2*Q) - damping
    T Rbell_ = T(1); // Mode-specific R for Bell (Zavalishin gain-dependent damping)
    T A_  = T(1);    // sqrt(gain) for shelving/bell
    Mode mode_ = Mode::LowPass;

    // Cached derived coefficients (recomputed by updateDerivedCoeffs()).
    T effG_ = T(0);  // possibly shelf-prewarped g
    T effR_ = T(1);  // mode-effective damping (R_ or Rbell_)
    T a1_ = T(1), a2_ = T(0), a3_ = T(0);

private:
    /**
     * @brief Clamps the stored cutoff and derives the TPT coefficient.
     *
     * The Nyquist clamp lives here (not in setCutoff) so that requests
     * stored before prepare() get normalised once the sample rate is known,
     * and getCutoff() always reports the clamped value.
     */
    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate > 0)
        {
            cutoff_ = std::clamp(cutoff_, T(20),
                                 static_cast<T>(spec_.sampleRate) * T(0.499));
            g_ = static_cast<T>(std::tan(pi<double> * static_cast<double>(cutoff_)
                                         / spec_.sampleRate));
        }
        updateBellR();
        updateDerivedCoeffs();
    }

    /** @brief Recomputes Rbell_ from R_, A_, gainDb_ for Bell mode. */
    void updateBellR() noexcept
    {
        // Zavalishin Bell EQ topology: modify damping by gain factor A
        // (A_ >= 1 always - setGain uses |dB| and rejects non-finite input).
        // Boost: decrease R (increase Q) => narrower peak compensates for gain spread
        // Cut:   increase R (decrease Q) => wider notch compensates for gain spread
        if (gainDb_ >= T(0))
            Rbell_ = R_ / A_;
        else
            Rbell_ = R_ * A_;
    }

    /**
     * @brief Recomputes the cached TPT coefficients (a1, a2, a3).
     *
     * Called from every setter, so the per-sample path never divides when the
     * parameters are static. Per-sample modulation still works: setCutoff()
     * (or any other setter) refreshes the cache at the same cost the old
     * per-sample computation had.
     *
     * Shelf modes pre-warp g by sqrt(A) (Simper's convention) so the
     * half-gain point of the shelf lands exactly on the nominal frequency.
     */
    void updateDerivedCoeffs() noexcept
    {
        effR_ = (mode_ == Mode::Bell) ? Rbell_ : R_;

        T gEff = g_;
        if (mode_ == Mode::LowShelf || mode_ == Mode::HighShelf)
        {
            const T sqrtAs = std::sqrt((gainDb_ >= T(0)) ? A_ : T(1) / A_);
            gEff = (mode_ == Mode::LowShelf) ? g_ / sqrtAs : g_ * sqrtAs;
        }

        effG_ = gEff;
        a1_ = T(1) / (T(1) + T(2) * effR_ * gEff + gEff * gEff);
        a2_ = gEff * a1_;
        a3_ = gEff * a2_;
    }

    /** @brief Selected-mode output for one sample; no bounds check (internal). */
    [[nodiscard]] T processOne(T input, int channel) noexcept
    {
        const MultiOutput m = processCore(input, channel);
        return selectOutput(input, m.lowpass, m.highpass, m.bandpass);
    }

    /**
     * @brief Core TPT SVF processing - produces all 3 outputs.
     *
     * Implements the Zavalishin/Simper TPT SVF topology:
     *   v3 = input - ic2eq
     *   v1 = a1*ic1eq + a2*v3          (bandpass)
     *   v2 = ic2eq + a2*ic1eq + a3*v3  (lowpass)
     *   ic1eq = 2*v1 - ic1eq           (trapezoidal update)
     *   ic2eq = 2*v2 - ic2eq
     *
     * where a1 = 1/(1 + 2R*g + g*g). The coefficients come pre-computed from
     * updateDerivedCoeffs(), so this hot path is division-free.
     */
    [[nodiscard]] MultiOutput processCore(T input, int channel) noexcept
    {
        auto& s = state_[channel];

        const T v3 = input - s.ic2eq;
        const T v1 = a1_ * s.ic1eq + a2_ * v3;
        const T v2 = s.ic2eq + a2_ * s.ic1eq + a3_ * v3;

        s.ic1eq = T(2) * v1 - s.ic1eq;
        s.ic2eq = T(2) * v2 - s.ic2eq;

        return { v2, input - T(2) * effR_ * v1 - v2, v1 };
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
                //   cut:   output = input + (1/A^2 - 1) * 2*Rbell * BP
                // At the centre frequency 2*Rbell*bp == input, so the mix factor
                // must be (A^2 - 1) to reach gain A^2 (boost) and (1/A^2 - 1) to
                // reach gain 1/A^2 (cut). The cut term must be NEGATIVE - the old
                // (1 - 1/A^2) was positive and boosted instead of cutting.
                const T k = T(2) * Rbell_;
                return (gainDb_ >= T(0))
                    ? input + (A_ * A_ - T(1)) * k * bp
                    : input + (T(1) / (A_ * A_) - T(1)) * k * bp;
            }
            case Mode::LowShelf:
            {
                // Canonical Simper low shelf. With As = A_ (boost) or 1/A_
                // (cut), where A_ = 10^(|dB|/40) = sqrt(linear gain), and the
                // pre-warped g from updateDerivedCoeffs():
                //   out = As^2 * LP + 2R * As * BP + HP
                // DC gain = As^2 (= linear gain), Nyquist gain = 1, and the
                // half-gain point sits exactly on the nominal frequency.
                const T As   = (gainDb_ >= T(0)) ? A_ : T(1) / A_;
                const T twoR = T(2) * R_;
                return As * As * lp + As * twoR * bp + hp;
            }
            case Mode::HighShelf:
            {
                // Canonical Simper high shelf (mirror of the low shelf):
                //   out = LP + 2R * As * BP + As^2 * HP
                const T As   = (gainDb_ >= T(0)) ? A_ : T(1) / A_;
                const T twoR = T(2) * R_;
                return lp + As * twoR * bp + As * As * hp;
            }
        }
        return lp;
    }
};

} // namespace dspark
