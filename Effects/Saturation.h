// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Saturation.h
 * @brief Professional multi-algorithm saturation processor — 100% standalone.
 *
 * Ten saturation algorithms from subtle analog warmth to aggressive digital distortion,
 * with full Mid/Side stereo processing, analog drift simulation, pre/post filtering,
 * dry/wet mixing, DC blocking, and artifact-free algorithm crossfading.
 *
 * Dependencies: DSP/Core headers only (C++20 STL).
 *
 * @tparam SampleType float or double.
 *
 * @code
 *   dspark::Saturation<float> sat;
 *   sat.prepare({ .sampleRate = 48000.0, .maxBlockSize = 512, .numChannels = 2 });
 *   sat.setAlgorithm(dspark::Saturation<float>::Algorithm::Tube);
 *   sat.setDrive(12.0f);
 *   sat.setMix(0.7f);
 *   sat.process(buffer.toView());
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/Oversampling.h"
#include "../Core/Smoothers.h"
#include "../Core/AnalogRandom.h"
#include "../Core/SpscQueue.h"
#include "../Core/SpinLock.h"
#include "MidSide.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

namespace dspark {

// ============================================================================
// Internal algorithm implementations
// ============================================================================
namespace detail {

template <typename T>
class SaturationAlgorithm
{
public:
    virtual ~SaturationAlgorithm() = default;
    virtual void prepare(const AudioSpec& spec) noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual void update(T /*driveGain*/, T /*character*/, const AudioSpec& /*spec*/) noexcept {}
    virtual T processSample(T sample, T drive, T character, int channel) noexcept = 0;

    virtual void processBlock(AudioBufferView<T> buffer, T drive, T character) noexcept
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            T* data = buffer.getChannel(ch);
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                data[i] = processSample(data[i], drive, character, ch);
        }
    }
};

// -- SoftClip (tanh) ---------------------------------------------------------
template <typename T>
class TanhAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    T processSample(T sample, T drive, T character, int) noexcept override
    {
        T x    = sample * drive;
        T bias = character * T(0.3);
        return fastTanh(x + bias) - fastTanh(bias);
    }
};

// -- Tube (12AX7-style asymmetric triode model) -----------------------------
// Bug-fix: previous implementation used `1 - exp(-x^1.5)` which is bounded in
// [0, 1] and never approaches identity for small inputs — meaning every signal
// was heavily attenuated even at unity drive (e.g. input 0.5 -> 0.193, ~8 dB
// loss before any "saturation" was intended).
//
// The replacement uses a tanh-based asymmetric saturator that:
//   - Is essentially transparent for small inputs (linear gain ≈ 1)
//   - Soft-saturates only as |x| approaches and exceeds 1 (0 dBFS)
//   - Negative half is slightly more compressed than positive — this
//     asymmetry generates the predominant 2nd-harmonic content that gives
//     single-ended triodes their characteristic "warmth"
//   - `character` (0..1) controls how pronounced that asymmetry is
template <typename T>
class TubeAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    T processSample(T sample, T drive, T character, int) noexcept override
    {
        T x = sample * drive;

        // Negative-half compression multiplier: 1.15 (subtle) to 1.65 (strong)
        T asym = T(1.15) + character * T(0.5);

        // Asymmetric tanh: positive half gets standard curve, negative half
        // gets pre-multiplied input so it saturates earlier.
        return (x >= T(0)) ? fastTanh(x) : fastTanh(x * asym);
    }
};

// -- HardClip ----------------------------------------------------------------
template <typename T>
class HardClipAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    T processSample(T sample, T drive, T character, int) noexcept override
    {
        T bias = character * T(0.3);
        T x = sample * drive + bias;
        T clipped = std::clamp(x, T(-1), T(1));
        return clipped - std::clamp(bias, T(-1), T(1));
    }
};

// -- Exciter (polynomial waveshaper) -----------------------------------------
template <typename T>
class ExciterAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    T processSample(T sample, T drive, T character, int) noexcept override
    {
        T x      = std::clamp(sample * drive, T(-10), T(10));
        T x2     = x * x;
        T x3     = x2 * x;
        T result = x + character * T(0.25) * x2 - T(0.15) * x3;
        return std::clamp(result, T(-1), T(1));
    }
};

// -- Wavefolder (sin + first-order ADAA) -------------------------------------
template <typename T>
class WavefolderAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 8;
    std::array<T, kMaxCh> lastX_ {};
    std::array<T, kMaxCh> lastF_ {};

public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    void reset() noexcept override
    {
        lastX_.fill(T(0));
        lastF_.fill(T(-1)); // -cos(0)
    }
    T processSample(T sample, T drive, T character, int ch) noexcept override
    {
        T x   = sample * drive * twoPi<T> + character * pi<T>;
        T F_x = -std::cos(x);
        T diff = x - lastX_[ch];

        T result;
        if (std::abs(diff) > T(1e-5))
            result = (F_x - lastF_[ch]) / diff;
        else
            result = std::sin(x);

        lastX_[ch] = x;
        lastF_[ch] = F_x;
        return result;
    }
};

// -- Bitcrusher (TPDF dither) ------------------------------------------------
template <typename T>
class BitcrusherAlgorithm final : public SaturationAlgorithm<T>
{
    AnalogRandom::Generator<T> ditherGen_;

public:
    void prepare(const AudioSpec& spec) noexcept override { ditherGen_.prepare(spec.sampleRate); }
    void reset() noexcept override {}
    T processSample(T sample, T drive, T /*character*/, int) noexcept override
    {
        T clamped  = std::clamp(drive, T(1), T(100));
        T bitDepth = mapRange(clamped, T(1), T(100), T(16), T(2));
        T steps    = std::pow(T(2), bitDepth);
        T invSteps = T(1) / steps;
        T dither   = (ditherGen_.getNextSample() - ditherGen_.getNextSample()) * invSteps;
        return invSteps * std::round((sample + dither) * steps);
    }
};

// -- Tape (hysteresis model + head bump + HF rolloff) ------------------------
// Reference: Jatin Chowdhury, "Real-time Physical Modelling for Analog Tape
// Machines" (DAFx 2019). Uses a simplified Langevin-based hysteresis model
// instead of the full Jiles-Atherton ODE, for RT performance.
//
// Key behaviors modelled:
//   - Magnetic hysteresis: output depends on history (asymmetric S-curve)
//   - Head bump: +1.5 dB resonance around 60-100 Hz (playback head proximity)
//   - HF rolloff: increasing drive causes more HF loss (gap losses, bias)
//   - Saturation curve follows Langevin function: L(x) = coth(x) - 1/x
template <typename T>
class TapeAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 8;
    std::array<Biquad<T, 1>, kMaxCh> preFilters_;   // Head bump
    std::array<Biquad<T, 1>, kMaxCh> postFilters_;  // HF rolloff
    std::array<T, kMaxCh>            M_ {};          // Magnetisation state (hysteresis)
    std::array<T, kMaxCh>            lastH_ {};      // Previous input H-field
    int numChannels_ = 0;

    // Langevin function: L(x) = coth(x) - 1/x
    // Approximation for numerical stability near x=0
    static T langevin(T x) noexcept
    {
        T ax = std::abs(x);
        if (ax < T(0.01))
            return x / T(3);  // Taylor series: L(x) ≈ x/3 for small x
        return T(1) / std::tanh(x) - T(1) / x;
    }

    // Derivative of Langevin: L'(x) = -csch²(x) + 1/x²
    static T langevinDeriv(T x) noexcept
    {
        T ax = std::abs(x);
        if (ax < T(0.01))
            return T(1) / T(3);
        T csch = T(1) / std::sinh(x);
        return -csch * csch + T(1) / (x * x);
    }

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = spec.numChannels;
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : preFilters_)  f.reset();
        for (auto& f : postFilters_) f.reset();
        M_.fill(T(0));
        lastH_.fill(T(0));
    }
    void update(T drive, T /*character*/, const AudioSpec& spec) noexcept override
    {
        auto driveDb = gainToDecibels(drive, T(-100));
        // Head bump: resonant peak at ~80Hz, more prominent with higher drive
        T bumpGain = T(1.5) + std::min(driveDb * T(0.05), T(3.0));
        auto peakCoeffs = BiquadCoeffs<T>::makePeak(spec.sampleRate, 80.0, 0.6,
                                                     static_cast<double>(bumpGain));
        // HF rolloff: more drive = more HF loss (simulates bias/gap losses)
        auto lpFreq = std::max(2000.0, 18000.0 - static_cast<double>(driveDb) * 250.0);
        auto lpCoeffs = BiquadCoeffs<T>::makeLowPass(spec.sampleRate, lpFreq, 0.55);

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            preFilters_[ch].setCoeffs(peakCoeffs);
            postFilters_[ch].setCoeffs(lpCoeffs);
        }
    }
    T processSample(T sample, T drive, T character, int ch) noexcept override
    {
        // Head bump pre-filter
        T filtered = preFilters_[ch].processSample(sample, 0);

        // Hysteresis model (simplified Chowdhury/Langevin)
        // Parameters based on tape formulation:
        //   Ms = saturation magnetisation (controls output level)
        //   a  = shape parameter (controls curve width)
        //   c  = inter-domain coupling (controls hysteresis width)
        //
        // Bug-fix: previous Ms=1 with a in [0.8, 1.2] gave a linear gain of
        // Ms/(3a) ≈ 0.28..0.42 — every signal was attenuated 7..11 dB before
        // any saturation behaviour was meant to occur. The relationship
        // Ms = 3*a keeps the linear gain at unity regardless of `character`,
        // while smaller `a` (higher character) makes the saturation knee
        // come in sooner.
        T a  = T(1.0) - character * T(0.25);   // 1.0 (clean) .. 0.75 (harder)
        T Ms = T(3) * a;                        // unity linear gain
        T c  = T(0.6) + character * T(0.2);     // 0.6 .. 0.8 hysteresis width

        T H = filtered * drive;  // Applied field
        T dH = H - lastH_[ch];

        // Effective field including inter-domain coupling
        T He = H + c * M_[ch];

        // Anhysteretic magnetisation (what M would be without hysteresis)
        T Man = Ms * langevin(He / a);

        // Differential: dM/dH with hysteresis
        T delta = (dH >= T(0)) ? T(1) : T(-1);
        T ManDiff = Man - M_[ch];
        T dManDH = Ms * langevinDeriv(He / a) / a;

        // Simplified implicit solve: M_new = M + dM
        T denominator = T(1) - c * dManDH;
        if (std::abs(denominator) < T(1e-10))
            denominator = std::copysign(T(1e-10), denominator);

        T dM;
        if (delta * ManDiff > T(0))
            dM = (ManDiff / denominator) * std::abs(dH) / (a + std::abs(dH));
        else
            dM = T(0); // Reversal: M stays (hysteresis memory)

        M_[ch] = std::clamp(M_[ch] + dM, -Ms, Ms);
        lastH_[ch] = H;

        // Post-filter: HF rolloff
        return postFilters_[ch].processSample(M_[ch], 0);
    }
};

// -- Transformer (frequency-dependent: heavy LF, light HF) ------------------
template <typename T>
class TransformerAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 8;
    std::array<Biquad<T, 1>, kMaxCh> lpFilters_;
    int numChannels_ = 0;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = spec.numChannels;
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : lpFilters_) f.reset();
    }
    void update(T /*drive*/, T /*character*/, const AudioSpec& spec) noexcept override
    {
        auto c = BiquadCoeffs<T>::makeLowPass(spec.sampleRate, 250.0, 0.707);
        for (int ch = 0; ch < numChannels_; ++ch)
            lpFilters_[ch].setCoeffs(c);
    }
    T processSample(T sample, T drive, T character, int ch) noexcept override
    {
        // Output-transformer behaviour: low frequencies hit core saturation
        // first (iron permeability is highest at LF); high frequencies pass
        // largely unaffected. The asymmetry between bands is what makes a
        // transformer sound different from a wide-band tanh.
        //
        // Bug-fix: previous version used drive*2 / drive*0.5 — at unity drive
        // (0 dB) the lows were already pushed through tanh with 6 dB of extra
        // gain, producing audible distortion on every mastered file. The
        // factors below keep the LF-bias-toward-saturation character but at
        // a fraction of the original aggression.
        T low  = lpFilters_[ch].processSample(sample, 0);
        T high = sample - low;
        T bias = character * T(0.2);
        T satLow  = fastTanh((low  + bias) * drive * T(1.3)) - fastTanh(bias);
        T satHigh = fastTanh((high + bias) * drive * T(0.7)) - fastTanh(bias);
        return satLow + satHigh;
    }
};

// -- Downsample (sample rate reduction) --------------------------------------
template <typename T>
class DownsampleAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 8;
    std::array<Biquad<T, 1>, kMaxCh> aaFilters_;
    std::array<T, kMaxCh>            lastSample_ {};
    std::array<int, kMaxCh>          counter_    {};
    int numChannels_ = 0;

    static int driveToReduction(T drive)
    {
        T clamped = std::clamp(drive, T(1), T(100));
        return std::max(1, static_cast<int>(mapRange(clamped, T(1), T(100), T(1), T(50))));
    }

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = spec.numChannels;
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : aaFilters_) f.reset();
        lastSample_.fill(T(0));
        counter_.fill(0);
    }
    void update(T drive, T, const AudioSpec& spec) noexcept override
    {
        int reduction = driveToReduction(drive);
        auto c = BiquadCoeffs<T>::makeLowPass(spec.sampleRate,
                     spec.sampleRate / (2.5 * reduction), 0.707);
        for (int ch = 0; ch < numChannels_; ++ch)
            aaFilters_[ch].setCoeffs(c);
    }
    T processSample(T sample, T drive, T /*character*/, int ch) noexcept override
    {
        int reduction = driveToReduction(drive);
        T filtered = aaFilters_[ch].processSample(sample, 0);
        if (counter_[ch] % reduction == 0)
            lastSample_[ch] = filtered;
        counter_[ch] = (counter_[ch] + 1) % (reduction * 1024);
        return lastSample_[ch];
    }
};

// -- MultiStage (Tube → Tape → Transformer cascade) -------------------------
template <typename T>
class MultiStageAlgorithm final : public SaturationAlgorithm<T>
{
    TubeAlgorithm<T>        tube_;
    TapeAlgorithm<T>        tape_;
    TransformerAlgorithm<T> xfmr_;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        tube_.prepare(spec);
        tape_.prepare(spec);
        xfmr_.prepare(spec);
    }
    void reset() noexcept override { tube_.reset(); tape_.reset(); xfmr_.reset(); }
    void update(T drive, T character, const AudioSpec& spec) noexcept override
    {
        tape_.update(drive * T(0.6), character, spec);
        xfmr_.update(drive * T(0.8), character, spec);
    }
    T processSample(T sample, T drive, T character, int ch) noexcept override
    {
        // Bug-fix: previous version multiplied the output of each stage by
        // gainComp = 1/sqrt(drive*0.5). For drive=1 (=0 dB) this was 1.414×
        // applied TWICE (≈ 2× total), and for low drive it grew unbounded
        // (drive=0.1 → 4.47×) — the inverse of what makeup gain should do.
        // The cascade is now drive-distributed only: each stage sees a
        // fraction of the total drive so the chain doesn't compound into
        // destructive distortion at unity drive.
        T tubeOut  = tube_.processSample(sample,  drive * T(0.5), character, ch);
        T tapeOut  = tape_.processSample(tubeOut, drive * T(0.6), character, ch);
        return       xfmr_.processSample(tapeOut, drive * T(0.8), character, ch);
    }
};

} // namespace detail

// ============================================================================
// Saturation — Public API
// ============================================================================

/**
 * @class Saturation
 * @brief Multi-algorithm saturation processor with full stereo and analog simulation.
 *
 * Thread-safe setters can be called from any thread (GUI, automation).
 * Parameters are delivered lock-free to the audio thread via an internal SPSC queue.
 * Algorithm switching is crossfaded for artifact-free transitions.
 *
 * Features excluded by design (belong in host/wrapper layer):
 * - Metering (use LevelFollower separately)
 * - Harmonic analysis (requires FFT — host responsibility)
 * - Task delegation (JUCE-specific concept)
 *
 * @tparam SampleType float or double.
 */
template <typename SampleType>
class Saturation
{
    static_assert(std::is_floating_point_v<SampleType>,
                  "Saturation: SampleType must be float or double.");

public:
    // -- Enums ---------------------------------------------------------------

    enum class Algorithm
    {
        Tube, Tape, Transformer, SoftClip, HardClip,
        Exciter, Wavefolder, Bitcrusher, Downsample, MultiStage
    };

    enum class ProcessingMode { Stereo, MidOnly, SideOnly, MidSide };

    enum class OutputMode { Normal, WetOnly, Delta };

    // -- Lifecycle -----------------------------------------------------------

    Saturation()
    {
        pool_[0]  = std::make_unique<detail::TubeAlgorithm<SampleType>>();
        pool_[1]  = std::make_unique<detail::TapeAlgorithm<SampleType>>();
        pool_[2]  = std::make_unique<detail::TransformerAlgorithm<SampleType>>();
        pool_[3]  = std::make_unique<detail::TanhAlgorithm<SampleType>>();
        pool_[4]  = std::make_unique<detail::HardClipAlgorithm<SampleType>>();
        pool_[5]  = std::make_unique<detail::ExciterAlgorithm<SampleType>>();
        pool_[6]  = std::make_unique<detail::WavefolderAlgorithm<SampleType>>();
        pool_[7]  = std::make_unique<detail::BitcrusherAlgorithm<SampleType>>();
        pool_[8]  = std::make_unique<detail::DownsampleAlgorithm<SampleType>>();
        pool_[9]  = std::make_unique<detail::MultiStageAlgorithm<SampleType>>();

        active_.store(pool_[static_cast<int>(Algorithm::SoftClip)].get());
        next_.store(nullptr);
    }

    ~Saturation() = default;

    Saturation(const Saturation&)            = delete;
    Saturation& operator=(const Saturation&) = delete;

    /**
     * @brief Prepares all internal resources. Call once in your prepare() method.
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;

        for (auto& algo : pool_)
            if (algo) algo->prepare(spec);

        preFilter_.reset();
        postFilter_.reset();
        dcBlocker_.reset();
        // One-shot DC blocker coefficients — depend only on sampleRate.
        dcBlocker_.setCoeffs(BiquadCoeffs<SampleType>::makeDcBlocker(spec.sampleRate));
        // Invalidate cached filter-parameter values so the first process()
        // block recomputes coefficients from the smoother's initial state.
        lastPreHpFreq_    = -1.0f;
        lastPostTiltFreq_ = -1.0f;
        lastPostTiltGain_ = std::numeric_limits<float>::quiet_NaN();
        dryWetMixer_.prepare(spec);
        if (oversampler_)
            oversampler_->prepare(spec);
        tempBuffer_.resize(spec.numChannels,
                          spec.maxBlockSize * std::max(1, oversamplingFactor_));

        auto sr = spec.sampleRate;
        driveSmoother_.reset(sr, 20.0f, 0.707f, 0.0f);
        mixSmoother_.reset(sr, 20.0f, 1.0f);
        characterSmoother_.reset(sr, 20.0f, 0.0f);
        driftSmoother_.reset(sr, 500.0f, 0.0f);
        preHpSmoother_.reset(sr, 30.0f, 0.707f, 20.0f);
        postTiltFreqSmoother_.reset(sr, 30.0f, 0.707f, 1000.0f);
        postTiltGainSmoother_.reset(sr, 30.0f, 0.0f);
        outputGainSmoother_.reset(sr, 20.0f, 0.0f);
        crossfader_.reset(sr, 10.0f, 1.0f);

        leftDrift_.prepare(sr);
        rightDrift_.prepare(sr);
        // Decorrelate L/R drift so the emulated analogue drift produces
        // stereo width instead of a mono-modulated signal. The two seeds
        // are arbitrary large primes separated by enough bits to avoid
        // splitmix64 collisions in the first few samples.
        leftDrift_.reseed(0x9E3779B97F4A7C15ULL);
        rightDrift_.reseed(0xBF58476D1CE4E5B9ULL);

        reset();

        prepared_ = true;
    }

    /** @brief Clears all internal state. Real-time safe. */
    void reset()
    {
        for (auto& algo : pool_)
            if (algo) algo->reset();

        preFilter_.reset();
        postFilter_.reset();
        dcBlocker_.reset();
        dryWetMixer_.reset();
        if (oversampler_)
            oversampler_->reset();

        driveSmoother_.skip();
        mixSmoother_.skip();
        characterSmoother_.skip();
        driftSmoother_.skip();
        preHpSmoother_.skip();
        postTiltFreqSmoother_.skip();
        postTiltGainSmoother_.skip();
        outputGainSmoother_.skip();
        crossfader_.setCurrentAndTargetValue(1.0f);

        prevBlendSample_.fill(SampleType(0));
        prevSlewSample_.fill(SampleType(0));
    }

    /**
     * @brief Processes audio in-place (AudioProcessor contract).
     * @param buffer Mutable audio buffer view.
     */
    void processBlock(AudioBufferView<SampleType> buffer) noexcept { if (!prepared_) return; process(buffer); }

    /**
     * @brief Processes audio in-place.
     *
     * The main audio processing call. Apply to your buffer every block.
     *
     * @param buffer Mutable audio buffer view.
     */
    void process(AudioBufferView<SampleType> buffer) noexcept
    {
        if (!prepared_) return;
        handleParameterChanges();

        // Capture dry signal for mix
        dryWetMixer_.pushDry(buffer);

        // Pre-filter (high-pass to remove low-end before saturation).
        //
        // C1+A1 fix (v2 — forensic): process in sub-blocks of kCoefRefresh
        // samples. Within each sub-block we advance the smoother by the
        // exact chunk size (so the effective ramp time matches the user's
        // configuration, not blockSize × config) and rebuild the biquad
        // coefficients only when the smoothed value actually changed.
        //
        // The previous revision called getNextValue() ONCE per block, which
        // advanced the smoother by a single sample → effective ramp time
        // ballooned by a factor of blockSize (~256×) and users perceived
        // the pre-filter as "stuck" relative to the control. That caused
        // the regressions reported after Phase 2 (pre-HP locked at a
        // stale freq → audible as a static aggressive filter).
        {
            const int numSamples = buffer.getNumSamples();
            constexpr int kCoefRefresh = 16;
            for (int i = 0; i < numSamples; i += kCoefRefresh)
            {
                const int chunk = std::min(kCoefRefresh, numSamples - i);
                float curFreq = lastPreHpFreq_;
                for (int k = 0; k < chunk; ++k)
                    curFreq = preHpSmoother_.getNextValue();

                if (curFreq != lastPreHpFreq_)
                {
                    auto c = BiquadCoeffs<SampleType>::makeHighPass(
                                 spec_.sampleRate, static_cast<double>(curFreq));
                    preFilter_.setCoeffs(c);
                    lastPreHpFreq_ = curFreq;
                }
                auto subView = buffer.getSubView(i, chunk);
                preFilter_.processBlock(subView);
            }
        }

        // Mid/Side encode if needed
        const bool isMidSide =
            (procMode_ == ProcessingMode::MidOnly ||
             procMode_ == ProcessingMode::SideOnly ||
             procMode_ == ProcessingMode::MidSide) &&
            buffer.getNumChannels() == 2;

        if (isMidSide) MidSide<SampleType>::encode(buffer);

        // Core saturation (with optional oversampling)
        if (oversamplingFactor_ > 1 && oversampler_)
        {
            auto upView = oversampler_->upsample(buffer);
            processSaturation(upView);
            oversampler_->downsample(buffer);
        }
        else
        {
            processSaturation(buffer);
        }

        if (isMidSide) MidSide<SampleType>::decode(buffer);

        // Post-filter (tilt EQ). Same sub-block strategy as the pre-filter:
        // advance the smoothers by the exact sub-block size, rebuild the
        // peak-EQ coefficients only when freq or gain actually changed.
        // makePeak() is one of the most expensive BiquadCoeffs factories
        // (tan + exp + sqrt), so gating it on a scalar compare is a clear
        // win for the steady-state path.
        {
            const int numSamples = buffer.getNumSamples();
            constexpr int kCoefRefresh = 16;
            for (int i = 0; i < numSamples; i += kCoefRefresh)
            {
                const int chunk = std::min(kCoefRefresh, numSamples - i);
                float curFreq = lastPostTiltFreq_;
                float curGain = lastPostTiltGain_;
                for (int k = 0; k < chunk; ++k)
                {
                    curFreq = postTiltFreqSmoother_.getNextValue();
                    curGain = postTiltGainSmoother_.getNextValue();
                }

                if (curFreq != lastPostTiltFreq_ || curGain != lastPostTiltGain_)
                {
                    auto c = BiquadCoeffs<SampleType>::makePeak(
                                 spec_.sampleRate,
                                 static_cast<double>(curFreq),
                                 0.707,
                                 static_cast<double>(curGain));
                    postFilter_.setCoeffs(c);
                    lastPostTiltFreq_ = curFreq;
                    lastPostTiltGain_ = curGain;
                }
                auto subView = buffer.getSubView(i, chunk);
                postFilter_.processBlock(subView);
            }
        }

        // DC blocker. Coefficients depend only on sampleRate (constant after
        // prepare), so the one-time setCoeffs now lives in prepare() and the
        // hot path just processes the block.
        if (dcBlockingEnabled_)
            dcBlocker_.processBlock(buffer);

        // Output gain
        applyOutputGain(buffer);

        // Output mode handling
        if (outputMode_ == OutputMode::WetOnly)
        {
            // Buffer already contains wet — skip dry/wet mix
        }
        else if (outputMode_ == OutputMode::Delta)
        {
            // Wet - Dry = harmonics only
            const int nCh = std::min(buffer.getNumChannels(), dryWetMixer_.getDryNumChannels());
            const int nS  = std::min(buffer.getNumSamples(), dryWetMixer_.getDryCapturedSamples());
            for (int ch = 0; ch < nCh; ++ch)
            {
                SampleType*       wet = buffer.getChannel(ch);
                const SampleType* dry = dryWetMixer_.getDryChannel(ch);
                for (int i = 0; i < nS; ++i)
                    wet[i] -= dry[i];
            }
        }
        else // Normal
        {
            dryWetMixer_.mixWet(buffer, static_cast<SampleType>(mixSmoother_.getTargetValue()));
        }
    }

    // -- Thread-safe setters -------------------------------------------------

    /** @brief Sets the saturation algorithm. @param algo Algorithm type (SoftClip, HardClip, Tube, Tape, etc.). */
    void setAlgorithm(Algorithm algo)       { pushParam([&](auto& p){ p.algorithm = algo; }); }

    /** @brief Sets the drive amount. @param dB Drive in decibels (0 = unity, higher = more saturation). */
    void setDrive(SampleType dB)            { pushParam([&](auto& p){ p.driveDb = dB; }); }

    /** @brief Sets the dry/wet mix. @param mix01 Mix amount (0 = fully dry, 1 = fully wet). */
    void setMix(SampleType mix01)           { pushParam([&](auto& p){ p.mix = mix01; }); }

    /** @brief Sets the saturation character (harmonic bias). @param c Character amount (0 = neutral). */
    void setCharacter(SampleType c)         { pushParam([&](auto& p){ p.character = c; }); }

    /** @brief Sets the processing mode. @param m Stereo, mono, or mid/side processing. */
    void setProcessingMode(ProcessingMode m){ pushParam([&](auto& p){ p.processingMode = m; }); }

    /** @brief Sets the output mode. @param m Normal or delta (difference) output. */
    void setOutputMode(OutputMode m)        { pushParam([&](auto& p){ p.outputMode = m; }); }

    /** @brief Sets the analog drift intensity. @param i Drift amount (0 = off, 1 = full drift). */
    void setAnalogDrift(SampleType i)       { pushParam([&](auto& p){ p.analogDrift = i; }); }

    /** @brief Sets the pre-filter high-pass frequency. @param hz Cutoff in Hz (removes DC/sub before saturation). */
    void setPreFilterHpFrequency(SampleType hz) { pushParam([&](auto& p){ p.preFilterHpFreq = hz; }); }

    /** @brief Sets the output gain. @param dB Gain in decibels applied after saturation. */
    void setOutputGain(SampleType dB)       { pushParam([&](auto& p){ p.outputGain = dB; }); }

    /** @brief Enables or disables the DC blocking filter. @param on True to enable. */
    void setDcBlocking(bool on)             { pushParam([&](auto& p){ p.dcBlocking = on; }); }

    /**
     * @brief Enables adaptive blend (PurestDrive-style).
     *
     * When enabled, saturation intensity adapts per-sample based on input
     * amplitude: loud peaks get full saturation, quiet signals stay clean.
     * Eliminates the "always-on" harshness of fixed-drive saturation.
     *
     * @param on True to enable adaptive blend.
     */
    void setAdaptiveBlend(bool on) noexcept
    {
        adaptiveBlend_.store(on, std::memory_order_relaxed);
    }

    /**
     * @brief Sets slew-dependent saturation intensity (HardVacuum-style).
     *
     * When > 0, high-frequency transients (large sample-to-sample deltas)
     * receive more saturation than static or low-frequency signals. This
     * naturally produces more harmonic content on treble without affecting bass.
     *
     * @param amount Slew sensitivity (0 = off, 0.5 = moderate, 1 = full).
     */
    void setSlewSensitivity(SampleType amount) noexcept
    {
        slewSensitivity_.store(std::clamp(amount, SampleType(0), SampleType(1)),
                               std::memory_order_relaxed);
    }

    /**
     * @brief Sets a post-saturation tilt EQ.
     * @param centerHz Center frequency in Hz.
     * @param amountDb Tilt amount in dB (positive = brighter, negative = darker).
     */
    void setPostFilterTilt(SampleType centerHz, SampleType amountDb)
    {
        pushParam([&](auto& p){
            p.postFilterTiltFreq = centerHz;
            p.postFilterTiltGain = amountDb;
        });
    }

    /**
     * @brief Enables oversampling for anti-aliased saturation.
     *
     * Higher factors reduce aliasing from harmonic generation at the cost
     * of CPU. Must be a power of two. Call before prepare(), or call
     * prepare() again after changing this setting.
     *
     * @param factor Oversampling factor (1 = off, 2, 4, 8, or 16).
     */
    void setOversampling(int factor)
    {
        if (factor < 1 || (factor & (factor - 1)) != 0) return;
        oversamplingFactor_ = factor;
        if (factor > 1)
        {
            oversampler_ = std::make_unique<Oversampling<SampleType>>(factor);
            if (spec_.sampleRate > 0)
                oversampler_->prepare(spec_);
        }
        else
        {
            oversampler_.reset();
        }
    }

    /** @brief Returns the current oversampling factor. */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return oversamplingFactor_; }

    // -- Thread-safe getters -------------------------------------------------

    /** @brief Returns the currently active saturation algorithm. */
    [[nodiscard]] Algorithm getCurrentAlgorithm() const noexcept
    {
        return currentAlgoType_.load(std::memory_order_relaxed);
    }

    /** @brief Returns the current gain reduction in dB (for metering). */
    [[nodiscard]] SampleType getGainReductionDb() const noexcept
    {
        return gainReductionDb_.load(std::memory_order_relaxed);
    }

protected:
    // -- Internal parameter struct -------------------------------------------

    struct Params
    {
        Algorithm      algorithm      = Algorithm::SoftClip;
        ProcessingMode processingMode = ProcessingMode::Stereo;
        OutputMode     outputMode     = OutputMode::Normal;
        SampleType     driveDb        = SampleType(0);
        SampleType     mix            = SampleType(1);
        SampleType     character      = SampleType(0);
        SampleType     analogDrift    = SampleType(0);
        SampleType     preFilterHpFreq    = SampleType(20);
        SampleType     postFilterTiltFreq = SampleType(1000);
        SampleType     postFilterTiltGain = SampleType(0);
        SampleType     outputGain         = SampleType(0);
        bool           dcBlocking         = true;
    };

    template <typename Fn>
    void pushParam(Fn&& mutate)
    {
        SpinLock::ScopedLock guard(paramsLock_);
        mutate(lastParams_);
        paramQueue_.push(lastParams_);
    }

    void handleParameterChanges()
    {
        Params p;
        while (paramQueue_.pop(p))
        {
            driveSmoother_.setTargetValue(std::clamp(static_cast<float>(p.driveDb), -24.0f, 48.0f));
            mixSmoother_.setTargetValue(std::clamp(static_cast<float>(p.mix), 0.0f, 1.0f));
            characterSmoother_.setTargetValue(std::clamp(static_cast<float>(p.character), -1.0f, 1.0f));
            driftSmoother_.setTargetValue(std::clamp(static_cast<float>(p.analogDrift), 0.0f, 1.0f));
            outputGainSmoother_.setTargetValue(static_cast<float>(p.outputGain));

            float nyquist = static_cast<float>(spec_.sampleRate) / 2.0f;
            preHpSmoother_.setTargetValue(std::clamp(static_cast<float>(p.preFilterHpFreq), 10.0f, nyquist));
            postTiltFreqSmoother_.setTargetValue(std::clamp(static_cast<float>(p.postFilterTiltFreq), 100.0f, nyquist));
            postTiltGainSmoother_.setTargetValue(std::clamp(static_cast<float>(p.postFilterTiltGain), -12.0f, 12.0f));

            dcBlockingEnabled_ = p.dcBlocking;
            procMode_   = p.processingMode;
            outputMode_ = p.outputMode;
            currentAlgoType_.store(p.algorithm, std::memory_order_relaxed);

            auto* requested = pool_[static_cast<int>(p.algorithm)].get();
            if (active_.load() != requested && next_.load() != requested)
            {
                next_.store(requested);
                crossfader_.setTargetValue(0.0f);
            }
        }
    }

    void processSaturation(AudioBufferView<SampleType> buffer)
    {
        auto* primary   = active_.load();
        auto* secondary = next_.load();
        bool  xfading   = secondary != nullptr && crossfader_.isSmoothing();

        auto driveDb   = static_cast<SampleType>(driveSmoother_.getTargetValue());
        auto driveGain = decibelsToGain(driveDb);
        auto character = static_cast<SampleType>(characterSmoother_.getTargetValue());

        // Use oversampled spec for correct algorithm filter frequencies
        AudioSpec updateSpec = spec_;
        if (oversamplingFactor_ > 1)
            updateSpec.sampleRate *= oversamplingFactor_;

        if (primary)   primary->update(driveGain, character, updateSpec);
        if (secondary) secondary->update(driveGain, character, updateSpec);

        auto driftIntensity = driftSmoother_.getTargetValue();
        bool useSlew = slewSensitivity_.load(std::memory_order_relaxed) > SampleType(0);
        bool useAdaptive = adaptiveBlend_.load(std::memory_order_relaxed);

        // Capture peak input BEFORE saturation (for GR metering).
        // We compare peakOut with peakIn * driveGain — the peak that would have
        // come out if the saturator were perfectly linear. The difference
        // measures how much the saturator clipped/compressed the driven signal.
        SampleType peakInOriginal = SampleType(0);
        {
            const int nCh = buffer.getNumChannels();
            const int nS  = buffer.getNumSamples();
            for (int ch = 0; ch < nCh; ++ch)
            {
                const SampleType* d = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    peakInOriginal = std::max(peakInOriginal, std::abs(d[i]));
            }
        }

        // Slew and adaptive blend require per-sample processing
        if (driftIntensity < 0.01f && !useSlew && !useAdaptive)
            processWithoutDrift(buffer, primary, secondary, xfading, driveGain, character);
        else
            processWithDrift(buffer, primary, secondary, xfading);

        // Capture peak output AFTER saturation, then compute GR.
        SampleType peakOut = SampleType(0);
        {
            const int nCh = buffer.getNumChannels();
            const int nS  = buffer.getNumSamples();
            for (int ch = 0; ch < nCh; ++ch)
            {
                const SampleType* d = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    peakOut = std::max(peakOut, std::abs(d[i]));
            }
        }

        SampleType peakInDriven = peakInOriginal * driveGain;
        if (peakInDriven > SampleType(1e-6))
        {
            SampleType ratio = std::min(peakOut / peakInDriven, SampleType(1));
            gainReductionDb_.store(gainToDecibels(ratio, SampleType(-100)),
                                   std::memory_order_relaxed);
        }
        else
        {
            gainReductionDb_.store(SampleType(0), std::memory_order_relaxed);
        }

        // Complete crossfade
        if (xfading && !crossfader_.isSmoothing())
        {
            active_.store(secondary);
            next_.store(nullptr);
            if (primary) primary->reset();
            crossfader_.setCurrentAndTargetValue(1.0f);
        }
    }

    void processWithoutDrift(AudioBufferView<SampleType> buffer,
                             detail::SaturationAlgorithm<SampleType>* primary,
                             detail::SaturationAlgorithm<SampleType>* secondary,
                             bool xfading, SampleType driveGain, SampleType character)
    {
        if (xfading)
        {
            auto temp = tempBuffer_.toView();
            // Copy only the relevant portion
            const int nCh = buffer.getNumChannels();
            const int nS  = buffer.getNumSamples();
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(temp.getChannel(ch), buffer.getChannel(ch),
                           static_cast<std::size_t>(nS) * sizeof(SampleType));

            auto tempView = AudioBufferView<SampleType>(
                tempBuffer_.toView().getSubView(0, nS));

            if (primary)   primary->processBlock(buffer, driveGain, character);
            if (secondary) secondary->processBlock(tempView, driveGain, character);

            for (int i = 0; i < nS; ++i)
            {
                auto fade = crossfader_.getNextValue();
                for (int ch = 0; ch < nCh; ++ch)
                {
                    SampleType* out     = buffer.getChannel(ch);
                    const SampleType* alt = tempView.getChannel(ch);
                    out[i] = out[i] * static_cast<SampleType>(fade) +
                             alt[i] * (SampleType(1) - static_cast<SampleType>(fade));
                }
            }
        }
        else
        {
            if (primary) primary->processBlock(buffer, driveGain, character);
        }
    }

    void processWithDrift(AudioBufferView<SampleType> buffer,
                          detail::SaturationAlgorithm<SampleType>* primary,
                          detail::SaturationAlgorithm<SampleType>* secondary,
                          bool xfading)
    {
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();

        // GR metering moved to processSaturation() so it covers both drift /
        // non-drift paths and uses the correct peak-driven reference.

        auto slewAmt  = slewSensitivity_.load(std::memory_order_relaxed);
        bool adaptive = adaptiveBlend_.load(std::memory_order_relaxed);

        for (int i = 0; i < nS; ++i)
        {
            auto driveDbSmoothed = static_cast<SampleType>(driveSmoother_.getNextValue());
            auto driveGainS = decibelsToGain(driveDbSmoothed);
            auto charS      = static_cast<SampleType>(characterSmoother_.getNextValue());
            auto driftS     = driftSmoother_.getNextValue();
            auto fade       = crossfader_.getNextValue();

            for (int ch = 0; ch < nCh; ++ch)
            {
                SampleType drift = (ch == 0)
                    ? SampleType(1) + static_cast<SampleType>(driftS) * static_cast<SampleType>(leftDrift_.getNextSample())
                    : SampleType(1) + static_cast<SampleType>(driftS) * static_cast<SampleType>(rightDrift_.getNextSample());

                SampleType* data = buffer.getChannel(ch);
                SampleType sample = data[i];
                SampleType dry = sample;

                // Slew-dependent saturation (HardVacuum-style):
                // High-frequency content (large deltas) gets more drive.
                // A3: use tanh(|delta|) instead of sin(|delta|) — sin folds
                // back (negative response) for |delta| > π/2 which inverts
                // the intended monotonic transient-boost behaviour and can
                // produce loud artifacts on very hot transients. tanh is
                // bounded in [0,1), monotonic, and saturates gracefully.
                if (slewAmt > SampleType(0))
                {
                    SampleType delta = sample - prevSlewSample_[ch];
                    sample += std::tanh(std::abs(delta)) * slewAmt * sample;
                    prevSlewSample_[ch] = dry;
                }

                SampleType wet = SampleType(0);
                if (primary)
                    wet = primary->processSample(sample, driveGainS * drift, charS * drift, ch);

                if (xfading && secondary)
                {
                    SampleType alt = secondary->processSample(sample, driveGainS * drift, charS * drift, ch);
                    wet = wet * static_cast<SampleType>(fade) + alt * (SampleType(1) - static_cast<SampleType>(fade));
                }

                // Adaptive blend (PurestDrive-style):
                // Saturation intensity follows input amplitude, not drive.
                // A4: previous formula `avg * driveGainS` pinned `apply` to 1
                // permanently at even moderate drive (e.g. 6 dB on avg=0.5 → 1),
                // producing "always saturated" behaviour irrespective of input.
                // Correct behaviour is: quiet passages stay mostly dry, loud
                // passages blend in more saturated signal. Drive influence is
                // already baked into `wet` via the primary saturator.
                if (adaptive)
                {
                    SampleType avg = (std::abs(prevBlendSample_[ch]) + std::abs(dry)) * SampleType(0.5);
                    SampleType apply = std::clamp(avg, SampleType(0), SampleType(1));
                    wet = dry * (SampleType(1) - apply) + wet * apply;
                    prevBlendSample_[ch] = dry;
                }

                // Respect processing mode for M/S
                if ((procMode_ == ProcessingMode::MidOnly && ch == 1) ||
                    (procMode_ == ProcessingMode::SideOnly && ch == 0))
                    data[i] = dry;
                else
                    data[i] = wet;
            }
        }

        // GR metering moved to processSaturation() — see comment there.
    }

    void applyOutputGain(AudioBufferView<SampleType> buffer)
    {
        auto targetGainDb = outputGainSmoother_.getTargetValue();
        if (std::abs(targetGainDb - outputGainSmoother_.getCurrentValue()) < 0.001f)
        {
            buffer.applyGain(decibelsToGain(static_cast<SampleType>(targetGainDb)));
        }
        else
        {
            const int nCh = buffer.getNumChannels();
            const int nS  = buffer.getNumSamples();
            for (int i = 0; i < nS; ++i)
            {
                auto gain = decibelsToGain(static_cast<SampleType>(outputGainSmoother_.getNextValue()));
                for (int ch = 0; ch < nCh; ++ch)
                    buffer.getChannel(ch)[i] *= gain;
            }
        }
    }

    // -- Members -------------------------------------------------------------

    AudioSpec spec_ {};
    bool prepared_ = false;

    // Algorithm pool (all pre-allocated)
    static constexpr int kNumAlgorithms = 10;
    std::array<std::unique_ptr<detail::SaturationAlgorithm<SampleType>>, kNumAlgorithms> pool_;
    std::atomic<detail::SaturationAlgorithm<SampleType>*> active_ { nullptr };
    std::atomic<detail::SaturationAlgorithm<SampleType>*> next_   { nullptr };

    // Parameter delivery
    SpscQueue<Params>  paramQueue_;
    Params             lastParams_;
    SpinLock           paramsLock_;

    // Processing state
    ProcessingMode procMode_   = ProcessingMode::Stereo;
    OutputMode     outputMode_ = OutputMode::Normal;
    bool           dcBlockingEnabled_ = true;
    std::atomic<Algorithm> currentAlgoType_ { Algorithm::SoftClip };

    // Smoothers
    Smoothers::StateVariableSmoother driveSmoother_, preHpSmoother_, postTiltFreqSmoother_;
    Smoothers::LinearSmoother       mixSmoother_, characterSmoother_, driftSmoother_,
                                    outputGainSmoother_, postTiltGainSmoother_;
    Smoothers::LinearSmoother       crossfader_;

    // Filters (multichannel)
    Biquad<SampleType> preFilter_, postFilter_, dcBlocker_;

    // Dry/Wet
    DryWetMixer<SampleType> dryWetMixer_;

    // Analog drift generators
    AnalogRandom::Generator<SampleType> leftDrift_, rightDrift_;

    // Temp buffer for crossfading
    AudioBuffer<SampleType> tempBuffer_;

    // Oversampling
    std::unique_ptr<Oversampling<SampleType>> oversampler_;
    int oversamplingFactor_ = 1;

    // Gain reduction tracking
    std::atomic<SampleType> gainReductionDb_ { SampleType(0) };

    // Adaptive blend (PurestDrive-style)
    std::atomic<bool> adaptiveBlend_ { false };
    static constexpr int kMaxCh = 8;
    std::array<SampleType, kMaxCh> prevBlendSample_ {};

    // Slew-dependent saturation (HardVacuum-style)
    std::atomic<SampleType> slewSensitivity_ { SampleType(0) };
    std::array<SampleType, kMaxCh> prevSlewSample_ {};

    // C1+A1 cache: last filter-parameter values that were pushed into the
    // biquad coefficient factories. Compared on every block; setCoeffs and
    // the underlying makeHighPass/makePeak call are skipped when unchanged.
    // Sentinel values (-1 Hz, NaN) guarantee a rebuild on the first block.
    float lastPreHpFreq_    = -1.0f;
    float lastPostTiltFreq_ = -1.0f;
    float lastPostTiltGain_ = std::numeric_limits<float>::quiet_NaN();
};

} // namespace dspark
