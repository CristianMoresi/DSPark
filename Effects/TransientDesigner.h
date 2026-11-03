// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TransientDesigner.h
 * @brief Transient shaping via dual-envelope analysis (VCA Logarithmic Model).
 *
 * Emulates classic analog transient shapers by calculating the difference
 * between a fast (peak) and slow (RMS) envelope in the logarithmic domain.
 * This effectively separates the attack and release phases, applying
 * independent gain adjustments to the transient and the body.
 *
 * Both envelopes follow max(|x|, analytic magnitude), the magnitude taken from
 * a zero-latency IIR Hilbert pair (Core/Hilbert.h): peaks still register on
 * the very sample they occur, but a steady tone reads as its constant
 * amplitude instead of a rectified wave. A plain rectifier made the fast
 * envelope ripple at twice the signal frequency, so steady bass notes were
 * gain-modulated as if they were transients (measured THD+N on a steady
 * 40 Hz tone: 2.5% at attack +50%, 4.5% at sustain -50%).
 *
 * Threading model:
 * - prepare() / reset() / getState() / setState(): setup or UI threads only,
 *   never concurrently with processBlock().
 * - setAttack / setSustain / setCharacter / setOutputDepRecovery: RT-safe
 *   atomic publications from any thread (non-finite values are ignored).
 *   The applied amounts ramp to the published ones at a unit per 10 ms.
 * - Non-finite input samples read as silence in the detector.
 * - Channels beyond the 16-channel state cap are left untouched.
 * - Before prepare() the envelope coefficients are zero, which makes
 *   processBlock() an exact pass-through by construction.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h,
 *               Hilbert.h, StateBlob.h.
 *
 * @code
 *   dspark::TransientDesigner<float> td;
 *   td.prepare(spec);
 *   td.setAttack(0.5f);    // +50% attack emphasis
 *   td.setSustain(-0.3f);  // -30% sustain reduction
 *   td.processBlock(buffer);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Hilbert.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class TransientDesigner
 * @brief Zero-allocation, thread-safe transient shaper.
 *
 * The per-channel envelope recursion is inherently serial (no SIMD); the
 * channel-outer loop keeps each channel's state in registers and the data
 * accesses cache-friendly.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TransientDesigner
{
public:
    ~TransientDesigner() = default; // non-virtual: leaf class (no virtual dispatch)

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the processor with the current audio specification.
     * @param spec Audio specification including sample rate (invalid specs are ignored).
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid()) return;
        prepare(spec.sampleRate);
    }

    /**
     * @brief Prepares the processor with a specific sample rate.
     * @param sampleRate The operating sample rate in Hz. Non-finite or
     *                   non-positive rates are ignored (a NaN rate used to
     *                   pass the coefficient guard and poison the envelopes
     *                   permanently).
     */
    void prepare(double sampleRate) noexcept
    {
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0) return;
        sampleRate_ = sampleRate;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place.
     * @param buffer View of the audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        const T attAmt = attackAmount_.load(std::memory_order_relaxed);
        const T susAmt = sustainAmount_.load(std::memory_order_relaxed);
        const bool odr = outputDepRecovery_.load(std::memory_order_relaxed);

        // Amount changes ramp at a fixed rate (a unit change over 10 ms) so a
        // setAttack()/setSustain() during a transient cannot step the gain by
        // up to 24 dB in one sample. The ramp is closed-form in the samples
        // elapsed since the change, so every channel sees the same values and
        // the result does not depend on the block partition.
        if (attAmt != rampToAttack_ || susAmt != rampToSustain_)
        {
            const T done = static_cast<T>(rampStep_ * rampPos_);
            rampFromAttack_ = moveTowards(rampFromAttack_, rampToAttack_, done);
            rampFromSustain_ = moveTowards(rampFromSustain_, rampToSustain_, done);
            rampToAttack_ = attAmt;
            rampToSustain_ = susAmt;
            rampPos_ = 0.0;
        }
        const bool ramping = (rampFromAttack_ != attAmt) || (rampFromSustain_ != susAmt);
        const T att0 = rampFromAttack_, sus0 = rampFromSustain_;
        const double pos0 = rampPos_;

        // Constants for VCA log-domain emulation
        constexpr T noiseFloor = T(1e-5);     // -100 dB floor avoids log(0) and denormals
        constexpr T maxGainLog = T(2.77258);  // approx +24dB max gain change

        // Outer loop: channels (state stays in registers)
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* const channelData = buffer.getChannel(ch);
            auto& analytic = analytic_[static_cast<size_t>(ch)];
            T fast = envFast_[ch];
            T slow = envSlow_[ch];
            T lastOut = lastOutput_[ch];

            // Inner loop: serial envelope recursion per sample, fed in chunks
            // by the analytic magnitude (computed section by section).
            for (int i = 0; i < nS; ++i)
            {
                if (i % kDetChunk == 0)
                {
                    const int n = std::min(kDetChunk, nS - i);
                    for (int j = 0; j < n; ++j)
                    {
                        // Non-finite samples read as silence in the detector
                        // (they would latch in the envelopes and the allpasses).
                        const T x = channelData[i + j];
                        detIn_[static_cast<size_t>(j)] = std::isfinite(x) ? static_cast<double>(x) : 0.0;
                    }
                    analytic.magnitudeBlock(detIn_.data(), detMag_.data(), n);
                }
                T sample = channelData[i];
                const T envelopeLevel = std::max(
                    static_cast<T>(std::abs(detIn_[static_cast<size_t>(i % kDetChunk)])),
                    static_cast<T>(detMag_[static_cast<size_t>(i % kDetChunk)]));
                T absSample = envelopeLevel + noiseFloor;

                // 1. Fast envelope (Peak)
                T fastCoeff = (absSample > fast) ? fastAttackCoeff_ : fastReleaseCoeff_;
                fast += fastCoeff * (absSample - fast);

                // 2. Slow envelope (Sustain/RMS tracker)
                T currentSlowRelCoeff = slowReleaseCoeff_;

                if (odr)
                {
                    // Corrected ODR: Higher output = LARGER coefficient = Faster release
                    T modifier = T(1) + std::abs(lastOut) * T(2.0);
                    currentSlowRelCoeff = std::min(fastReleaseCoeff_, currentSlowRelCoeff * modifier);
                }

                T slowCoeff = (absSample > slow) ? slowAttackCoeff_ : currentSlowRelCoeff;
                slow += slowCoeff * (absSample - slow);

                // 3. Log-domain VCA computation (fastLog/fastExp: relative
                // error < 2e-7 - far below audibility, ~3x faster than libm)
                T diffLog = fastLog(fast) - fastLog(slow);

                // Attack kicks in when fast > slow (diffLog > 0). Sustain when slow > fast (diffLog < 0)
                const T stepNow = static_cast<T>(rampStep_ * (pos0 + static_cast<double>(i + 1)));
                const T att = ramping ? moveTowards(att0, attAmt, stepNow) : attAmt;
                const T sus = ramping ? moveTowards(sus0, susAmt, stepNow) : susAmt;
                T gainLog = (diffLog > T(0)) ? (diffLog * att) : (-diffLog * sus);

                gainLog = std::clamp(gainLog, -maxGainLog, maxGainLog);

                T gain = fastExp(gainLog);

                const T out = sample * gain;
                lastOut = std::isfinite(out) ? out : T(0);
                channelData[i] = out;
            }

            // Save state
            envFast_[ch] = fast;
            envSlow_[ch] = slow;
            lastOutput_[ch] = lastOut;
        }
        if (ramping)
        {
            rampPos_ += static_cast<double>(nS);
            if (rampStep_ * rampPos_ >= 2.0) // the whole [-1, 1] span has elapsed
            {
                rampFromAttack_ = attAmt;
                rampFromSustain_ = susAmt;
                rampPos_ = 0.0;
            }
        }
    }

    // -- Parameters ----------------------------------------------------------
    // All setters ignore non-finite values: a NaN amount used to storm the
    // output for as long as it was published, and under output-dependent
    // recovery the NaN last-output silently switched the slow envelope to the
    // fast release, leaving the envelope history diverged after recovery.

    /**
     * @brief Sets attack (transient) emphasis.
     * @param amount -100 to +100 (%). Positive = boost transients, negative = soften.
     */
    void setAttack(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        attackAmount_.store(std::clamp(amount, T(-100), T(100)) / T(100), std::memory_order_relaxed);
    }

    /**
     * @brief Sets sustain (body) emphasis.
     * @param amount -100 to +100 (%). Positive = boost sustain, negative = reduce.
     */
    void setSustain(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        sustainAmount_.store(std::clamp(amount, T(-100), T(100)) / T(100), std::memory_order_relaxed);
    }

    /**
     * @brief Enables output-dependent recovery (ODR).
     * @param enabled If true, slow envelope release speed scales with output level.
     */
    void setOutputDepRecovery(bool enabled) noexcept
    {
        outputDepRecovery_.store(enabled, std::memory_order_relaxed);
    }

    /**
     * @brief Sets character as a single macro-knob (overwrites attack AND sustain).
     * @param amount Range [-1.0, 1.0]. -1 = soften transients/boost sustain, +1 = boost transients/reduce sustain.
     */
    void setCharacter(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        T c = std::clamp(amount, T(-1), T(1));
        attackAmount_.store(c, std::memory_order_relaxed);
        sustainAmount_.store(-c * T(0.5), std::memory_order_relaxed);
    }

    /**
     * @brief Clears all internal buffers and state. Must be lock-free.
     */
    void reset() noexcept
    {
        envFast_.fill(T(1e-5)); // Init to noise floor
        envSlow_.fill(T(1e-5));
        lastOutput_.fill(T(0));
        for (auto& a : analytic_) a.reset();
        rampFromAttack_ = rampToAttack_ = attackAmount_.load(std::memory_order_relaxed);
        rampFromSustain_ = rampToSustain_ = sustainAmount_.load(std::memory_order_relaxed);
        rampPos_ = 0.0;
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TDES"), 1);
        // static_cast<float>: the blob stores float (percent units), and
        // StateWriter's overload set is ambiguous for a double argument.
        w.write("attack", static_cast<float>(attackAmount_.load(std::memory_order_relaxed)) * 100.0f);
        w.write("sustain", static_cast<float>(sustainAmount_.load(std::memory_order_relaxed)) * 100.0f);
        w.write("outputDep", outputDepRecovery_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TDES")) return false;
        setAttack(static_cast<T>(r.read("attack", 0.0f)));
        setSustain(static_cast<T>(r.read("sustain", 0.0f)));
        setOutputDepRecovery(r.read("outputDep", false));
        return true;
    }

private:
    static constexpr int kMaxChannels = 16;

    void updateCoefficients() noexcept
    {
        if (!(sampleRate_ > 0.0)) return;
        T fs = static_cast<T>(sampleRate_);
        fastAttackCoeff_  = T(1) - std::exp(T(-1) / (fs * T(0.0001)));
        fastReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.005)));
        slowAttackCoeff_  = T(1) - std::exp(T(-1) / (fs * T(0.020)));
        slowReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.200)));
        rampStep_ = 1.0 / (sampleRate_ * 0.010); // a unit amount change over 10 ms
    }

    double sampleRate_ = 48000.0;

    // Atomic parameters (Lock-free thread safety)
    std::atomic<T> attackAmount_ { T(0) };
    std::atomic<T> sustainAmount_ { T(0) };
    std::atomic<bool> outputDepRecovery_ { false };

    // Envelope coefficients (written by prepare/updateCoefficients on the
    // setup thread, read by the audio thread)
    T fastAttackCoeff_ = T(0), fastReleaseCoeff_ = T(0);
    T slowAttackCoeff_ = T(0), slowReleaseCoeff_ = T(0);
    // Amount ramp (audio thread): from/to values and samples since the change.
    double rampStep_ = 1.0;      ///< Amount change per sample (instant before prepare()).
    double rampPos_ = 0.0;
    T rampFromAttack_ = T(0), rampToAttack_ = T(0);
    T rampFromSustain_ = T(0), rampToSustain_ = T(0);

    // Per-channel state (scalar recursion; no SIMD kernel touches these)
    std::array<T, kMaxChannels> envFast_ {};
    std::array<T, kMaxChannels> envSlow_ {};
    std::array<T, kMaxChannels> lastOutput_ {};
    std::array<HilbertIIR<double>, kMaxChannels> analytic_ {}; ///< Ripple-free envelope source.
    static constexpr int kDetChunk = 64;
    std::array<double, kDetChunk> detIn_ {};    ///< Detector chunk scratch (input).
    std::array<double, kDetChunk> detMag_ {};   ///< Detector chunk scratch (analytic magnitude).
};

} // namespace dspark
