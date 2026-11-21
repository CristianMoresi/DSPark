// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DeEsser.h
 * @brief Frequency-selective dynamic processor for sibilance reduction.
 *
 * A dynamic-EQ style de-esser: a bandpass sidechain detects energy in a
 * configurable frequency band and a dynamic peak-cut bell applies the gain
 * reduction at that band. Optimized for zero inner-loop allocations and
 * trigonometric eliminations during dynamic EQ modulation.
 *
 * Architecture:
 * 1. Bandpass sidechain (Biquad) isolates the sibilant band.
 * 2. Envelope follower (peak with attack/release) tracks maximum stereo sibilance.
 * 3. A matched peak-cut bell (BiquadCoeffs::makePeakMatched, Vicanek) applies
 *    the reduction; its width follows the analog prototype up to Nyquist (a
 *    -12 dB, one-octave cut at 16 kHz reads -9.4 dB at 20 kHz at 48 kHz,
 *    prototype -8.25; the cookbook form read -2.1). setMixedBell() (off by
 *    default) runs it instead as one fixed filter whose share follows the
 *    reduction sample by sample, with nothing designed while it moves.
 *
 * Non-finite samples read as silence in the detector and bypass the cut for
 * that sample, so one NaN or Inf cannot latch in the filter state.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Biquad.h, DenormalGuard.h, DspMath.h, AudioSpec.h,
 *               AudioBuffer.h, StateBlob.h.
 *
 * @code
 *   dspark::DeEsser<float> deesser;
 *   deesser.prepare(spec);
 *   deesser.setFrequency(7000.0f);
 *   deesser.setThreshold(-20.0f);
 *   deesser.setReduction(12.0f);
 *   deesser.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/NormalizedLadder.h"
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
 * @class DeEsser
 * @brief Stereo-linked, CPU-optimized dynamic-EQ de-esser.
 *
 * Detection is stereo-linked (one shared envelope) to preserve imaging.
 * Channels beyond the first two are left untouched (pass-through), as is the
 * whole buffer before prepare().
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DeEsser
{
public:
    enum class DetectionMode
    {
        Bandpass,    ///< Standard bandpass filter detection.
        Derivative   ///< Sum of absolute derivatives. Sensitive to sustained high frequencies.
    };

    /**
     * @brief Prepares the de-esser state and filters.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        removalScratch_.assign(static_cast<size_t>(std::min(numChannels_, kMaxChannels)),
                             std::vector<T>(static_cast<size_t>(std::max(1, spec.maxBlockSize)), T(0)));
        mixScratch_.assign(static_cast<size_t>(kMaxChannels),
                           std::vector<T>(static_cast<size_t>(std::max(1, spec.maxBlockSize)), T(0)));
        mixWeights_ = mixScratch_;
        endDesigned_ = false;
        updateEnvelopeCoeffs();
        coefSmoothCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * 0.001)));
        autoFall_ = std::pow(10.0, -kAutoRefFallDb / (20.0 * sampleRate_));

        reset();

        filterParamsDirty_.store(true, std::memory_order_release);
        updateFiltersIfDirty();
    }

    /**
     * @brief Processes audio in-place. Stereo-linked to preserve imaging.
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (removalListen_.load(std::memory_order_relaxed) && !removalScratch_.empty())
        {
            // Removal listen: the input minus the de-essed signal (no latency),
            // in chunks as long as the scratch.
            const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(removalScratch_.size()));
            const int step = static_cast<int>(removalScratch_[0].size());
            for (int off = 0; off < buffer.getNumSamples(); off += step)
            {
                const int m = std::min(step, buffer.getNumSamples() - off);
                auto part = buffer.getSubView(off, m);
                for (int ch = 0; ch < nCh; ++ch) std::copy_n(part.getChannel(ch), m, removalScratch_[static_cast<size_t>(ch)].data());
                processWet(part);
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* d = part.getChannel(ch);
                    const T* in = removalScratch_[static_cast<size_t>(ch)].data();
                    for (int i = 0; i < m; ++i) d[i] = in[i] - d[i];
                }
            }
            return;
        }
        processWet(buffer);
    }

    /**
     * @brief Removal listen: the output is what the de-esser takes out. A
     *        monitoring control, not in the state.
     */
    void setRemovalListen(bool on) noexcept { removalListen_.store(on, std::memory_order_relaxed); }
    /** @return Whether the output is what the de-esser takes out. */
    [[nodiscard]] bool isRemovalListen() const noexcept { return removalListen_.load(std::memory_order_relaxed); }

    /** @brief How the cut bell is realised (see setStructure()). */
    enum class Structure
    {
        DirectForm,  ///< Direct form (the default).
        Ladder       ///< Normalized ladder: passive while the reduction moves the bell.
    };

    /**
     * @brief Chooses how the cut bell is realised.
     *
     * The bell is designed again every 16 samples as the reduction moves.
     * Ladder runs it as a normalized ladder (NormalizedLadder): the same
     * response, and a memory that cannot gain energy at the jumps, for
     * about twice the arithmetic of the bell. Safe from any thread; the
     * switch restarts the bell's memory at the next block.
     *
     * @param structure DirectForm (default) or Ladder.
     */
    void setStructure(Structure structure) noexcept
    {
        structure_.store(static_cast<Structure>(std::clamp(static_cast<int>(structure), 0, 1)), std::memory_order_relaxed);
    }

    /** @return How the cut bell is realised. */
    [[nodiscard]] Structure getStructure() const noexcept { return structure_.load(std::memory_order_relaxed); }

    /**
     * @brief Runs the cut as one fixed filter mixed by the reduction.
     *
     * The default bell is designed again every 16 samples as the reduction
     * moves, a gain that moves in steps. Mixed, the bell is the signal and
     * one fixed filter at the end of the range, of which only the share heard
     * changes, sample by sample:
     * ```
     *   y = x + w * (end(x) - x),   w = (1 - 10^(gr / 20)) / (1 - 10^(-range / 20))
     * ```
     * so the centre of the band sits exactly on the reduction of the moment
     * and nothing is designed while it moves. The end is the bell of the
     * band's Q at half the range, carried to the whole range at its centre
     * (the choice DynamicEQ's mixed bands make): nearer the rest the cut is
     * wider than nominal and at the end narrower, by 10^(range / 80) in Q at
     * most. The end filter runs as a normalized ladder and glides 20 ms to a
     * new design when frequency, bandwidth or range move. With no reduction
     * the output is the input, bit for bit. Safe from any thread; the switch
     * restarts the filter memories at the next block. Off by default.
     *
     * Measured at 48 kHz on a 6 kHz tone that jumps 28 dB 25 times a second
     * under a bell of 12 dB: with a release of 1 or 5 ms the default bell
     * leaves lines 3 kHz away at -46 to -52 dBc and the mixed one at -64 to
     * -96 (the floor the tone's own edges set); with the default 20 ms both
     * sit on that floor. It costs about 2.4 times the default bell (18 to 34
     * against 41 to 81 ns per stereo frame).
     *
     * State key "mixedBell" (absent = off).
     */
    void setMixedBell(bool on) noexcept { mixedWanted_.store(on, std::memory_order_relaxed); }

    /** @return Whether the cut is a fixed filter mixed by the reduction (setMixedBell()). */
    [[nodiscard]] bool isMixedBell() const noexcept { return mixedWanted_.load(std::memory_order_relaxed); }

private:
    void processWet(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(), std::min(numChannels_, kMaxChannels));
        int numSamples = buffer.getNumSamples();
        if (numSamples == 0 || numCh == 0) return;
        DenormalGuard guard;

        updateFiltersIfDirty();

        // A change of structure restarts the bell's memory.
        const bool ladder = structure_.load(std::memory_order_relaxed) == Structure::Ladder;
        if (ladder != ladderActive_)
        {
            ladderActive_ = ladder;
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                reduction_[ch].reset();
                ladder_[ch].reset();
                ladder_[ch].setCoeffs(reduction_[ch].getCoeffs());
            }
        }

        if (envCoefsDirty_.exchange(false, std::memory_order_acquire))
            updateEnvelopeCoeffs();

        T thresh = threshold_.load(std::memory_order_relaxed);
        T maxRed = maxReduction_.load(std::memory_order_relaxed);
        auto mode = detectionMode_.load(std::memory_order_relaxed);
        relative_ = autoThreshold_.load(std::memory_order_relaxed);

        constexpr int kCoefRefreshInterval = 16;
        const T coefSmooth = coefSmoothCoeff_;
        T blockMaxGr = T(0);

        // A change of engine restarts the filter memories.
        const bool mixed = mixedWanted_.load(std::memory_order_relaxed) && !mixScratch_.empty();
        if (mixed != mixedActive_)
        {
            mixedActive_ = mixed;
            end_.reset();
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                reduction_[ch].reset();
                ladder_[ch].reset();
            }
        }

        if (mixedActive_)
        {
            designEnd(static_cast<double>(maxRed));
            const int step = static_cast<int>(mixScratch_[0].size());
            for (int off = 0; off < numSamples; off += step)
            {
                const int m = std::min(step, numSamples - off);
                auto part = buffer.getSubView(off, m);
                if (mode == DetectionMode::Derivative)
                    processInternal<true, true>(part, numCh, m, thresh, maxRed, coefSmooth, kCoefRefreshInterval, blockMaxGr);
                else
                    processInternal<false, true>(part, numCh, m, thresh, maxRed, coefSmooth, kCoefRefreshInterval, blockMaxGr);
                mixEnd(part, numCh, m);
            }
        }
        // Branch hoisting: Process samples depending on the mode to avoid inner-loop conditionals.
        else if (mode == DetectionMode::Derivative)
        {
            processInternal<true>(buffer, numCh, numSamples, thresh, maxRed, coefSmooth, kCoefRefreshInterval, blockMaxGr);
        }
        else
        {
            processInternal<false>(buffer, numCh, numSamples, thresh, maxRed, coefSmooth, kCoefRefreshInterval, blockMaxGr);
        }

        gainReduction_.store(blockMaxGr, std::memory_order_relaxed);
    }

public:
    /** @brief Resets internal DSP state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            detector_[ch].reset();
            reduction_[ch].reset();
            ladder_[ch].reset();
            smoothedGrDb_[ch] = T(0);
            derivShift_[ch].fill(T(0));
        }
        end_.reset();
        envelope_ = T(0);
        autoRef_ = kAutoRefFloor;
        refreshCounter_ = 0;
        gainReduction_.store(T(0), std::memory_order_relaxed);
    }

    // Setters

    /** @brief Sets the detector attack, clamped to [0.1, 20] ms. Non-finite values are ignored. */
    void setAttack(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        attackMs_.store(std::clamp(ms, T(0.1), T(20)), std::memory_order_relaxed);
        envCoefsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the detector release, clamped to [1, 500] ms. Non-finite values are ignored. */
    void setRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::clamp(ms, T(1), T(500)), std::memory_order_relaxed);
        envCoefsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the sibilance centre frequency in Hz.
     *
     * Clamped against the prepared sample rate when the filters rebuild
     * (effective range [1, 0.499 * fs]). Non-finite values are ignored.
     */
    void setFrequency(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        frequency_.store(hz, std::memory_order_relaxed);
        filterParamsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the detection/cut bandwidth in octaves (floored at 0.1). Non-finite values are ignored. */
    void setBandwidth(T octaves) noexcept
    {
        if (!std::isfinite(octaves)) return;
        T bw = std::max(octaves, T(0.1));
        T q = T(1) / (T(2) * std::sinh(T(0.34657359) * bw));
        bandwidth_.store(q, std::memory_order_relaxed);
        filterParamsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the detection threshold in dBFS. Non-finite values are ignored. */
    void setThreshold(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        threshold_.store(db, std::memory_order_relaxed);
    }

    /**
     * @brief Counts the threshold from the level of the whole input.
     *
     * A reference follows the largest sample of the input, all of its band:
     * it rises at once and falls 1 dB per second, and the threshold is added
     * to it. The de-esser then acts on how loud the sibilance is against the
     * voice that carries it, the same at any input level: a threshold of
     * -20 dB reduces a band that comes within 20 dB of the voice's recent
     * peaks. Off by default: the threshold is in dBFS.
     *
     * State key "autoThreshold" (absent = off).
     */
    void setAutoThreshold(bool on) noexcept { autoThreshold_.store(on, std::memory_order_relaxed); }

    /** @brief Whether the threshold counts from the level of the input (see setAutoThreshold()). */
    [[nodiscard]] bool isAutoThreshold() const noexcept { return autoThreshold_.load(std::memory_order_relaxed); }

    /** @brief Sets the maximum gain reduction in dB (sign is ignored). Non-finite values are ignored. */
    void setReduction(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        maxReduction_.store(std::abs(db), std::memory_order_relaxed);
    }

    /** @brief Sets the detection mode. Out-of-range values are clamped so the getter stays honest. */
    void setDetectionMode(DetectionMode mode) noexcept
    {
        const int v = std::clamp(static_cast<int>(mode),
                                 static_cast<int>(DetectionMode::Bandpass),
                                 static_cast<int>(DetectionMode::Derivative));
        detectionMode_.store(static_cast<DetectionMode>(v), std::memory_order_relaxed);
    }

    // Getters
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReduction_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }
    [[nodiscard]] DetectionMode getDetectionMode() const noexcept { return detectionMode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getAttack() const noexcept { return attackMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }

    /** @return The detection/cut bandwidth in octaves (inverse of the stored Q). */
    [[nodiscard]] T getBandwidth() const noexcept
    {
        return static_cast<T>(std::asinh(1.0 / (2.0 * static_cast<double>(
            bandwidth_.load(std::memory_order_relaxed)))) / 0.34657359);
    }

    /** @return The maximum gain reduction in dB (positive). */
    [[nodiscard]] T getReduction() const noexcept { return maxReduction_.load(std::memory_order_relaxed); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float (setState reads float back); the explicit
        // casts also keep this overload resolvable when T is double.
        StateWriter w(stateId("DEES"), 1);
        w.write("frequency", static_cast<float>(frequency_.load(std::memory_order_relaxed)));
        // bandwidth_ stores Q = 1/(2 sinh(ln2/2 * oct)); serialize back in octaves.
        w.write("bandwidth", static_cast<float>(getBandwidth()));
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        w.write("reduction", static_cast<float>(maxReduction_.load(std::memory_order_relaxed)));
        w.write("attack", static_cast<float>(attackMs_.load(std::memory_order_relaxed)));
        w.write("release", static_cast<float>(releaseMs_.load(std::memory_order_relaxed)));
        w.write("detection", static_cast<int32_t>(detectionMode_.load(std::memory_order_relaxed)));
        w.write("structure", static_cast<int32_t>(getStructure()));
        w.write("autoThreshold", autoThreshold_.load(std::memory_order_relaxed));
        w.write("mixedBell", isMixedBell());
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DEES")) return false;
        setFrequency(static_cast<T>(r.read("frequency", 7000.0f)));
        // Missing key: the constructor default, Q = 2 (0.714 octaves). The
        // fallback used to be 2 octaves, a different filter.
        setBandwidth(static_cast<T>(r.read("bandwidth", 0.7140373f)));
        setThreshold(static_cast<T>(r.read("threshold", -20.0f)));
        setReduction(static_cast<T>(r.read("reduction", 12.0f)));
        setAttack(static_cast<T>(r.read("attack", 0.5f)));
        setRelease(static_cast<T>(r.read("release", 20.0f)));
        setDetectionMode(static_cast<DetectionMode>(r.read("detection", 0)));
        setStructure(static_cast<Structure>(r.read("structure", 0)));
        setAutoThreshold(r.read("autoThreshold", false));
        setMixedBell(r.read("mixedBell", false));
        return true;
    }

private:
    template <bool IsDerivative, bool Mixed = false>
    void processInternal(AudioBufferView<T>& buffer, int numCh, int numSamples,
                         T thresh, T maxRed, T coefSmooth, int refreshInterval, T& blockMaxGr) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            T maxLevel = T(0);
            T maxInput = T(0);

            // 1. Stereo-linked detection
            for (int ch = 0; ch < numCh; ++ch)
            {
                T dataIn = buffer.getChannel(ch)[i];
                if (!std::isfinite(dataIn)) dataIn = T(0); // would latch in the detector
                maxInput = std::max(maxInput, std::abs(dataIn));
                T level = T(0);

                if constexpr (IsDerivative)
                {
                    auto& sr = derivShift_[ch];
                    for (int k = kDerivLen - 1; k > 0; --k) sr[k] = sr[k - 1];
                    sr[0] = detector_[ch].processSample(dataIn, 0);

                    T sumDiffs = T(0);
                    for (int k = 0; k < kDerivLen - 1; ++k)
                    {
                        sumDiffs += std::abs(sr[k] - sr[k + 1]);
                    }
                    level = sumDiffs / T(kDerivLen - 1);
                }
                else
                {
                    level = std::abs(detector_[ch].processSample(dataIn, 0));
                }

                if (level > maxLevel) maxLevel = level;
            }

            // 2. Single linked envelope
            T coeff = (maxLevel > envelope_) ? attackCoeff_ : releaseCoeff_;
            envelope_ += coeff * (maxLevel - envelope_);

            // 3. Compute GR
            T envDb = gainToDecibels(envelope_ + T(1e-30));
            T threshNow = thresh;
            if (relative_)
            {
                // The threshold counts from a reference that follows the
                // whole input: up at once, down 1 dB per second.
                autoRef_ = std::max({ static_cast<double>(maxInput), autoRef_ * autoFall_, kAutoRefFloor });
                threshNow = thresh + gainToDecibels(static_cast<T>(autoRef_));
            }
            T overDb = envDb - threshNow;
            T grDb = (overDb > T(0)) ? -std::min(overDb, maxRed) : T(0);

            if (-grDb > blockMaxGr) blockMaxGr = -grDb;

            // 4. Apply processing per channel. The refresh grid is a
            //    free-running counter (not the block index), so the output
            //    does not depend on how the host cuts the stream.
            if constexpr (Mixed)
            {
                // Mixed: only the share of the fixed end filter is computed
                // here; mixEnd() runs the filter over the chunk afterwards.
                (void)refreshInterval;
                for (int ch = 0; ch < numCh; ++ch)
                {
                    T& smoothedGr = smoothedGrDb_[ch];
                    smoothedGr += coefSmooth * (grDb - smoothedGr);
                    double w = 0.0;
                    if (smoothedGr < T(0) && endShare_ > 0.0)
                        w = std::clamp((1.0 - std::pow(10.0, static_cast<double>(smoothedGr) / 20.0)) / endShare_, 0.0, 1.0);
                    mixWeights_[static_cast<size_t>(ch)][static_cast<size_t>(i)] = static_cast<T>(w);
                }
            }
            else
            {
                const bool refresh = (refreshCounter_ & static_cast<unsigned>(refreshInterval - 1)) == 0u;
                ++refreshCounter_;
                for (int ch = 0; ch < numCh; ++ch)
                {
                    T& smoothedGr = smoothedGrDb_[ch];
                    smoothedGr += coefSmooth * (grDb - smoothedGr);

                    if (refresh)
                    {
                        // Channels share the linked gain: design once per frame.
                        if (ch > 0 && smoothedGr == smoothedGrDb_[0])
                        {
                            reduction_[ch].setCoeffsNow(reduction_[0].getCoeffs());
                            if (ladderActive_) ladder_[ch].setLadder(ladder_[0].getLadder());
                        }
                        else
                            updateDynamicPeakCoeffs(ch, smoothedGr);
                    }

                    T* channelData = buffer.getChannel(ch);
                    const T x = channelData[i];
                    // A non-finite sample passes without entering the filter state.
                    if (!std::isfinite(x)) continue;
                    channelData[i] = ladderActive_ ? static_cast<T>(ladder_[ch].processSample(static_cast<double>(x), 0))
                                                   : reduction_[ch].processSample(x, 0);
                }
            }
        }
    }

    /**
     * @brief Updates Biquad filters if frequency or bandwidth changed.
     * Precomputes trigonometric constants for the inner-loop Peak filter.
     */
    void updateFiltersIfDirty() noexcept
    {
        if (!filterParamsDirty_.exchange(false, std::memory_order_acquire))
            return;

        T freq = frequency_.load(std::memory_order_relaxed);
        T bw = bandwidth_.load(std::memory_order_relaxed);

        // One shared effective frequency for BOTH filters. The bandpass
        // factory clamps internally, but the precomputed peak terms did not:
        // beyond Nyquist sin(w0) goes negative, alpha flips sign and the
        // dynamic bell turns unstable under gain reduction.
        const double fEff = std::clamp(static_cast<double>(freq), 1.0,
                                       std::max(1.0, sampleRate_ * 0.499));

        // Stream-owner direct writes (setCoeffsNow): prepare() and the audio
        // thread are the only callers, and the coefficients are for their own
        // use (docs/threading.md, the origin rule).
        // Matched to the analog band-pass, like the bell it drives: a
        // bilinear one narrows toward Nyquist and would hear less than the
        // bell cuts, right where the sibilance is.
        auto c = BiquadCoeffs::makeBandPassMatched(sampleRate_, fEff, static_cast<double>(bw));
        for (int ch = 0; ch < kMaxChannels; ++ch)
            detector_[ch].setCoeffsNow(c);

        peakFreq_ = fEff;
        peakQ_ = static_cast<double>(bw);
    }

    /**
     * @brief Designs the dynamic cut bell for the given gain (matched design).
     * @param ch Channel index.
     * @param gainDb Gain to apply (negative).
     */
    void updateDynamicPeakCoeffs(int ch, T gainDb) noexcept
    {
        // Stream-owner direct write: computed on the audio thread for its own
        // use, so it bypasses the staged cross-thread channel.
        const BiquadCoeffs c = BiquadCoeffs::makePeakMatched(sampleRate_, peakFreq_, peakQ_, static_cast<double>(gainDb));
        reduction_[ch].setCoeffsNow(c);
        if (ladderActive_) ladder_[ch].setCoeffs(c);
    }

    /**
     * @brief Designs the fixed end filter of the mixed bell when frequency,
     *        bandwidth or range moved (a 20 ms glide; at once the first time).
     *
     * The end is the bell of the band's Q at half the range, H, carried to
     * the whole range at its centre: 1 + (1 + r) (H - 1), r the half range as
     * an amplitude. Same poles as H; at the centre H = r, so the end is r^2.
     */
    void designEnd(double rangeDb) noexcept
    {
        if (endDesigned_ && peakFreq_ == endFreq_ && peakQ_ == endQ_ && rangeDb == endRange_) return;
        const BiquadCoeffs half = BiquadCoeffs::makePeakMatched(sampleRate_, peakFreq_, peakQ_, -0.5 * rangeDb);
        const double reach = 1.0 + std::pow(10.0, -rangeDb / 40.0);
        BiquadCoeffs end = half;
        end.b0 = 1.0 + reach * (half.b0 - 1.0);
        end.b1 = half.a1 + reach * (half.b1 - half.a1);
        end.b2 = half.a2 + reach * (half.b2 - half.a2);
        if (endDesigned_) end_.rampCoeffs(end, static_cast<int>(std::lround(kMixedRampMs * 0.001 * sampleRate_)));
        else              end_.setCoeffs(end);
        endDesigned_ = true;
        endFreq_ = peakFreq_;
        endQ_ = peakQ_;
        endRange_ = rangeDb;
        endShare_ = rangeDb > 0.0 ? 1.0 - std::pow(10.0, -rangeDb / 20.0) : 0.0;
    }

    /**
     * @brief The mixed bell over one chunk: the end filter on a copy of the
     *        input (non-finite samples read as silence), then each sample
     *        moved towards it by its share.
     */
    void mixEnd(AudioBufferView<T>& part, int numCh, int n) noexcept
    {
        T* copies[kMaxChannels] = {};
        for (int ch = 0; ch < numCh; ++ch)
        {
            const T* x = part.getChannel(ch);
            T* c = mixScratch_[static_cast<size_t>(ch)].data();
            for (int i = 0; i < n; ++i) c[i] = std::isfinite(x[i]) ? x[i] : T(0);
            copies[ch] = c;
        }
        end_.processBlock(AudioBufferView<T>(copies, numCh, n));
        for (int ch = 0; ch < numCh; ++ch)
        {
            T* y = part.getChannel(ch);
            const T* e = copies[ch];
            const T* w = mixWeights_[static_cast<size_t>(ch)].data();
            for (int i = 0; i < n; ++i)
                if (w[i] > T(0) && std::isfinite(y[i])) y[i] += w[i] * (e[i] - y[i]);
        }
    }

    void updateEnvelopeCoeffs() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        const double atkSec = static_cast<double>(attackMs_.load(std::memory_order_relaxed)) * 0.001;
        const double relSec = static_cast<double>(releaseMs_.load(std::memory_order_relaxed)) * 0.001;
        attackCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * std::max(atkSec, 1e-6))));
        releaseCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * std::max(relSec, 1e-6))));
    }

    static constexpr int kMaxChannels = 2;
    static constexpr int kDerivLen = 8;

    double sampleRate_ = 44100.0;
    int numChannels_ = 0;

    std::atomic<T> frequency_ { T(7000) };
    std::atomic<T> bandwidth_ { T(2) };
    std::atomic<T> threshold_ { T(-20) };
    std::atomic<T> maxReduction_ { T(12) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> releaseMs_ { T(20) };
    std::atomic<DetectionMode> detectionMode_ { DetectionMode::Bandpass };

    std::atomic<bool> envCoefsDirty_ { false };
    std::atomic<bool> filterParamsDirty_ { true };

    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    T coefSmoothCoeff_ = T(0);
    std::atomic<T> gainReduction_ { T(0) };

    Biquad<T, 1> detector_[kMaxChannels]{};
    Biquad<T, 1> reduction_[kMaxChannels]{};
    NormalizedLadder<T, 1> ladder_[kMaxChannels]{};      ///< The bell as a ladder (setStructure()).
    std::atomic<Structure> structure_ { Structure::DirectForm }; ///< setStructure().
    bool ladderActive_ = false;                          ///< The audio thread runs the ladder.

    // Mixed bell (setMixedBell()).
    static constexpr double kMixedRampMs = 20.0;         ///< Glide of the end filter to a new design.
    std::atomic<bool> mixedWanted_ { false };            ///< setMixedBell().
    bool mixedActive_ = false;                           ///< The engine the audio thread runs.
    NormalizedLadder<T, kMaxChannels> end_;              ///< The fixed filter at the end of the range.
    std::vector<std::vector<T>> mixScratch_;             ///< The end filter's input and output, per channel.
    std::vector<std::vector<T>> mixWeights_;             ///< Each sample's share of the end, per channel.
    bool endDesigned_ = false;                           ///< The end filter has a design.
    double endFreq_ = 0.0, endQ_ = 0.0, endRange_ = 0.0; ///< What it was designed for.
    double endShare_ = 0.0;                              ///< 1 - the end's gain at the centre.

    // Automatic threshold (setAutoThreshold()).
    static constexpr double kAutoRefFallDb = 1.0;  ///< dB per second its reference falls.
    static constexpr double kAutoRefFloor = 1e-5;  ///< The reference never goes under -100 dBFS.
    std::atomic<bool> autoThreshold_ { false };    ///< The threshold counts from the level of the input.
    std::atomic<bool> removalListen_ { false };            ///< The output is what the de-esser takes out.
    std::vector<std::vector<T>> removalScratch_;     ///< The input of the block, for the removal listen.
    bool relative_ = false;                        ///< Its value for the block (audio thread).
    double autoRef_ = kAutoRefFloor;               ///< The reference (linear).
    double autoFall_ = 1.0;                        ///< Its fall per sample (a factor).

    T envelope_{T(0)}; // Stereo-linked envelope
    T smoothedGrDb_[kMaxChannels]{};

    std::array<std::array<T, kDerivLen>, kMaxChannels> derivShift_ {};

    // Bell design inputs (clamped centre and Q), set with the detector.
    double peakFreq_ = 7000.0;
    double peakQ_ = 2.0;
    unsigned refreshCounter_ = 0; ///< Free-running coefficient-refresh grid.
};

} // namespace dspark
