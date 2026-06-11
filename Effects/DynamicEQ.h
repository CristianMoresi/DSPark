// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DynamicEQ.h
 * @brief Dynamic parametric equalizer with per-band level detection and true VCA ballistics.
 *
 * Implements a high-performance dynamic EQ. Unlike standard envelope followers,
 * this implementation applies attack/release ballistics directly to the gain reduction
 * signal, accurately modeling analog VCA behavior and allowing true independent
 * times for Above and Below threshold processing.
 *
 * Architecture per band:
 * ```
 *   Sidechain → [Detector: BP / LP / HP per shape] → Peak Detect → Gain Computer → Ballistics → [Bell / Shelf EQ]
 * ```
 *
 * @note Thread-safe parameter updates via `std::atomic` ensuring zero-locking
 *       in the real-time audio thread.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/Oversampling.h"
#include "../Core/RingBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <type_traits>

namespace dspark {

/**
 * @class DynamicEQ
 * @brief Dynamic parametric EQ with dual above/below threshold per band.
 *
 * @tparam T        Sample type (float or double).
 * @tparam MaxBands Maximum number of bands (compile-time, default 8).
 */
template <FloatType T, int MaxBands = 8>
class DynamicEQ
{
public:
    /**
     * @brief Full configuration for a single dynamic EQ band.
     * @note Must remain trivially copyable to allow lock-free atomic updates.
     */
    /** @brief Shape of the dynamic gain filter (and its detector region). */
    enum class BandShape
    {
        Bell,       ///< Parametric bell; detector is a bandpass at freq/Q.
        LowShelf,   ///< Dynamic low shelf; detector hears below freq.
        HighShelf   ///< Dynamic high shelf; detector hears above freq.
    };

    struct BandConfig
    {
        T frequency      = T(1000);
        T q              = T(1.0);
        T threshold      = T(-20);
        BandShape shape  = BandShape::Bell;
        bool enabled     = true;

        T aboveRatio     = T(1);
        T aboveAttackMs  = T(5);
        T aboveReleaseMs = T(50);
        T aboveRangeDb   = T(12);
        bool aboveBoost  = false;

        T belowRatio     = T(1);
        T belowAttackMs  = T(10);
        T belowReleaseMs = T(100);
        T belowRangeDb   = T(12);
        bool belowBoost  = false;
    };

    static_assert(std::is_trivially_copyable_v<BandConfig>, "BandConfig must be trivially copyable for std::atomic");

    DynamicEQ()
    {
        for (int i = 0; i < MaxBands; ++i) {
            configs_[i] = BandConfig{};
            configSeq_[i].store(0, std::memory_order_relaxed);
            paramsDirty_[i].store(true, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Initializes the dynamic EQ, allocating ring buffers and oversamplers.
     * @param spec Audio specification containing sample rate and block size.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        sampleRate_ = spec.sampleRate;

        if (oversamplingFactor_ > 1)
        {
            // Two oversamplers: the audio-path one owns the buffer that we
            // both process and downsample back; the sidechain one only acts
            // as a level-detection upsampler so its buffer is read-only.
            oversampler_   = std::make_unique<Oversampling<T>>(oversamplingFactor_);
            oversamplerSc_ = std::make_unique<Oversampling<T>>(oversamplingFactor_);
            oversampler_  ->prepare(spec);
            oversamplerSc_->prepare(spec);
        }
        else
        {
            oversampler_.reset();
            oversamplerSc_.reset();
        }

        int maxLaSamples = static_cast<int>(sampleRate_ * oversamplingFactor_ * 0.01) + 1;
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].prepare(maxLaSamples);

        updateLookahead();
        reset();
        isPrepared_ = true;
    }

    /**
     * @brief Processes audio in-place using self-sidechain.
     * @param buffer In/Out audio buffer.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlock(buffer, buffer);
    }

    /**
     * @brief Processes audio with an external sidechain.
     * @param audio In/Out audio buffer.
     * @param sidechain Reference signal used for level detection.
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        if (!isPrepared_) return;
        DenormalGuard guard;

        if (oversamplingFactor_ > 1 && oversampler_ && oversamplerSc_)
        {
            // Up-sample audio and sidechain through their dedicated oversamplers.
            // Each Oversampling instance owns one internal high-rate buffer, so
            // we cannot share one between the two streams; processing them
            // separately keeps each stream's polyphase filter state consistent.
            auto upAudio = oversampler_->upsample(audio);
            auto upSc    = oversamplerSc_->upsample(sidechain);

            processCore(upAudio, upSc, sampleRate_ * oversamplingFactor_);

            oversampler_->downsample(audio);
        }
        else
        {
            // Standard processing path
            processCore(audio, sidechain, sampleRate_);
        }
    }

    /**
     * @brief Thread-safe configuration update for a specific band.
     */
    void setBand(int band, const BandConfig& config) noexcept
    {
        if (band < 0 || band >= MaxBands) return;
        // Seqlock publish (lock-free, no atomic<BigStruct> mutex).
        configSeq_[band].fetch_add(1, std::memory_order_acq_rel); // odd
        configs_[band] = config;
        configSeq_[band].fetch_add(1, std::memory_order_release); // even
        paramsDirty_[band].store(true, std::memory_order_release);
    }

    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 1, MaxBands), std::memory_order_relaxed);
    }

    void setOversampling(int factor) noexcept
    {
        oversamplingFactor_ = std::clamp(factor, 1, 4);
        if (oversamplingFactor_ == 3) oversamplingFactor_ = 4;
        isPrepared_ = false; // Forces user to call prepare()
    }

    void setLookahead(T ms) noexcept
    {
        lookaheadMs_ = std::clamp(ms, T(0), T(10));
        updateLookahead();
    }

    [[nodiscard]] T getBandGainDb(int band) const noexcept
    {
        if (band < 0 || band >= MaxBands) return T(0);
        return meterGainDb_[band].load(std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        for (int b = 0; b < MaxBands; ++b)
        {
            bandDetector_[b].reset();
            bandFilter_[b].reset();
            currentGainDb_[b] = T(0);
            meterGainDb_[b].store(T(0), std::memory_order_relaxed);
            paramsDirty_[b].store(true, std::memory_order_relaxed);
        }
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].reset();
    }


    /** @brief Serializes bands and modes (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DYEQ"), 1);
        const int n = numBands_.load(std::memory_order_relaxed);
        w.write("numBands", n);
        char key[28];
        for (int i = 0; i < n; ++i)
        {
            const BandConfig& c = configs_[static_cast<size_t>(i)];
            std::snprintf(key, sizeof(key), "b%d.freq", i);
            w.write(key, static_cast<float>(c.frequency));
            std::snprintf(key, sizeof(key), "b%d.q", i);
            w.write(key, static_cast<float>(c.q));
            std::snprintf(key, sizeof(key), "b%d.thresh", i);
            w.write(key, static_cast<float>(c.threshold));
            std::snprintf(key, sizeof(key), "b%d.shape", i);
            w.write(key, static_cast<int>(c.shape));
            std::snprintf(key, sizeof(key), "b%d.on", i);
            w.write(key, c.enabled);
            std::snprintf(key, sizeof(key), "b%d.aRatio", i);
            w.write(key, static_cast<float>(c.aboveRatio));
            std::snprintf(key, sizeof(key), "b%d.aAtk", i);
            w.write(key, static_cast<float>(c.aboveAttackMs));
            std::snprintf(key, sizeof(key), "b%d.aRel", i);
            w.write(key, static_cast<float>(c.aboveReleaseMs));
            std::snprintf(key, sizeof(key), "b%d.aRange", i);
            w.write(key, static_cast<float>(c.aboveRangeDb));
            std::snprintf(key, sizeof(key), "b%d.aBoost", i);
            w.write(key, c.aboveBoost);
            std::snprintf(key, sizeof(key), "b%d.bRatio", i);
            w.write(key, static_cast<float>(c.belowRatio));
            std::snprintf(key, sizeof(key), "b%d.bAtk", i);
            w.write(key, static_cast<float>(c.belowAttackMs));
            std::snprintf(key, sizeof(key), "b%d.bRel", i);
            w.write(key, static_cast<float>(c.belowReleaseMs));
            std::snprintf(key, sizeof(key), "b%d.bRange", i);
            w.write(key, static_cast<float>(c.belowRangeDb));
            std::snprintf(key, sizeof(key), "b%d.bBoost", i);
            w.write(key, c.belowBoost);
        }
        return w.blob();
    }

    /** @brief Restores bands from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DYEQ")) return false;
        const int n = std::clamp(r.read("numBands", 0), 0, MaxBands);
        char key[28];
        for (int i = 0; i < n; ++i)
        {
            BandConfig c;
            std::snprintf(key, sizeof(key), "b%d.freq", i);
            c.frequency = static_cast<T>(r.read(key, 1000.0f));
            std::snprintf(key, sizeof(key), "b%d.q", i);
            c.q = static_cast<T>(r.read(key, 1.0f));
            std::snprintf(key, sizeof(key), "b%d.thresh", i);
            c.threshold = static_cast<T>(r.read(key, -20.0f));
            std::snprintf(key, sizeof(key), "b%d.shape", i);
            c.shape = static_cast<BandShape>(std::clamp(r.read(key, 0), 0, 2));
            std::snprintf(key, sizeof(key), "b%d.on", i);
            c.enabled = r.read(key, true);
            std::snprintf(key, sizeof(key), "b%d.aRatio", i);
            c.aboveRatio = static_cast<T>(r.read(key, 1.0f));
            std::snprintf(key, sizeof(key), "b%d.aAtk", i);
            c.aboveAttackMs = static_cast<T>(r.read(key, 5.0f));
            std::snprintf(key, sizeof(key), "b%d.aRel", i);
            c.aboveReleaseMs = static_cast<T>(r.read(key, 50.0f));
            std::snprintf(key, sizeof(key), "b%d.aRange", i);
            c.aboveRangeDb = static_cast<T>(r.read(key, 12.0f));
            std::snprintf(key, sizeof(key), "b%d.aBoost", i);
            c.aboveBoost = r.read(key, false);
            std::snprintf(key, sizeof(key), "b%d.bRatio", i);
            c.belowRatio = static_cast<T>(r.read(key, 1.0f));
            std::snprintf(key, sizeof(key), "b%d.bAtk", i);
            c.belowAttackMs = static_cast<T>(r.read(key, 10.0f));
            std::snprintf(key, sizeof(key), "b%d.bRel", i);
            c.belowReleaseMs = static_cast<T>(r.read(key, 100.0f));
            std::snprintf(key, sizeof(key), "b%d.bRange", i);
            c.belowRangeDb = static_cast<T>(r.read(key, 12.0f));
            std::snprintf(key, sizeof(key), "b%d.bBoost", i);
            c.belowBoost = r.read(key, false);
            setBand(i, c);
        }
        setNumBands(n);
        return true;
    }

private:
    static constexpr int kMaxChannels = 16;
    static constexpr T kMinLevelDb = T(-100.0);
    static constexpr T kMinEnvelope = T(1e-12); // Prevents NaN in log10

    struct BandState
    {
        BandConfig cfg;
        T aboveAtkCoeff, aboveRelCoeff;
        T belowAtkCoeff, belowRelCoeff;
    };

    void processCore(AudioBufferView<T>& audio, AudioBufferView<T>& sidechain, double currentFs) noexcept
    {
        const int nCh  = std::min(audio.getNumChannels(), kMaxChannels);
        const int scCh = sidechain.getNumChannels();
        const int nS   = audio.getNumSamples();
        if (scCh <= 0) return; // a 0-channel sidechain would index getChannel(-1)
        const int nb   = numBands_.load(std::memory_order_relaxed);
        const int laSamples = lookaheadSamples_.load(std::memory_order_relaxed);

        // 1. Thread-Safe State Update
        for (int b = 0; b < nb; ++b)
        {
            if (paramsDirty_[b].exchange(false, std::memory_order_acquire))
                updateBandInternalState(b, currentFs);
        }

        // 2. Audio Processing Loop
        for (int i = 0; i < nS; ++i)
        {
            for (int b = 0; b < nb; ++b)
            {
                if (!states_[b].cfg.enabled) continue;
                
                T maxLevelDb = kMinLevelDb;
                
                // Sidechain Detection (Stereo Linked by Max Peak)
                for (int ch = 0; ch < nCh; ++ch)
                {
                    int sc = std::min(ch, scCh - 1);
                    T scSample = sidechain.getChannel(sc)[i];
                    
                    T detected = std::abs(bandDetector_[b].processSample(scSample, ch));
                    T levelDb = gainToDecibels(std::max(detected, kMinEnvelope));
                    
                    if (levelDb > maxLevelDb) maxLevelDb = levelDb;
                }

                // Gain Computer
                T targetGainDb = computeTargetGain(states_[b].cfg, maxLevelDb);

                // Gain Ballistics (Attack/Release applied to the Gain itself)
                T& currentGain = currentGainDb_[b];
                T diff = targetGainDb - currentGain;
                
                T coeff;
                if (maxLevelDb > states_[b].cfg.threshold) {
                    coeff = (std::abs(targetGainDb) > std::abs(currentGain)) 
                            ? states_[b].aboveAtkCoeff : states_[b].aboveRelCoeff;
                } else {
                    coeff = (std::abs(targetGainDb) > std::abs(currentGain)) 
                            ? states_[b].belowAtkCoeff : states_[b].belowRelCoeff;
                }
                
                currentGain += coeff * diff;

                // Refresh gain-filter coefficients every 16 samples — the gain
                // envelope is slow enough that this granularity is inaudible.
                // Bells use the precomputed freq/Q trig (a single pow() per
                // refresh, F-059 performance fix); shelves run their full
                // design, which at 1/16th rate stays negligible.
                if ((i & 15) == 0)
                {
                    if (std::abs(currentGain) > T(0.01))
                    {
                        switch (states_[b].cfg.shape)
                        {
                        case BandShape::Bell:
                            updateDynamicPeakCoeffs(b, currentGain);
                            break;
                        case BandShape::LowShelf:
                            bandFilter_[b].setCoeffs(BiquadCoeffs<T>::makeLowShelf(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(currentGain)));
                            break;
                        case BandShape::HighShelf:
                            bandFilter_[b].setCoeffs(BiquadCoeffs<T>::makeHighShelf(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(currentGain)));
                            break;
                        }
                    }
                    else
                        bandFilter_[b].setCoeffs(BiquadCoeffs<T>{}); // Bypass
                }

                if ((i & 63) == 0) // Sub-sample metering update
                    meterGainDb_[b].store(currentGain, std::memory_order_relaxed);
            }

            // Apply Filters
            for (int ch = 0; ch < nCh; ++ch)
            {
                T audioSample = audio.getChannel(ch)[i];
                
                if (laSamples > 0) {
                    lookaheadBuf_[ch].push(audioSample);
                    audioSample = lookaheadBuf_[ch].read(laSamples);
                }

                for (int b = 0; b < nb; ++b) {
                    if (states_[b].cfg.enabled) {
                        audioSample = bandFilter_[b].processSample(audioSample, ch);
                    }
                }
                audio.getChannel(ch)[i] = audioSample;
            }
        }
    }

    [[nodiscard]] T computeTargetGain(const BandConfig& cfg, T levelDb) const noexcept
    {
        T gainDb = T(0);

        if (levelDb > cfg.threshold)
        {
            if (cfg.aboveRatio > T(1.001)) {
                T overDb = levelDb - cfg.threshold;
                T amount = std::min(overDb * (T(1) - T(1) / cfg.aboveRatio), cfg.aboveRangeDb);
                gainDb += cfg.aboveBoost ? amount : -amount;
            }
        }
        else
        {
            if (cfg.belowRatio > T(1.001)) {
                T underDb = cfg.threshold - levelDb;
                T amount = std::min(underDb * (T(1) - T(1) / cfg.belowRatio), cfg.belowRangeDb);
                gainDb += cfg.belowBoost ? amount : -amount;
            }
        }
        return gainDb;
    }

    void updateBandInternalState(int b, double fs) noexcept
    {
        // Seqlock read of the published config (retry on a torn/in-progress write).
        BandConfig cfg;
        unsigned s0, s1;
        do {
            s0  = configSeq_[b].load(std::memory_order_acquire);
            cfg = configs_[b];
            s1  = configSeq_[b].load(std::memory_order_acquire);
        } while ((s0 & 1u) != 0u || s0 != s1);
        states_[b].cfg = cfg;

        // Detector listens where the gain filter acts: bandpass for bells,
        // the corresponding half of the spectrum for shelves.
        switch (cfg.shape)
        {
        case BandShape::Bell:
            bandDetector_[b].setCoeffs(BiquadCoeffs<T>::makeBandPass(fs, cfg.frequency, cfg.q));
            break;
        case BandShape::LowShelf:
            bandDetector_[b].setCoeffs(BiquadCoeffs<T>::makeLowPass(fs, cfg.frequency, T(0.707)));
            break;
        case BandShape::HighShelf:
            bandDetector_[b].setCoeffs(BiquadCoeffs<T>::makeHighPass(fs, cfg.frequency, T(0.707)));
            break;
        }

        // Precompute the freq/Q-dependent peak-EQ terms ONCE per parameter change
        // (cos w0 and alpha), so the per-block dynamic update needs only a pow().
        const double w0 = 2.0 * 3.14159265358979323846 * static_cast<double>(cfg.frequency) / fs;
        precomputedCos_[b]   = static_cast<T>(std::cos(w0));
        precomputedAlpha_[b] = static_cast<T>(std::sin(w0) / (2.0 * std::max(static_cast<double>(cfg.q), 0.001)));

        auto calcCoeff = [fs](T ms) -> T {
            const double tauSec = std::max(static_cast<double>(ms), 0.01) / 1000.0;
            return static_cast<T>(1.0 - std::exp(-1.0 / (fs * tauSec)));
        };

        states_[b].aboveAtkCoeff = calcCoeff(cfg.aboveAttackMs);
        states_[b].aboveRelCoeff = calcCoeff(cfg.aboveReleaseMs);
        states_[b].belowAtkCoeff = calcCoeff(cfg.belowAttackMs);
        states_[b].belowRelCoeff = calcCoeff(cfg.belowReleaseMs);
    }

    /** @brief Fast peak-EQ coefficient update from precomputed cos/alpha + gain. */
    void updateDynamicPeakCoeffs(int b, T gainDb) noexcept
    {
        const T A     = std::pow(T(10), gainDb / T(40));
        const T cosw  = precomputedCos_[b];
        const T alpha = precomputedAlpha_[b];
        const T a0Inv = T(1) / (T(1) + alpha / A);

        BiquadCoeffs<T> c;
        c.b0 = (T(1) + alpha * A) * a0Inv;
        c.b1 = (T(-2) * cosw)     * a0Inv;
        c.b2 = (T(1) - alpha * A) * a0Inv;
        c.a1 = (T(-2) * cosw)     * a0Inv;
        c.a2 = (T(1) - alpha / A) * a0Inv;
        bandFilter_[b].setCoeffs(c);
    }

    void updateLookahead() noexcept
    {
        if (sampleRate_ > 0) {
            int samples = static_cast<int>(sampleRate_ * oversamplingFactor_ * lookaheadMs_ / T(1000));
            lookaheadSamples_.store(samples, std::memory_order_relaxed);
        }
    }

    // -- State & Mem ---------------------------------------------------------
    bool isPrepared_ = false;
    AudioSpec spec_ {};
    double sampleRate_ = 0;
    
    std::atomic<int> numBands_ { 0 };
    // BandConfig (~50 bytes) is NOT lock-free as std::atomic, so publish it via a
    // per-band seqlock instead (single producer = control thread, single consumer
    // = audio thread). configSeq_ odd = write in progress.
    std::array<BandConfig, MaxBands> configs_ {};
    std::array<std::atomic<unsigned>, MaxBands> configSeq_ {};
    std::array<std::atomic<bool>, MaxBands> paramsDirty_ {};
    std::array<BandState, MaxBands> states_ {};
    // Precomputed freq/Q-dependent peak-EQ trig terms (per band) so the per-block
    // dynamic gain update needs only a pow(), not cos/sin/pow every sample.
    std::array<T, MaxBands> precomputedCos_ {};
    std::array<T, MaxBands> precomputedAlpha_ {};

    // Biquads must span the class's full channel capacity (kMaxChannels); the
    // default Biquad<T> is only 8 channels, which would index its per-channel state
    // out of bounds when processing 9..16-channel (surround/immersive) audio.
    std::array<Biquad<T, kMaxChannels>, MaxBands> bandDetector_ {};
    std::array<Biquad<T, kMaxChannels>, MaxBands> bandFilter_ {};
    std::array<T, MaxBands> currentGainDb_ {};
    std::array<std::atomic<T>, MaxBands> meterGainDb_ {};

    int oversamplingFactor_ = 1;
    std::unique_ptr<Oversampling<T>> oversampler_;     // audio path
    std::unique_ptr<Oversampling<T>> oversamplerSc_;   // sidechain path

    T lookaheadMs_ = T(0);
    std::atomic<int> lookaheadSamples_ { 0 };
    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuf_ {};
};

} // namespace dspark
