// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

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
 *   Sidechain -> [Detector: BP / LP / HP per shape] -> Peak Detect -> Gain Computer -> Ballistics -> [Bell / Shelf EQ]
 * ```
 *
 * Threading model (per method; the model is docs/threading.md):
 * - prepare() / reset() / setOversampling() / setState(): setup or UI threads
 *   only, never concurrently with processBlock().
 * - setBand() / setNumBands() / setLookahead(): RT-safe from one control
 *   thread; the audio thread applies them at the next block. Each band is
 *   published through its own seqlock over std::atomic words, so the audio
 *   thread adopts a whole band configuration or none of it. Non-finite band
 *   fields keep the previous value. The audio thread's read of that seqlock is
 *   BOUNDED (StagedBand::tryRead): it never waits for the control thread to
 *   finish publishing, and a band whose read gives up keeps its current state
 *   while its dirty flag is re-armed, so the change lands one block later
 *   instead of holding the callback open.
 * - getState() reads the same seqlock into a private copy, so it is safe from
 *   any thread (it allocates, so keep it off the audio thread). It uses the
 *   UNBOUNDED StagedBand::read(): a readout has no previously adopted copy to
 *   keep, and it is not real-time.
 * - getBandGainDb() is an atomic metering read; getLatency() is safe from any
 *   thread once prepared.
 * - One-master designation (Core/Biquad.h, setCoeffsNow()): the embedded
 *   bandFilter_/bandDetector_ Biquads are AUDIO-MODULATED -- the audio thread
 *   writes their coefficients directly via Biquad::setCoeffsNow(), and nothing
 *   ever calls Biquad::setCoeffs() on them. Control parameters arrive only
 *   through this class's own StagedBand seqlock, adopted at processCore entry.
 * - Lookahead changes take effect immediately and may click (configure
 *   before playback).
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, Biquad.h, DspMath.h,
 *               Oversampling.h, RingBuffer.h, DenormalGuard.h, StateBlob.h.
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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

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
    /** @brief Shape of the dynamic gain filter (and its detector region). */
    enum class BandShape
    {
        Bell,       ///< Parametric bell; detector is a bandpass at freq/Q.
        LowShelf,   ///< Dynamic low shelf; detector hears below freq.
        HighShelf   ///< Dynamic high shelf; detector hears above freq.
    };

    /**
     * @brief Full configuration for a single dynamic EQ band.
     * @note Must remain trivially copyable for the lock-free seqlock publish.
     */
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
    static_assert(std::atomic<T>::is_always_lock_free,
                  "audio-thread stores must not lock");

    DynamicEQ()
    {
        for (int i = 0; i < MaxBands; ++i) {
            masterConfigs_[i] = BandConfig{};
            staged_[static_cast<size_t>(i)].publish(masterConfigs_[i]);
            paramsDirty_[i].store(true, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Initializes the dynamic EQ, allocating ring buffers and oversamplers.
     * @param spec Audio specification containing sample rate and block size.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // invalid specs are ignored (state kept)
        isPrepared_.store(false, std::memory_order_relaxed); // basic guarantee
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
        isPrepared_.store(true, std::memory_order_relaxed);
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
        if (!isPrepared_.load(std::memory_order_relaxed)) return;

        // A sidechain shorter than the audio block would be read past its
        // end; fall back to self-keying instead of over-reading the caller.
        if (sidechain.getNumChannels() <= 0 ||
            sidechain.getNumSamples() < audio.getNumSamples())
        {
            if (audio.getNumChannels() <= 0) return;
            sidechain = audio;
        }

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
     *
     * Fields are sanitized on the way in: non-finite values keep the band's
     * previously published value (a NaN frequency/Q/time used to poison the
     * detector or the gain ballistics permanently), wild shape enums are
     * clamped, and the range fields are floored at 0.
     */
    void setBand(int band, const BandConfig& config) noexcept
    {
        if (band < 0 || band >= MaxBands) return;

        BandConfig c = config;
        // The fallback reads the control thread's own private master, never the
        // published words: a setter must not race the audio thread to decide
        // what "the band's previously published value" is.
        const BandConfig& prev = masterConfigs_[static_cast<size_t>(band)];
        auto keep = [](T v, T fallback) { return std::isfinite(v) ? v : fallback; };
        c.frequency      = std::max(keep(c.frequency, prev.frequency), T(1));
        c.q              = keep(c.q, prev.q);
        c.threshold      = keep(c.threshold, prev.threshold);
        c.shape          = static_cast<BandShape>(std::clamp(static_cast<int>(c.shape), 0, 2));
        c.aboveRatio     = keep(c.aboveRatio, prev.aboveRatio);
        c.aboveAttackMs  = keep(c.aboveAttackMs, prev.aboveAttackMs);
        c.aboveReleaseMs = keep(c.aboveReleaseMs, prev.aboveReleaseMs);
        c.aboveRangeDb   = std::max(keep(c.aboveRangeDb, prev.aboveRangeDb), T(0));
        c.belowRatio     = keep(c.belowRatio, prev.belowRatio);
        c.belowAttackMs  = keep(c.belowAttackMs, prev.belowAttackMs);
        c.belowReleaseMs = keep(c.belowReleaseMs, prev.belowReleaseMs);
        c.belowRangeDb   = std::max(keep(c.belowRangeDb, prev.belowRangeDb), T(0));

        masterConfigs_[static_cast<size_t>(band)] = c;
        staged_[static_cast<size_t>(band)].publish(c);
        paramsDirty_[band].store(true, std::memory_order_release);
    }

    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 1, MaxBands), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the internal oversampling factor.
     *
     * Transparency policy: the factor is host-visible, fully configurable and
     * can be turned OFF.
     * - Supported factors: {1, 2, 4}. **DEFAULT is 1 = OFF** (no internal
     *   resampling, zero added latency, no hidden cascaded resampler). 3 rounds
     *   up to 4 (the underlying Oversampling engine works in powers of two);
     *   values outside [1,4] are clamped.
     * - CPU/latency cost per factor (relative to 1x=off): 2x roughly doubles the
     *   per-band detector+filter work and adds the half-band polyphase group
     *   delay; 4x roughly quadruples it and adds ~2x that group delay. The exact
     *   added latency for the active factor is reported by getLatency() so the
     *   host can apply correct PDC. Prefer 1x and oversample the whole nonlinear
     *   section once (docs/cookbook.md) when a chain already oversamples.
     *
     * @note Setup threads only: this un-prepares the processor and requires a
     *       prepare() call before the new factor takes effect.
     */
    void setOversampling(int factor) noexcept
    {
        oversamplingFactor_ = std::clamp(factor, 1, 4);
        if (oversamplingFactor_ == 3) oversamplingFactor_ = 4;
        isPrepared_.store(false, std::memory_order_relaxed); // Forces user to call prepare()
    }

    /** @brief Sets the lookahead (0..10 ms). Applied immediately (may click);
     *         non-finite values are ignored. */
    void setLookahead(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        lookaheadMs_ = std::clamp(ms, T(0), T(10));
        updateLookahead();
    }

    /**
     * @brief Total latency in samples at the base rate.
     *
     * Lookahead delay plus the oversampler's group delay. Neither was
     * reported before, so hosts could not compensate (PDC). Safe to call
     * from the main thread once prepared (the oversampler only changes
     * inside prepare()).
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        const int factor = std::max(oversamplingFactor_, 1);
        const int la = lookaheadSamples_.load(std::memory_order_relaxed) / factor;
        const int os = oversampler_ ? oversampler_->getLatency() : 0;
        return la + os;
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
            // Seqlock read into a private copy: reading the band by reference
            // here made getState() a plain cross-thread read of the words
            // setBand() writes.
            const BandConfig c = staged_[static_cast<size_t>(i)].read();
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

                // Refresh gain-filter coefficients every 16 samples - the gain
                // envelope is slow enough that this granularity is inaudible.
                // Bells use the precomputed freq/Q trig, so a refresh costs
                // one pow() instead of a full sin/cos/pow redesign; shelves
                // run their full design, which at 1/16th rate stays negligible.
                // Stream-owner direct writes (setCoeffsNow): these values are
                // computed HERE, on the audio thread, for this thread's own
                // use, so they never touch the staged cross-thread channel.
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
                            bandFilter_[b].setCoeffsNow(BiquadCoeffs::makeLowShelf(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(currentGain)));
                            break;
                        case BandShape::HighShelf:
                            bandFilter_[b].setCoeffsNow(BiquadCoeffs::makeHighShelf(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(currentGain)));
                            break;
                        }
                    }
                    else
                        bandFilter_[b].setCoeffsNow(BiquadCoeffs{}); // Bypass
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
        // Bounded seqlock read of the published config into a thread-private
        // plain copy. Every word crossing threads is std::atomic (see
        // StagedBand): copying the struct itself here, even under a correct
        // counter, was a plain concurrent read of words the control thread
        // writes and so a data race by the C++ model, not merely a torn-value
        // hazard. This runs on the audio thread, so the read is bounded: it
        // never waits for the control thread to finish publishing.
        BandConfig cfg;
        if (!staged_[static_cast<size_t>(b)].tryRead(cfg))
        {
            // Gave up: leave states_[b] exactly as it is -- the band keeps
            // running on its current config -- and re-arm so the publication is
            // adopted on a later block.
            paramsDirty_[b].store(true, std::memory_order_release);
            return;
        }

        // A band re-enabled after being disabled would replay arbitrarily old
        // filter history and gain state: start it clean.
        const bool wasEnabled = states_[b].cfg.enabled;
        if (cfg.enabled && !wasEnabled)
        {
            bandDetector_[b].reset();
            bandFilter_[b].reset();
            currentGainDb_[b] = T(0);
        }
        states_[b].cfg = cfg;

        // Detector listens where the gain filter acts: bandpass for bells,
        // the corresponding half of the spectrum for shelves.
        // Direct writes (setCoeffsNow): this runs on the audio thread, which
        // owns these Biquads (one-master rule, see the threading block), so
        // even this cold reconfiguration must not self-publish through the
        // staged channel.
        switch (cfg.shape)
        {
        case BandShape::Bell:
            bandDetector_[b].setCoeffsNow(BiquadCoeffs::makeBandPass(fs, cfg.frequency, cfg.q));
            break;
        case BandShape::LowShelf:
            bandDetector_[b].setCoeffsNow(BiquadCoeffs::makeLowPass(fs, cfg.frequency, T(0.707)));
            break;
        case BandShape::HighShelf:
            bandDetector_[b].setCoeffsNow(BiquadCoeffs::makeHighPass(fs, cfg.frequency, T(0.707)));
            break;
        }

        // Precompute the freq/Q-dependent peak-EQ terms ONCE per parameter change
        // (cos w0 and alpha), so the per-block dynamic update needs only a pow().
        // Double, like the rest of the coefficient path: the design is only
        // ever as good as the arithmetic that builds it.
        const double w0 = 2.0 * 3.14159265358979323846 * static_cast<double>(cfg.frequency) / fs;
        precomputedCos_[b]   = std::cos(w0);
        precomputedAlpha_[b] = std::sin(w0) / (2.0 * std::max(static_cast<double>(cfg.q), 0.001));

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
        const double A     = std::pow(10.0, static_cast<double>(gainDb) / 40.0);
        const double cosw  = precomputedCos_[b];
        const double alpha = precomputedAlpha_[b];
        const double a0Inv = 1.0 / (1.0 + alpha / A);

        BiquadCoeffs c;
        c.b0 = (1.0 + alpha * A) * a0Inv;
        c.b1 = (-2.0 * cosw)     * a0Inv;
        c.b2 = (1.0 - alpha * A) * a0Inv;
        c.a1 = (-2.0 * cosw)     * a0Inv;
        c.a2 = (1.0 - alpha / A) * a0Inv;
        // Stream-owner direct write: computed on the audio thread for its
        // own use, so it bypasses the staged cross-thread channel.
        bandFilter_[b].setCoeffsNow(c);
    }

    void updateLookahead() noexcept
    {
        if (sampleRate_ > 0) {
            int samples = static_cast<int>(sampleRate_ * oversamplingFactor_ * lookaheadMs_ / T(1000));
            lookaheadSamples_.store(samples, std::memory_order_relaxed);
        }
    }

    // -- State & Mem ---------------------------------------------------------
    std::atomic<bool> isPrepared_ { false };
    AudioSpec spec_ {};
    double sampleRate_ = 0;

    std::atomic<int> numBands_ { 0 };

    /**
     * @brief One band's published configuration: a seqlock over atomic words.
     *
     * A BandConfig is a multi-word set with an invariant, so it uses the
     * canonical seqlock of docs/threading.md, matching Core/Biquad.h and
     * Core/FIRFilter.h frame for frame. Every word crossing threads is
     * std::atomic: staging the struct itself, even inside a correct seqlock,
     * is a plain concurrent read/write and therefore a data race by the C++
     * memory model regardless of what the counter says.
     */
    struct StagedBand
    {
        std::atomic<T>    frequency      { T(1000) };
        std::atomic<T>    q              { T(1.0) };
        std::atomic<T>    threshold      { T(-20) };
        std::atomic<int>  shape          { static_cast<int>(BandShape::Bell) };
        std::atomic<bool> enabled        { true };

        std::atomic<T>    aboveRatio     { T(1) };
        std::atomic<T>    aboveAttackMs  { T(5) };
        std::atomic<T>    aboveReleaseMs { T(50) };
        std::atomic<T>    aboveRangeDb   { T(12) };
        std::atomic<bool> aboveBoost     { false };

        std::atomic<T>    belowRatio     { T(1) };
        std::atomic<T>    belowAttackMs  { T(10) };
        std::atomic<T>    belowReleaseMs { T(100) };
        std::atomic<T>    belowRangeDb   { T(12) };
        std::atomic<bool> belowBoost     { false };

        std::atomic<unsigned> seq        { 0 };

        /**
         * @brief Publishes a whole band configuration (control thread only).
         *
         * An odd sequence number marks a write in progress; a reader that
         * observes one retries. The release fence is mandatory: an
         * acquire/release counter alone does not order the relaxed word stores
         * against it, so without it a reader could observe a mid-publish word
         * while the counter still looked even ([atomics.fences]/2; Boehm, "Can
         * Seqlocks Get Along With Programming Language Memory Models?", MSPC
         * 2012). Control-thread only: the audio path pays nothing for it.
         */
        void publish(const BandConfig& c) noexcept
        {
            seq.fetch_add(1, std::memory_order_acq_rel);   // -> odd
            std::atomic_thread_fence(std::memory_order_release);
            frequency.store(c.frequency, std::memory_order_relaxed);
            q.store(c.q, std::memory_order_relaxed);
            threshold.store(c.threshold, std::memory_order_relaxed);
            shape.store(static_cast<int>(c.shape), std::memory_order_relaxed);
            enabled.store(c.enabled, std::memory_order_relaxed);
            aboveRatio.store(c.aboveRatio, std::memory_order_relaxed);
            aboveAttackMs.store(c.aboveAttackMs, std::memory_order_relaxed);
            aboveReleaseMs.store(c.aboveReleaseMs, std::memory_order_relaxed);
            aboveRangeDb.store(c.aboveRangeDb, std::memory_order_relaxed);
            aboveBoost.store(c.aboveBoost, std::memory_order_relaxed);
            belowRatio.store(c.belowRatio, std::memory_order_relaxed);
            belowAttackMs.store(c.belowAttackMs, std::memory_order_relaxed);
            belowReleaseMs.store(c.belowReleaseMs, std::memory_order_relaxed);
            belowRangeDb.store(c.belowRangeDb, std::memory_order_relaxed);
            belowBoost.store(c.belowBoost, std::memory_order_relaxed);
            seq.fetch_add(1, std::memory_order_release);   // -> even
        }

        /// Attempt bound for the audio-thread read (tryRead). Three is enough
        /// that attempt 1 succeeds whenever the reader does not collide with a
        /// publish, with two spare attempts to absorb one unlucky collision;
        /// past that, deferring the update by one call costs less than spending
        /// more of the block budget on it.
        static constexpr int kSeqlockMaxAttempts = 3;

        /** @brief Relaxed copy of the fifteen published words (no ordering of
         *  its own; the caller's fences and counter checks provide that).
         *
         *  Shared by both readers below so the word list exists exactly once: a
         *  field added to the band cannot be picked up by one reader and missed
         *  by the other. */
        void loadWordsRelaxed(BandConfig& c) const noexcept
        {
            c.frequency      = frequency.load(std::memory_order_relaxed);
            c.q              = q.load(std::memory_order_relaxed);
            c.threshold      = threshold.load(std::memory_order_relaxed);
            c.shape          = static_cast<BandShape>(shape.load(std::memory_order_relaxed));
            c.enabled        = enabled.load(std::memory_order_relaxed);
            c.aboveRatio     = aboveRatio.load(std::memory_order_relaxed);
            c.aboveAttackMs  = aboveAttackMs.load(std::memory_order_relaxed);
            c.aboveReleaseMs = aboveReleaseMs.load(std::memory_order_relaxed);
            c.aboveRangeDb   = aboveRangeDb.load(std::memory_order_relaxed);
            c.aboveBoost     = aboveBoost.load(std::memory_order_relaxed);
            c.belowRatio     = belowRatio.load(std::memory_order_relaxed);
            c.belowAttackMs  = belowAttackMs.load(std::memory_order_relaxed);
            c.belowReleaseMs = belowReleaseMs.load(std::memory_order_relaxed);
            c.belowRangeDb   = belowRangeDb.load(std::memory_order_relaxed);
            c.belowBoost     = belowBoost.load(std::memory_order_relaxed);
        }

        /**
         * @brief The bounded seqlock read for the AUDIO thread.
         *
         * Makes at most kSeqlockMaxAttempts validation attempts and then gives
         * up rather than spin until the control thread finishes publishing --
         * an audio-thread loop whose exit depends on another thread's progress
         * is a stall of that thread's scheduling quantum, inside the callback.
         * The accept test is the same one read() uses, so bounding can never
         * make a torn set adoptable; it can only defer an adoption. `out` is
         * written only once a copy has validated, so a caller may pass storage
         * it is still using and a give-up leaves it intact.
         *
         * An odd counter means the writer is mid-publish: skip the copy
         * entirely instead of copying fifteen words that would be discarded.
         *
         * @param out Receives the band configuration if and only if true is returned.
         * @return true if a consistent configuration was adopted; false if the
         *         attempt bound was reached (the caller keeps what it has and
         *         should re-arm its dirty flag to retry on a later call).
         */
        [[nodiscard]] bool tryRead(BandConfig& out) const noexcept
        {
            for (int attempt = 0; attempt < kSeqlockMaxAttempts; ++attempt)
            {
                const unsigned s0 = seq.load(std::memory_order_acquire);
                if ((s0 & 1u) != 0u) continue;   // writer mid-publish: do not copy
                BandConfig c;
                loadWordsRelaxed(c);
                std::atomic_thread_fence(std::memory_order_acquire);
                if (s0 == seq.load(std::memory_order_relaxed))
                {
                    out = c;                     // commit only on validation
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Reads the band into a caller-private plain copy (readouts).
         *
         * Control- and GUI-thread entry point, and NOT for the audio thread:
         * this retry is unbounded on purpose. A readout has no previously
         * adopted copy to fall back on, so "sorry, try again" would be a worse
         * API than a microsecond of spinning; the audio path uses tryRead().
         *
         * The acquire fence keeps the word copy from sinking below the re-read
         * of the counter: an acquire LOAD orders only later accesses, so
         * without the fence a torn copy could pass the s0 == s1 check on a
         * weakly ordered target or after compiler reordering. It also pairs
         * with the writer's release fence, so a copy that read any mid-publish
         * word cannot validate.
         */
        [[nodiscard]] BandConfig read() const noexcept
        {
            BandConfig c;
            unsigned s0, s1;
            do {
                s0 = seq.load(std::memory_order_acquire);
                loadWordsRelaxed(c);
                std::atomic_thread_fence(std::memory_order_acquire);
                s1 = seq.load(std::memory_order_relaxed);
            } while ((s0 & 1u) != 0u || s0 != s1);
            return c;
        }
    };

    /// Published band configurations. The ONLY band storage both threads reach.
    std::array<StagedBand, MaxBands> staged_ {};

    /// Control-thread-private master copy. Never read by the audio thread or by
    /// any readout: it exists so a setter can fall back to the band's current
    /// value for a non-finite field without reading published storage. Single
    /// writer by contract, so it needs no synchronization of its own.
    std::array<BandConfig, MaxBands> masterConfigs_ {};
    std::array<std::atomic<bool>, MaxBands> paramsDirty_ {};
    std::array<BandState, MaxBands> states_ {};
    // Precomputed freq/Q-dependent peak-EQ trig terms (per band) so the per-block
    // dynamic gain update needs only a pow(), not cos/sin/pow every sample.
    std::array<double, MaxBands> precomputedCos_ {};
    std::array<double, MaxBands> precomputedAlpha_ {};

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
