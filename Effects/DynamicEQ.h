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
 * The peak detect is max(|x|, analytic magnitude) of the detector output, the
 * magnitude from a zero-latency IIR Hilbert pair (Core/Hilbert.h): a steady
 * tone reads as its flat amplitude. The rectified |x| alone rippled at twice
 * the tone frequency, and the gain computer turned that ripple into
 * distortion on sustained low notes (measured THD+N on a steady 60 Hz tone,
 * -30 dB threshold, 3:1: 0.22% bell, 0.30% low shelf; now below 0.001%).
 *
 * Bells use BiquadCoeffs::makePeakMatched() (Vicanek's matched design): the
 * width follows the analog prototype up to Nyquist instead of cramping (a
 * +12 dB bell at 16 kHz gives +10.0 dB at 20 kHz at 48 kHz, prototype +9.65;
 * the cookbook form gave +3.4), and a centre above Nyquist is clamped
 * instead of producing an unstable filter.
 *
 * Non-finite samples read as silence in the detectors and bypass the gain
 * filters for that sample, so one NaN or Inf cannot latch in their state.
 *
 * A band rests at its own gain (BandConfig::gain, 0 dB by default) and the
 * dynamic part moves it from there, up to the range of each side.
 *
 * Mixed bands (setMixedBands(), off by default): the gain of a band at rest
 * is then a filter of its own, and the part that moves is the signal and one
 * fixed filter per side (the end of the range), with only how much of the
 * filter is heard changing:
 * ```
 *   y = x + w * (end(x) - x)
 * ```
 * with w from the gain in amplitude, so the centre of a bell (the plateau of
 * a shelf) sits exactly on the gain the ballistics ask for, sample by
 * sample. No recursion has its coefficients moved while it plays, and
 * nothing is designed while the gain moves: the default engine designs the
 * band again every 16 samples, which is a gain that moves in steps (a steady
 * 6 kHz tone under a band that opens and closes 25 times a second grows
 * lines 3 kHz away at -66 dBc; mixed, they are at -104). Two fixed curves
 * cannot be the nominal band at every depth, so the end is chosen for the
 * band to be exactly the bell or shelf of its Q, fitted to the analog curve
 * up to Nyquist, half way through the range in decibels; nearer the rest a
 * cut is wider than nominal and at the end of the range narrower (by a
 * factor of 10^(range / 80) in Q at most: 1.41 for 12 dB), and a boost the
 * other way round. A band that rests at 0 dB and is not acting returns its
 * input bit for bit.
 *
 * Linear phase (setLinearPhase(), off by default; a setup choice): the
 * filters of a band are then zero-phase kernels, run by one partitioned
 * convolution (Core/FirBank.h), and the gain sets how much of each is added
 * to the signal, which is late by the latency of the mode (half the kernel
 * plus one partition: 2560 samples at 44.1 and 48 kHz). The band changes the
 * level and nothing of the phase; its gain is held back to meet the audio it
 * was computed from; and the bands add up side by side, each one heard from
 * the input, so the mode has one latency whatever the number of bands. The
 * ends are chosen as in the mixed engine, on magnitudes: half way through
 * the range the band has exactly the magnitude of the nominal one.
 *
 * What the gain follows, per band (all off by default):
 * - BandConfig::knee: a soft knee around the threshold, or an automatic one
 *   that grows with the width of the band and with its range.
 * - BandConfig::timing = Program: the band listens through a filter 1.5
 *   times narrower and three followers replace the attack and release
 *   fields: a fast one, a slow one that follows the fast, and a last one
 *   over the larger of the two. Their times come from the frequency and the
 *   width of the band (a band cannot know its level sooner than its own
 *   filter lets it). A short event releases at the fast rate and a held one
 *   at the slow rate.
 * - BandConfig::relativeThreshold: the threshold counts from the band's own
 *   recent level (a reference that rises at once and falls 2 dB per second)
 *   instead of from full scale, so the band acts the same on a quiet take
 *   and on a loud one. When the band is moved to another zone a second
 *   reference listens there for half a second and then takes over.
 * - BandConfig::scope: the band acts on every channel (linked by the
 *   loudest), on the left or the right alone, or on the middle or the sides
 *   of the first pair, and listens to the same part of the sidechain.
 *
 * Threading model (per method; the model is docs/threading.md):
 * - prepare() / reset() / setOversampling() / setLinearPhase() /
 *   setLinearPhaseKernelLength() / setState(): setup or UI threads only,
 *   never concurrently with processBlock().
 * - setMixedBands(): any thread; the audio thread changes engine at the next
 *   block and the bands start clean there (may click: choose it before
 *   playback).
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
 * Dependencies: MidSide.h, AudioBuffer.h, AudioSpec.h, Biquad.h,
 *               BiquadCascade.h, DspMath.h, FFT.h, FilterDesign.h, FirBank.h,
 *               Hilbert.h,
 *               Oversampling.h, RingBuffer.h, DenormalGuard.h, StateBlob.h.
 */

#include "MidSide.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/BiquadCascade.h"
#include "../Core/DspMath.h"
#include "../Core/FilterDesign.h"
#include "../Core/FirBank.h"
#include "../Core/FFT.h"
#include "../Core/Hilbert.h"
#include "../Core/NormalizedLadder.h"
#include "../Core/Oversampling.h"
#include "../Core/RingBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numbers>
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

    /** @brief Where in a stereo pair a band acts and listens. */
    enum class Scope
    {
        Stereo, ///< Every channel, linked by the loudest. Default.
        Left,   ///< The first channel only.
        Right,  ///< The second channel only.
        Mid,    ///< The middle of the first two channels: what they share.
        Side    ///< Their sides: what sets them apart.
    };

    /** @brief What sets how fast the gain of a band moves. */
    enum class Timing
    {
        Manual,  ///< The attack and release fields, applied to the gain. Default.
        Program  ///< Three followers of the level, timed from the band's frequency and width.
    };

    /** @brief BandConfig::knee value that asks for the automatic knee. */
    static constexpr T kAutomaticKnee = T(-1);

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

        T gain           = T(0);            ///< Gain of the band at rest, in dB (-24 to 24). The dynamic part moves from here.
        T knee           = T(0);            ///< Width of the knee around the threshold in dB (0 to 40); negative (kAutomaticKnee): from the band's width and range.
        Scope scope = Scope::Stereo;   ///< The channels the band acts on and listens to.
        Timing timing    = Timing::Manual;  ///< Program: the times come from the band, and the attack and release fields are not used.
        bool relativeThreshold = false;     ///< The threshold counts from the band's own recent level instead of from full scale.

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
        if (oversamplingAutomatic_)
            oversamplingFactor_ = Oversampling<T>::factorForRate(spec.sampleRate, kAutomaticInternalRate, 4);

        if (oversamplingFactor_ > 1 && !linearWanted_)
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

        analytic_.assign(static_cast<size_t>(MaxBands * kMaxChannels), HilbertIIR<double> {});

        // The mixed engine: three fixed filters per band, and the rate their
        // designs are fitted at (the one the bands run at).
        mixedEngine_ = std::make_unique<MixedEngine>();
        for (auto& band : mixedEngine_->bands)
        {
            band.rest.setNumSections(1);
            band.cut.setNumSections(1);
            band.lift.setNumSections(1);
        }
        designBand_.prepare(sampleRate_ * runFactor());

        // The linear-phase mode: a bank of kernels (one per band and branch),
        // the signal and the weights late by its latency, and the transform
        // the kernels are designed with. linear_ gates the mode: set last.
        linear_.reset();
        if (linearWanted_)
        {
            auto engine = std::make_unique<LinearEngine>();
            LinearEngine& e = *engine;
            e.kernel = resolveLinearTaps(spec.sampleRate);
            e.channels = std::min(spec.numChannels, kMaxChannels);
            e.bank.prepare(e.kernel, e.kernel / kLinearPartitions, e.channels + (e.channels >= 2 ? 2 : 0),
                           MaxBands * kLinearBranches);
            e.latency = e.kernel / 2 + e.bank.getLatency();
            e.designFft = std::make_unique<FFTReal<T>>(static_cast<size_t>(e.kernel));
            e.spectrum.assign(static_cast<size_t>(e.kernel + 2), T(0));
            e.time.assign(static_cast<size_t>(e.kernel), T(0));
            e.taps.assign(static_cast<size_t>(e.kernel), T(0));
            e.window.resize(static_cast<size_t>(e.kernel));
            for (int m = 0; m < e.kernel; ++m)
            {
                const double t = 2.0 * std::numbers::pi * m / e.kernel;
                e.window[static_cast<size_t>(m)] = static_cast<T>(
                    0.35875 - 0.48829 * std::cos(t) + 0.14128 * std::cos(2.0 * t) - 0.01168 * std::cos(3.0 * t));
            }
            for (int ch = 0; ch < kMaxChannels; ++ch) e.dry[ch].prepare(e.latency + 1);
            int ring = 1;
            while (ring < e.latency + kDetChunk + 1) ring <<= 1;
            e.weightMask = ring - 1;
            e.weights.assign(static_cast<size_t>(MaxBands * 2) * static_cast<size_t>(ring), T(0));
            linear_ = std::move(engine);
        }

        int maxLaSamples = static_cast<int>(sampleRate_ * runFactor() * 0.01) + 1;
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].prepare(maxLaSamples);

        updateLookahead();
        reset();
        for (int ch = 0; ch < std::min(spec.numChannels, kMaxChannels); ++ch)
            deltaRing_[static_cast<size_t>(ch)].prepare(kRemovalCapacity + std::max(1, spec.maxBlockSize));
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
        if (!removalListen_.load(std::memory_order_relaxed))
        {
            processImpl(audio, sidechain);
            return;
        }
        // Removal listen: the input, late by the latency, minus the output, in
        // chunks of the prepared block.
        const bool selfKeyed = sidechain.getNumChannels() <= 0 || sidechain.getNumSamples() < audio.getNumSamples()
                            || sidechain.getChannel(0) == audio.getChannel(0);
        const int nCh = std::min(audio.getNumChannels(), kMaxChannels);
        const int step = std::max(1, spec_.maxBlockSize);
        for (int off = 0; off < audio.getNumSamples(); off += step)
        {
            const int m = std::min(step, audio.getNumSamples() - off);
            auto part = audio.getSubView(off, m);
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* d = part.getChannel(ch);
                for (int i = 0; i < m; ++i) deltaRing_[static_cast<size_t>(ch)].push(d[i]);
            }
            const int late = std::min(getLatency(), kRemovalCapacity);
            processImpl(part, selfKeyed ? part : sidechain.getSubView(off, m));
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = part.getChannel(ch);
                const auto& ring = deltaRing_[static_cast<size_t>(ch)];
                for (int i = 0; i < m; ++i) d[i] = ring.read(late + (m - 1 - i)) - d[i];
            }
        }
    }

    /**
     * @brief Removal listen: the output is what the bands take out (or add),
     *        the input delayed by the latency minus the output. A monitoring
     *        control, not in the state.
     */
    void setRemovalListen(bool on) noexcept { removalListen_.store(on, std::memory_order_relaxed); }
    /** @return Whether the output is what the bands change. */
    [[nodiscard]] bool isRemovalListen() const noexcept { return removalListen_.load(std::memory_order_relaxed); }

private:
    void processImpl(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {

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
            // The oversamplers hold at most maxBlockSize base samples, so a
            // longer block runs in chunks (its tail used to pass dry).
            const int total = audio.getNumSamples();
            const int chunk = std::max(1, spec_.maxBlockSize);
            for (int off = 0; off < total; off += chunk)
            {
                const int m = std::min(chunk, total - off);
                auto audioChunk = audio.getSubView(off, m);
                auto upAudio = oversampler_->upsample(audioChunk);
                auto upSc    = oversamplerSc_->upsample(sidechain.getSubView(off, m));

                processCore(upAudio, upSc, sampleRate_ * oversamplingFactor_);

                oversampler_->downsample(audioChunk);
            }
        }
        else
        {
            // Standard processing path
            processCore(audio, sidechain, sampleRate_);
        }
    }

public:
    /**
     * @brief Thread-safe configuration update for a specific band.
     *
     * Fields are sanitized on the way in: non-finite values keep the band's
     * previously published value (a NaN frequency/Q/time used to poison the
     * detector or the gain ballistics permanently), wild shape enums are
     * clamped (a scope or a timing that is not one keeps the band's own),
     * the gain at rest is held within 24 dB, the knee within 40 dB (any
     * negative knee is the automatic one), and the range fields are floored
     * at 0. Configuring a band at or
     * beyond the current band count raises the count to band + 1 (it used to
     * leave the band silent until setNumBands()).
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
        c.gain           = std::clamp(keep(c.gain, prev.gain), T(-24), T(24));
        c.knee           = std::min(keep(c.knee, prev.knee), T(40));
        if (c.knee < T(0)) c.knee = kAutomaticKnee;
        if (static_cast<int>(c.scope) < 0 || static_cast<int>(c.scope) > static_cast<int>(Scope::Side))
            c.scope = prev.scope;
        if (static_cast<int>(c.timing) < 0 || static_cast<int>(c.timing) > static_cast<int>(Timing::Program))
            c.timing = prev.timing;
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
        // Single control writer: a plain load/store pair cannot lose a write.
        if (numBands_.load(std::memory_order_relaxed) < band + 1)
            numBands_.store(band + 1, std::memory_order_relaxed);
    }

    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 1, MaxBands), std::memory_order_relaxed);
    }

    /**
     * @brief Chooses the engine of the bands: fixed filters mixed by the
     *        gain, or one filter designed again as the gain moves (the
     *        default).
     *
     * A mixed band is the signal and one fixed filter per side (the end of
     * the range), none of which ever changes while the gain moves; the gain
     * sets how much of the filter is heard, sample by sample (see the file
     * notes). Safe from any thread; the audio thread changes engine at
     * the next block and every band starts clean there, which may click:
     * choose it before playback.
     *
     * @param on True for the mixed engine.
     */
    void setMixedBands(bool on) noexcept { mixedWanted_.store(on, std::memory_order_relaxed); }

    /** @brief Whether the bands are mixed from fixed filters (setMixedBands()). */
    [[nodiscard]] bool isMixedBands() const noexcept { return mixedWanted_.load(std::memory_order_relaxed); }

    /** @brief What a stereo band's linked level reads (see setLinkMode()). */
    enum class LinkMode
    {
        Loudest,  ///< The louder channel's level (the default).
        Dominant  ///< The level of the pair's dominant component in the band.
    };

    /**
     * @brief Chooses how the bands placed on the stereo pair link it.
     *
     * Dominant reads, in each band, the level of the pair's dominant
     * component: the two channels' band levels and their correlation over
     * 20 ms make a covariance whose larger eigenvalue is read (see
     * dominantPairLevel()). A source reads the same wherever an equal-power
     * pan puts it, and uncorrelated channels read the louder 3 dB down. It
     * applies to Scope::Stereo bands on two channels. Safe from any thread.
     *
     * @param mode Loudest (default) or Dominant.
     */
    void setLinkMode(LinkMode mode) noexcept
    {
        linkMode_.store(static_cast<LinkMode>(std::clamp(static_cast<int>(mode), 0, 1)), std::memory_order_relaxed);
    }

    /** @return How the stereo bands link the pair. */
    [[nodiscard]] LinkMode getLinkMode() const noexcept { return linkMode_.load(std::memory_order_relaxed); }

    /** @brief How the band filter is realised (see setStructure()). */
    enum class Structure
    {
        DirectForm,  ///< Direct form (the default).
        Ladder       ///< Normalized ladder: passive while the gain moves the filter.
    };

    /**
     * @brief Chooses how the filter of each band is realised in the default engine.
     *
     * The default engine designs each band again every 16 samples as its
     * gain moves. Ladder runs that filter as a normalized ladder
     * (NormalizedLadder): the same response, and a memory that cannot gain
     * energy at the jumps however fast the gain swings, for about twice
     * the arithmetic of the filter. The mixed engine (setMixedBands()) and
     * the linear phase keep their own filters. Safe from any thread; the
     * switch restarts the bands' memory at the next block.
     *
     * @param structure DirectForm (default) or Ladder.
     */
    void setStructure(Structure structure) noexcept
    {
        structure_.store(static_cast<Structure>(std::clamp(static_cast<int>(structure), 0, 1)), std::memory_order_relaxed);
    }

    /** @return How the filter of each band is realised. */
    [[nodiscard]] Structure getStructure() const noexcept { return structure_.load(std::memory_order_relaxed); }

    /**
     * @brief Linear phase: the bands change the level of the signal and nothing of its phase.
     *
     * Each band is then zero-phase filters (its rest, and the end of each
     * side, as in the mixed engine), run by one partitioned convolution, and
     * the gain sets how much of each is added to the signal, late by the
     * latency of the mode: half the kernel plus one partition (2560 samples
     * at 44.1 and 48 kHz with the default kernel; getLatency() reports it).
     * The bands add up side by side, each one heard from the input, so two
     * bands that overlap add their changes instead of multiplying them. The
     * detector runs ahead of the audio by that latency, and the gain is held
     * back to meet it (less the lookahead, which costs no latency here). The
     * kernel resolves bands down to about its own length: with the default
     * one a bell narrower than some 100 Hz comes out wider and shallower
     * (setLinearPhaseKernelLength()). The mode runs at the host rate:
     * oversampling is not used with it.
     *
     * @note Setup threads only: like setOversampling() this un-prepares the
     *       processor until the next prepare().
     *
     * @param on True for linear phase.
     */
    void setLinearPhase(bool on) noexcept
    {
        if (on == linearWanted_) return;
        linearWanted_ = on;
        isPrepared_.store(false, std::memory_order_relaxed);
    }

    /** @brief Whether the bands run in linear phase (setLinearPhase()). */
    [[nodiscard]] bool isLinearPhase() const noexcept { return linearWanted_; }

    /**
     * @brief Asks for the length of the linear-phase kernels, in taps.
     *
     * A power of two from 1024 to 65536 (other values go up to the next
     * one); 0, the default, is 4096 at 44.1 and 48 kHz and in proportion at
     * higher rates. Longer kernels resolve lower and narrower bands and add
     * latency (half the kernel plus an eighth). Setup threads only: it
     * un-prepares the processor when the mode is on.
     *
     * @param taps Kernel length, or 0 for the automatic one.
     */
    void setLinearPhaseKernelLength(int taps) noexcept
    {
        taps = std::clamp(taps, 0, 65536);
        if (taps == linearTaps_) return;
        linearTaps_ = taps;
        if (linearWanted_) isPrepared_.store(false, std::memory_order_relaxed);
    }

    /** @brief The kernel length in use (after prepare()), or the one asked for. */
    [[nodiscard]] int getLinearPhaseKernelLength() const noexcept { return linear_ ? linear_->kernel : linearTaps_; }

    /**
     * @brief The magnitude response of a band with its dynamic part at a
     *        given gain, as the engine in use plays it.
     *
     * For an interface: draw it at getBandGainDb() for the live curve, or at
     * the range of a side for where the band can go. Reads the published
     * band; setup or UI threads.
     *
     * @param band          Band index.
     * @param dynamicGainDb Gain of the dynamic part, in dB from the rest.
     * @param frequencies   Array of frequencies in Hz.
     * @param magnitudes    Output array (same size, linear scale).
     * @param numPoints     Number of frequency points.
     */
    void getBandMagnitude(int band, T dynamicGainDb, const T* frequencies, T* magnitudes, int numPoints) const noexcept
    {
        if (frequencies == nullptr || magnitudes == nullptr) return;
        for (int i = 0; i < numPoints; ++i) magnitudes[i] = T(1);
        if (band < 0 || band >= MaxBands || sampleRate_ <= 0.0 || !std::isfinite(dynamicGainDb)) return;

        const BandConfig cfg = staged_[static_cast<size_t>(band)].read();
        const double fs = sampleRate_ * runFactor();
        const double rest = static_cast<double>(cfg.gain), moved = static_cast<double>(dynamicGainDb);

        if (linearWanted_)
        {
            // Linear phase: the magnitudes of the prototypes, mixed as the mode mixes them.
            const double range = rangeTowards(cfg, moved > 0.0);
            const bool acts = range > kRestDb && moved != 0.0;
            const double side = moved > 0.0 ? range : -range;
            const double ratio = std::pow(10.0, moved / 20.0), endRatio = std::pow(10.0, side / 20.0);
            const double weight = acts ? std::clamp((ratio - 1.0) / (endRatio - 1.0), 0.0, 1.0) : 0.0;
            const double reach = 1.0 + std::pow(10.0, side / 40.0);
            const AnalogSection atRest = prototypeOf(cfg, rest), atHalf = prototypeOf(cfg, 0.5 * side);
            for (int i = 0; i < numPoints; ++i)
            {
                const double omega = 2.0 * std::numbers::pi * static_cast<double>(frequencies[i]);
                const double base = std::abs(rest) > kRestDb ? std::sqrt(atRest.magnitudeSquared(omega)) : 1.0;
                const double moving = acts ? 1.0 + weight * reach * (std::sqrt(atHalf.magnitudeSquared(omega)) - 1.0) : 1.0;
                magnitudes[i] = static_cast<T>(base * moving);
            }
            return;
        }

        if (!isMixedBands())
        {
            const double total = rest + moved;
            BiquadCoeffs c;
            if (std::abs(total) > 0.01)
                switch (cfg.shape)
                {
                    case BandShape::LowShelf:  c = BiquadCoeffs::makeLowShelf(fs, static_cast<double>(cfg.frequency), total); break;
                    case BandShape::HighShelf: c = BiquadCoeffs::makeHighShelf(fs, static_cast<double>(cfg.frequency), total); break;
                    default:                   c = BiquadCoeffs::makePeakMatched(fs, static_cast<double>(cfg.frequency),
                                                                                 static_cast<double>(cfg.q), total); break;
                }
            for (int i = 0; i < numPoints; ++i)
                magnitudes[i] = static_cast<T>(std::abs(responseOf(c, static_cast<double>(frequencies[i]), fs)));
            return;
        }

        // Mixed: the filter at rest, and after it the dry signal plus a share
        // of what the end of that side adds to it.
        const double range = rangeTowards(cfg, moved > 0.0);
        const bool acts = range > kRestDb && moved != 0.0;
        const double side = moved > 0.0 ? range : -range;
        const double ratio = std::pow(10.0, moved / 20.0), endRatio = std::pow(10.0, side / 20.0);
        const double weight = acts ? std::clamp((ratio - 1.0) / (endRatio - 1.0), 0.0, 1.0) : 0.0;
        const bool rests = std::abs(rest) > kRestDb;
        const BiquadCoeffs atRest = rests ? designShape(cfg, rest) : BiquadCoeffs {};
        const BiquadCoeffs atEnd = acts ? designEnd(cfg, side) : BiquadCoeffs {};
        for (int i = 0; i < numPoints; ++i)
        {
            const double hz = static_cast<double>(frequencies[i]);
            const std::complex<double> moving = 1.0 + weight * (responseOf(atEnd, hz, fs) - 1.0);
            magnitudes[i] = static_cast<T>(std::abs(responseOf(atRest, hz, fs) * moving));
        }
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
     *   delay (64 samples); 4x roughly quadruples it and adds 70 (the second
     *   stage runs a much shorter half-band). The exact
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
        oversamplingAutomatic_ = false;
        isPrepared_.store(false, std::memory_order_relaxed); // Forces user to call prepare()
    }

    /**
     * @brief Lets the host rate choose the factor (setup threads only: like
     *  setOversampling() it un-prepares the processor until the next prepare()).
     *
     * With it on, every prepare() takes the smallest factor that brings
     * the gain stage to an internal rate of at least 88.2 kHz: 2x at 44.1 and 48 kHz and
     * none from 88.2 kHz up. The
     * nonlinearity then runs at about the same rate whatever the session
     * rate, and a high-rate session stops paying for a factor it does not
     * need. Off by default; an explicit setOversampling() turns it off.
     *
     * @param on True to choose the factor from the rate.
     */
    void setAutomaticOversampling(bool on) noexcept
    {
        if (on == oversamplingAutomatic_) return;
        oversamplingAutomatic_ = on;
        isPrepared_.store(false, std::memory_order_relaxed);
    }

    /** @brief Whether the factor follows the host rate (setAutomaticOversampling()). */
    [[nodiscard]] bool isOversamplingAutomatic() const noexcept { return oversamplingAutomatic_; }

    /** @brief Active oversampling factor (1 = off), as the last prepare() or setter left it. */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return oversamplingFactor_; }

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
        if (linear_) return linear_->latency;      // the lookahead comes out of it
        const int factor = runFactor();
        const int la = lookaheadSamples_.load(std::memory_order_relaxed) / factor;
        const int os = oversampler_ ? oversampler_->getLatency() : 0;
        return la + os;
    }

    /** @brief The gain the dynamic part of a band is applying, in dB from its rest (metering; any thread). */
    [[nodiscard]] T getBandGainDb(int band) const noexcept
    {
        if (band < 0 || band >= MaxBands) return T(0);
        return meterGainDb_[band].load(std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        // The analytic detectors and the oversamplers hold signal history
        // too: leaving it made reset() replay the old programme (a silent
        // input produced a 0.84 peak at 2x and a -13 dB gain reading).
        for (auto& c : bandCov_) c = {};
        for (int b = 0; b < MaxBands; ++b)
        {
            cleanBand(b);
            meterGainDb_[b].store(T(0), std::memory_order_relaxed);
            paramsDirty_[b].store(true, std::memory_order_relaxed);
        }
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].reset();
        if (linear_)
        {
            linear_->bank.reset();
            for (auto& line : linear_->dry) line.reset();
            std::fill(linear_->weights.begin(), linear_->weights.end(), T(0));
            linear_->weightPos = 0;
            linear_->played.fill(false);
            linear_->snap = true;
        }
        if (oversampler_) oversampler_->reset();
        if (oversamplerSc_) oversamplerSc_->reset();
        refreshCounter_ = 0;
        lastNumBands_ = MaxBands;
    }


    /** @brief Serializes bands and modes (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DYEQ"), 1);
        const int n = numBands_.load(std::memory_order_relaxed);
        w.write("numBands", n);
        w.write("lookahead", static_cast<float>(lookaheadMs_));
        w.write("mixed", isMixedBands());
        w.write("linkMode", static_cast<int32_t>(getLinkMode()));
        w.write("structure", static_cast<int32_t>(getStructure()));
        w.write("linear", linearWanted_);
        w.write("linearTaps", linearTaps_);
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
            std::snprintf(key, sizeof(key), "b%d.gain", i);
            w.write(key, static_cast<float>(c.gain));
            std::snprintf(key, sizeof(key), "b%d.knee", i);
            w.write(key, static_cast<float>(c.knee));
            std::snprintf(key, sizeof(key), "b%d.scope", i);
            w.write(key, static_cast<int>(c.scope));
            std::snprintf(key, sizeof(key), "b%d.timing", i);
            w.write(key, static_cast<int>(c.timing));
            std::snprintf(key, sizeof(key), "b%d.relative", i);
            w.write(key, c.relativeThreshold);
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
            std::snprintf(key, sizeof(key), "b%d.gain", i);
            c.gain = static_cast<T>(r.read(key, 0.0f));
            std::snprintf(key, sizeof(key), "b%d.knee", i);
            c.knee = static_cast<T>(r.read(key, 0.0f));
            std::snprintf(key, sizeof(key), "b%d.scope", i);
            c.scope = static_cast<Scope>(std::clamp(r.read(key, 0), 0, static_cast<int>(Scope::Side)));
            std::snprintf(key, sizeof(key), "b%d.timing", i);
            c.timing = static_cast<Timing>(std::clamp(r.read(key, 0), 0, static_cast<int>(Timing::Program)));
            std::snprintf(key, sizeof(key), "b%d.relative", i);
            c.relativeThreshold = r.read(key, false);
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
        // Blobs written before the key existed were rendered without lookahead.
        setLookahead(static_cast<T>(r.read("lookahead", 0.0f)));
        setMixedBands(r.read("mixed", false));
        setLinkMode(static_cast<LinkMode>(r.read("linkMode", 0)));
        setStructure(static_cast<Structure>(r.read("structure", 0)));
        setLinearPhaseKernelLength(r.read("linearTaps", 0));
        setLinearPhase(r.read("linear", false));
        return true;
    }

private:
    static constexpr int kMaxChannels = 16;
    static constexpr T kMinLevelDb = T(-100.0);
    static constexpr T kMinEnvelope = T(1e-12); // Prevents NaN in log10
    static constexpr int kDetChunk = 64;

    // Program timing. The followers count their times in units of the time
    // constant of the detector's own envelope (Q / (pi f) for a band-pass),
    // each with a floor: a band cannot know its level sooner than its filter
    // lets it, and a narrow or low band is slower for that reason alone.
    static constexpr double kDetectorNarrowing = 1.5;   ///< The detector's Q over the band's.
    static constexpr double kFastRiseFloor = 0.0003;    ///< Seconds; the fast follower rises in one unit.
    static constexpr double kFastFallUnits = 12.0;
    static constexpr double kFastFallFloor = 0.020;
    static constexpr double kSlowRiseUnits = 40.0;
    static constexpr double kSlowRiseFloor = 0.080;
    static constexpr double kSlowFallUnits = 150.0;
    static constexpr double kSlowFallFloor = 0.400;
    static constexpr double kLastFallShare = 0.25;      ///< Fall of the last follower, as a share of the fast one's.
    static constexpr double kDbPerNeper = 8.685889638065037;

    static constexpr double kReferenceFallDb = 2.0;     ///< dB per second the reference level falls.
    static constexpr double kReferenceWatch = 0.5;      ///< Seconds a second reference listens to a new zone.

    static constexpr double kKneePerOctave = 3.0;       ///< Automatic knee: dB per (1 + octaves of width), at a range of 12 dB.
    static constexpr double kMixedRampMs = 20.0;        ///< Glide of a mixed band's filters to a new design.
    static constexpr double kRestDb = 0.01;             ///< A gain or a range closer to 0 dB than this is no filter.

    struct BandState
    {
        BandConfig cfg;
        T aboveAtkCoeff = T(0), aboveRelCoeff = T(0);
        T belowAtkCoeff = T(0), belowRelCoeff = T(0);
        T knee = T(0);                      ///< Knee in use (the automatic one resolved).

        // Program timing: the three followers, in amplitude.
        double fastUp = 1.0, fastDown = 1.0, slowUp = 1.0, slowDown = 1.0, lastDown = 1.0;
        double fast = 0.0, slow = 0.0, heard = 0.0;

        // Relative threshold: the reference level and the one that listens to a new zone.
        double reference = -100.0, candidate = -100.0;
        double referenceFall = 0.0;         ///< dB per sample.
        int watching = 0;                   ///< Samples the candidate still listens.

        // Mixed engine.
        bool rests = false;                 ///< The band has a filter at rest (its gain is not 0 dB).
        bool cuts = false, lifts = false;   ///< It has an end below its rest, and one above.
        double cutScale = 0.0, liftScale = 0.0;   ///< 1 / (1 - end) and 1 / (end - 1), the ends as amplitude ratios to the rest.
        bool designed = false;              ///< Its filters carry a design: the next one glides.
    };

    /** @brief The fixed filters of a mixed band: its rest (in series), and the end of the range on each side. */
    struct MixedBand
    {
        BiquadCascade<T, 1, kMaxChannels> rest, cut, lift;
    };

    /** @brief Everything the mixed engine needs, allocated in prepare(). */
    struct MixedEngine
    {
        std::array<MixedBand, MaxBands> bands;
        std::array<std::array<T, kDetChunk>, kMaxChannels> cut {}, lift {};   ///< What the ends give for the chunk.
        std::array<std::array<T, kDetChunk>, kMaxChannels> held {};           ///< Non-finite samples kept out of the filters.
        std::array<double, kDetChunk> cutWeight {}, liftWeight {};
    };

    [[nodiscard]] static bool ofThePair(const BandConfig& cfg) noexcept
    {
        return cfg.scope == Scope::Mid || cfg.scope == Scope::Side;
    }

    /** @brief The factor the bands run at: the linear-phase mode runs at the host rate. */
    [[nodiscard]] int runFactor() const noexcept { return linearWanted_ ? 1 : std::max(oversamplingFactor_, 1); }

    /** @brief Whether a band has something to act on: one channel has no right and no sides. */
    [[nodiscard]] static bool plays(const BandConfig& cfg, int numChannels) noexcept
    {
        if (!cfg.enabled) return false;
        return numChannels >= 2 || (cfg.scope != Scope::Right && cfg.scope != Scope::Side);
    }

    void processCore(AudioBufferView<T>& audio, AudioBufferView<T>& sidechain, double currentFs) noexcept
    {
        const int nCh  = std::min(audio.getNumChannels(), kMaxChannels);
        const int scCh = sidechain.getNumChannels();
        const int nS   = audio.getNumSamples();
        if (scCh <= 0) return; // a 0-channel sidechain would index getChannel(-1)
        const int nb   = numBands_.load(std::memory_order_relaxed);
        const int laSamples = lookaheadSamples_.load(std::memory_order_relaxed);
        dominantLink_ = linkMode_.load(std::memory_order_relaxed) == LinkMode::Dominant && nCh == 2;
        if (dominantLink_)
        {
            if (currentFs != linkRate_)
            {
                linkRate_ = currentFs;
                linkCoeff_ = 1.0 - std::exp(-1.0 / (0.02 * currentFs));
            }
            for (auto& c : bandCov_)
                for (double& v : c) if (std::abs(v) < 1e-30) v = 0.0;   // no subnormals in silence
        }

        // A change of structure: the bands' filters restart in the new one.
        const bool ladder = structure_.load(std::memory_order_relaxed) == Structure::Ladder;
        if (ladder != ladderActive_)
        {
            ladderActive_ = ladder;
            for (int b = 0; b < MaxBands; ++b)
            {
                bandFilter_[b].reset();
                bandLadder_[static_cast<size_t>(b)].reset();
                bandLadder_[static_cast<size_t>(b)].setCoeffs(bandFilter_[b].getCoeffs());
            }
        }

        // A change of engine: every band starts clean in the new one.
        const bool mixed = mixedWanted_.load(std::memory_order_relaxed) && mixedEngine_ != nullptr && !linear_;
        if (mixed != mixedActive_)
        {
            mixedActive_ = mixed;
            for (int b = 0; b < MaxBands; ++b) restartBand(b);
        }

        // Bands re-activated by a band-count increase start clean: they would
        // otherwise replay the gain and filter state from when they were last
        // active (a boosted band re-added on a 0.001 tone peaked at 5.4).
        if (nb > lastNumBands_)
            for (int b = lastNumBands_; b < nb; ++b) restartBand(b);
        lastNumBands_ = nb;

        // 1. Thread-Safe State Update
        for (int b = 0; b < nb; ++b)
        {
            if (paramsDirty_[b].exchange(false, std::memory_order_acquire))
                updateBandInternalState(b, currentFs);
        }

        // The bands that play, in the order they are applied: the stereo, left
        // and right ones first, then the ones of the middle and the sides.
        numFirst_ = numSecond_ = 0;
        for (int b = 0; b < nb; ++b)
        {
            playing_[static_cast<size_t>(b)] = plays(states_[b].cfg, nCh);
            if (!playing_[static_cast<size_t>(b)]) continue;
            if (ofThePair(states_[b].cfg)) second_[static_cast<size_t>(numSecond_++)] = b;
            else first_[static_cast<size_t>(numFirst_++)] = b;
        }
        if (linear_) serviceLinear(nb);

        // 2. Chunks: the detection and the gain of every band run first, band
        //    by band and channel by channel, so each detector (and its allpass
        //    pair) keeps its state in registers; the filters then run over
        //    the chunk. Feed-forward, so the order changes nothing.
        for (int chunkStart = 0; chunkStart < nS; chunkStart += kDetChunk)
        {
            const int chunkLen = std::min(kDetChunk, nS - chunkStart);

            for (int b = 0; b < nb; ++b)
            {
                if (!playing_[static_cast<size_t>(b)]) continue;
                detectBand(b, sidechain, chunkStart, chunkLen, nCh, scCh);
                computeGains(b, chunkLen);
            }

            if (linear_) applyLinear(audio, chunkStart, chunkLen, nCh, laSamples);
            else if (mixedActive_) applyMixed(audio, chunkStart, chunkLen, nCh, laSamples);
            else applyRedesigned(audio, chunkStart, chunkLen, nCh, nb, laSamples, currentFs);
        }
    }

    /** The band filter takes a design (and its ladder, when it runs). */
    void setBandCoeffs(int b, const BiquadCoeffs& c) noexcept
    {
        bandFilter_[b].setCoeffsNow(c);
        if (ladderActive_) bandLadder_[static_cast<size_t>(b)].setCoeffs(c);
    }

    /** One sample of one channel through a band's filter, in the structure in use. */
    T runBand(int b, T x, int ch) noexcept
    {
        if (ladderActive_)
            return static_cast<T>(bandLadder_[static_cast<size_t>(b)].processSample(static_cast<double>(x), ch));
        return bandFilter_[b].processSample(x, ch);
    }

    /**
     * @brief The level a band hears over one chunk: every channel (the
     *        loudest), one channel, or the middle or the sides of the first
     *        pair of the sidechain.
     *
     * Leaves decibels in detLevelDb_ (manual timing) or amplitudes in
     * detLinear_ (program timing).
     */
    void detectBand(int b, AudioBufferView<T>& sidechain, int start, int n, int nCh, int scCh) noexcept
    {
        const BandConfig& cfg = states_[b].cfg;
        const bool program = cfg.timing == Timing::Program;
        T* levels = detLevelDb_[static_cast<size_t>(b)].data();
        std::fill(levels, levels + n, kMinLevelDb);
        if (program) std::fill(detLinear_.begin(), detLinear_.begin() + n, 0.0);

        const Scope where = cfg.scope;
        const int listeners = where == Scope::Stereo ? nCh : 1;
        const bool dominant = dominantLink_ && where == Scope::Stereo;
        for (int k = 0; k < listeners; ++k)
        {
            int slot = k;
            const T* first = sidechain.getChannel(std::min(k, scCh - 1)) + start;
            const T* second = nullptr;
            T sign = T(1);
            if (where == Scope::Right)
            {
                slot = 1;
                first = sidechain.getChannel(std::min(1, scCh - 1)) + start;
            }
            else if (where == Scope::Mid || where == Scope::Side)
            {
                if (scCh >= 2) second = sidechain.getChannel(1) + start;
                else if (where == Scope::Side) break;   // one channel has no sides: silence
                sign = where == Scope::Mid ? T(1) : T(-1);
            }

            if (second == nullptr)
            {
                for (int j = 0; j < n; ++j)
                    detBand_[static_cast<size_t>(j)] = static_cast<double>(bandDetector_[b].processSample(
                        std::isfinite(first[j]) ? first[j] : T(0), slot)); // non-finite would latch
            }
            else
            {
                for (int j = 0; j < n; ++j)
                {
                    const T one = std::isfinite(first[j]) ? first[j] : T(0);
                    const T other = std::isfinite(second[j]) ? second[j] : T(0);
                    detBand_[static_cast<size_t>(j)] = static_cast<double>(
                        bandDetector_[b].processSample((one + sign * other) * T(0.5), slot));
                }
            }
            analytic_[static_cast<size_t>(b * kMaxChannels + slot)].magnitudeBlock(
                detBand_.data(), detMag_.data(), n);

            // The dominant link reads the pair once the second channel is in.
            if (dominant)
            {
                if (k == 0)
                {
                    std::copy_n(detBand_.begin(), n, pairBand_.begin());
                    std::copy_n(detMag_.begin(), n, pairMag_.begin());
                    continue;
                }
                auto& cov = bandCov_[static_cast<size_t>(b)];
                for (int j = 0; j < n; ++j)
                {
                    const size_t s = static_cast<size_t>(j);
                    const double l = pairBand_[s], r = detBand_[s];
                    cov[0] += linkCoeff_ * (l * l - cov[0]);
                    cov[1] += linkCoeff_ * (r * r - cov[1]);
                    cov[2] += linkCoeff_ * (l * r - cov[2]);
                    const double den = cov[0] * cov[1];
                    const double rho2 = den > 1e-60 ? std::min(1.0, cov[2] * cov[2] / den) : 0.0;
                    const double amp = dominantPairLevel(std::max(std::abs(l), pairMag_[s]),
                                                         std::max(std::abs(r), detMag_[s]), rho2);
                    if (program) detLinear_[s] = amp;
                    else levels[j] = gainToDecibels(std::max(static_cast<T>(amp), kMinEnvelope));
                }
                continue;
            }

            // max(|x|, analytic magnitude): a steady tone reads as its flat
            // amplitude, not a rectified wave (see @file).
            if (program)
            {
                for (int j = 0; j < n; ++j)
                    detLinear_[static_cast<size_t>(j)] = std::max(detLinear_[static_cast<size_t>(j)],
                        std::max(std::abs(detBand_[static_cast<size_t>(j)]), detMag_[static_cast<size_t>(j)]));
            }
            else
            {
                for (int j = 0; j < n; ++j)
                {
                    const T detected = std::max(static_cast<T>(std::abs(detBand_[static_cast<size_t>(j)])),
                                                static_cast<T>(detMag_[static_cast<size_t>(j)]));
                    const T levelDb = gainToDecibels(std::max(detected, kMinEnvelope));
                    if (levelDb > levels[j]) levels[j] = levelDb;
                }
            }
        }
    }

    /**
     * @brief The dynamic gain of a band over one chunk, in decibels from its
     *        rest, into dynDb_.
     */
    void computeGains(int b, int n) noexcept
    {
        BandState& st = states_[b];
        const BandConfig& cfg = st.cfg;
        const bool program = cfg.timing == Timing::Program;
        const T* levels = detLevelDb_[static_cast<size_t>(b)].data();
        T* gains = dynDb_[static_cast<size_t>(b)].data();
        T& currentGain = currentGainDb_[b];

        for (int j = 0; j < n; ++j)
        {
            T levelDb;
            if (program)
            {
                // Fast; slow, which follows the fast; and a last one over the
                // larger of the two, which takes the corner out of the hand-over.
                const double x = detLinear_[static_cast<size_t>(j)];
                st.fast += (x > st.fast ? st.fastUp : st.fastDown) * (x - st.fast);
                st.slow += (st.fast > st.slow ? st.slowUp : st.slowDown) * (st.fast - st.slow);
                const double larger = std::max(st.fast, st.slow);
                st.heard = larger > st.heard ? larger : st.heard + st.lastDown * (larger - st.heard);
                levelDb = std::max(static_cast<T>(fastLog(std::max(st.heard, 1e-12)) * kDbPerNeper), kMinLevelDb);
            }
            else
            {
                levelDb = levels[j];
            }

            T threshold = cfg.threshold;
            if (cfg.relativeThreshold)
            {
                const double level = static_cast<double>(levelDb);
                st.reference = std::max(level, st.reference - st.referenceFall);
                if (st.watching > 0)
                {
                    st.candidate = std::max(level, st.candidate - st.referenceFall);
                    if (--st.watching == 0) st.reference = st.candidate;
                }
                threshold = static_cast<T>(st.reference + static_cast<double>(cfg.threshold));
            }

            // Gain Computer
            const T targetGainDb = computeTargetGain(cfg, levelDb, threshold, st.knee);

            if (program)
            {
                currentGain = targetGainDb;     // the followers have set the pace
            }
            else
            {
                // Gain Ballistics (Attack/Release applied to the Gain itself)
                const T diff = targetGainDb - currentGain;
                T coeff;
                if (levelDb > threshold) {
                    coeff = (std::abs(targetGainDb) > std::abs(currentGain))
                            ? st.aboveAtkCoeff : st.aboveRelCoeff;
                } else {
                    coeff = (std::abs(targetGainDb) > std::abs(currentGain))
                            ? st.belowAtkCoeff : st.belowRelCoeff;
                }
                currentGain += coeff * diff;
            }
            gains[j] = currentGain;
        }

        // A follower left to fall for long reaches numbers too small to carry.
        if (st.fast < 1e-30) st.fast = 0.0;
        if (st.slow < 1e-30) st.slow = 0.0;
        if (st.heard < 1e-30) st.heard = 0.0;
    }

    /**
     * @brief The default engine over one chunk: one filter per band, designed
     *        again at the gain of the moment every 16 samples.
     */
    void applyRedesigned(AudioBufferView<T>& audio, int start, int n, int nCh, int nb,
                         int laSamples, double currentFs) noexcept
    {
        for (int j = 0; j < n; ++j)
        {
            const int i = start + j;
            for (int b = 0; b < nb; ++b)
            {
                if (!playing_[static_cast<size_t>(b)]) continue;
                const T currentGain = dynDb_[static_cast<size_t>(b)][static_cast<size_t>(j)];

                // Refresh gain-filter coefficients every 16 samples - the gain
                // envelope is slow enough that this granularity is inaudible.
                // The grid is a free-running counter, not the block index, so
                // the output does not depend on how the host cuts the stream.
                // Bells run the matched design; at 1/16th rate it stays
                // negligible (+1.6% CPU over the cookbook form).
                // Stream-owner direct writes (setCoeffsNow): these values are
                // computed HERE, on the audio thread, for this thread's own
                // use, so they never touch the staged cross-thread channel.
                if ((refreshCounter_ & 15u) == 0u)
                {
                    const T total = states_[b].cfg.gain + currentGain;
                    if (std::abs(total) > T(0.01))
                    {
                        switch (states_[b].cfg.shape)
                        {
                        case BandShape::Bell:
                            setBandCoeffs(b, BiquadCoeffs::makePeakMatched(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(states_[b].cfg.q), static_cast<double>(total)));
                            break;
                        case BandShape::LowShelf:
                            setBandCoeffs(b, BiquadCoeffs::makeLowShelf(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(total)));
                            break;
                        case BandShape::HighShelf:
                            setBandCoeffs(b, BiquadCoeffs::makeHighShelf(
                                currentFs, static_cast<double>(states_[b].cfg.frequency),
                                static_cast<double>(total)));
                            break;
                        }
                    }
                    else
                        setBandCoeffs(b, BiquadCoeffs{}); // Bypass
                }

                if ((refreshCounter_ & 63u) == 0u) // Sub-sample metering update
                    meterGainDb_[b].store(currentGain, std::memory_order_relaxed);
            }
            ++refreshCounter_;

            // The frame, delayed by the lookahead. A non-finite sample passes
            // without entering the filter state, where it would latch for good.
            T frame[kMaxChannels] = {};
            bool finite[kMaxChannels] = {};
            for (int ch = 0; ch < nCh; ++ch)
            {
                T audioSample = audio.getChannel(ch)[i];
                if (laSamples > 0) {
                    lookaheadBuf_[ch].push(audioSample);
                    audioSample = lookaheadBuf_[ch].read(laSamples);
                }
                frame[ch] = audioSample;
                finite[ch] = std::isfinite(audioSample);
            }

            for (int k = 0; k < numFirst_; ++k)
            {
                const int b = first_[static_cast<size_t>(k)];
                switch (states_[b].cfg.scope)
                {
                case Scope::Left:
                    if (finite[0]) frame[0] = runBand(b, frame[0], 0);
                    break;
                case Scope::Right:
                    if (finite[1]) frame[1] = runBand(b, frame[1], 1);
                    break;
                default:
                    for (int ch = 0; ch < nCh; ++ch)
                        if (finite[ch]) frame[ch] = runBand(b, frame[ch], ch);
                    break;
                }
            }

            if (numSecond_ > 0 && finite[0] && (nCh < 2 || finite[1]))
            {
                // The middle and the sides of the pair (one channel is its own middle).
                T middle = frame[0], sides = T(0);
                if (nCh >= 2)
                {
                    middle = (frame[0] + frame[1]) * T(0.5);
                    sides = (frame[0] - frame[1]) * T(0.5);
                }
                for (int k = 0; k < numSecond_; ++k)
                {
                    const int b = second_[static_cast<size_t>(k)];
                    if (states_[b].cfg.scope == Scope::Mid)
                        middle = runBand(b, middle, 0);
                    else
                        sides = runBand(b, sides, 0);
                }
                if (nCh >= 2)
                {
                    frame[0] = middle + sides;
                    frame[1] = middle - sides;
                }
                else
                {
                    frame[0] = middle;
                }
            }

            for (int ch = 0; ch < nCh; ++ch)
                audio.getChannel(ch)[i] = frame[ch];
        }
    }

    /**
     * @brief The mixed engine over one chunk: the bands in series, each one
     *        its fixed filters mixed by its gain.
     */
    void applyMixed(AudioBufferView<T>& audio, int start, int n, int nCh, int laSamples) noexcept
    {
        MixedEngine& engine = *mixedEngine_;
        T* channels[kMaxChannels] = {};
        bool bad[kMaxChannels] = {};     // channels with a non-finite sample in this chunk
        bool anyBad = false;

        // The chunk, delayed by the lookahead. A non-finite sample stays out
        // of the filters and goes back to its place at the end.
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = audio.getChannel(ch) + start;
            channels[ch] = data;
            if (laSamples > 0)
                for (int j = 0; j < n; ++j)
                {
                    lookaheadBuf_[ch].push(data[j]);
                    data[j] = lookaheadBuf_[ch].read(laSamples);
                }
            for (int j = 0; j < n; ++j)
                if (!std::isfinite(data[j]))
                {
                    auto& held = engine.held[static_cast<size_t>(ch)];
                    if (!bad[ch]) held.fill(T(0));
                    held[static_cast<size_t>(j)] = data[j];
                    data[j] = T(0);
                    bad[ch] = anyBad = true;
                }
        }

        for (int k = 0; k < numFirst_; ++k)
        {
            const int b = first_[static_cast<size_t>(k)];
            const Scope where = states_[b].cfg.scope;
            if (where == Scope::Stereo)
            {
                mixBand(b, AudioBufferView<T>(channels, nCh, n), n);
            }
            else
            {
                T* single[1] = { channels[where == Scope::Left ? 0 : 1] };
                mixBand(b, AudioBufferView<T>(single, 1, n), n);
            }
        }

        if (numSecond_ > 0)
        {
            AudioBufferView<T> pair(channels, nCh, n);
            if (nCh >= 2) MidSide<T>::encode(pair);
            for (int k = 0; k < numSecond_; ++k)
            {
                const int b = second_[static_cast<size_t>(k)];
                T* single[1] = { channels[states_[b].cfg.scope == Scope::Mid ? 0 : 1] };
                mixBand(b, AudioBufferView<T>(single, 1, n), n);
            }
            if (nCh >= 2) MidSide<T>::decode(pair);
        }

        if (anyBad)
            for (int ch = 0; ch < nCh; ++ch)
            {
                if (!bad[ch]) continue;
                const auto& held = engine.held[static_cast<size_t>(ch)];
                for (int j = 0; j < n; ++j)
                    if (!std::isfinite(held[static_cast<size_t>(j)])) channels[ch][j] = held[static_cast<size_t>(j)];
            }
    }

    /**
     * @brief One mixed band over one chunk of the channels it acts on.
     *
     * The weights come from the gain in amplitude: with the band at d dB
     * from its rest and the end of that side at e dB, w = (1 - 10^(d/20)) /
     * (1 - 10^(e/20)), which puts the centre of the band exactly d dB from
     * the rest.
     */
    void mixBand(int b, AudioBufferView<T> where, int n) noexcept
    {
        const BandState& st = states_[b];
        MixedEngine& engine = *mixedEngine_;
        MixedBand& filters = engine.bands[static_cast<size_t>(b)];
        const T* gains = dynDb_[static_cast<size_t>(b)].data();
        const int count = where.getNumChannels();

        bool down = false, up = false;
        for (int j = 0; j < n; ++j)
        {
            const double d = static_cast<double>(gains[j]);
            const double cut = cutWeight(st, d), lift = liftWeight(st, d);
            engine.cutWeight[static_cast<size_t>(j)] = cut;
            engine.liftWeight[static_cast<size_t>(j)] = lift;
            down = down || cut > 0.0;
            up = up || lift > 0.0;
        }
        meterGainDb_[b].store(gains[n - 1], std::memory_order_relaxed);

        // The filter at rest is a filter in series. The ends hear what comes
        // out of it, on copies: always, moving or not, so that their memory
        // is the signal's.
        if (st.rests) filters.rest.processBlock(where);
        T* copies[kMaxChannels] = {};
        if (st.cuts)
        {
            for (int ch = 0; ch < count; ++ch)
            {
                copies[ch] = engine.cut[static_cast<size_t>(ch)].data();
                std::copy_n(where.getChannel(ch), n, copies[ch]);
            }
            filters.cut.processBlock(AudioBufferView<T>(copies, count, n));
        }
        if (st.lifts)
        {
            for (int ch = 0; ch < count; ++ch)
            {
                copies[ch] = engine.lift[static_cast<size_t>(ch)].data();
                std::copy_n(where.getChannel(ch), n, copies[ch]);
            }
            filters.lift.processBlock(AudioBufferView<T>(copies, count, n));
        }

        for (int ch = 0; ch < count; ++ch)
        {
            T* y = where.getChannel(ch);
            if (down)
            {
                const T* end = engine.cut[static_cast<size_t>(ch)].data();
                for (int j = 0; j < n; ++j)
                    y[j] += static_cast<T>(engine.cutWeight[static_cast<size_t>(j)]) * (end[j] - y[j]);
            }
            if (up)
            {
                const T* end = engine.lift[static_cast<size_t>(ch)].data();
                for (int j = 0; j < n; ++j)
                    y[j] += static_cast<T>(engine.liftWeight[static_cast<size_t>(j)]) * (end[j] - y[j]);
            }
        }
    }

    // -- Linear phase ---------------------------------------------------------

    static constexpr int kLinearBranches = 3;        ///< Per band: its rest, the end of its cuts, the end of its lifts.
    static constexpr int kLinearDefaultTaps = 4096;  ///< Kernel at 44.1 and 48 kHz; it grows with the rate.
    static constexpr int kLinearPartitions = 8;      ///< Partitions per kernel: the bank adds one of latency.

    /** @brief Everything the linear-phase mode needs, allocated in prepare(). */
    struct LinearEngine
    {
        FirBank<T> bank;                                         ///< One kernel per band and branch.
        std::unique_ptr<FFTReal<T>> designFft;
        std::vector<T> spectrum, time, taps, window;
        std::array<RingBuffer<T>, kMaxChannels> dry;             ///< The signal, late.
        std::vector<T> weights;                                  ///< Per band and side: the weights, late.
        int weightMask = 0, weightPos = 0;
        std::array<std::array<bool, kLinearBranches>, MaxBands> pending {};   ///< Branches that wait for their design.
        std::array<bool, MaxBands> played {};                    ///< Bands that played in the last block.
        std::array<std::array<T, kDetChunk>, kMaxChannels + 2> lines {};      ///< What the bank hears of a run.
        int next = 0;                                            ///< Where the search for a design resumes.
        int kernel = 0, latency = 0, channels = 0;
        bool snap = true;                                        ///< Every design that waits lands on the next block, at once.
    };

    /** @brief The kernel length the mode runs: the one asked for, or 4096 taps at 44.1 and 48 kHz and in proportion above. */
    [[nodiscard]] int resolveLinearTaps(double sampleRate) const noexcept
    {
        int taps = linearTaps_;
        if (taps <= 0)
        {
            taps = kLinearDefaultTaps;
            while (taps < 65536 && sampleRate > 1.5 * 48000.0 * (taps / kLinearDefaultTaps)) taps <<= 1;
        }
        int size = 1024;
        while (size < taps && size < 65536) size <<= 1;
        return size;
    }

    /** @brief The input lines of the bank that hear a band: its channels, or the middle or the sides of the pair. */
    [[nodiscard]] unsigned linearMask(const BandConfig& cfg) const noexcept
    {
        const int channels = linear_->channels;
        switch (cfg.scope)
        {
            case Scope::Left:  return 1u;
            case Scope::Right: return channels >= 2 ? 2u : 0u;
            case Scope::Mid:   return channels >= 2 ? 1u << channels : 1u;
            case Scope::Side:  return channels >= 2 ? 1u << (channels + 1) : 0u;
            default:               return (1u << channels) - 1u;
        }
    }

    /**
     * @brief Designs the kernel of one branch of a band and hands it to the bank.
     *
     * Zero phase, from the magnitude of the analog prototypes: the rest of
     * the band minus one, or what the end of a side adds to the signal
     * (through the rest). The end is chosen as in the mixed engine: half way
     * through the range the band has exactly the magnitude of the nominal
     * one.
     */
    void designLinearBranch(int b, int branch) noexcept
    {
        LinearEngine& e = *linear_;
        const BandState& st = states_[b];
        const BandConfig& cfg = st.cfg;
        const int index = b * kLinearBranches + branch;
        const bool has = branch == 0 ? st.rests : (branch == 1 ? st.cuts : st.lifts);
        if (!has)
        {
            e.bank.removeKernel(index);
            return;
        }

        const AnalogSection rest = prototypeOf(cfg, static_cast<double>(cfg.gain));
        const double range = branch == 1 ? -rangeTowards(cfg, false) : rangeTowards(cfg, true);
        const AnalogSection half = prototypeOf(cfg, 0.5 * range);
        const double reach = 1.0 + std::pow(10.0, range / 40.0);
        const int bins = e.kernel / 2;
        const double step = 2.0 * std::numbers::pi * sampleRate_ / static_cast<double>(e.kernel);
        for (int k = 0; k <= bins; ++k)
        {
            const double omega = step * static_cast<double>(k);
            const double atRest = st.rests ? std::sqrt(rest.magnitudeSquared(omega)) : 1.0;
            const double level = branch == 0 ? atRest - 1.0
                                             : atRest * reach * (std::sqrt(half.magnitudeSquared(omega)) - 1.0);
            e.spectrum[static_cast<size_t>(2 * k)] = static_cast<T>(level);
            e.spectrum[static_cast<size_t>(2 * k + 1)] = T(0);
        }
        e.designFft->inverse(e.spectrum.data(), e.time.data());
        // Centred on half the kernel, and windowed.
        const int centre = e.kernel / 2;
        for (int m = 0; m < e.kernel; ++m)
            e.taps[static_cast<size_t>(m)] = e.time[static_cast<size_t>((m + centre) % e.kernel)] * e.window[static_cast<size_t>(m)];
        e.bank.setKernel(index, e.taps.data(), linearMask(cfg));
    }

    /**
     * @brief Housekeeping of the mode, once per block: bands that stop
     *        playing leave the bank, and one design that waits is done (all
     *        of them on the first block, where nothing has sounded yet).
     */
    void serviceLinear(int nb) noexcept
    {
        LinearEngine& e = *linear_;
        for (int b = 0; b < MaxBands; ++b)
        {
            const auto at = static_cast<size_t>(b);
            const bool on = b < nb && playing_[at];
            if (on && !e.played[at])
                e.pending[at].fill(true);
            if (!on && e.played[at])
                for (int branch = 0; branch < kLinearBranches; ++branch)
                    e.bank.removeKernel(b * kLinearBranches + branch);
            e.played[at] = on;
        }

        const int total = MaxBands * kLinearBranches;
        for (int step = 0; step < total; ++step)
        {
            const int index = (e.next + step) % total;
            const int b = index / kLinearBranches, branch = index % kLinearBranches;
            if (!e.pending[static_cast<size_t>(b)][static_cast<size_t>(branch)]) continue;
            e.pending[static_cast<size_t>(b)][static_cast<size_t>(branch)] = false;
            if (!e.played[static_cast<size_t>(b)]) continue;
            designLinearBranch(b, branch);
            if (e.snap) continue;
            e.next = (index + 1) % total;
            break;
        }
        if (e.snap)
        {
            e.bank.reset();     // nothing has sounded: the kernels land with no fade
            e.snap = false;
        }
    }

    /**
     * @brief The linear-phase mode over one chunk: the signal, late, plus
     *        what each branch of each band adds to it, weighted by its gain.
     *
     * The bands add up side by side, each one heard from the input, so the
     * mode has one latency whatever the number of bands. The weights are as
     * late as the audio minus the lookahead.
     */
    void applyLinear(AudioBufferView<T>& audio, int start, int n, int nCh, int laSamples) noexcept
    {
        LinearEngine& e = *linear_;
        const size_t ring = static_cast<size_t>(e.weightMask) + 1u;
        const int lineChannels = std::min(nCh, e.channels);
        const bool pair = e.channels >= 2;
        const int delay = e.latency - std::clamp(laSamples, 0, e.latency);

        // The weights of this chunk, into their delay.
        for (int k = 0; k < numFirst_ + numSecond_; ++k)
        {
            const int b = k < numFirst_ ? first_[static_cast<size_t>(k)] : second_[static_cast<size_t>(k - numFirst_)];
            const BandState& st = states_[b];
            const T* gains = dynDb_[static_cast<size_t>(b)].data();
            T* cuts = e.weights.data() + static_cast<size_t>(b * 2) * ring;
            T* lifts = cuts + ring;
            for (int j = 0; j < n; ++j)
            {
                const double d = static_cast<double>(gains[j]);
                const auto at = static_cast<size_t>((e.weightPos + j) & e.weightMask);
                cuts[at] = static_cast<T>(cutWeight(st, d));
                lifts[at] = static_cast<T>(liftWeight(st, d));
            }
            meterGainDb_[b].store(gains[n - 1], std::memory_order_relaxed);
        }

        int done = 0;
        while (done < n)
        {
            const int run = std::min(n - done, e.bank.room());

            // What the bank hears: the channels (a non-finite sample as
            // silence), and the middle and the sides of the pair.
            const T* heard[kMaxChannels + 2] = {};
            for (int ch = 0; ch < lineChannels; ++ch)
            {
                const T* x = audio.getChannel(ch) + start + done;
                T* line = e.lines[static_cast<size_t>(ch)].data();
                for (int j = 0; j < run; ++j) line[j] = std::isfinite(x[j]) ? x[j] : T(0);
                heard[ch] = line;
            }
            if (pair && lineChannels >= 1)
            {
                T* middle = e.lines[static_cast<size_t>(kMaxChannels)].data();
                T* sides = e.lines[static_cast<size_t>(kMaxChannels + 1)].data();
                const T* left = e.lines[0].data();
                if (lineChannels >= 2)
                {
                    const T* right = e.lines[1].data();
                    for (int j = 0; j < run; ++j)
                    {
                        middle[j] = (left[j] + right[j]) * T(0.5);
                        sides[j] = (left[j] - right[j]) * T(0.5);
                    }
                    heard[e.channels + 1] = sides;
                }
                else
                {
                    std::copy_n(left, run, middle);     // one channel is its own middle
                }
                heard[e.channels] = middle;
            }

            // The signal, late.
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* y = audio.getChannel(ch) + start + done;
                for (int j = 0; j < run; ++j)
                {
                    e.dry[ch].push(y[j]);
                    y[j] = e.dry[ch].read(e.latency);
                }
            }

            // What each branch of each band adds.
            for (int k = 0; k < numFirst_ + numSecond_; ++k)
            {
                const int b = k < numFirst_ ? first_[static_cast<size_t>(k)] : second_[static_cast<size_t>(k - numFirst_)];
                const Scope where = states_[b].cfg.scope;
                const T* sideWeights[kLinearBranches] = { nullptr, e.weights.data() + static_cast<size_t>(b * 2) * ring,
                                                          e.weights.data() + static_cast<size_t>(b * 2 + 1) * ring };
                for (int branch = 0; branch < kLinearBranches; ++branch)
                {
                    const int index = b * kLinearBranches + branch;
                    if (e.bank.isKernelSilent(index)) continue;
                    const T* weights = sideWeights[branch];
                    const auto add = [&](int line, int ch, T sign) noexcept {
                        if (ch >= nCh) return;
                        const T* wet = e.bank.tap(index, line);
                        T* y = audio.getChannel(ch) + start + done;
                        if (weights == nullptr)
                        {
                            for (int j = 0; j < run; ++j) y[j] += sign * wet[j];
                            return;
                        }
                        for (int j = 0; j < run; ++j)
                            y[j] += sign * weights[static_cast<size_t>((e.weightPos + done + j - delay) & e.weightMask)] * wet[j];
                    };
                    switch (where)
                    {
                        case Scope::Left:  add(0, 0, T(1)); break;
                        case Scope::Right: add(1, 1, T(1)); break;
                        case Scope::Mid:
                            add(pair ? e.channels : 0, 0, T(1));
                            if (pair) add(e.channels, 1, T(1));
                            break;
                        case Scope::Side:
                            if (pair) { add(e.channels + 1, 0, T(1)); add(e.channels + 1, 1, T(-1)); }
                            break;
                        default:
                            for (int ch = 0; ch < lineChannels; ++ch) add(ch, ch, T(1));
                            break;
                    }
                }
            }

            e.bank.push(heard, run);
            done += run;
        }
        e.weightPos = (e.weightPos + n) & e.weightMask;
    }

    /**
     * @brief The gain the level asks for, in decibels from the band's rest.
     *
     * With a knee of W dB the slope of each side comes in over W around the
     * threshold, as a parabola, instead of at a corner.
     */
    [[nodiscard]] static T computeTargetGain(const BandConfig& cfg, T levelDb, T threshold, T knee) noexcept
    {
        T gainDb = T(0);

        if (knee > T(0))
        {
            const T over = levelDb - threshold;
            const T half = knee * T(0.5);
            if (cfg.aboveRatio > T(1.001) && over > -half)
            {
                const T bent = over < half ? (over + half) * (over + half) / (T(2) * knee) : over;
                const T amount = std::min(bent * (T(1) - T(1) / cfg.aboveRatio), cfg.aboveRangeDb);
                gainDb += cfg.aboveBoost ? amount : -amount;
            }
            if (cfg.belowRatio > T(1.001) && over < half)
            {
                const T under = -over;
                const T bent = under < half ? (under + half) * (under + half) / (T(2) * knee) : under;
                const T amount = std::min(bent * (T(1) - T(1) / cfg.belowRatio), cfg.belowRangeDb);
                gainDb += cfg.belowBoost ? amount : -amount;
            }
            return gainDb;
        }

        if (levelDb > threshold)
        {
            if (cfg.aboveRatio > T(1.001)) {
                T overDb = levelDb - threshold;
                T amount = std::min(overDb * (T(1) - T(1) / cfg.aboveRatio), cfg.aboveRangeDb);
                gainDb += cfg.aboveBoost ? amount : -amount;
            }
        }
        else
        {
            if (cfg.belowRatio > T(1.001)) {
                T underDb = threshold - levelDb;
                T amount = std::min(underDb * (T(1) - T(1) / cfg.belowRatio), cfg.belowRangeDb);
                gainDb += cfg.belowBoost ? amount : -amount;
            }
        }
        return gainDb;
    }

    /** @brief Clears everything a band remembers of the signal (its configuration stays). */
    void cleanBand(int b) noexcept
    {
        bandDetector_[b].reset();
        bandFilter_[b].reset();
        bandLadder_[static_cast<size_t>(b)].reset();
        currentGainDb_[b] = T(0);
        BandState& st = states_[b];
        st.fast = st.slow = st.heard = 0.0;
        st.reference = st.candidate = static_cast<double>(kMinLevelDb);
        st.watching = 0;
        st.designed = false;
        if (!analytic_.empty())     // allocated in prepare()
            for (int ch = 0; ch < kMaxChannels; ++ch)
                analytic_[static_cast<size_t>(b * kMaxChannels + ch)].reset();
        if (mixedEngine_)
        {
            MixedBand& filters = mixedEngine_->bands[static_cast<size_t>(b)];
            filters.rest.reset();
            filters.cut.reset();
            filters.lift.reset();
        }
        if (linear_)
        {
            const size_t ring = static_cast<size_t>(linear_->weightMask) + 1u;
            std::fill_n(linear_->weights.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(b * 2) * ring),
                        2u * ring, T(0));
        }
    }

    /** @brief A band that comes into play: clean, and designed again at the next update. */
    void restartBand(int b) noexcept
    {
        cleanBand(b);
        meterGainDb_[b].store(T(0), std::memory_order_relaxed);
        paramsDirty_[b].store(true, std::memory_order_release);
    }

    /** @brief The complex response of a section at one frequency. */
    [[nodiscard]] static std::complex<double> responseOf(const BiquadCoeffs& c, double hz, double fs) noexcept
    {
        const double w = 2.0 * std::numbers::pi * std::clamp(hz, 0.0, 0.5 * fs) / fs;
        const std::complex<double> z1 = std::polar(1.0, -w), z2 = std::polar(1.0, -2.0 * w);
        return (c.b0 + c.b1 * z1 + c.b2 * z2) / (1.0 + c.a1 * z1 + c.a2 * z2);
    }

    /** @brief The larger range among the sides of a band that act in one direction. */
    [[nodiscard]] static double rangeTowards(const BandConfig& cfg, bool boost) noexcept
    {
        double range = 0.0;
        if (cfg.aboveRatio > T(1.001) && cfg.aboveBoost == boost)
            range = std::max(range, static_cast<double>(cfg.aboveRangeDb));
        if (cfg.belowRatio > T(1.001) && cfg.belowBoost == boost)
            range = std::max(range, static_cast<double>(cfg.belowRangeDb));
        return range;
    }

    /** @brief The analog prototype of a band's shape at one gain. */
    [[nodiscard]] static AnalogSection prototypeOf(const BandConfig& cfg, double gainDb) noexcept
    {
        const double hz = std::clamp(static_cast<double>(cfg.frequency), 1.0, 80000.0);
        const double q = std::max(static_cast<double>(cfg.q), 0.025);
        const double gain = std::clamp(gainDb, -96.0, 96.0);
        switch (cfg.shape)
        {
            case BandShape::LowShelf:  return AnalogSection::lowShelf(hz, gain);
            case BandShape::HighShelf: return AnalogSection::highShelf(hz, gain);
            default:                   return AnalogSection::peak(hz, q, gain);
        }
    }

    /** @brief The filter of a band's shape at one gain, fitted to its analog curve. */
    [[nodiscard]] BiquadCoeffs designShape(const BandConfig& cfg, double gainDb) const noexcept
    {
        return FilterDesign::digital(prototypeOf(cfg, gainDb), designBand_);
    }

    /** @brief What a band is made of with fixed filters: whether it rests away from 0 dB, and the end of each side. */
    struct Sides
    {
        bool rests = false, cuts = false, lifts = false;
        double cutRange = 0.0, liftRange = 0.0;
    };

    [[nodiscard]] static Sides sidesOf(const BandConfig& cfg) noexcept
    {
        Sides s;
        s.cutRange = rangeTowards(cfg, false);
        s.liftRange = rangeTowards(cfg, true);
        s.cuts = s.cutRange > kRestDb;
        s.lifts = s.liftRange > kRestDb;
        s.rests = std::abs(static_cast<double>(cfg.gain)) > kRestDb;
        return s;
    }

    /** @brief Stores the sides of a band and the scales that turn its gain into the weight of each end. */
    static void adoptSides(BandState& st, const Sides& s) noexcept
    {
        st.rests = s.rests;
        st.cuts = s.cuts;
        st.lifts = s.lifts;
        st.cutScale = s.cuts ? 1.0 / (1.0 - std::pow(10.0, -s.cutRange / 20.0)) : 0.0;
        st.liftScale = s.lifts ? 1.0 / (std::pow(10.0, s.liftRange / 20.0) - 1.0) : 0.0;
        st.designed = true;
    }

    /** @brief The weight of the end of the cuts for a gain of d dB from the rest (0 for a lift). */
    [[nodiscard]] static double cutWeight(const BandState& st, double d) noexcept
    {
        return (d < 0.0 && st.cuts) ? std::min((1.0 - fastPow10(d * 0.05)) * st.cutScale, 1.0) : 0.0;
    }

    /** @brief The weight of the end of the lifts for a gain of d dB from the rest (0 for a cut). */
    [[nodiscard]] static double liftWeight(const BandState& st, double d) noexcept
    {
        return (d > 0.0 && st.lifts) ? std::min((fastPow10(d * 0.05) - 1.0) * st.liftScale, 1.0) : 0.0;
    }

    /**
     * @brief The fixed filter at the end of one side of a mixed band.
     *
     * Not the band at that gain: the filter that makes the band, half way
     * through the range in decibels, exactly the bell or shelf of that gain
     * and of its Q. With H the band at half the range and r its gain there
     * as an amplitude, the end is 1 + (1 + r) (H - 1): the same poles, and
     * the whole range at its centre. A fixed pair cannot be the nominal
     * curve at every depth, and this choice splits the difference: nearer
     * the rest a cut is wider than nominal and at the end narrower, by a
     * factor of 10^(range / 80) in Q at most (1.41 for 12 dB); a boost, the
     * other way round.
     *
     * @param cfg     The band.
     * @param rangeDb The range of the side: negative for a cut.
     */
    [[nodiscard]] BiquadCoeffs designEnd(const BandConfig& cfg, double rangeDb) const noexcept
    {
        const BiquadCoeffs half = designShape(cfg, 0.5 * rangeDb);
        const double reach = 1.0 + std::pow(10.0, rangeDb / 40.0);
        BiquadCoeffs end = half;
        end.b0 = 1.0 + reach * (half.b0 - 1.0);
        end.b1 = half.a1 + reach * (half.b1 - half.a1);
        end.b2 = half.a2 + reach * (half.b2 - half.a2);
        return end;
    }

    /** @brief Designs the fixed filters of a mixed band: at rest and at the end of each side. */
    void designMixedBand(int b, double fs) noexcept
    {
        BandState& st = states_[b];
        const BandConfig& cfg = st.cfg;
        MixedBand& filters = mixedEngine_->bands[static_cast<size_t>(b)];
        const double rest = static_cast<double>(cfg.gain);
        const Sides sides = sidesOf(cfg);
        const bool rests = sides.rests, cuts = sides.cuts, lifts = sides.lifts;
        const double cutRange = sides.cutRange, liftRange = sides.liftRange;

        // A filter that was out of the band comes in clean; one that was
        // playing glides to its new design.
        const int ramp = static_cast<int>(kMixedRampMs * 0.001 * fs);
        const auto place = [&](BiquadCascade<T, 1, kMaxChannels>& filter, const BiquadCoeffs& design, bool was, bool is) noexcept {
            if (!is) return;
            if (!was || !st.designed) filter.reset();
            filter.rampCoeffs(0, design, was && st.designed ? ramp : 0);
        };
        if (rests) place(filters.rest, designShape(cfg, rest), st.rests, rests);
        if (cuts) place(filters.cut, designEnd(cfg, -cutRange), st.cuts, cuts);
        if (lifts) place(filters.lift, designEnd(cfg, liftRange), st.lifts, lifts);

        adoptSides(st, sides);
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

        BandState& st = states_[b];
        const BandConfig old = st.cfg;

        // A band re-enabled after being disabled would replay arbitrarily old
        // filter history and gain state: start it clean. So does one that
        // moves to other channels, whose memory is of another signal.
        if ((cfg.enabled && !old.enabled) || cfg.scope != old.scope)
            cleanBand(b);

        // The zone the band listens to has moved: with a relative threshold,
        // a second reference listens there before it takes over.
        const bool moved = cfg.frequency != old.frequency || cfg.q != old.q || cfg.shape != old.shape
                           || cfg.timing != old.timing;
        if (cfg.relativeThreshold && old.relativeThreshold && moved)
        {
            // The detector starts clean there: what it still rings of the old
            // zone is not the level of the new one.
            bandDetector_[b].reset();
            for (int ch = 0; ch < kMaxChannels; ++ch)
                analytic_[static_cast<size_t>(b * kMaxChannels + ch)].reset();
            st.fast = st.slow = st.heard = 0.0;
            st.candidate = static_cast<double>(kMinLevelDb);
            st.watching = std::max(1, static_cast<int>(kReferenceWatch * fs));
        }
        else if (cfg.relativeThreshold && !old.relativeThreshold)
        {
            st.reference = st.candidate = static_cast<double>(kMinLevelDb);
            st.watching = 0;
        }
        st.referenceFall = kReferenceFallDb / fs;
        st.cfg = cfg;

        // Detector listens where the gain filter acts: bandpass for bells,
        // the corresponding half of the spectrum for shelves. With program
        // timing the band-pass is narrower than the band, so what sets the
        // gain is the middle of it.
        // Direct writes (setCoeffsNow): this runs on the audio thread, which
        // owns these Biquads (one-master rule, see the threading block), so
        // even this cold reconfiguration must not self-publish through the
        // staged channel.
        const bool program = cfg.timing == Timing::Program;
        const T detectorQ = program ? cfg.q * static_cast<T>(kDetectorNarrowing) : cfg.q;
        switch (cfg.shape)
        {
        case BandShape::Bell:
            bandDetector_[b].setCoeffsNow(BiquadCoeffs::makeBandPass(fs, cfg.frequency, detectorQ));
            break;
        case BandShape::LowShelf:
            bandDetector_[b].setCoeffsNow(BiquadCoeffs::makeLowPass(fs, cfg.frequency, T(0.707)));
            break;
        case BandShape::HighShelf:
            bandDetector_[b].setCoeffsNow(BiquadCoeffs::makeHighPass(fs, cfg.frequency, T(0.707)));
            break;
        }

        auto calcCoeff = [fs](T ms) -> T {
            const double tauSec = std::max(static_cast<double>(ms), 0.01) / 1000.0;
            return static_cast<T>(1.0 - std::exp(-1.0 / (fs * tauSec)));
        };

        st.aboveAtkCoeff = calcCoeff(cfg.aboveAttackMs);
        st.aboveRelCoeff = calcCoeff(cfg.aboveReleaseMs);
        st.belowAtkCoeff = calcCoeff(cfg.belowAttackMs);
        st.belowRelCoeff = calcCoeff(cfg.belowReleaseMs);

        // Program timing: the time constant of the detector's envelope is the unit.
        const double hz = std::clamp(static_cast<double>(cfg.frequency), 10.0, 0.45 * fs);
        const double unit = cfg.shape == BandShape::Bell
            ? std::max(static_cast<double>(detectorQ), 0.1) / (std::numbers::pi * hz)
            : 1.0 / (std::numbers::pi * hz);
        const auto pole = [fs](double seconds) noexcept { return 1.0 - std::exp(-1.0 / (fs * seconds)); };
        const double fastFall = std::max(kFastFallUnits * unit, kFastFallFloor);
        st.fastUp = pole(std::max(unit, kFastRiseFloor));
        st.fastDown = pole(fastFall);
        st.slowUp = pole(std::max(kSlowRiseUnits * unit, kSlowRiseFloor));
        st.slowDown = pole(std::max(kSlowFallUnits * unit, kSlowFallFloor));
        st.lastDown = pole(kLastFallShare * fastFall);

        // The knee: the one asked for, or one that grows with the width of the
        // band (a wide band is a tone control and wants a gentle onset) and
        // with the root of its range.
        if (cfg.knee >= T(0))
        {
            st.knee = cfg.knee;
        }
        else
        {
            const double range = std::max(rangeTowards(cfg, false), rangeTowards(cfg, true));
            const double octaves = cfg.shape == BandShape::Bell
                ? 2.0 / std::numbers::ln2 * std::asinh(0.5 / std::max(static_cast<double>(cfg.q), 0.05))
                : 2.0;
            st.knee = static_cast<T>(std::min(kKneePerOctave * (1.0 + octaves) * std::sqrt(range / 12.0), 40.0));
        }

        if (linear_)
        {
            // The kernels of the band are designed again, one per block.
            adoptSides(st, sidesOf(cfg));
            linear_->pending[static_cast<size_t>(b)].fill(true);
        }
        else if (mixedActive_)
        {
            designMixedBand(b, fs);
        }
    }

    void updateLookahead() noexcept
    {
        if (sampleRate_ > 0) {
            int samples = static_cast<int>(sampleRate_ * runFactor() * lookaheadMs_ / T(1000));
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
        std::atomic<T>    gain           { T(0) };
        std::atomic<T>    knee           { T(0) };
        std::atomic<int>  scope      { static_cast<int>(Scope::Stereo) };
        std::atomic<int>  timing         { static_cast<int>(Timing::Manual) };
        std::atomic<bool> relative       { false };

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
            gain.store(c.gain, std::memory_order_relaxed);
            knee.store(c.knee, std::memory_order_relaxed);
            scope.store(static_cast<int>(c.scope), std::memory_order_relaxed);
            timing.store(static_cast<int>(c.timing), std::memory_order_relaxed);
            relative.store(c.relativeThreshold, std::memory_order_relaxed);
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

        /** @brief Relaxed copy of the twenty published words (no ordering of
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
            c.gain           = gain.load(std::memory_order_relaxed);
            c.knee           = knee.load(std::memory_order_relaxed);
            c.scope      = static_cast<Scope>(scope.load(std::memory_order_relaxed));
            c.timing         = static_cast<Timing>(timing.load(std::memory_order_relaxed));
            c.relativeThreshold = relative.load(std::memory_order_relaxed);
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
         * entirely instead of copying twenty words that would be discarded.
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
    unsigned refreshCounter_ = 0;   ///< Free-running coefficient-refresh grid (audio thread).
    int lastNumBands_ = MaxBands;   ///< Audio-thread tracker for band re-activation resets.

    // Biquads must span the class's full channel capacity (kMaxChannels); the
    // default Biquad<T> is only 8 channels, which would index its per-channel state
    // out of bounds when processing 9..16-channel (surround/immersive) audio.
    std::array<Biquad<T, kMaxChannels>, MaxBands> bandDetector_ {};
    std::array<Biquad<T, kMaxChannels>, MaxBands> bandFilter_ {};
    std::array<NormalizedLadder<T, kMaxChannels>, MaxBands> bandLadder_ {}; ///< The band filters as ladders (setStructure()).
    std::atomic<Structure> structure_ { Structure::DirectForm };            ///< setStructure().
    bool ladderActive_ = false;                                             ///< The audio thread runs the ladders.
    std::array<T, MaxBands> currentGainDb_ {};
    std::array<std::atomic<T>, MaxBands> meterGainDb_ {};

    int oversamplingFactor_ = 1;
    bool oversamplingAutomatic_ = false;               ///< The factor follows the host rate.
    static constexpr double kAutomaticInternalRate = 88200.0; ///< Rate the automatic factor reaches.
    std::unique_ptr<Oversampling<T>> oversampler_;     // audio path
    std::unique_ptr<Oversampling<T>> oversamplerSc_;   // sidechain path

    T lookaheadMs_ = T(0);
    std::atomic<int> lookaheadSamples_ { 0 };
    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuf_ {};
    std::vector<HilbertIIR<double>> analytic_; ///< Per band and channel: ripple-free detector level.
    std::array<double, kDetChunk> detBand_ {};                    ///< Detector chunk scratch.
    std::array<double, kDetChunk> detMag_ {};                     ///< Its analytic magnitude.
    std::array<double, kDetChunk> pairBand_ {};                   ///< The first channel's band (dominant link).
    std::array<double, kDetChunk> pairMag_ {};                    ///< Its analytic magnitude.
    std::array<std::array<double, 3>, MaxBands> bandCov_ {};      ///< Covariance of the pair per band (dominant link).
    std::atomic<LinkMode> linkMode_ { LinkMode::Loudest };        ///< setLinkMode().
    bool dominantLink_ = false;                                   ///< This block links by the dominant component.
    double linkRate_ = 0.0;                                       ///< Rate the link coefficient was made for.
    double linkCoeff_ = 0.0;                                      ///< One-pole coefficient of the covariance (20 ms).
    std::array<std::array<T, kDetChunk>, MaxBands> detLevelDb_ {}; ///< Linked level per band.
    std::array<double, kDetChunk> detLinear_ {};                  ///< Linked amplitude of the band in hand (program timing).
    std::array<std::array<T, kDetChunk>, MaxBands> dynDb_ {};     ///< Dynamic gain per band, in dB from its rest.

    std::array<bool, MaxBands> playing_ {};      ///< Bands with something to act on in this block.
    std::array<int, MaxBands> first_ {};         ///< The stereo, left and right ones, in order.
    std::array<int, MaxBands> second_ {};        ///< The ones of the middle and the sides, in order.
    int numFirst_ = 0, numSecond_ = 0;

    std::atomic<bool> mixedWanted_ { false };    ///< setMixedBands().
    static constexpr int kRemovalCapacity = 32768; ///< Longest latency the removal listen lines up.
    std::atomic<bool> removalListen_ { false };          ///< The output is what the bands take out.
    std::array<RingBuffer<T>, kMaxChannels> deltaRing_ {};
    bool mixedActive_ = false;                   ///< The engine the audio thread is running.
    std::unique_ptr<MixedEngine> mixedEngine_;
    std::unique_ptr<LinearEngine> linear_;       ///< The linear-phase mode; null when it is off.
    bool linearWanted_ = false;                  ///< setLinearPhase(): applied by prepare().
    int linearTaps_ = 0;                         ///< Kernel length asked for; 0 is automatic.
    FilterDesign::Band designBand_;              ///< The frequencies the mixed bands are fitted over.
};

} // namespace dspark
