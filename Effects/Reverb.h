// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Reverb.h
 * @brief Convolution reverb with one-line IR loading and progressive API.
 *
 * Wraps the Convolver engine into a complete reverb effect with dry/wet mix,
 * pre-delay, and automatic IR management. Supports loading impulse responses
 * from WAV files or from raw sample data. The dry path is delay-compensated
 * against the convolution engine's latency, so dry and wet stay sample-aligned
 * at any mix setting (getLatency() reports the shared latency to the host).
 *
 * IR shaping (setDecayScale / setStretch) reshapes the loaded impulse
 * response without touching the stored original, so the controls are
 * always relative to the file as loaded:
 *
 * - **Decay scale** multiplies the IR's own T60 (estimated from its
 *   Schroeder decay curve). Values below 1 also trim the now-silent tail,
 *   which directly reduces convolution CPU cost.
 * - **Stretch** resamples the IR (tape-speed style): above 1 the space
 *   gets larger and darker, below 1 smaller and brighter.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `reverb.loadIR("hall.wav"); reverb.setMix(0.3f);`
 * - **Level 2 (intermediate):** Pre-delay, IR decay scale / stretch.
 * - **Level 3 (expert):** Direct access to Convolver and DryWetMixer internals.
 *
 * Threading: prepare() belongs to the setup thread (allocates; never call it
 * concurrently with processing). processBlock() and reset() belong to the
 * audio thread. loadIR(), setDecayScale(), setStretch() and setState()
 * rebuild the convolver bank (they allocate) on the calling GUI/setup thread
 * and publish it atomically: safe while audio runs, one writer at a time.
 * setMix()/setPreDelay() are lock-free atomic publications, safe from any
 * thread. Non-finite setter arguments are ignored. Loading an IR changes
 * getLatency(): hosts must be notified.
 *
 * File loading (loadIR from a path) is excluded when DSPARK_NO_FILE_IO is
 * defined; the raw-data overload keeps working on embedded targets.
 *
 * Dependencies: Convolver.h, DryWetMixer.h, RingBuffer.h, AudioSpec.h,
 *               AudioBuffer.h, DspMath.h, Resampler.h, StateBlob.h,
 *               WavFile.h (only without DSPARK_NO_FILE_IO).
 *
 * @code
 *   dspark::Reverb<float> reverb;
 *   reverb.prepare(spec);
 *   reverb.loadIR("hall.wav");   // One line, done
 *   reverb.setMix(0.3f);         // 30% wet
 *   reverb.processBlock(buffer);
 *
 *   // With pre-delay:
 *   reverb.setPreDelay(20.0f);   // 20 ms
 * @endcode
 */

#include "../Core/Convolver.h"
#include "../Core/DryWetMixer.h"
#include "../Core/RingBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DspMath.h"
#include "../Core/Resampler.h"
#include "../Core/StateBlob.h"
#ifndef DSPARK_NO_FILE_IO
#include "../IO/WavFile.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class Reverb
 * @brief Convolution reverb with IR loading, dry/wet, and pre-delay.
 *
 * Internally manages one Convolver per channel. The IR is automatically
 * resampled if its sample rate differs from the processing sample rate.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Reverb
{
public:
    ~Reverb() = default; // non-virtual: leaf class (no virtual dispatch)

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the reverb for processing.
     *
     * Allocates internal buffers, sets up the dry/wet mixer (including the
     * dry-path compensation for the convolution latency) and the pre-delay.
     * If an IR was loaded before prepare(), it will be re-applied.
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        spec_ = spec;

        // The convolution engine partitions at the next power of two of the
        // max block size (>= 2, matching Convolver's own normalisation), and
        // that is exactly its processing latency. Clamp before the round-up
        // loop so an absurd block size cannot overflow the shift.
        const int blockSize = std::clamp(spec.maxBlockSize, 1, 1 << 20);
        int fftBlock = 2;
        while (fftBlock < blockSize) fftBlock <<= 1;
        fftBlockSize_ = fftBlock;

        // Delay the dry path by the same amount so dry and wet stay aligned
        // at any mix (the wet is late by the convolver latency; without this
        // the mix comb-filtered against the shifted dry).
        mixer_.prepare(spec);
        mixer_.setLatencyCompensation(fftBlockSize_);

        // Pre-delay ring buffers (one per channel, max 500ms)
        int maxDelaySamples = static_cast<int>(spec.sampleRate * 0.5) + 1;
        preDelayBuffers_.resize(static_cast<size_t>(spec.numChannels));
        for (auto& rb : preDelayBuffers_)
            rb.prepare(maxDelaySamples);

        updatePreDelay();

        // Re-apply IR if one was already loaded. applyIR() rebuilds the bank
        // on this (GUI) thread and publishes it atomically.
        if (!irStorage_.empty())
            applyIR();
    }

    /**
     * @brief Processes audio through the reverb.
     *
     * Flow: pushDry -> pre-delay -> convolve -> mixWet (with the dry delayed
     * by the convolution latency so both paths stay aligned). Without a
     * loaded IR the audio passes through untouched (and getLatency() is 0).
     * Channels beyond the prepared count pass through untouched.
     *
     * Thread-safety: the ConvolverBank is published atomically by
     * loadIR/applyIR. We snapshot the current bank once at the top of the
     * block into a local shared_ptr, so even if the GUI thread publishes a
     * replacement mid-block, the audio thread keeps using a stable bank
     * until the block completes. No resize of a live vector, no torn reads.
     *
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        // Design note: the bank swap is guarded by a one-flag spinlock (see
        // loadBank()): the only writer is loadIR(), a rare, user-initiated
        // event, so contention is effectively zero. A manual RCU scheme
        // would remove the spinlock at the cost of a real use-after-free
        // hazard under racing loads; correctness wins here.
        auto bank = loadBank();
        if (!bank || bank->convolvers.empty()) return;

        const int nCh = std::min(buffer.getNumChannels(),
                                 static_cast<int>(bank->convolvers.size()));
        const int nS  = buffer.getNumSamples();

        mixer_.pushDry(buffer);

        int preDelSamp = preDelaySamples_.load(std::memory_order_relaxed);
        T mixVal = mix_.load(std::memory_order_relaxed);

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = buffer.getChannel(ch);

            if (preDelSamp > 0)
            {
                auto& ring = preDelayBuffers_[static_cast<size_t>(ch)];
                for (int i = 0; i < nS; ++i)
                {
                    ring.push(data[i]);
                    data[i] = ring.read(preDelSamp);
                }
            }

            bank->convolvers[static_cast<size_t>(ch)].processInPlace(data, nS);
        }

        mixer_.mixWet(buffer, mixVal);
    }

    /**
     * @brief Resets the DSP state (convolver tails, pre-delay, mixer). RT-Safe.
     *
     * The loaded IR stays loaded: resetting used to drop the convolver bank
     * entirely, silently unloading the reverb (a host reset on stop/start
     * left it as a dry passthrough until the next loadIR()).
     */
    void reset() noexcept
    {
        // Reset the snapshot we can see; if a concurrent load publishes a
        // replacement bank it arrives freshly zeroed anyway.
        if (auto bank = loadBank())
            for (auto& conv : bank->convolvers)
                conv.reset();
        for (auto& rb : preDelayBuffers_)
            rb.reset();
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

#ifndef DSPARK_NO_FILE_IO
    /**
     * @brief Loads an impulse response from a WAV file.
     *
     * The IR is automatically resampled if the WAV sample rate differs from
     * the processing sample rate. Multi-channel IRs are supported (one
     * convolver per channel); mono IRs are duplicated across all channels.
     * Empty or degenerate files are rejected.
     *
     * @param wavFilePath Path to the WAV file.
     * @return True if the IR was loaded successfully.
     */
    bool loadIR(const char* wavFilePath)
    {
        WavFile wav;
        if (!wav.openRead(wavFilePath))
            return false;

        auto info = wav.getInfo();
        if (info.numSamples <= 0 || info.numChannels <= 0
            || info.numSamples > (static_cast<int64_t>(1) << 30)
            || !(info.sampleRate > 0))
        {
            wav.close();
            return false;
        }

        AudioBuffer<T> irBuf;
        irBuf.resize(info.numChannels, static_cast<int>(info.numSamples));
        wav.readSamples(irBuf.toView());
        wav.close();

        // Store channel 0 (or all channels)
        irChannels_ = info.numChannels;
        int irLen = static_cast<int>(info.numSamples);

        irStorage_.resize(static_cast<size_t>(irChannels_) * static_cast<size_t>(irLen));
        for (int ch = 0; ch < irChannels_; ++ch)
        {
            const T* src = irBuf.getChannel(ch);
            T* dst = irStorage_.data() + static_cast<size_t>(ch) * static_cast<size_t>(irLen);
            std::copy_n(src, irLen, dst);
        }
        irLength_ = irLen;
        irSampleRate_ = info.sampleRate;

        if (spec_.sampleRate > 0)
            applyIR();

        return true;
    }
#endif // DSPARK_NO_FILE_IO

    /**
     * @brief Sets the dry/wet mix.
     * @param dryWet 0.0 = fully dry (no reverb), 1.0 = fully wet.
     *               Non-finite values are ignored.
     */
    void setMix(T dryWet) noexcept
    {
        if (!std::isfinite(dryWet)) return;
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed);
    }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Loads an IR from raw sample data.
     *
     * @param data         Pointer to mono IR samples.
     * @param length       Number of samples (must be > 0).
     * @param irSampleRate Sample rate of the IR data (must be > 0 and finite).
     * @return True if the IR was accepted (invalid arguments are rejected).
     */
    bool loadIR(const T* data, int length, double irSampleRate)
    {
        if (data == nullptr || length <= 0
            || !std::isfinite(irSampleRate) || !(irSampleRate > 0.0))
            return false;

        irChannels_ = 1;
        irLength_ = length;
        irSampleRate_ = irSampleRate;

        irStorage_.assign(data, data + length);

        if (spec_.sampleRate > 0)
            applyIR();

        return true;
    }

    /**
     * @brief Sets the pre-delay time in milliseconds.
     *
     * Pre-delay adds a gap before the reverb tail starts, creating a sense
     * of room size. Typical values: 0-50 ms. Applied immediately (a live
     * change may click); intended as a setup control, not for automation.
     *
     * @param ms Pre-delay in milliseconds, clamped to [0, 500] (the ring
     *           buffer allocation). Non-finite values are ignored.
     */
    void setPreDelay(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        preDelayMs_.store(std::clamp(ms, T(0), T(500)), std::memory_order_relaxed);
        updatePreDelay();
    }

    /**
     * @brief Scales the decay time (T60) of the loaded IR.
     *
     * The IR's own decay rate is estimated from its Schroeder backward
     * energy curve (T20 fit between -5 dB and -25 dB), then an exponential
     * envelope is applied from the direct-sound peak onward so the shaped
     * IR decays at `scale` times the original T60. The direct sound and
     * early part are preserved.
     *
     * Values below 1 shorten the tail AND trim the IR where its energy
     * falls below -100 dB, so the convolution gets proportionally cheaper
     * (about half the CPU at 0.5 on a typical exponential hall). Values
     * above 1 lengthen the tail; this also lifts whatever noise floor the
     * recording has, so moderate boosts (up to 2x) are the useful range.
     * IRs without a broadly exponential tail (gated/reversed effects) are
     * left unshaped.
     *
     * Setup/UI threads only: rebuilds the convolver bank (allocates) and
     * publishes it atomically, exactly like loadIR(). Not meant for
     * per-block automation. Non-finite values are ignored.
     *
     * @param scale T60 multiplier, clamped to [0.25, 2]. 1 = as loaded.
     */
    void setDecayScale(T scale)
    {
        if (!std::isfinite(scale)) return;
        decayScale_.store(std::clamp(scale, T(0.25), T(2)),
                          std::memory_order_relaxed);
        if (spec_.sampleRate > 0 && !irStorage_.empty())
            applyIR();
    }

    /**
     * @brief Stretches the loaded IR in time (tape-speed style).
     *
     * The IR is resampled by the given ratio: 2.0 doubles its length
     * (larger, darker space, roughly an octave down in coloration), 0.5
     * halves it (smaller, brighter). Decay scale and stretch compose:
     * effective T60 is approximately original * decayScale * stretch.
     *
     * Setup/UI threads only: rebuilds the convolver bank (allocates) and
     * publishes it atomically, exactly like loadIR(). Non-finite values are
     * ignored.
     *
     * @param ratio Time-stretch ratio, clamped to [0.5, 2]. 1 = as loaded.
     */
    void setStretch(T ratio)
    {
        if (!std::isfinite(ratio)) return;
        stretch_.store(std::clamp(ratio, T(0.5), T(2)),
                       std::memory_order_relaxed);
        if (spec_.sampleRate > 0 && !irStorage_.empty())
            applyIR();
    }

    /** @brief Returns the current IR decay scale. */
    [[nodiscard]] T getDecayScale() const noexcept { return decayScale_.load(std::memory_order_relaxed); }

    /** @brief Returns the current IR stretch ratio. */
    [[nodiscard]] T getStretch() const noexcept { return stretch_.load(std::memory_order_relaxed); }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Direct access to a channel's Convolver (GUI thread only).
     *
     * NOTE: the live convolver bank is published atomically and may be
     * replaced by a future loadIR() call. This accessor returns a reference
     * into the currently-published bank and MUST NOT be used from the audio
     * thread. Before an IR is loaded (or with an out-of-range channel) it
     * returns an inert fallback engine instead of dereferencing a null bank.
     *
     * @param channel Channel index (clamped into the bank's range).
     */
    Convolver<T>& getConvolver(int channel = 0)
    {
        auto bank = loadBank();
        if (!bank || bank->convolvers.empty())
            return fallbackConvolver_;
        const int n = static_cast<int>(bank->convolvers.size());
        channel = std::clamp(channel, 0, n - 1);
        return bank->convolvers[static_cast<size_t>(channel)];
    }

    /** @brief Direct access to the DryWetMixer. */
    DryWetMixer<T>& getMixer() { return mixer_; }

    /** @brief Returns true if an IR has been loaded and applied. */
    [[nodiscard]] bool isLoaded() const noexcept
    {
        return static_cast<bool>(loadBank());
    }

    /** @brief Returns the current mix value. */
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Returns the current pre-delay in ms. */
    [[nodiscard]] T getPreDelay() const noexcept { return preDelayMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Returns the convolution latency in samples.
     *
     * 0 without an IR (the audio passes through untouched); the convolver's
     * partition latency once an IR is loaded. The dry path is internally
     * delayed by the same amount, so this is the whole effect's latency.
     * Hosts must re-read it after loading an IR.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        auto bank = loadBank();
        return (bank && !bank->convolvers.empty())
               ? bank->convolvers.front().getLatency() : 0;
    }


    /** @brief Serializes the parameter state. The impulse response itself is
     *  content (load it with loadIR), not a preset parameter. */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("CRVB"), 1);
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("preDelay", preDelayMs_.load(std::memory_order_relaxed));
        w.write("decayScale", decayScale_.load(std::memory_order_relaxed));
        w.write("stretch", stretch_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("CRVB")) return false;
        setMix(static_cast<T>(r.read("mix", 0.3f)));
        setPreDelay(static_cast<T>(r.read("preDelay", 0.0f)));
        // Store both shaping values first, then rebuild once (each setter
        // would otherwise trigger its own IR rebuild). Non-finite blob values
        // keep the current settings.
        T ds = static_cast<T>(r.read("decayScale", 1.0f));
        T st = static_cast<T>(r.read("stretch", 1.0f));
        if (!std::isfinite(ds)) ds = decayScale_.load(std::memory_order_relaxed);
        if (!std::isfinite(st)) st = stretch_.load(std::memory_order_relaxed);
        ds = std::clamp(ds, T(0.25), T(2));
        st = std::clamp(st, T(0.5), T(2));
        const bool shapeChanged =
            ds != decayScale_.load(std::memory_order_relaxed)
            || st != stretch_.load(std::memory_order_relaxed);
        decayScale_.store(ds, std::memory_order_relaxed);
        stretch_.store(st, std::memory_order_relaxed);
        if (shapeChanged && spec_.sampleRate > 0 && !irStorage_.empty())
            applyIR();
        return true;
    }

protected:
    void updatePreDelay() noexcept
    {
        if (spec_.sampleRate > 0)
        {
            // The pre-delay ring buffers hold 500 ms; clamp so an over-range pre-delay
            // can't read past the buffer (RingBuffer::read would wrap to a wrong sample).
            const int maxSamp = static_cast<int>(spec_.sampleRate * 0.5);
            const int samp = static_cast<int>(static_cast<T>(spec_.sampleRate)
                                              * preDelayMs_.load(std::memory_order_relaxed) / T(1000));
            preDelaySamples_.store(std::clamp(samp, 0, maxSamp), std::memory_order_relaxed);
        }
    }

    void applyIR()
    {
        if (irStorage_.empty() || irLength_ <= 0 || fftBlockSize_ <= 0) return;

        int numCh = spec_.numChannels;

        // Build the new bank in a local shared_ptr. The old bank (if any)
        // stays live until the last audio-thread snapshot releases it.
        auto newBank = std::make_shared<ConvolverBank>();
        newBank->convolvers.resize(static_cast<size_t>(numCh));

        // IR shaping controls, always applied to the stored original.
        // Stretch works by declaring a scaled source rate and letting the
        // resampling stage do the time-scaling (tape-speed semantics).
        const double dScale =
            static_cast<double>(decayScale_.load(std::memory_order_relaxed));
        const double stretch =
            static_cast<double>(stretch_.load(std::memory_order_relaxed));
        const bool doShape = std::abs(dScale - 1.0) > 1e-6;
        const double effIrRate = irSampleRate_ / std::max(stretch, 0.01);

        std::vector<T> shaped;   // lazy decay-shaped copy of one IR channel
        int shapedCh = -1;

        for (int ch = 0; ch < numCh; ++ch)
        {
            // Pick IR channel: use corresponding channel if available, else mono (ch 0)
            int irCh = (ch < irChannels_) ? ch : 0;
            const T* irData = irStorage_.data()
                            + static_cast<size_t>(irCh) * static_cast<size_t>(irLength_);
            int irLen = irLength_;

            if (doShape)
            {
                if (shapedCh != irCh)
                {
                    shaped = shapeDecay(irData, irLength_, dScale);
                    shapedCh = irCh;
                }
                if (!shaped.empty())
                {
                    irData = shaped.data();
                    irLen = static_cast<int>(shaped.size());
                }
            }

            auto& conv = newBank->convolvers[static_cast<size_t>(ch)];

            // Resample if the (stretch-adjusted) IR rate differs from the engine
            if (std::abs(effIrRate - spec_.sampleRate) > 1.0)
            {
                Resampler<T> resampler;
                resampler.prepare(effIrRate, spec_.sampleRate);
                // Size from the resampler's own bound (INT_MAX-safe): the old
                // floor(n*ratio)+1 arithmetic could overflow the int cast with
                // an extreme rate ratio.
                std::vector<T> resampled(
                    static_cast<size_t>(resampler.getMaxOutputSamples(irLen)));
                int produced = resampler.processBlock(irData, irLen,
                                                      resampled.data());

                conv.prepare(fftBlockSize_, resampled.data(), produced);
            }
            else
            {
                conv.prepare(fftBlockSize_, irData, irLen);
            }
        }

        // Atomic release-store: any subsequent acquire-load in processBlock
        // will observe the fully-constructed bank.
        storeBank(newBank);
    }

    /**
     * @brief Returns a decay-scaled copy of one IR channel (see setDecayScale).
     *
     * Estimates the source decay rate from the Schroeder backward-energy
     * curve (T20 fit: -5 dB to -25 dB crossings of the EDC), then applies
     * exp(-k * (n - peak)) so the result decays at `factor` times the
     * original T60. The direct sound (up to the peak) is untouched.
     * Returns an empty vector when the IR has no measurable exponential
     * decay (too short, or gated); the caller then uses the original data.
     */
    [[nodiscard]] std::vector<T> shapeDecay(const T* ir, int len,
                                            double factor) const
    {
        if (!ir || len < 64) return {};

        // Total energy + direct-sound peak (double accumulation).
        double total = 0.0;
        double peakMag = 0.0;
        int peak = 0;
        for (int n = 0; n < len; ++n)
        {
            const double v = static_cast<double>(ir[n]);
            total += v * v;
            const double m = std::abs(v);
            if (m > peakMag) { peakMag = m; peak = n; }
        }
        if (total <= 1e-30 || peakMag <= 0.0) return {};

        // Schroeder EDC crossings at -5 dB and -25 dB (energy ratios).
        constexpr double r5  = 0.31622776601683794;    // 10^(-5/10)
        constexpr double r25 = 0.0031622776601683794;  // 10^(-25/10)
        int t5 = -1, t25 = -1;
        double tail = total;
        for (int n = 0; n < len; ++n)
        {
            const double ratio = tail / total;
            if (t5 < 0 && ratio <= r5 && n > peak) t5 = n;
            if (ratio <= r25 && n > peak) { t25 = n; break; }
            const double v = static_cast<double>(ir[n]);
            tail -= v * v;
        }
        if (t5 < 0 || t25 < 0 || t25 - t5 < 32) return {};  // no usable slope

        // Amplitude decay rate: the EDC drops 20 dB over (t25 - t5) samples,
        // so exp(-beta * t) with beta = ln(10) / (t25 - t5).
        const double beta = 2.302585092994046 / static_cast<double>(t25 - t5);
        const double k = beta * (1.0 / factor - 1.0);

        std::vector<T> out(static_cast<size_t>(len));
        const double gStep = std::exp(-k);
        double g = 1.0;
        for (int n = 0; n < len; ++n)
        {
            out[static_cast<size_t>(n)] = (n <= peak)
                ? ir[n]
                : static_cast<T>(static_cast<double>(ir[n]) * g);
            if (n >= peak) g *= gStep;
        }

        if (factor < 1.0)
        {
            // Trim where the shaped energy falls below -100 dB of its total:
            // the removed stretch is inaudible, and a shorter IR means fewer
            // convolution partitions (the CPU saving the shaping is for).
            double sTotal = 0.0;
            for (const T v : out) sTotal += static_cast<double>(v) * v;
            if (sTotal > 1e-30)
            {
                double sTail = sTotal;
                int cut = len;
                for (int n = 0; n < len; ++n)
                {
                    if (sTail / sTotal <= 1e-10) { cut = n; break; }
                    const double v = static_cast<double>(out[static_cast<size_t>(n)]);
                    sTail -= v * v;
                }
                cut = std::max(cut, 64);
                if (cut < len) out.resize(static_cast<size_t>(cut));
            }
        }
        else if (factor > 1.0)
        {
            // A raised envelope would end in a cliff at the IR boundary:
            // fade the final stretch (up to 20 ms at 48 kHz) with a raised
            // cosine so the lengthened tail closes cleanly.
            const int fade = std::min(len / 8, 960);
            const int start = len - fade;
            for (int i = 0; i < fade; ++i)
            {
                const double w = 0.5 * (1.0 + std::cos(3.141592653589793
                                        * static_cast<double>(i + 1)
                                        / static_cast<double>(fade)));
                const size_t idx = static_cast<size_t>(start + i);
                out[idx] = static_cast<T>(static_cast<double>(out[idx]) * w);
            }
        }
        return out;
    }

    // Holds the current convolver set. Published by applyIR() and snapshotted
    // by the audio thread at the top of processBlock, so an in-flight block
    // keeps a stable bank alive.
    struct ConvolverBank
    {
        std::vector<Convolver<T>> convolvers;
    };

    AudioSpec spec_ {};
    int fftBlockSize_ = 0; ///< Convolver partition size = engine latency (set in prepare()).
    std::atomic<T> mix_ { T(0.3) };
    std::atomic<T> preDelayMs_ { T(0) };
    std::atomic<int> preDelaySamples_ { 0 };
    std::atomic<T> decayScale_ { T(1) };
    std::atomic<T> stretch_ { T(1) };

    // IR storage (GUI-thread only: rebuild source of truth)
    std::vector<T> irStorage_;
    int irLength_ = 0;
    int irChannels_ = 0;
    double irSampleRate_ = 0;

    // Processing (audio-thread visible state).
    //
    // Portable stand-in for std::atomic<std::shared_ptr>: libc++ (macOS,
    // Emscripten) does not ship the C++20 specialization. A one-flag
    // spinlock guards only the pointer copy/swap (nanoseconds on the audio
    // thread against one UI-initiated store per IR load), preserving the
    // exact snapshot semantics: an in-flight processBlock keeps its own
    // shared_ptr alive, and replaced banks destruct on the UI thread
    // (the swap drops them outside the lock).
    [[nodiscard]] std::shared_ptr<ConvolverBank> loadBank() const noexcept
    {
        while (bankLock_.test_and_set(std::memory_order_acquire)) {}
        auto copy = bankPtr_;
        bankLock_.clear(std::memory_order_release);
        return copy;
    }
    void storeBank(std::shared_ptr<ConvolverBank> next) noexcept
    {
        while (bankLock_.test_and_set(std::memory_order_acquire)) {}
        bankPtr_.swap(next);
        bankLock_.clear(std::memory_order_release);
        // `next` (the previous bank) destructs here, outside the lock.
    }

    mutable std::atomic_flag bankLock_ = ATOMIC_FLAG_INIT;
    std::shared_ptr<ConvolverBank> bankPtr_;
    std::vector<RingBuffer<T>> preDelayBuffers_;
    DryWetMixer<T> mixer_;
    Convolver<T> fallbackConvolver_; ///< Inert engine for getConvolver() with no bank.
};

} // namespace dspark
