// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralDenoiser.h
 * @brief Spectral noise reduction with a learnable noise profile.
 *
 * Broadcast-style noise reduction on the SpectralProcessor STFT pipeline:
 *
 * 1. **Learn**: while learning is enabled, the per-bin power of the incoming
 *    signal (noise only - e.g. a second of room tone) accumulates into the
 *    noise profile: the running mean noise power per bin.
 * 2. **Reduce**: each bin gets the Wiener gain xi / (1 + xi), where the a
 *    priori SNR xi is the decision-directed estimate of Ephraim and Malah
 *    (weight 0.98 per quarter of the transform) against the profile scaled
 *    by `threshold^2`,
 *    signal and profile read alike in a band of setDetail() around the bin
 *    (up to an octave; by default the bin itself).
 *    Gains never fall below the `reduction` floor. The decision-directed
 *    estimate follows onsets at once but ignores random noise flicker, so
 *    the residual noise stays a smooth, attenuated copy of the noise instead
 *    of the "musical noise" (isolated tones) a hard per-bin gate leaves.
 *
 * A non-finite input reaches neither the profile nor the gain memory: the
 * frames that contain it come out silent and the next clean frame continues.
 *
 * Limits of the detail: a bin is as fine as it gets (23 Hz with the 2048 of
 * 48 kHz), and noise is taken whole between held lines a few bins apart: a
 * 24th of an octave at 10 kHz, a sixth at 4 kHz; at 8192 a sixth at 1 kHz.
 * Nearer than that, as in the low end, the noise between the lines stays
 * under them. A longer transform trades its latency and the fidelity of what
 * moves for that (its frames are a sixteenth of it apart, which kept the
 * images of the lines between them within 1.1 dB over the noise there in
 * what was measured).
 * A sixth of an octave at 50 Hz is not separated with 65536 either.
 * getDisplayGain() is the gain applied to the first channel.
 *
 * Options, all off by default:
 * setGainRelease() gives the gains of each bin a release, slower in the low
 * end (by the square root of the frequency against 1 kHz); a sensitivity
 * curve (setSensitivityBand(), six bands: cuts, bells, shelves, tilt) weights
 * the reduction frequency by frequency; setRemovalListen(true) outputs what is taken
 * out; prepare(spec, 0) picks the transform size by the rate (2048 at 44.1
 * and 48 kHz, doubling by octaves); setFilterMode(true) shortens what is
 * spread ahead of an attack to one short frame: the gains stay zero phase
 * (so up to 5 ms can still be spread ahead), and where an attack is ahead
 * (seen in the latency) frames of 5 ms play in place of the long ones, until
 * 11 ms after it, at the same latency. Anywhere else it is the default path
 * to the bit. There setLink() takes the power each channel is read by part
 * of the way to the mean of the pair, bin by bin and within the same frame
 * (1: one detector and one gain for both).
 *
 * Latency is the STFT's (fftSize samples). All per-bin state is per channel;
 * the profile is shared. The decision-directed weight is per quarter of the
 * transform, so its memory in milliseconds scales with fftSize/sampleRate.
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored). prepare() is setup-thread
 * only (allocates; invalid specs are ignored and an unprepared instance
 * passes audio through). reset() and clearProfile() belong to the stream
 * owner (the profile is written by the audio thread while learning).
 * getState()/setState() are setup/UI threads. Channels beyond the prepared
 * count pass through dry (SpectralProcessor behaviour).
 *
 * Dependencies: Core/SpectralProcessor.h, Core/SpectralFilter.h (the
 * readout), Core/FFT.h, Core/WindowFunctions.h, Core/KeyEqualizer.h,
 * Core/AudioSpec.h, Core/AudioBuffer.h, Core/DspMath.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/KeyEqualizer.h"
#include "../Core/SpectralFilter.h"
#include "../Core/SpectralProcessor.h"
#include "../Core/StateBlob.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class SpectralDenoiser
 * @brief Learn-a-profile spectral noise reduction (hiss/hum/room-tone).
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SpectralDenoiser
{
public:
    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the STFT pipeline and the per-channel bin state.
     *
     * Invalid specs (non-positive/non-finite rate, block size or channel
     * count) are ignored: the previous state is kept and an unprepared
     * instance stays pass-through. fftSize is sanitized by the STFT engine
     * (power of two in [4, 1 << 20]).
     *
     * @param spec    Audio environment specification.
     * @param fftSize STFT size (default 2048; larger = finer hum notching;
     *                0 = by the rate: 2048 at 44.1 and 48 kHz, doubling by
     *                octaves). One over the rate's own runs its frames a
     *                sixteenth of it apart in place of a quarter, at some
     *                four times the cost: a quarter apart, gains that differ
     *                from bin to bin leave images of a line between the lines.
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        numChannels_ = spec.numChannels;
        sampleRate_ = spec.sampleRate;
        const int octaves = std::clamp(static_cast<int>(std::lround(std::log2(std::max(spec.sampleRate, 44100.0) / 44100.0))), 0, 3);
        const int rateSize = 2048 << octaves;   // the rate's own
        if (fftSize <= 0) fftSize = rateSize;
        stft_.prepare(spec, fftSize, fftSize / 4);
        const int size = stft_.getFFTSize();
        dense_ = size > rateSize;
        if (dense_) stft_.prepare(spec, size, size / kDenseFrames);
        // The same memories in seconds at either spacing of the frames.
        longAlpha_ = dense_ ? static_cast<float>(std::pow(static_cast<double>(kDDAlpha), 4.0 / kDenseFrames)) : kDDAlpha;
        learnMost_ = dense_ ? kMaxLearnFrames * kDenseFrames / 4 : kMaxLearnFrames;
        displayEvery_ = dense_ ? kDenseFrames / 2 : 2;
        numBins_ = stft_.getNumBins();
        prepareShort(spec);

        profile_.assign(static_cast<size_t>(numBins_), 0.0f);
        learnFrames_ = 0;
        cleanPower_.assign(static_cast<size_t>(numChannels_),
                           std::vector<float>(static_cast<size_t>(numBins_), 0.0f));
        heldGain_.assign(static_cast<size_t>(numChannels_),
                         std::vector<float>(static_cast<size_t>(numBins_), 1.0f));
        for (auto& v : pairPower_) v.assign(static_cast<size_t>(numBins_), 0.0f);
        const auto most = static_cast<size_t>(numBins_);
        gains_.assign(most, 1.0f);
        for (auto* v : { &detect_, &noiseHere_ }) v->assign(most, 0.0f);
        sum_.assign(most + 1, 0.0);
        stftBands_.prepare(numBins_);
        shortBands_.prepare(shortBins_);
        bandsFor_ = -1.0;
        noiseHolds_ = false;
        stftReadout_.prepare(spec.sampleRate / static_cast<double>(stft_.getFFTSize()), numBins_);
        shownGains_ = std::vector<std::atomic<float>>(static_cast<size_t>(numBins_));
        for (auto& g : shownGains_) g.store(1.0f, std::memory_order_relaxed);
        weight_.assign(static_cast<size_t>(numBins_), 1.0f);
        release_.assign(static_cast<size_t>(numBins_), 0.0f);
        shortRelease_.assign(static_cast<size_t>(shortBins_), 0.0f);
        shortWeight_.assign(static_cast<size_t>(shortBins_), 1.0f);
        sensitivity_.prepare(spec.sampleRate);
        weightDirty_.store(true, std::memory_order_release);
        releaseFor_ = -1.0f;

        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears signal state and per-bin gain memories (keeps profile). */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        stft_.reset();
        for (auto& c : cleanPower_)
            std::fill(c.begin(), c.end(), 0.0f);
        for (auto& c : heldGain_) std::fill(c.begin(), c.end(), 1.0f);
        // The stream starts in silence: the short frames miss nothing of it.
        resetShort(0);
        shortRuns_ = filterMode_.load(std::memory_order_relaxed);
        displayHop_ = 0;
        for (auto& g : shownGains_) g.store(1.0f, std::memory_order_relaxed);
    }

    /** @brief Forgets the learned noise profile (stream-owner thread). */
    void clearProfile() noexcept
    {
        std::fill(profile_.begin(), profile_.end(), 0.0f);
        noiseHolds_ = false;
        shortNoiseHolds_ = false;
        learnFrames_ = 0;
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief While true, incoming audio trains the noise profile. */
    void setLearning(bool learning) noexcept
    {
        learning_.store(learning, std::memory_order_relaxed);
    }

    /** @brief Maximum attenuation of noise bins in dB [0, 40] (default 18):
     *  the floor of the per-bin gain. Non-finite values are ignored. */
    void setReduction(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        reduction_.store(std::clamp(db, T(0), T(40)), std::memory_order_relaxed);
    }

    /** @brief Noise over-subtraction factor over the learned profile, in
     *  magnitude [1, 8] (default 2): the gain rule sees the learned noise
     *  scaled by it, so higher values remove more noise and more low-level
     *  signal. Non-finite values are ignored. */
    void setThreshold(T factor) noexcept
    {
        if (!std::isfinite(factor)) return;
        threshold_.store(std::clamp(factor, T(1), T(8)), std::memory_order_relaxed);
    }

    /**
     * @brief The detail: the band, in octaves, in which the signal and the
     *        noise are read at each bin, from 1 (broad, the steadiest
     *        residual) down to 0 (default: each bin by itself, the finest
     *        there is, between the partials of a note). A band is never
     *        narrower than a bin. It moves while playing, at the same
     *        latency, and the profile learned holds at any detail.
     */
    void setDetail(T octaves) noexcept
    {
        if (!std::isfinite(octaves)) return;
        detail_.store(std::clamp(octaves, T(0), T(1)), std::memory_order_relaxed);
    }
    /** @brief Release of each bin's gain at 1 kHz, 0 (off, default) to 2000 ms; slower below, faster above. */
    void setGainRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::clamp(ms, T(0), T(2000)), std::memory_order_relaxed);
    }
    /** @brief One band of the sensitivity curve: the reduction is scaled by its gain in dB / 12 + 1, clamped 0 to 2. */
    void setSensitivityBand(int index, const typename KeyEqualizer<T, 2>::Band& band) noexcept
    {
        sensitivity_.setBand(index, band);
        weightDirty_.store(true, std::memory_order_release);
    }
    /** @return One band of the sensitivity curve. */
    [[nodiscard]] typename KeyEqualizer<T, 2>::Band getSensitivityBand(int index) const noexcept { return sensitivity_.getBand(index); }
    /** @brief Outputs what is taken out instead of what is kept. */
    void setRemovalListen(bool on) noexcept { removalListen_.store(on, std::memory_order_relaxed); }
    /**
     * @brief Spreads no further ahead of an attack than one short frame of
     *        5 ms (see the file comment): the default path, with those
     *        frames where an attack is ahead and for 11 ms after it. The
     *        latency is the same, and it can be switched while playing.
     */
    void setFilterMode(bool on) noexcept { filterMode_.store(on, std::memory_order_relaxed); }
    /** @brief In filter mode, how far the power each channel is read by goes to the mean of the pair, 0 (each its own, default) to 1 (one detector and one filter). */
    void setLink(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        link_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @return The detail, in octaves. */
    [[nodiscard]] T getDetail() const noexcept { return detail_.load(std::memory_order_relaxed); }
    /** @brief Points of the display curve, evenly in log frequency from 20 Hz to 20 kHz. */
    static constexpr int kDisplayPoints = SpectralReadout<T>::kPoints;
    /** @return The frequency of a display point, in hertz. */
    [[nodiscard]] static double getDisplayFrequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }
    /** @return The gain applied to the first channel at a display point, in dB (any thread). */
    [[nodiscard]] T getDisplayGain(int point) const noexcept
    {
        return stftReadout_.gainPoint(point, [this](int bin) noexcept { return shownGains_[static_cast<size_t>(bin)].load(std::memory_order_relaxed); });
    }
    [[nodiscard]] T getGainRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getRemovalListen() const noexcept { return removalListen_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getFilterMode() const noexcept { return filterMode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLink() const noexcept { return link_.load(std::memory_order_relaxed); }

    [[nodiscard]] bool getLearning() const noexcept { return learning_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getReduction() const noexcept { return reduction_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }

    /** @brief Latency in samples (the STFT pipeline's); 0 before prepare()
     *  succeeds, while the instance passes audio through. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return prepared_.load(std::memory_order_relaxed) ? stft_.getLatency() : 0;
    }

    /** @brief Serializes the parameter state (the learned profile is material-
     *  dependent content, not a preset, and is intentionally not included). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DNSE"), 2);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("reduction", static_cast<float>(reduction_.load(std::memory_order_relaxed)));
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        w.write("detailOctaves", static_cast<float>(getDetail()));
        w.write("release", static_cast<float>(getGainRelease()));
        w.write("filterMode", getFilterMode());
        w.write("link", static_cast<float>(getLink()));
        sensitivity_.writeState(w, "sens");
        return w.blob();
    }

    /**
     * @brief Restores parameters from a blob (tolerant; rejects foreign ids).
     *        The smoothing of the first version, which was in bins, becomes
     *        the band as wide at 1 kHz.
     */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DNSE")) return false;
        setReduction(static_cast<T>(r.read("reduction", 18.0f)));
        setThreshold(static_cast<T>(r.read("threshold", 2.0f)));
        if (r.processorVersion() < 2)
        {
            // A pole a both ways spreads sqrt(2a) / (1 - a) bins, a box as wide sqrt(12) of that.
            const double a = std::clamp(static_cast<double>(r.read("smoothing", 0.0f)), 0.0, 0.95);
            const double wide = std::sqrt(24.0 * a) / (1.0 - a) * 48000.0 / 2048.0;
            setDetail(static_cast<T>(std::log2(1.0 + wide / 1000.0)));
        }
        else setDetail(static_cast<T>(r.read("detailOctaves", 0.0f)));
        setGainRelease(static_cast<T>(r.read("release", 0.0f)));
        setFilterMode(r.read("filterMode", false));
        setLink(static_cast<T>(r.read("link", 0.0f)));
        sensitivity_.readState(r, "sens");
        weightDirty_.store(true, std::memory_order_release);
        return true;
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Processes a block in-place. Pass-through until prepare() succeeds. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;

        const bool learning = learning_.load(std::memory_order_relaxed);
        const float floorGain = std::pow(
            10.0f, -static_cast<float>(reduction_.load(std::memory_order_relaxed)) / 20.0f);
        const float thresh = static_cast<float>(threshold_.load(std::memory_order_relaxed));

        // The STFT processes min(buffer, prepared) channels.
        const int nChEff = std::min(buffer.getNumChannels(), numChannels_);
        const int samples = buffer.getNumSamples();
        if (nChEff <= 0 || samples <= 0) return;

        Now now;
        now.channels = nChEff;
        now.learning = learning;
        now.floorGain = floorGain;
        // Over-subtraction: the learned noise power is scaled by threshold^2.
        now.overSub = thresh * thresh;
        now.reductionDb = static_cast<float>(reduction_.load(std::memory_order_relaxed));
        now.removal = removalListen_.load(std::memory_order_relaxed);
        const bool spreadsNothing = filterMode_.load(std::memory_order_relaxed);
        now.link = spreadsNothing && nChEff == 2 ? static_cast<float>(link_.load(std::memory_order_relaxed)) : 0.0f;

        refreshOptions();
        if (spreadsNothing && !shortRuns_)
        {
            // Switched on while playing: nothing of it until it has seen the stream.
            resetShort(stft_.getFFTSize() + kPastFrames * shortHop_);
            shortRuns_ = true;
        }
        if (!shortRuns_)
        {
            processLong(buffer, now);
            return;
        }
        // Switched off, it stays until what it is playing is over.
        if (!spreadsNothing && hold_ <= 0 && ramp_ <= 0)
        {
            shortRuns_ = false;
            processLong(buffer, now);
            return;
        }
        for (int done = 0; done < samples;)
        {
            const int count = std::min(samples - done, shortHop_ - shortPhase_);
            playAhead(buffer.getSubView(done, count), now, spreadsNothing);
            done += count;
            shortPhase_ += count;
            if (shortPhase_ == shortHop_) shortPhase_ = 0;
        }
    }

private:
    /** @brief Picks up the options and rebuilds what depends on them. */
    void refreshOptions() noexcept
    {
        const double width = static_cast<double>(detail_.load(std::memory_order_relaxed));
        if (width != bandsFor_)
        {
            bandsFor_ = width;
            const double share = std::exp2(0.5 * width) - std::exp2(-0.5 * width);
            stftBands_.lay(numBins_, share);
            shortBands_.lay(shortBins_, share);
            noiseHolds_ = false;
            shortNoiseHolds_ = false;
        }
        releaseNow_ = static_cast<float>(releaseMs_.load(std::memory_order_relaxed));
        if (weightDirty_.exchange(false, std::memory_order_acquire))
        {
            weighted_ = sensitivity_.hasBands();
            const auto response = sensitivity_.getResponse();
            const double binHz = sampleRate_ / static_cast<double>(stft_.getFFTSize());
            for (int k = 0; k < numBins_; ++k)
                weight_[static_cast<size_t>(k)] = weighted_
                    ? static_cast<float>(std::clamp(1.0 + response.magnitudeDb(std::max(1.0, k * binHz)) / 12.0, 0.0, 2.0))
                    : 1.0f;
            const double shortHz = sampleRate_ / static_cast<double>(shortSize_);
            for (int k = 0; k < shortBins_; ++k)
                shortWeight_[static_cast<size_t>(k)] = weighted_
                    ? static_cast<float>(std::clamp(1.0 + response.magnitudeDb(std::max(1.0, k * shortHz)) / 12.0, 0.0, 2.0))
                    : 1.0f;
        }
        if (releaseNow_ != releaseFor_)
        {
            releaseFor_ = releaseNow_;
            const auto fill = [&](std::vector<float>& r, double binHz, double hop) {
                for (size_t k = 0; k < r.size(); ++k)
                {
                    const double f = std::max(20.0, static_cast<double>(k) * binHz);
                    const double tau = 0.001 * static_cast<double>(releaseNow_) * std::sqrt(1000.0 / f);
                    r[k] = releaseNow_ > 0.0f ? static_cast<float>(std::exp(-hop / std::max(tau, 1e-6))) : 0.0f;
                }
            };
            // One hop of either: a part of the transform.
            fill(release_, sampleRate_ / static_cast<double>(stft_.getFFTSize()),
                 static_cast<double>(stft_.getFFTSize()) / (dense_ ? static_cast<double>(kDenseFrames) : 4.0) / sampleRate_);
            fill(shortRelease_, sampleRate_ / static_cast<double>(shortSize_), static_cast<double>(shortHop_) / sampleRate_);
        }
    }

    /**
     * @brief The gain of one bin (the decision-directed Wiener rule) with
     *        its floor.
     */
    static float binGain(float power, float noise, float prevClean, float overSub, float floorGain, float alpha) noexcept
    {
        float gain = 1.0f;
        const float lambda = overSub * noise;
        if (lambda > 0.0f)
        {
            const float post = power / lambda;
            const float prio = std::max(alpha * prevClean / lambda
                                        + (1.0f - alpha) * std::max(post - 1.0f, 0.0f),
                                        kMinPrioriSnr);
            gain = std::max(prio / (1.0f + prio), floorGain);
        }
        return gain;
    }

    /// The band of each bin at the detail, on the bins of one path.
    struct Bands
    {
        struct Edge
        {
            double part = 0.0;
            int bin = 0;
        };

        void prepare(int bins)
        {
            edges_.assign(2 * static_cast<size_t>(bins), Edge {});
            scale_.assign(static_cast<size_t>(bins), 1.0);
            first_ = bins;
        }

        /** @brief The bands as wide as `share` of their centres, none narrower than a bin. */
        void lay(int bins, double share) noexcept
        {
            first_ = share * static_cast<double>(bins - 1) > 1.0 ? static_cast<int>(1.0 / share) + 1 : bins;
            for (int k = first_; k < bins; ++k)
            {
                const double middle = k + 0.5, half = 0.5 * k * share;
                const double from = std::max(0.0, middle - half), to = std::min(static_cast<double>(bins), middle + half);
                const auto u = static_cast<size_t>(k);
                edges_[2 * u] = place(from, bins);
                edges_[2 * u + 1] = place(to, bins);
                scale_[u] = 1.0 / (to - from);
            }
        }

        [[nodiscard]] static Edge place(double at, int bins) noexcept
        {
            const int bin = std::min(static_cast<int>(at), bins - 1);
            return Edge { at - bin, bin };
        }

        /** @brief Each bin's mean over its band, in place. */
        void read(float* v, int bins, double* sum) const noexcept
        {
            if (first_ >= bins) return;
            double total = 0.0;
            for (int k = 0; k < bins; ++k)
            {
                sum[k] = total;
                if (std::isfinite(v[k])) total += static_cast<double>(v[k]);
            }
            sum[bins] = total;
            for (int k = first_; k < bins; ++k)
            {
                const auto u = static_cast<size_t>(k);
                const Edge a = edges_[2 * u], b = edges_[2 * u + 1];
                const double low = sum[a.bin] + a.part * (sum[a.bin + 1] - sum[a.bin]);
                const double high = sum[b.bin] + b.part * (sum[b.bin + 1] - sum[b.bin]);
                v[k] = static_cast<float>(std::max(0.0, (high - low) * scale_[u]));
            }
        }

        std::vector<Edge> edges_;
        std::vector<double> scale_;
        int first_ = 0;   ///< The bins under it are bands by themselves.
    };

    /** @brief Every half transform: some 47 drawings a second at 2048, as often as a screen shows. */
    [[nodiscard]] bool isDisplayDue() noexcept
    {
        const bool due = displayHop_ == 0;
        if (++displayHop_ >= displayEvery_) displayHop_ = 0;
        return due;
    }

    /** @brief The gains of the first channel for a display: the points are its reader's to make. */
    void publish(int n) noexcept
    {
        if (n != static_cast<int>(shownGains_.size())) return;
        for (int k = 0; k < n; ++k) shownGains_[static_cast<size_t>(k)].store(gains_[static_cast<size_t>(k)], std::memory_order_relaxed);
    }

    /** @brief The release on a gain curve. */
    void shapeGains(float* g, int n, std::vector<float>& held, const std::vector<float>& release) noexcept
    {
        if (releaseNow_ > 0.0f)
            for (int k = 0; k < n; ++k)
            {
                float& h = held[static_cast<size_t>(k)];
                h = g[k] < h ? g[k] : g[k] + release[static_cast<size_t>(k)] * (h - g[k]);
                g[k] = h;
            }
    }

    /** @brief The settings of a block. */
    struct Now
    {
        int channels = 0;
        bool learning = false, removal = false;
        float floorGain = 1.0f, overSub = 1.0f, reductionDb = 0.0f, link = 0.0f;
    };

    /** @brief What one kind of frame keeps. */
    struct Frames
    {
        std::vector<std::vector<float>>& clean;
        std::vector<std::vector<float>>& held;
        std::array<std::vector<float>, 2>& pair;
        const std::vector<float>& weight;
        const std::vector<float>& release;
        const Bands& bands;
        float* noise;
        float alpha;
    };

    /** @brief The power of a bin, 0 where it is not finite. */
    static float powerOf(const T* bins, int k) noexcept
    {
        const float re = static_cast<float>(bins[2 * k]), im = static_cast<float>(bins[2 * k + 1]);
        const float p = re * re + im * im;
        return std::isfinite(p) ? p : 0.0f;
    }

    /** @brief Both channels' powers of a hop, for the link. */
    static void observePair(int ch, const T* bins, int count, const Now& now, std::array<std::vector<float>, 2>& pair) noexcept
    {
        if (now.link <= 0.0f || ch >= 2) return;
        float* p = pair[static_cast<size_t>(ch)].data();
        for (int k = 0; k < count; ++k) p[k] = powerOf(bins, k);
    }

    /** @brief The rule on one frame of a channel, in place (either kind of frame). */
    void reduceFrame(int ch, T* bins, int count, const Now& now, const Frames& f) noexcept
    {
        const auto c = static_cast<size_t>(ch);
        float* power = detect_.data();
        if (now.link > 0.0f && ch < 2)
        {
            // Read part of the way to the mean of the pair, in this same hop;
            // at 1 one reading, the same to the bit on both.
            const float* own = f.pair[c].data();
            const float* other = f.pair[1 - c].data();
            const float part = 0.5f * now.link;
            if (now.link >= 1.0f)
                for (int k = 0; k < count; ++k) power[k] = 0.5f * (own[k] + other[k]);
            else
                for (int k = 0; k < count; ++k) power[k] = own[k] + part * (other[k] - own[k]);
        }
        else
            for (int k = 0; k < count; ++k) power[k] = powerOf(bins, k);
        f.bands.read(power, count, sum_.data());

        auto& clean = f.clean[c];
        for (int k = 0; k < count; ++k)
        {
            const auto u = static_cast<size_t>(k);
            const float fl = weighted_ ? std::pow(10.0f, -now.reductionDb * f.weight[u] / 20.0f) : now.floorGain;
            gains_[u] = binGain(power[k], f.noise[k], clean[u], now.overSub, fl, f.alpha);
        }
        shapeGains(gains_.data(), count, f.held[c], f.release);
        for (int k = 0; k < count; ++k)
        {
            const auto u = static_cast<size_t>(k);
            float& g = gains_[u];
            const float re = static_cast<float>(bins[2 * k]), im = static_cast<float>(bins[2 * k + 1]);
            // The gain memory is recursive: a NaN in it would stay for the whole stream.
            if (!std::isfinite(re * re + im * im))
            {
                g = 0.0f;
                clean[u] = 0.0f;
                bins[2 * k] = T(0);
                bins[2 * k + 1] = T(0);
                continue;
            }
            clean[u] = g * g * power[k];
            const float applied = now.removal ? 1.0f - g : g;
            bins[2 * k] = static_cast<T>(re * applied);
            bins[2 * k + 1] = static_cast<T>(im * applied);
        }
    }

    /**
     * @brief The rule on a long frame with every option at rest, in one pass
     *        over the bins: what reduceFrame() gives there, to the bit.
     */
    template <bool Learning, bool Shown>
    void reduceFramePlain(int ch, T* bins, int count, const Now& now, float learnRate) noexcept
    {
        float* clean = cleanPower_[static_cast<size_t>(ch)].data();
        float* learned = profile_.data();
        std::atomic<float>* shown = shownGains_.data();
        const float floorGain = now.floorGain, overSub = now.overSub, alpha = longAlpha_;
        for (int k = 0; k < count; ++k)
        {
            const float re = static_cast<float>(bins[2 * k]), im = static_cast<float>(bins[2 * k + 1]);
            const float power = re * re + im * im;
            if (!std::isfinite(power))
            {
                if constexpr (Shown) shown[k].store(0.0f, std::memory_order_relaxed);
                clean[k] = 0.0f;
                bins[2 * k] = T(0);
                bins[2 * k + 1] = T(0);
                continue;
            }
            if constexpr (Learning) learned[k] += learnRate * (power - learned[k]);
            const float g = binGain(power, learned[k], clean[k], overSub, floorGain, alpha);
            if constexpr (Shown) shown[k].store(g, std::memory_order_relaxed);
            clean[k] = g * g * power;
            bins[2 * k] = static_cast<T>(re * g);
            bins[2 * k + 1] = static_cast<T>(im * g);
        }
    }

    /** @brief The long frames: the default path, and all of the filter mode but at attacks. */
    void processLong(AudioBufferView<T> buffer, const Now& now) noexcept
    {
        // Nothing to read in bands or by the pair, and nothing to shape a gain.
        const bool plain = now.link <= 0.0f && !now.removal && !weighted_ && !(releaseNow_ > 0.0f)
                           && stftBands_.first_ >= numBins_;
        const auto process = [&](int ch, T* bins, int count) noexcept
        {
            if (plain)
            {
                // The gains go to the display only when one is due, as they are made.
                const bool shown = ch == 0 && isDisplayDue() && count == static_cast<int>(shownGains_.size());
                if (now.learning)
                {
                    learnFrames_ = std::min(learnFrames_ + 1, learnMost_);
                    const float rate = 1.0f / static_cast<float>(learnFrames_);
                    if (shown) reduceFramePlain<true, true>(ch, bins, count, now, rate);
                    else reduceFramePlain<true, false>(ch, bins, count, now, rate);
                    noiseHolds_ = false;
                    shortNoiseHolds_ = false;
                }
                else if (shown) reduceFramePlain<false, true>(ch, bins, count, now, 0.0f);
                else reduceFramePlain<false, false>(ch, bins, count, now, 0.0f);
                return;
            }
            if (now.learning)
            {
                // The running mean of the noise power in each bin, an
                // exponential one once kMaxLearnFrames are in.
                learnFrames_ = std::min(learnFrames_ + 1, learnMost_);
                const float rate = 1.0f / static_cast<float>(learnFrames_);
                for (int k = 0; k < count; ++k)
                {
                    const float re = static_cast<float>(bins[2 * k]), im = static_cast<float>(bins[2 * k + 1]);
                    const float p = re * re + im * im;
                    auto& learned = profile_[static_cast<size_t>(k)];
                    if (std::isfinite(p)) learned += rate * (p - learned);
                }
                shortNoiseHolds_ = false;
            }
            // The profile in the bands holds until it or they move.
            if (now.learning || !noiseHolds_)
            {
                noiseHolds_ = true;
                std::copy_n(profile_.data(), count, noiseHere_.data());
                stftBands_.read(noiseHere_.data(), count, sum_.data());
            }
            const Frames frames { cleanPower_, heldGain_, pairPower_, weight_, release_, stftBands_, noiseHere_.data(), longAlpha_ };
            reduceFrame(ch, bins, count, now, frames);
            if (ch == 0 && isDisplayDue()) publish(count);
        };
        if (now.link > 0.0f)
        {
            stft_.processBlock(buffer, [&](int ch, const T* bins, int count) noexcept { observePair(ch, bins, count, now, pairPower_); }, process);
            return;
        }
        // Channel by channel, through the one spectrum: the callback runs once
        // per channel processed and hop, in channel order.
        int call = 0;
        stft_.processBlock(buffer, [&](T* bins, int count) noexcept
        {
            process(call % now.channels, bins, count);
            ++call;
        });
    }

    // -- The frames that play where an attack is ahead -------------------------

    void prepareShort(const AudioSpec& spec)
    {
        const int size = stft_.getFFTSize();
        // About 5 ms, and shorter than the long frames.
        const double wanted = std::min(kShortSeconds * spec.sampleRate, 0.5 * static_cast<double>(size));
        int n = std::min(16, size);
        while (n * 1.45 < wanted) n *= 2;
        short_.prepare(spec, n, n / 4);
        shortSize_ = short_.getFFTSize();
        shortHop_ = short_.getHopSize();
        shortBins_ = short_.getNumBins();
        const int longHop = dense_ ? size / 4 : stft_.getHopSize();   // the hop the memory is stated for
        shortAlpha_ = static_cast<float>(std::pow(static_cast<double>(kDDAlpha), static_cast<double>(shortHop_) / static_cast<double>(longHop)));
        fadeIn_ = std::max(1, shortSize_);
        after_ = std::min(size, static_cast<int>(kAfterSeconds * spec.sampleRate));
        lag_ = size - shortSize_;

        const auto channels = static_cast<size_t>(numChannels_);
        const auto bins = static_cast<size_t>(shortBins_);
        shortClean_.assign(channels, std::vector<float>(bins, 0.0f));
        shortHeld_.assign(channels, std::vector<float>(bins, 1.0f));
        shortDelay_.assign(channels, std::vector<T>(static_cast<size_t>(lag_), T(0)));
        shortAudio_.resize(std::min(numChannels_, kMostChannels), shortHop_);
        for (auto& v : shortPair_) v.assign(bins, 0.0f);
        shortNoise_.assign(bins, 0.0f);
        shortNoiseRead_.assign(bins, 0.0f);
        past_.assign(channels * kAttackBands * static_cast<size_t>(kPastFrames), 0.0f);

        // What each of the four bands of the detector holds.
        const double edges[] = { 750.0, 3000.0, 8000.0 };
        bandEdge_[0] = 0;
        for (int b = 1; b < kAttackBands; ++b)
            bandEdge_[static_cast<size_t>(b)] = std::clamp(static_cast<int>(std::lround(edges[b - 1] * shortSize_ / spec.sampleRate)),
                                                           bandEdge_[static_cast<size_t>(b - 1)], shortBins_);
        bandEdge_[static_cast<size_t>(kAttackBands)] = shortBins_;

        // A short bin of the profile: the long bins under the short window's lobe.
        ratio_ = size / shortSize_;
        spreadReach_ = std::min(kLobeBins * ratio_, size / 2);
        FFTReal<double> fft(static_cast<size_t>(size));
        std::vector<double> time(static_cast<size_t>(size), 0.0), spectrum(static_cast<size_t>(size + 2), 0.0);
        WindowFunctions<double>::hann(time.data(), shortSize_, true);
        for (int i = 0; i < shortSize_; ++i) time[static_cast<size_t>(i)] = std::sqrt(time[static_cast<size_t>(i)]);
        fft.forward(time.data(), spectrum.data());
        spread_.assign(static_cast<size_t>(spreadReach_ + 1), 0.0f);
        double whole = 0.0;
        for (int d = 0; d <= spreadReach_; ++d)
        {
            const double re = spectrum[static_cast<size_t>(2 * d)], im = spectrum[static_cast<size_t>(2 * d + 1)];
            whole += (d == 0 ? 1.0 : 2.0) * (re * re + im * im);
        }
        // White noise reads by the energy of each window.
        const double scale = static_cast<double>(shortSize_) / (static_cast<double>(size) * whole);
        for (int d = 0; d <= spreadReach_; ++d)
        {
            const double re = spectrum[static_cast<size_t>(2 * d)], im = spectrum[static_cast<size_t>(2 * d + 1)];
            spread_[static_cast<size_t>(d)] = static_cast<float>(scale * (re * re + im * im));
        }
        shortNoiseHolds_ = false;
    }

    /** @param unseen Samples before an attack counts (those it has not seen). */
    void resetShort(int unseen) noexcept
    {
        short_.reset();
        for (auto& c : shortClean_) std::fill(c.begin(), c.end(), 0.0f);
        for (auto& c : shortHeld_) std::fill(c.begin(), c.end(), 1.0f);
        for (auto& d : shortDelay_) std::fill(d.begin(), d.end(), T(0));
        std::fill(past_.begin(), past_.end(), 0.0f);
        delayAt_ = 0;
        pastAt_ = 0;
        shortPhase_ = 0;
        hold_ = 0;
        ramp_ = 0;
        unseen_ = unseen;
        attack_ = false;
    }

    /** @brief Whether an attack starts in a short frame: a band over the noise, 6 dB over the most of its last 100 ms. */
    [[nodiscard]] bool startsAttack(int ch, const T* bins) noexcept
    {
        float* past = past_.data() + static_cast<size_t>(ch) * static_cast<size_t>(kAttackBands * kPastFrames);
        bool starts = false;
        for (int b = 0; b < kAttackBands; ++b, past += kPastFrames)
        {
            float energy = 0.0f, noise = 0.0f;
            for (int k = bandEdge_[static_cast<size_t>(b)]; k < bandEdge_[static_cast<size_t>(b + 1)]; ++k)
            {
                energy += powerOf(bins, k);
                noise += shortNoise_[static_cast<size_t>(k)];
            }
            if (energy > kOverNoise * noise)
            {
                // Not the two frames just before: they overlap this one.
                float most = 0.0f;
                for (int age = 3; age <= kPastFrames; ++age)
                {
                    const int at = pastAt_ - age < 0 ? pastAt_ - age + kPastFrames : pastAt_ - age;
                    most = std::max(most, past[at]);
                    if (energy <= kRise * most) break;
                }
                starts = starts || energy > kRise * most;
            }
            past[pastAt_] = energy;
        }
        return starts;
    }

    /** @brief A chunk up to a short hop: the long frames, and the short ones where an attack is ahead. */
    void playAhead(AudioBufferView<T> chunk, const Now& now, bool seeks) noexcept
    {
        const int count = chunk.getNumSamples(), channels = std::min(now.channels, shortAudio_.getNumChannels());
        T* of[kMostChannels] {};
        for (int ch = 0; ch < channels; ++ch)
        {
            of[ch] = shortAudio_.getChannel(ch);
            std::copy_n(chunk.getChannel(ch), count, of[ch]);
        }
        processLong(chunk, now);

        const auto observe = [&](int ch, const T* bins, int bins2) noexcept
        {
            if (!shortNoiseHolds_) readShortNoise();
            observePair(ch, bins, bins2, now, shortPair_);
            attack_ = startsAttack(ch, bins) || attack_;
            if (ch == channels - 1) pastAt_ = (pastAt_ + 1) % kPastFrames;
        };
        const auto process = [&](int ch, T* bins, int bins2) noexcept
        {
            const Frames frames { shortClean_, shortHeld_, shortPair_, shortWeight_, shortRelease_, shortBands_, shortNoiseRead_.data(), shortAlpha_ };
            reduceFrame(ch, bins, bins2, now, frames);
        };
        // What plays now was settled by the hops before this chunk's.
        const int hold = hold_, ramp = ramp_;
        short_.processBlock(AudioBufferView<T>(of, channels, count), observe, process);

        int at = delayAt_;
        for (int ch = 0; ch < channels; ++ch)
        {
            T* in = shortAudio_.getChannel(ch);
            T* line = shortDelay_[static_cast<size_t>(ch)].data();
            at = delayAt_;
            for (int i = 0; i < count && lag_ > 0; ++i)
            {
                const T late = line[at];
                line[at] = in[i];
                in[i] = late;
                if (++at == lag_) at = 0;
            }
        }
        delayAt_ = at;

        if (hold > 0 || ramp > 0)
        {
            int held = hold, r = ramp;
            for (int i = 0; i < count; ++i)
            {
                if (held > 0)
                {
                    --held;
                    r = std::min(r + 1, fadeIn_);
                }
                else if (r > 0) --r;
                if (r <= 0) continue;
                const T share = r >= fadeIn_ ? T(1)
                    : T(0.5) - T(0.5) * std::cos(std::numbers::pi_v<T> * static_cast<T>(r) / static_cast<T>(fadeIn_));
                for (int ch = 0; ch < channels; ++ch)
                {
                    T& out = chunk.getChannel(ch)[i];
                    out += share * (shortAudio_.getChannel(ch)[i] - out);
                }
            }
            hold_ = held;
            ramp_ = r;
        }
        if (unseen_ > 0) unseen_ -= count;
        if (attack_)
        {
            attack_ = false;
            // Its start is a hop back, and is heard the latency later.
            if (seeks && unseen_ <= 0) hold_ = std::max(hold_, stft_.getFFTSize() - shortHop_ + after_);
        }
    }

    /** @brief The learned profile as the short frames read it. */
    void readShortNoise() noexcept
    {
        shortNoiseHolds_ = true;
        const int last = numBins_ - 1;
        for (int j = 0; j < shortBins_; ++j)
        {
            const int centre = j * ratio_;
            float sum = 0.0f;
            for (int d = -spreadReach_; d <= spreadReach_; ++d)
            {
                int k = centre + d;
                if (k < 0) k = -k;
                if (k > last) k = 2 * last - k;
                sum += spread_[static_cast<size_t>(d < 0 ? -d : d)] * profile_[static_cast<size_t>(std::clamp(k, 0, last))];
            }
            shortNoise_[static_cast<size_t>(j)] = sum;
        }
        std::copy(shortNoise_.begin(), shortNoise_.end(), shortNoiseRead_.begin());
        shortBands_.read(shortNoiseRead_.data(), shortBins_, sum_.data());
    }

    static constexpr int kMostChannels = 16;          ///< A view's.
    static constexpr int kAttackBands = 4;            ///< The detector's.
    static constexpr int kPastFrames = 72;            ///< The short hops an attack rises over: about 100 ms.
    static constexpr int kLobeBins = 4;               ///< The short window's lobe and first side lobes, in its bins.
    static constexpr float kRise = 4.0f;              ///< An attack is this far over them, in power.
    static constexpr float kOverNoise = 8.0f;         ///< And over the learned noise.
    static constexpr double kShortSeconds = 0.00533;  ///< A short frame.
    static constexpr double kAfterSeconds = 0.0107;   ///< How long they stay after an attack.

    SpectralProcessor<T> stft_, short_;
    KeyEqualizer<T, 2> sensitivity_;
    int numChannels_ = 0;
    int numBins_ = 0;
    int shortSize_ = 256, shortHop_ = 64, shortBins_ = 129, lag_ = 0, delayAt_ = 0, shortPhase_ = 0;
    int fadeIn_ = 1, after_ = 0, hold_ = 0, ramp_ = 0, unseen_ = 0, pastAt_ = 0, ratio_ = 1, spreadReach_ = 0;
    float shortAlpha_ = 0.98f;
    bool shortRuns_ = false, attack_ = false, shortNoiseHolds_ = false, noiseHolds_ = false;
    std::array<int, static_cast<size_t>(kAttackBands + 1)> bandEdge_ {};
    std::array<std::vector<float>, 2> pairPower_, shortPair_;
    std::vector<std::vector<float>> shortClean_, shortHeld_;
    std::vector<std::vector<T>> shortDelay_;
    AudioBuffer<T, kMostChannels> shortAudio_;
    std::vector<float> shortNoise_, shortNoiseRead_, shortWeight_, shortRelease_, spread_, past_;
    double sampleRate_ = 48000.0;
    std::atomic<bool> prepared_ { false };
    std::atomic<T> detail_ { T(0) }, releaseMs_ { T(0) }, link_ { T(0) };
    std::atomic<bool> removalListen_ { false }, filterMode_ { false }, weightDirty_ { true };
    bool weighted_ = false;
    float releaseNow_ = 0.0f, releaseFor_ = -1.0f;
    Bands stftBands_, shortBands_;
    double bandsFor_ = -1.0;             ///< The detail they are laid for.
    int displayHop_ = 0, displayEvery_ = 2;
    bool dense_ = false;                 ///< A transform longer than the rate's own.
    float longAlpha_ = 0.98f;            ///< kDDAlpha at the spacing of the long frames.
    int learnMost_ = 4096;               ///< kMaxLearnFrames at that spacing.
    std::vector<std::vector<float>> heldGain_;
    std::vector<float> gains_, weight_, release_, detect_, noiseHere_;
    std::vector<double> sum_;
    SpectralReadout<T> stftReadout_;     ///< The map from bins to the display's points.
    std::vector<std::atomic<float>> shownGains_;   ///< The first channel's gains of the last frame shown.

    static constexpr float kDDAlpha = 0.98f;           ///< Decision-directed weight per hop.
    static constexpr float kMinPrioriSnr = 0.003162f;  ///< -25 dB a priori SNR floor.
    static constexpr int kMaxLearnFrames = 4096;       ///< Averaging horizon of the profile.
    static constexpr int kDenseFrames = 16;            ///< Frames in a transform longer than the rate's own.

    std::vector<float> profile_;                  ///< Learned mean noise power per bin.
    int learnFrames_ = 0;                         ///< Frames in the running mean.
    std::vector<std::vector<float>> cleanPower_;  ///< Per-channel previous clean power estimate.

    std::atomic<bool> learning_ { false };
    std::atomic<T> reduction_ { T(18) };
    std::atomic<T> threshold_ { T(2) };
};

} // namespace dspark
