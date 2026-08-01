// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file OnsetDetector.h
 * @brief Causal SuperFlux onset detector with a Boeck-2012 adaptive peak picker.
 *
 * Implements the SuperFlux onset-detection function (Boeck & Widmer, DAFx-13):
 * a log-frequency triangular filterbank magnitude spectrogram, a frequency
 * maximum filter that suppresses vibrato/tremolo false positives, and a
 * half-wave-rectified spectral flux to the mu-th previous frame. The onset
 * strength envelope (ODF) is peak-picked with the online-capable rule of
 * Boeck, Krebs & Schedl (ISMIR 2012) -- the same recipe used by
 * librosa.util.peak_pick. Two simpler ODFs are also provided:
 * plain SpectralFlux and a rectified ComplexDomain function (Dixon, DAFx-06).
 *
 * The STFT front-end is built directly on FFTReal + WindowFunctions with a
 * mirrored analysis ring (the same technique as PitchDetector), which gives
 * exact, block-size-independent control over frame timing -- a precondition
 * for the deterministic latency contract below. It is the shared onset
 * front-end consumed by BeatTracker (ADR-009).
 *
 * Latency (ONE definition, D-2). Frame N (default 2048, Hann), hop =
 * round(fs/200) (221 samples at 44.1 kHz, 240 at 48 kHz; ~5 ms). The detector
 * reports every onset at a
 * single fixed causal reporting latency
 *
 *     L = fftSize + hop        (getLatencySamples()).
 *
 * At 44.1 kHz with the defaults this is 2048 + 221 = 2269 samples (~51.4 ms;
 * 2048 + 240 = 2288 at 48 kHz).
 * The onsetDetected() latch asserts exactly L samples after the onset's
 * reference sample (the analysis-frame centre that localises the transient).
 * Detection completes internally earlier (~fftSize/2 + hop); events are held
 * and released at the fixed offset L so a caller sees one deterministic,
 * block-size-independent latency regardless of where in a frame the onset
 * falls (see OA-6). The adaptive threshold's pre_avg look-back (100 ms) is a
 * backward-looking warm-up, NOT part of L: it is history the moving mean needs
 * before its first valid decision and adds zero per-onset reporting latency.
 *
 * Picker defaults (D-2): pre_max = post_max = 30 ms, pre_avg = 100 ms,
 * post_avg = 70 ms (offline only), combination width = 30 ms. In the causal
 * streaming path post_avg = 0 and post_max is one hop -- the single-frame
 * confirmation that a candidate is a maximum, which is exactly the +hop term
 * of L. detectOffline() uses the symmetric (post_* > 0) picker.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with the audio path).
 * - processBlock() / pushSamples() / reset(): audio thread (stream owner);
 *   reset() is not concurrent with pushSamples().
 * - onsetDetected() / getOnsetStrength() / getLastOnsetSample() /
 *   getLatencySamples(): any thread, lock-free (published as atomics; the
 *   words are independent, so a reader overlapping a release may pair a
 *   fresh reference sample with the previous strength -- benign for
 *   triggering/metering).
 * - setMethod() / setThreshold() / setAdaptiveWhitening(): control thread
 *   (independent single-word relaxed atomics; non-finite thresholds are
 *   ignored).
 * - detectOffline(): offline convenience (allocates); not an audio-thread call.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O, so it is unaffected by DSPARK_NO_FILE_IO.
 *
 * Dependencies: DspMath.h, FFT.h, WindowFunctions.h, AudioBuffer.h, AudioSpec.h.
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class OnsetDetector
 * @brief Causal SuperFlux onset detector with lock-free readout.
 *
 * Role: analysis readout. It consumes const audio (AudioBufferView<const T> or
 * a raw span) and never mutates it. All heap use happens in prepare(); the
 * audio path (processBlock/pushSamples) allocates nothing, takes no lock and
 * throws nothing.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class OnsetDetector final
{
public:
    /** @brief Onset-detection function family. Default SuperFlux. */
    enum class Method { SpectralFlux, ComplexDomain, SuperFlux };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Allocates all state and configures the STFT front-end.
     *
     * Not real-time safe (allocates). Any previous stream state is cleared.
     * An invalid spec (non-finite or non-positive sample rate) is ignored,
     * preserving the previous configuration.
     *
     * @param spec    Audio environment (only sampleRate is used; the detector
     *                is mono -- feed channel 0, or mix down before pushing).
     * @param fftSize Analysis frame size; rounded up to a power of two in
     *                [64, 1<<16]. Default 2048.
     * @param hop     Hop in samples; hop <= 0 selects round(fs/200) (~5 ms).
     *                Clamped to [1, fftSize].
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048, int hop = 0)
    {
        if (!(spec.sampleRate > 0.0) || !std::isfinite(spec.sampleRate))
            return;

        fft_.reset(); // gate OFF: the audio path is a no-op while rebuilding

        sampleRate_ = spec.sampleRate;

        int fs = std::clamp(fftSize, kMinFft, kMaxFft);
        int pow2 = kMinFft;
        while (pow2 < fs) pow2 <<= 1;
        fftSize_ = pow2;
        numBins_ = fftSize_ / 2 + 1;

        if (hop <= 0)
            hop_ = std::max(1, static_cast<int>(std::lround(sampleRate_ / 200.0)));
        else
            hop_ = hop;
        hop_ = std::clamp(hop_, 1, fftSize_);

        latencySamples_.store(static_cast<int64_t>(fftSize_) + hop_,
                              std::memory_order_relaxed);

        // Analysis-window group-delay compensation: half-wave-rectified spectral
        // flux peaks on the rising flank of a transient, ~kLocalizationLead*N
        // before the windowed-energy peak. Adding it back centres the reported
        // onset on the transient (calibrated on band-limited clicks; softer
        // onsets localise slightly late but stay within the acceptance window).
        localizationOffset_ = static_cast<int>(std::lround(kLocalizationLead
                                                           * static_cast<double>(fftSize_)));
        // Warm-up: suppress onsets until the analysis ring is fully primed so
        // the silence->first-input ramp cannot fire a spurious onset.
        primeFrames_ = fftSize_ / hop_ + 2;

        // Analysis window (periodic Hann) and its ring (mirrored for a
        // contiguous most-recent-fftSize window without a modulo).
        window_.assign(static_cast<size_t>(fftSize_), T(0));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        ring_.assign(static_cast<size_t>(fftSize_) * 2, T(0));

        fft_time_.assign(static_cast<size_t>(fftSize_), T(0));
        fft_spec_.assign(static_cast<size_t>(fftSize_) + 2, T(0));
        mag_.assign(static_cast<size_t>(numBins_), T(0));
        phase_.assign(static_cast<size_t>(numBins_), T(0));
        prevPhase_.assign(static_cast<size_t>(numBins_), T(0));
        prevPhase2_.assign(static_cast<size_t>(numBins_), T(0));
        prevMag_.assign(static_cast<size_t>(numBins_), T(0));
        whitenPeak_.assign(static_cast<size_t>(numBins_), T(0));

        buildFilterBank();

        // Band history for the SuperFlux flux-to-mu-th-previous-frame. mu is
        // one hop by construction (adjacent frames at 200 fps), so two frames
        // of history are enough; keep a small ring keyed on frame index.
        muFrames_ = 1;
        bandCur_.assign(static_cast<size_t>(numBands_), T(0));
        bandPrev_.assign(static_cast<size_t>(numBands_), T(0));
        bandMaxPrev_.assign(static_cast<size_t>(numBands_), T(0));

        // Peak-picker windows in frames (derived from the ms defaults, D-2).
        preMaxFrames_  = msToFrames(30.0);
        postMaxFrames_ = msToFrames(30.0);   // offline; causal uses 1
        preAvgFrames_  = msToFrames(100.0);
        postAvgFrames_ = msToFrames(70.0);   // offline; causal uses 0
        waitFrames_    = msToFrames(30.0);

        // Causal ODF history: enough for the pre_avg look-back plus the
        // single-frame causal confirmation.
        odfHistLen_ = std::max(preAvgFrames_, preMaxFrames_) + 4;
        odfHist_.assign(static_cast<size_t>(odfHistLen_), T(0));

        // Pending-onset ring: onsets are held from detection until their
        // release sample (reference + L). At most ceil(L/hop)+2 can be
        // in flight; size generously to a power-of-two-ish bound.
        const int maxPending = (fftSize_ + hop_) / hop_ + 4;
        pending_.assign(static_cast<size_t>(std::max(8, maxPending)),
                        PendingOnset{});
        pendingHead_ = 0;
        pendingCount_ = 0;

        resetState();

        method_.store(Method::SuperFlux, std::memory_order_relaxed);
        // A conservative default delta tuned against the D-6 synthetic corpus
        // (OA-1..OA-5). Callers can override per material.
        threshold_.store(kDefaultDelta, std::memory_order_relaxed);
        whitening_.store(false, std::memory_order_relaxed);

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize_)); // gate ON
    }

    /** @brief Selects the ODF family. Lock-free. */
    void setMethod(Method m) noexcept { method_.store(m, std::memory_order_relaxed); }

    /**
     * @brief Sets the adaptive peak-pick delta (margin above the moving mean).
     * Non-finite values are ignored; negative values are clamped to 0.
     */
    void setThreshold(T deltaAboveMean) noexcept
    {
        if (!std::isfinite(deltaAboveMean)) return;
        threshold_.store(std::max(T(0), deltaAboveMean), std::memory_order_relaxed);
    }

    /** @brief Enables Stowell-Plumbley adaptive whitening (default off). */
    void setAdaptiveWhitening(bool on) noexcept
    {
        whitening_.store(on, std::memory_order_relaxed);
    }

    // -- Audio path (causal, RT-safe) ---------------------------------------

    /**
     * @brief Feeds a mono block; reads channel 0 only. Const, never mutated.
     *
     * Lock-free and allocation-free. Safe no-op before prepare().
     */
    void processBlock(AudioBufferView<const T> in) noexcept
    {
        if (fft_ == nullptr || in.getNumChannels() < 1) return;
        const T* ch0 = in.getChannel(0);
        pushSamples(std::span<const T>(ch0, static_cast<size_t>(in.getNumSamples())));
    }

    /**
     * @brief Feeds a mono stream of samples. Lock-free, allocation-free.
     *
     * Onsets fire at the fixed reporting latency L = fftSize + hop after their
     * reference sample; onsetDetected() latches per processing call (see the
     * file header). Safe no-op before prepare().
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (fft_ == nullptr) return;

        bool firedThisCall = false;

        for (const T s : samples)
        {
            const T x = std::isfinite(s) ? s : T(0);

            ring_[static_cast<size_t>(writePos_)] = x;
            ring_[static_cast<size_t>(writePos_ + fftSize_)] = x;
            if (++writePos_ >= fftSize_) writePos_ = 0;

            ++totalSamples_;

            // Release any pending onset whose reporting sample has arrived.
            while (pendingCount_ > 0)
            {
                const PendingOnset& p = pending_[static_cast<size_t>(pendingHead_)];
                if (totalSamples_ - 1 >= p.reportSample)
                {
                    lastOnsetSample_.store(p.referenceSample, std::memory_order_relaxed);
                    onsetStrength_.store(p.strength, std::memory_order_relaxed);
                    firedThisCall = true;
                    pendingHead_ = (pendingHead_ + 1) % static_cast<int>(pending_.size());
                    --pendingCount_;
                }
                else break;
            }

            if (++hopCounter_ >= hop_)
            {
                hopCounter_ = 0;
                analyzeFrame();
            }
        }

        onsetLatched_.store(firedThisCall, std::memory_order_relaxed);
    }

    // -- Readout (lock-free) -------------------------------------------------

    /** @brief True if an onset was reported during the most recent call. */
    [[nodiscard]] bool onsetDetected() const noexcept
    {
        return onsetLatched_.load(std::memory_order_relaxed);
    }

    /** @brief Onset strength (ODF value) of the most recent reported onset. */
    [[nodiscard]] T getOnsetStrength() const noexcept
    {
        return onsetStrength_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reference sample index (frame centre) of the most recent onset.
     * The latch fires exactly getLatencySamples() samples after this index.
     */
    [[nodiscard]] int64_t getLastOnsetSample() const noexcept
    {
        return lastOnsetSample_.load(std::memory_order_relaxed);
    }

    /** @brief The single causal reporting latency, L = fftSize + hop (samples). */
    [[nodiscard]] int getLatencySamples() const noexcept
    {
        return static_cast<int>(latencySamples_.load(std::memory_order_relaxed));
    }

    [[nodiscard]] int getFftSize() const noexcept { return fftSize_; }
    [[nodiscard]] int getHopSize() const noexcept { return hop_; }
    [[nodiscard]] int getNumBands() const noexcept { return numBands_; }

    // -- Offline convenience -------------------------------------------------

    /**
     * @brief Offline detection over a whole mono buffer (channel 0).
     *
     * Runs the same ODF with the symmetric (post_max/post_avg > 0) picker for
     * slightly higher F, and returns onset sample positions (frame-centre
     * references, ascending). Allocates -- not an audio-thread call. Resets
     * the streaming state on entry.
     */
    std::vector<int64_t> detectOffline(AudioBufferView<const T> whole)
    {
        std::vector<int64_t> out;
        if (fft_ == nullptr || whole.getNumChannels() < 1) return out;

        const int n = whole.getNumSamples();
        const T* x = whole.getChannel(0);

        // Build the full ODF envelope (offline: allocation allowed).
        std::vector<T> odf;
        std::vector<int64_t> odfRef; // reference sample per frame
        odf.reserve(static_cast<size_t>(n / std::max(1, hop_) + 2));
        odfRef.reserve(odf.capacity());

        resetState();
        const Method m = method_.load(std::memory_order_relaxed);
        const bool whiten = whitening_.load(std::memory_order_relaxed);

        int64_t total = 0;
        int hopc = 0;
        for (int i = 0; i < n; ++i)
        {
            const T v = std::isfinite(x[i]) ? x[i] : T(0);
            ring_[static_cast<size_t>(writePos_)] = v;
            ring_[static_cast<size_t>(writePos_ + fftSize_)] = v;
            if (++writePos_ >= fftSize_) writePos_ = 0;
            ++total;
            if (++hopc >= hop_)
            {
                hopc = 0;
                const T value = computeOdf(m, whiten);
                odf.push_back(value);
                odfRef.push_back(referenceSample(total));
            }
        }

        // Symmetric peak-pick over the whole envelope.
        const T delta = threshold_.load(std::memory_order_relaxed);
        const int nf = static_cast<int>(odf.size());
        int64_t lastFrame = -kBig;
        for (int f = 0; f < nf; ++f)
        {
            if (f < primeFrames_) continue; // warm-up guard (ring priming)
            bool isMax = true;
            for (int j = f - preMaxFrames_; j <= f + postMaxFrames_; ++j)
            {
                if (j < 0 || j >= nf) continue;
                if (odf[static_cast<size_t>(j)] > odf[static_cast<size_t>(f)]) { isMax = false; break; }
            }
            if (!isMax) continue;

            T sum = T(0); int cnt = 0;
            for (int j = f - preAvgFrames_; j <= f + postAvgFrames_; ++j)
            {
                if (j < 0 || j >= nf) continue;
                sum += odf[static_cast<size_t>(j)]; ++cnt;
            }
            const T mean = (cnt > 0) ? sum / static_cast<T>(cnt) : T(0);
            if (odf[static_cast<size_t>(f)] < mean + delta) continue;
            if (f - lastFrame <= waitFrames_) continue;

            out.push_back(odfRef[static_cast<size_t>(f)]);
            lastFrame = f;
        }

        resetState();
        return out;
    }

    /** @brief Clears all streaming state. Not concurrent with pushSamples(). */
    void reset() noexcept { resetState(); }

private:
    // -- Constants -----------------------------------------------------------
    static constexpr int kMinFft = 64;
    static constexpr int kMaxFft = 1 << 16;
    static constexpr int64_t kBig = int64_t(1) << 60;
    static constexpr T kDefaultDelta = T(0.03);
    static constexpr double kLocalizationLead = 0.34; ///< Flux-to-energy lead (fraction of N).
    static constexpr double kFMin = 27.5;      ///< Filterbank low edge (Hz).
    static constexpr double kFMaxHz = 16000.0; ///< Filterbank high edge (Hz).
    static constexpr int kBandsPerOctave = 24; ///< Quarter-tone resolution.

    struct PendingOnset
    {
        int64_t referenceSample = 0; ///< Frame-centre reference (localisation).
        int64_t reportSample = 0;    ///< reference + L (latch release point).
        T strength = T(0);
    };

    // -- Frame timing --------------------------------------------------------

    /** @brief Reference sample (frame centre) of the frame that fires when
     *  totalSamples == firePos. The window covers [firePos-fftSize, firePos-1]. */
    [[nodiscard]] int64_t referenceSample(int64_t firePos) const noexcept
    {
        return firePos - static_cast<int64_t>(fftSize_) / 2
             + static_cast<int64_t>(localizationOffset_);
    }

    [[nodiscard]] int msToFrames(double ms) const noexcept
    {
        return std::max(1, static_cast<int>(std::lround(ms * 0.001 * sampleRate_
                                                        / static_cast<double>(hop_))));
    }

    // -- STFT + ODF ----------------------------------------------------------

    /** @brief Windows the current most-recent-fftSize ring window, FFTs it,
     *  and fills mag_/phase_. */
    void computeSpectrum() noexcept
    {
        const T* w = window_.data();
        const T* r = &ring_[static_cast<size_t>(writePos_)]; // oldest..newest, contiguous
        for (int k = 0; k < fftSize_; ++k)
            fft_time_[static_cast<size_t>(k)] = r[k] * w[k];

        fft_->forward(fft_time_.data(), fft_spec_.data());

        for (int k = 0; k < numBins_; ++k)
        {
            const T re = fft_spec_[static_cast<size_t>(2 * k)];
            const T im = fft_spec_[static_cast<size_t>(2 * k + 1)];
            mag_[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
            phase_[static_cast<size_t>(k)] = std::atan2(im, re);
        }
    }

    /** @brief Applies per-bin adaptive whitening (Stowell-Plumbley) to mag_. */
    void applyWhitening() noexcept
    {
        for (int k = 0; k < numBins_; ++k)
        {
            T& pk = whitenPeak_[static_cast<size_t>(k)];
            const T decayed = pk * kWhitenDecay;
            const T m = mag_[static_cast<size_t>(k)];
            pk = std::max({ m, kWhitenFloor, decayed });
            mag_[static_cast<size_t>(k)] = m / pk;
        }
    }

    /** @brief Computes the ODF value for the current frame and rotates the
     *  per-frame history buffers. Assumes computeSpectrum() was called. */
    T computeOdf(Method m, bool whiten) noexcept
    {
        computeSpectrum();
        if (whiten) applyWhitening();

        T odf = T(0);
        switch (m)
        {
            case Method::SpectralFlux:
            {
                for (int k = 0; k < numBins_; ++k)
                {
                    const T d = mag_[static_cast<size_t>(k)] - prevMag_[static_cast<size_t>(k)];
                    if (d > T(0)) odf += d;
                }
                odf /= static_cast<T>(numBins_);
                break;
            }
            case Method::ComplexDomain:
            {
                // Rectified complex-domain deviation (Dixon 2006): phase-predict
                // each bin, sum |X - Xhat| where magnitude increased.
                for (int k = 0; k < numBins_; ++k)
                {
                    const T target = princArg(T(2) * prevPhase_[static_cast<size_t>(k)]
                                              - prevPhase2_[static_cast<size_t>(k)]);
                    const T pm = prevMag_[static_cast<size_t>(k)];
                    const T cm = mag_[static_cast<size_t>(k)];
                    const T re = cm * std::cos(phase_[static_cast<size_t>(k)])
                               - pm * std::cos(target);
                    const T im = cm * std::sin(phase_[static_cast<size_t>(k)])
                               - pm * std::sin(target);
                    if (cm >= pm) odf += std::sqrt(re * re + im * im);
                }
                odf /= static_cast<T>(numBins_);
                break;
            }
            case Method::SuperFlux:
            {
                // Log-filtered magnitude bands, flux to the mu-th previous
                // frame after a frequency maximum filter on the reference.
                filterLogBands(bandCur_);
                for (int b = 0; b < numBands_; ++b)
                {
                    const T d = bandCur_[static_cast<size_t>(b)]
                              - bandMaxPrev_[static_cast<size_t>(b)];
                    if (d > T(0)) odf += d;
                }
                odf /= static_cast<T>(numBands_);
                // Rotate: previous <- current, and rebuild the max-filtered
                // reference from the (new) previous frame.
                bandPrev_ = bandCur_;
                maxFilterFreq(bandPrev_, bandMaxPrev_);
                break;
            }
        }

        // Rotate per-bin history (prevPhase2_ <- prevPhase_ <- phase_) and the
        // previous magnitude, used by SpectralFlux/ComplexDomain next frame.
        rotatePhaseHistory();
        std::copy(mag_.begin(), mag_.end(), prevMag_.begin());

        return odf;
    }

    /** @brief prevPhase2_ <- prevPhase_ <- phase_ (correct 2-frame ring). */
    void rotatePhaseHistory() noexcept
    {
        std::copy(prevPhase_.begin(), prevPhase_.end(), prevPhase2_.begin());
        std::copy(phase_.begin(), phase_.end(), prevPhase_.begin());
    }

    /** @brief One causal analysis frame: ODF + online peak-pick + scheduling. */
    void analyzeFrame() noexcept
    {
        const Method m = method_.load(std::memory_order_relaxed);
        const bool whiten = whitening_.load(std::memory_order_relaxed);
        const T value = computeOdf(m, whiten);

        // Push into the causal ODF history ring.
        odfHist_[static_cast<size_t>(odfWrite_)] = value;
        odfWrite_ = (odfWrite_ + 1) % odfHistLen_;
        ++frameIndex_;

        // Causal peak-pick with a single-frame confirmation: we decide whether
        // the PREVIOUS frame was a maximum now that we have the current one.
        // This one-hop confirmation is exactly the +hop term of L.
        if (frameIndex_ < 2) { lastConfirmOdf_ = value; return; }

        const T prev = odfAt(1);      // previous frame's ODF (candidate)
        const T curr = value;         // current frame (confirmation)

        // (1) prev is a causal local max over [prev-preMax, prev+1].
        bool isMax = (prev >= curr);
        if (isMax)
        {
            for (int j = 2; j <= preMaxFrames_ + 1 && j < frameIndex_; ++j)
            {
                if (odfAt(j) > prev) { isMax = false; break; }
            }
        }

        if (isMax)
        {
            // (2) prev exceeds the backward moving mean + delta.
            T sum = T(0); int cnt = 0;
            for (int j = 1; j <= preAvgFrames_ && j < frameIndex_; ++j)
            {
                sum += odfAt(j); ++cnt;
            }
            const T mean = (cnt > 0) ? sum / static_cast<T>(cnt) : T(0);
            const T delta = threshold_.load(std::memory_order_relaxed);

            // (3) combination width since the last onset, and past warm-up.
            const int64_t candFrame = frameIndex_ - 1;
            if (candFrame >= primeFrames_ && prev >= mean + delta
                && candFrame - lastOnsetFrame_ > waitFrames_)
            {
                // The candidate is the previous frame; its fire position was
                // totalSamples_ - hop (one hop before the current frame's fire).
                const int64_t candFirePos = totalSamples_ - hop_;
                const int64_t ref = referenceSample(candFirePos);
                scheduleOnset(ref, prev);
                lastOnsetFrame_ = candFrame;
            }
        }
    }

    /** @brief ODF value j frames back (j>=1) from the most recent write. */
    [[nodiscard]] T odfAt(int j) const noexcept
    {
        int idx = odfWrite_ - 1 - j;
        idx %= odfHistLen_;
        if (idx < 0) idx += odfHistLen_;
        return odfHist_[static_cast<size_t>(idx)];
    }

    void scheduleOnset(int64_t referenceSample, T strength) noexcept
    {
        if (pendingCount_ >= static_cast<int>(pending_.size())) return; // saturate
        const int tail = (pendingHead_ + pendingCount_) % static_cast<int>(pending_.size());
        PendingOnset& p = pending_[static_cast<size_t>(tail)];
        p.referenceSample = referenceSample;
        p.reportSample = referenceSample + latencySamples_.load(std::memory_order_relaxed);
        p.strength = strength;
        ++pendingCount_;
    }

    // -- Filterbank ----------------------------------------------------------

    /** @brief Builds the log-frequency triangular filterbank (quarter-tone,
     *  peak-normalised, not area-normalised). Filter count depends on
     *  fftSize/fs; ~138 at 44.1 kHz / 2048. */
    void buildFilterBank()
    {
        fbStart_.clear();
        fbWeights_.clear();
        fbOffset_.clear();

        const double binHz = sampleRate_ / static_cast<double>(fftSize_);
        const double fMax = std::min(kFMaxHz, sampleRate_ * 0.5 * 0.999);

        // Quarter-tone centre bins, strictly increasing and unique.
        std::vector<int> centres;
        for (int i = 0; ; ++i)
        {
            const double f = kFMin * std::pow(2.0, static_cast<double>(i)
                                              / static_cast<double>(kBandsPerOctave));
            if (f > fMax) break;
            int bin = static_cast<int>(std::lround(f / binHz));
            bin = std::clamp(bin, 0, numBins_ - 1);
            if (centres.empty() || bin > centres.back())
                centres.push_back(bin);
        }

        // Triangular filters over consecutive triples (b[j-1], b[j], b[j+1]).
        numBands_ = 0;
        for (size_t j = 1; j + 1 < centres.size(); ++j)
        {
            const int lo = centres[j - 1];
            const int ce = centres[j];
            const int hi = centres[j + 1];
            if (!(lo < ce && ce < hi)) continue;

            fbStart_.push_back(lo);
            fbOffset_.push_back(static_cast<int>(fbWeights_.size()));
            for (int k = lo; k <= hi; ++k)
            {
                T wv;
                if (k <= ce)
                    wv = static_cast<T>(static_cast<double>(k - lo)
                                        / static_cast<double>(ce - lo));
                else
                    wv = static_cast<T>(static_cast<double>(hi - k)
                                        / static_cast<double>(hi - ce));
                fbWeights_.push_back(wv);
            }
            ++numBands_;
        }
        fbCount_.clear();
        for (int b = 0; b < numBands_; ++b)
        {
            const int off = fbOffset_[static_cast<size_t>(b)];
            const int nextOff = (b + 1 < numBands_)
                                ? fbOffset_[static_cast<size_t>(b + 1)]
                                : static_cast<int>(fbWeights_.size());
            fbCount_.push_back(nextOff - off);
        }
        if (numBands_ < 1) numBands_ = 1; // degenerate guard (tiny fftSize)
    }

    /** @brief Applies the filterbank to mag_ and takes log10(x+1) per band. */
    void filterLogBands(std::vector<T>& out) noexcept
    {
        for (int b = 0; b < numBands_ && b < static_cast<int>(fbStart_.size()); ++b)
        {
            const int start = fbStart_[static_cast<size_t>(b)];
            const int off = fbOffset_[static_cast<size_t>(b)];
            const int cnt = fbCount_[static_cast<size_t>(b)];
            T acc = T(0);
            for (int i = 0; i < cnt; ++i)
            {
                const int k = start + i;
                if (k >= 0 && k < numBins_)
                    acc += mag_[static_cast<size_t>(k)]
                         * fbWeights_[static_cast<size_t>(off + i)];
            }
            out[static_cast<size_t>(b)] = std::log10(acc + T(1));
        }
    }

    /** @brief 3-neighbour frequency maximum filter (SuperFlux vibrato guard). */
    void maxFilterFreq(const std::vector<T>& in, std::vector<T>& out) const noexcept
    {
        for (int b = 0; b < numBands_; ++b)
        {
            T mx = in[static_cast<size_t>(b)];
            if (b > 0) mx = std::max(mx, in[static_cast<size_t>(b - 1)]);
            if (b + 1 < numBands_) mx = std::max(mx, in[static_cast<size_t>(b + 1)]);
            out[static_cast<size_t>(b)] = mx;
        }
    }

    static T princArg(T x) noexcept
    {
        // Wrap to (-pi, pi].
        const T twoPiT = twoPi<T>;
        T y = x - twoPiT * std::floor(x / twoPiT + T(0.5));
        return y;
    }

    void resetState() noexcept
    {
        std::fill(ring_.begin(), ring_.end(), T(0));
        std::fill(prevMag_.begin(), prevMag_.end(), T(0));
        std::fill(prevPhase_.begin(), prevPhase_.end(), T(0));
        std::fill(prevPhase2_.begin(), prevPhase2_.end(), T(0));
        std::fill(whitenPeak_.begin(), whitenPeak_.end(), T(0));
        std::fill(bandPrev_.begin(), bandPrev_.end(), T(0));
        std::fill(bandMaxPrev_.begin(), bandMaxPrev_.end(), T(0));
        std::fill(odfHist_.begin(), odfHist_.end(), T(0));

        writePos_ = 0;
        hopCounter_ = 0;
        totalSamples_ = 0;
        frameIndex_ = 0;
        odfWrite_ = 0;
        lastConfirmOdf_ = T(0);
        lastOnsetFrame_ = -kBig;
        pendingHead_ = 0;
        pendingCount_ = 0;

        onsetLatched_.store(false, std::memory_order_relaxed);
        onsetStrength_.store(T(0), std::memory_order_relaxed);
        lastOnsetSample_.store(-1, std::memory_order_relaxed);
    }

    // -- Whitening constants -------------------------------------------------
    static constexpr T kWhitenDecay = T(0.9995);
    static constexpr T kWhitenFloor = T(1e-4);

    // -- Members -------------------------------------------------------------
    double sampleRate_ = 44100.0;
    int fftSize_ = 2048;
    int numBins_ = 1025;
    int hop_ = 221;
    int numBands_ = 1;
    int muFrames_ = 1;
    int localizationOffset_ = 0;
    int primeFrames_ = 12;

    std::unique_ptr<FFTReal<T>> fft_; // doubles as the "prepared" gate

    std::vector<T> window_;
    std::vector<T> ring_;      // size 2*fftSize
    std::vector<T> fft_time_;  // size fftSize
    std::vector<T> fft_spec_;  // size fftSize+2
    std::vector<T> mag_;       // numBins
    std::vector<T> phase_;     // numBins
    std::vector<T> prevPhase_, prevPhase2_, prevMag_, whitenPeak_;

    // Filterbank (CSR-style: start bin, flat weights, per-band offset/count).
    std::vector<int> fbStart_;
    std::vector<T> fbWeights_;
    std::vector<int> fbOffset_;
    std::vector<int> fbCount_;

    std::vector<T> bandCur_, bandPrev_, bandMaxPrev_;

    // Peak-picker windows (frames).
    int preMaxFrames_ = 6, postMaxFrames_ = 6;
    int preAvgFrames_ = 20, postAvgFrames_ = 14;
    int waitFrames_ = 6;

    // Causal ODF history ring.
    std::vector<T> odfHist_;
    int odfHistLen_ = 32;
    int odfWrite_ = 0;
    int64_t frameIndex_ = 0;
    T lastConfirmOdf_ = T(0);
    int64_t lastOnsetFrame_ = -kBig;

    // Streaming counters.
    int writePos_ = 0;
    int hopCounter_ = 0;
    int64_t totalSamples_ = 0;

    // Pending-onset ring (held from detection to report point).
    std::vector<PendingOnset> pending_;
    int pendingHead_ = 0;
    int pendingCount_ = 0;

    // Atomic parameters / readouts.
    std::atomic<Method> method_ { Method::SuperFlux };
    std::atomic<T> threshold_ { kDefaultDelta };
    std::atomic<bool> whitening_ { false };
    std::atomic<bool> onsetLatched_ { false };
    std::atomic<T> onsetStrength_ { T(0) };
    std::atomic<int64_t> lastOnsetSample_ { -1 };
    std::atomic<int64_t> latencySamples_ { 2269 };
};

} // namespace dspark
