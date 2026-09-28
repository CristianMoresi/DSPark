// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DelayEstimator.h
 * @brief Delay between two recordings of the same material, to a fraction of
 *        a sample, with a confidence, in memory bounded by the search range.
 *
 * Aligning a delivery against its reference - a master from another tool, a
 * re-recording, a stem exported elsewhere - needs the offset between them
 * before any sample-by-sample comparison means anything. This estimates it.
 *
 * Method. Both signals are cut into Hann-windowed frames (half overlap) and
 * the cross-spectrum delivery x conj(reference) is averaged over every frame
 * pushed, so memory is a few frames whatever the length of the program. The
 * delay is then read off in three steps:
 *
 * 1. The generalised cross-correlation with phase transform (GCC-PHAT: every
 *    bin weighted to unit magnitude, so a loud low end cannot drown the
 *    timing information of the rest of the band) is searched within
 *    +-maxDelay for its largest magnitude: the whole-sample delay, and the
 *    polarity (a negative peak is an inverted delivery).
 * 2. A parabola through that peak and its neighbours brings it within a
 *    small fraction of a sample.
 * 3. The residual is solved from the phase of the averaged cross-spectrum,
 *    whose slope over frequency IS the delay: a least-squares fit weighted
 *    by each bin's cross-spectral magnitude, iterated to convergence.
 *
 * Frames are aligned to the reference and the delivery is framed at the
 * whole-sample delay found so far (the spectrum already accumulated is
 * rotated to match whenever that moves), so a long
 * delay does not leave the windows looking at partly different material:
 * the estimate carries no bias from the window at any delay in range.
 *
 * Gain is irrelevant (the phase transform discards it) and so is polarity
 * (reported, not confused with a delay). Channels are averaged to one.
 *
 * Confidence, in [0, 1], is the share of the averaged cross-spectrum that is
 * consistent with the one delay reported: |sum S(k) e^{j w_k d}| / sum |S(k)|.
 * A pure delay reads 1. Noise in the delivery that is independent of the
 * reference averages out of the cross-spectrum as frames accumulate, so it
 * lowers the reading less the more material has been pushed (measured 0.98
 * with the noise as loud as the signal over 11 s, where the delay itself is
 * found within 0.002 samples); unrelated material reads near 0 (0.02 over
 * the same 11 s). Read it as "is there one delay here", not as a
 * signal-to-noise ratio.
 *
 * Measured over 11 s at 48 kHz, with the delay applied exactly (in the
 * frequency domain) at 0, 0.5, 12.37, 37, -250.4 and 4000.25 samples and a
 * 0.5 gain: white noise within 0.0005 samples down to 0 dB SNR; low-pass
 * coloured noise within 0.003 samples at 20 dB SNR and 0.025 at 0 dB.
 *
 * Sign: a positive delay means the delivery is LATE - delivery[n] matches
 * reference[n - delay].
 *
 * Threading: owner-managed; prepare() allocates (setup thread); push() and
 * estimate() allocate nothing and belong to the thread that owns the
 * instance. Not an audio-thread component by design, but real-time safe.
 *
 * Dependencies: DspMath.h, AudioBuffer.h, FFT.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioBuffer.h"
#include "../Core/FFT.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class DelayEstimator
 * @brief Streaming GCC-PHAT delay estimator with sub-sample refinement.
 *
 * @code
 *   dspark::DelayEstimator<float> est;
 *   est.prepare(48000.0, 48000);                 // search +-1 s
 *   for (each block) est.push(referenceBlock, deliveryBlock);
 *   const auto r = est.estimate();
 *   if (r.valid && r.confidence > 0.5) align(delivery, r.delaySamples);
 * @endcode
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DelayEstimator
{
public:
    /** @brief What estimate() returns. */
    struct Result
    {
        double delaySamples = 0.0;  ///< Positive: the delivery is late.
        double delaySeconds = 0.0;  ///< The same in seconds.
        double confidence = 0.0;    ///< Share of the cross-spectrum consistent with it, 0..1.
        bool inverted = false;      ///< The delivery is polarity-inverted.
        bool valid = false;         ///< Enough material was pushed to estimate.
    };

    /**
     * @brief Allocates for a search range.
     * @param sampleRate      Sample rate of both signals.
     * @param maxDelaySamples Largest delay searched, either way (>= 1). The
     *                        frame is the power of two at least eight times
     *                        this (and at least 4096), and so is the memory.
     */
    void prepare(double sampleRate, int maxDelaySamples)
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) return;
        sampleRate_ = sampleRate;
        maxDelay_ = std::max(1, maxDelaySamples);
        frame_ = 4096;
        while (frame_ < 8 * maxDelay_ && frame_ < (1 << 26)) frame_ <<= 1;
        hop_ = frame_ / 2;
        fft_ = std::make_unique<FFTReal<double>>(static_cast<size_t>(frame_));
        window_.resize(static_cast<size_t>(frame_));
        for (int i = 0; i < frame_; ++i)
            window_[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * pi<double> * i / frame_);
        // The reference needs one frame; the delivery a frame plus the delay
        // it is framed at, either side.
        historyLength_ = frame_ + 2 * maxDelay_ + 2;
        refHistory_.assign(static_cast<size_t>(historyLength_), 0.0);
        delHistory_.assign(static_cast<size_t>(historyLength_), 0.0);
        bufA_.assign(static_cast<size_t>(frame_ + 2), 0.0);
        bufB_.assign(static_cast<size_t>(frame_ + 2), 0.0);
        cross_.assign(static_cast<size_t>(frame_ / 2 + 1), std::complex<double> {});
        corr_.assign(static_cast<size_t>(frame_ + 2), 0.0);
        prepared_ = true;
        reset();
    }

    /** @brief Forgets everything pushed. */
    void reset() noexcept
    {
        std::fill(refHistory_.begin(), refHistory_.end(), 0.0);
        std::fill(delHistory_.begin(), delHistory_.end(), 0.0);
        std::fill(cross_.begin(), cross_.end(), std::complex<double> {});
        written_ = 0;
        nextFrameEnd_ = static_cast<int64_t>(frame_) + maxDelay_;
        frames_ = 0;
        offset_ = 0;
    }

    /**
     * @brief Feeds the next stretch of both signals (same length; channels
     *        averaged). Allocation free.
     */
    void push(AudioBufferView<const T> reference, AudioBufferView<const T> delivery) noexcept
    {
        if (!prepared_) return;
        const int n = std::min(reference.getNumSamples(), delivery.getNumSamples());
        const int rc = reference.getNumChannels(), dc = delivery.getNumChannels();
        if (n <= 0 || rc <= 0 || dc <= 0) return;
        for (int i = 0; i < n; ++i)
        {
            double r = 0.0, d = 0.0;
            for (int c = 0; c < rc; ++c) r += static_cast<double>(reference.getChannel(c)[i]);
            for (int c = 0; c < dc; ++c) d += static_cast<double>(delivery.getChannel(c)[i]);
            const size_t slot = static_cast<size_t>(written_ % historyLength_);
            refHistory_[slot] = std::isfinite(r) ? r / rc : 0.0;
            delHistory_[slot] = std::isfinite(d) ? d / dc : 0.0;
            ++written_;
            if (written_ == nextFrameEnd_)
            {
                accumulateFrame();
                nextFrameEnd_ += hop_;
            }
        }
    }

    /** @brief The delay over everything pushed so far. Allocation free. */
    [[nodiscard]] Result estimate() noexcept
    {
        Result out;
        if (!prepared_ || frames_ < 1) return out;
        double lag = 0.0;
        bool inverted = false;
        if (!searchPeak(lag, inverted)) return out;
        const double sign = inverted ? -1.0 : 1.0;

        // Refine on the phase slope of the averaged cross-spectrum. The
        // delivery was framed offset_ late, so the spectrum holds the delay
        // minus offset_.
        double d = lag - static_cast<double>(offset_);
        for (int it = 0; it < 8; ++it)
        {
            double num = 0.0, den = 0.0;
            for (int k = 1; k < frame_ / 2; ++k)
            {
                const double w = 2.0 * pi<double> * k / frame_;
                const std::complex<double> s = cross_[static_cast<size_t>(k)] * sign
                    * std::polar(1.0, w * d);
                const double mag = std::abs(s);
                if (!(mag > 0.0)) continue;
                const double phi = std::arg(s);   // ~ -w * (delay - d)
                num += mag * w * phi;
                den += mag * w * w;
            }
            if (!(den > 0.0)) break;
            const double step = -num / den;
            d += step;
            if (std::abs(step) < 1e-9) break;
        }

        std::complex<double> agree {};
        double total = 0.0;
        for (int k = 1; k < frame_ / 2; ++k)
        {
            const double w = 2.0 * pi<double> * k / frame_;
            const std::complex<double> s = cross_[static_cast<size_t>(k)] * sign;
            agree += s * std::polar(1.0, w * d);
            total += std::abs(s);
        }
        out.delaySamples = d + static_cast<double>(offset_);
        out.delaySeconds = out.delaySamples / sampleRate_;
        out.confidence = (total > 0.0) ? std::clamp(std::abs(agree) / total, 0.0, 1.0) : 0.0;
        out.inverted = inverted;
        out.valid = total > 0.0;
        return out;
    }

    /** @brief Frame length in samples (the analysis resolution in time). */
    [[nodiscard]] int getFrameSize() const noexcept { return frame_; }

    /** @brief Whole-signal convenience: one estimate over two buffers. Allocates. */
    static Result estimate(AudioBufferView<const T> reference, AudioBufferView<const T> delivery,
                           double sampleRate, int maxDelaySamples)
    {
        DelayEstimator est;
        est.prepare(sampleRate, maxDelaySamples);
        est.push(reference, delivery);
        // Flush: the last frames need the tail of the delivery.
        const int flush = est.frame_ + 2 * est.maxDelay_;
        std::vector<T> zeros(static_cast<size_t>(std::min(flush, 1 << 16)), T(0));
        const T* z = zeros.data();
        for (int left = flush; left > 0; left -= static_cast<int>(zeros.size()))
        {
            const int n = std::min(left, static_cast<int>(zeros.size()));
            est.push(AudioBufferView<const T>(&z, 1, n), AudioBufferView<const T>(&z, 1, n));
        }
        return est.estimate();
    }

private:
    /** Frames the reference [end - maxDelay - frame, end - maxDelay) and the
     *  delivery the same span offset_ later, and adds their cross-spectrum. */
    void accumulateFrame() noexcept
    {
        // Frame the delivery at the whole-sample delay found so far (see the
        // file header); it settles within the first frames that carry
        // material and moves only if the evidence moves it.
        if (frames_ >= 1)
        {
            double lag = 0.0;
            bool inv = false;
            if (searchPeak(lag, inv))
            {
                const int newOffset = static_cast<int>(std::lround(lag));
                if (newOffset != offset_)
                {
                    // The spectrum so far was taken at the old offset: its
                    // phase holds (delay - old offset). Rotate it to the new.
                    for (int k = 0; k <= frame_ / 2; ++k)
                    {
                        const double w = 2.0 * pi<double> * k / frame_;
                        cross_[static_cast<size_t>(k)] *= std::polar(1.0, w * (newOffset - offset_));
                    }
                    offset_ = newOffset;
                }
            }
        }

        const int64_t refStart = written_ - maxDelay_ - frame_;
        const int64_t delStart = refStart + offset_;
        for (int i = 0; i < frame_; ++i)
        {
            const double w = window_[static_cast<size_t>(i)];
            bufA_[static_cast<size_t>(i)] = w * sample(refHistory_, refStart + i);
            bufB_[static_cast<size_t>(i)] = w * sample(delHistory_, delStart + i);
        }
        fft_->forward(bufA_.data(), bufA_.data());
        fft_->forward(bufB_.data(), bufB_.data());
        for (int k = 0; k <= frame_ / 2; ++k)
        {
            const std::complex<double> a(bufA_[static_cast<size_t>(2 * k)], bufA_[static_cast<size_t>(2 * k + 1)]);
            const std::complex<double> b(bufB_[static_cast<size_t>(2 * k)], bufB_[static_cast<size_t>(2 * k + 1)]);
            cross_[static_cast<size_t>(k)] += b * std::conj(a);
        }
        ++frames_;
    }

    [[nodiscard]] double sample(const std::vector<double>& h, int64_t index) const noexcept
    {
        if (index < 0 || index >= written_ || index < written_ - historyLength_) return 0.0;
        return h[static_cast<size_t>(index % historyLength_)];
    }

    /** GCC-PHAT peak within +-maxDelay (absolute lag), parabola-refined. */
    bool searchPeak(double& lag, bool& inverted) noexcept
    {
        double peakMag = 0.0;
        for (const auto& s : cross_) peakMag = std::max(peakMag, std::abs(s));
        if (!(peakMag > 0.0)) return false;
        const double floor = peakMag * 1e-9;
        for (int k = 0; k <= frame_ / 2; ++k)
        {
            // Undo the framing offset so the correlation is in absolute lag.
            const double w = 2.0 * pi<double> * k / frame_;
            std::complex<double> s = cross_[static_cast<size_t>(k)] * std::polar(1.0, -w * offset_);
            const double m = std::abs(s);
            s = (m > floor) ? s / m : std::complex<double> {};
            corr_[static_cast<size_t>(2 * k)] = s.real();
            corr_[static_cast<size_t>(2 * k + 1)] = s.imag();
        }
        fft_->inverse(corr_.data(), corr_.data());
        auto at = [&](int tau) {
            const int m = ((tau % frame_) + frame_) % frame_;
            return corr_[static_cast<size_t>(m)];
        };
        int best = 0;
        double bestAbs = -1.0;
        for (int tau = -maxDelay_; tau <= maxDelay_; ++tau)
        {
            const double v = std::abs(at(tau));
            if (v > bestAbs) { bestAbs = v; best = tau; }
        }
        inverted = at(best) < 0.0;
        const double s = inverted ? -1.0 : 1.0;
        const double ym = s * at(best - 1), y0 = s * at(best), yp = s * at(best + 1);
        const double den = ym - 2.0 * y0 + yp;
        const double frac = (den < 0.0) ? std::clamp(0.5 * (ym - yp) / den, -0.5, 0.5) : 0.0;
        lag = static_cast<double>(best) + frac;
        return true;
    }

    double sampleRate_ = 48000.0;
    int maxDelay_ = 1;
    int frame_ = 4096;
    int hop_ = 2048;
    int historyLength_ = 0;
    bool prepared_ = false;
    std::unique_ptr<FFTReal<double>> fft_;
    std::vector<double> window_, refHistory_, delHistory_, bufA_, bufB_, corr_;
    std::vector<std::complex<double>> cross_;
    int64_t written_ = 0;
    int64_t nextFrameEnd_ = 0;
    int64_t frames_ = 0;
    int offset_ = 0;
};

} // namespace dspark
