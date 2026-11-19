// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralFilter.h
 * @brief A time-varying filter designed per frame from a magnitude curve and
 *        applied as a minimum-phase FIR: no resynthesis window, no time
 *        aliasing, no pre-echo.
 *
 * SpectralProcessor modifies a spectrum and resynthesizes it through a
 * window; that is the right tool for a change of the spectrum itself, but a
 * gain per bin applied that way smears in time (each frame's change spreads
 * over the whole window, before the event as well as after it) and wraps
 * around the frame. SpectralFilter is its filtering counterpart: every hop
 * it analyses the last `window` samples (Hann), zero-padded to `padding`
 * times that length; a callback reads the power of each bin and writes the
 * gain it wants there; the gain curve becomes a minimum-phase filter by the
 * real cepstrum (the log magnitude folded onto positive quefrencies), the
 * filter is cut in time to the room the padding leaves (with a raised-cosine
 * tail over its last quarter), and the windowed frame is filtered by it (a
 * linear convolution: the frame and the filter fit in the padded transform).
 * The frames overlap-add to the input times the filters. A gain of one
 * everywhere is the input delayed by the window (latency `window` samples),
 * to the rounding of the transforms.
 *
 * Quality: the padding is the spectral resolution of the curve (2, 4 or 8);
 * the overlap (2 to 16 frames per window) is its time resolution. With
 * setLinked(true) the callback runs once per hop on the mean power of the
 * channels and one filter serves them all (half the cost in stereo).
 * setDelta(true) applies one minus the filter: what it takes out. A key
 * (processBlock(buffer, &key, design)) replaces the programme in the power
 * the callback reads; the filter still runs on the programme.
 *
 * Threading: prepare() on the setup thread (allocates); processBlock(),
 * reset() and the setters on the audio thread (owner-managed).
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DenormalGuard.h, DspMath.h,
 * FFT.h, WindowFunctions.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "DenormalGuard.h"
#include "DspMath.h"
#include "FFT.h"
#include "WindowFunctions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class SpectralFilter
 * @brief Minimum-phase filtering by a gain curve designed every hop.
 *
 * @tparam T Sample type (float or double); the transforms run in double.
 */
template <FloatType T>
class SpectralFilter
{
public:
    static constexpr int kMinWindow = 64;      ///< Shortest analysis window.
    static constexpr int kMaxWindow = 32768;   ///< Longest analysis window.

    /**
     * @brief Allocates for a layout.
     * @param spec       Rate and channels (an invalid spec is ignored).
     * @param windowSize Analysis window, rounded up to a power of two (64 to 32768).
     * @param padding    Transform over window: 2, 4 or 8 (others are rounded down).
     * @param overlap    Frames per window: 2 to 16 (rounded down to a power of two).
     */
    void prepare(const AudioSpec& spec, int windowSize = 2048, int padding = 2, int overlap = 4)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        fs_ = spec.sampleRate;
        int n = kMinWindow;
        while (n < std::clamp(windowSize, kMinWindow, kMaxWindow)) n <<= 1;
        n_ = n;
        pad_ = padding >= 8 ? 8 : (padding >= 4 ? 4 : 2);
        m_ = n_ * pad_;
        int r = 2;
        while (r * 2 <= std::clamp(overlap, 2, 16)) r <<= 1;
        hop_ = n_ / r;
        norm_ = 2.0 / static_cast<double>(r);
        bins_ = m_ / 2 + 1;
        kernelLength_ = m_ - n_ + 1;
        fft_ = std::make_unique<FFTReal<double>>(static_cast<size_t>(m_));
        window_.resize(static_cast<size_t>(n_));
        WindowFunctions<double>::hann(window_.data(), n_, true);
        taper_.assign(static_cast<size_t>(kernelLength_), 1.0);
        {
            const int fade = std::max(1, kernelLength_ / 4);
            for (int i = 0; i < fade; ++i)
                taper_[static_cast<size_t>(kernelLength_ - fade + i)] = 0.5 + 0.5 * std::cos(3.141592653589793 * (i + 1) / (fade + 1));
        }
        const int ch = spec.numChannels;
        ring_.assign(static_cast<size_t>(ch), std::vector<double>(static_cast<size_t>(n_), 0.0));
        keyRing_.assign(static_cast<size_t>(ch), std::vector<double>(static_cast<size_t>(n_), 0.0));
        keySpectrum_.assign(static_cast<size_t>(m_ + 2), 0.0);
        accum_.assign(static_cast<size_t>(ch), std::vector<double>(static_cast<size_t>(2 * m_), 0.0));
        spectra_.assign(static_cast<size_t>(ch), std::vector<double>(static_cast<size_t>(m_ + 2), 0.0));
        writePos_ = readPos_ = 0;
        hopCounter_ = 0;
        frame_.assign(static_cast<size_t>(m_ + 2), 0.0);
        kernel_.assign(static_cast<size_t>(m_ + 2), 0.0);
        power_.assign(static_cast<size_t>(bins_), T(0));
        gain_.assign(static_cast<size_t>(bins_), T(1));
        channels_ = ch;
        prepared_ = true;
    }

    /** @brief One filter for every channel, designed on their mean power. */
    void setLinked(bool on) noexcept { linked_ = on; }
    [[nodiscard]] bool isLinked() const noexcept { return linked_; }
    /** @brief Applies one minus the filter (what it takes out). */
    void setDelta(bool on) noexcept { delta_ = on; }
    [[nodiscard]] bool isDelta() const noexcept { return delta_; }

    [[nodiscard]] int getLatency() const noexcept { return n_; }
    [[nodiscard]] int getWindowSize() const noexcept { return n_; }
    [[nodiscard]] int getTransformSize() const noexcept { return m_; }
    [[nodiscard]] int getHopSize() const noexcept { return hop_; }
    [[nodiscard]] int getNumBins() const noexcept { return bins_; }
    [[nodiscard]] double getBinHz() const noexcept { return fs_ / static_cast<double>(m_); }
    [[nodiscard]] double getSampleRate() const noexcept { return fs_; }

    /** @brief Clears the input, the output and the hop phase. */
    void reset() noexcept
    {
        for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
        for (auto& r : keyRing_) std::fill(r.begin(), r.end(), 0.0);
        for (auto& a : accum_) std::fill(a.begin(), a.end(), 0.0);
        writePos_ = readPos_ = 0;
        hopCounter_ = 0;
    }

    /**
     * @brief Filters a block in place.
     *
     * @tparam Design `void(int channel, const T* power, T* gain, int numBins)`:
     *         reads the power of each bin of the frame (a full-scale sine
     *         reads about 1 in its bin) and writes the gain wanted there
     *         (linear, preset to 1; a gain under -120 dB is taken as -120).
     *         The channel is -1 when linked. Must not throw.
     */
    template <typename Design>
    void processBlock(AudioBufferView<T> buffer, Design&& design) noexcept
    {
        processBlock(buffer, nullptr, design);
    }

    /**
     * @brief Filters a block in place, the callback reading the power of a
     *        key (the same length or longer; its last channel serves the
     *        programme's extra ones; nullptr reads the programme).
     */
    template <typename Design>
    void processBlock(AudioBufferView<T> buffer, const AudioBufferView<T>* key, Design&& design) noexcept
    {
        if (!prepared_) return;
        if (key != nullptr && (key->getNumChannels() <= 0 || key->getNumSamples() < buffer.getNumSamples())) key = nullptr;
        keyActive_ = key != nullptr;
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        const int nS = buffer.getNumSamples();
        const int mask = n_ - 1, accMask = 2 * m_ - 1;
        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, hop_ - hopCounter_);
            // Push, then drain, then the hop (the frames stay anchored to the
            // stream whatever the blocks; in place is safe).
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* in = buffer.getChannel(ch) + i;
                auto& r = ring_[static_cast<size_t>(ch)];
                for (int k = 0; k < chunk; ++k) r[static_cast<size_t>((writePos_ + k) & mask)] = static_cast<double>(in[k]);
                const T* kin = key != nullptr ? key->getChannel(std::min(ch, key->getNumChannels() - 1)) + i : in;
                auto& kr = keyRing_[static_cast<size_t>(ch)];
                for (int k = 0; k < chunk; ++k) kr[static_cast<size_t>((writePos_ + k) & mask)] = static_cast<double>(kin[k]);
            }
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = buffer.getChannel(ch) + i;
                auto& a = accum_[static_cast<size_t>(ch)];
                for (int k = 0; k < chunk; ++k)
                {
                    const auto p = static_cast<size_t>((readPos_ + k) & accMask);
                    out[k] = static_cast<T>(a[p]);
                    a[p] = 0.0;
                }
            }
            writePos_ = (writePos_ + chunk) & mask;
            readPos_ = (readPos_ + chunk) & accMask;
            hopCounter_ += chunk;
            if (hopCounter_ >= hop_)
            {
                hopCounter_ = 0;
                hop(nCh, design);
            }
            i += chunk;
        }
    }

private:
    template <typename Design>
    void hop(int nCh, Design& design) noexcept
    {
        const double scale = 16.0 / (static_cast<double>(n_) * static_cast<double>(n_));
        // Analysis of every channel: the last n samples, windowed, zero-padded.
        for (int ch = 0; ch < nCh; ++ch)
        {
            const auto& r = ring_[static_cast<size_t>(ch)];
            for (int k = 0; k < n_; ++k)
                frame_[static_cast<size_t>(k)] = r[static_cast<size_t>((writePos_ + k) & (n_ - 1))] * window_[static_cast<size_t>(k)];
            std::fill(frame_.begin() + n_, frame_.begin() + m_, 0.0);
            fft_->forward(frame_.data(), spectra_[static_cast<size_t>(ch)].data());
        }
        // The spectrum the design reads: the programme's, or the key's.
        const auto detectorSpectrum = [&](int ch) -> const double* {
            if (!keyActive_) return spectra_[static_cast<size_t>(ch)].data();
            const auto& r = keyRing_[static_cast<size_t>(ch)];
            for (int k = 0; k < n_; ++k)
                frame_[static_cast<size_t>(k)] = r[static_cast<size_t>((writePos_ + k) & (n_ - 1))] * window_[static_cast<size_t>(k)];
            std::fill(frame_.begin() + n_, frame_.begin() + m_, 0.0);
            fft_->forward(frame_.data(), keySpectrum_.data());
            return keySpectrum_.data();
        };
        const auto powerOf = [&](int ch) {
            const double* s = detectorSpectrum(ch);
            for (int k = 0; k < bins_; ++k)
                power_[static_cast<size_t>(k)] = static_cast<T>(scale * (s[2 * k] * s[2 * k] + s[2 * k + 1] * s[2 * k + 1]));
        };
        if (linked_ && nCh > 1)
        {
            std::fill(power_.begin(), power_.end(), T(0));
            for (int ch = 0; ch < nCh; ++ch)
            {
                const double* s = detectorSpectrum(ch);
                for (int k = 0; k < bins_; ++k)
                    power_[static_cast<size_t>(k)] += static_cast<T>(scale * (s[2 * k] * s[2 * k] + s[2 * k + 1] * s[2 * k + 1]) / nCh);
            }
            std::fill(gain_.begin(), gain_.end(), T(1));
            design(-1, power_.data(), gain_.data(), bins_);
            const bool unity = buildKernel();
            for (int ch = 0; ch < nCh; ++ch) apply(ch, unity);
        }
        else
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                powerOf(ch);
                std::fill(gain_.begin(), gain_.end(), T(1));
                design(ch, power_.data(), gain_.data(), bins_);
                apply(ch, buildKernel());
            }
        }
    }

    /**
     * @brief The minimum-phase filter of gain_ into kernel_ (its spectrum),
     *        cut to kernelLength_ samples. @return Whether the gain is one
     *        everywhere (then kernel_ is left alone and not used).
     */
    bool buildKernel() noexcept
    {
        bool unity = true;
        for (int k = 0; k < bins_ && unity; ++k) unity = gain_[static_cast<size_t>(k)] == T(1);
        if (unity) return true;
        double* c = frame_.data();
        double* h = kernel_.data();
        // The log magnitude as a real, even spectrum; its inverse is the
        // real cepstrum.
        for (int k = 0; k < bins_; ++k)
        {
            h[2 * k] = std::log(std::max(static_cast<double>(gain_[static_cast<size_t>(k)]), 1e-6));
            h[2 * k + 1] = 0.0;
        }
        fft_->inverse(h, c);
        // Fold onto the positive quefrencies: the minimum-phase cepstrum.
        for (int q = 1; q < m_ / 2; ++q) c[q] *= 2.0;
        std::fill(c + m_ / 2 + 1, c + m_, 0.0);
        fft_->forward(c, h);
        for (int k = 0; k < bins_; ++k)
        {
            const double mag = std::exp(h[2 * k]), ph = h[2 * k + 1];
            h[2 * k] = mag * std::cos(ph);
            h[2 * k + 1] = mag * std::sin(ph);
        }
        // Cut to the room the padding leaves, with a raised-cosine tail.
        fft_->inverse(h, c);
        for (int q = 0; q < kernelLength_; ++q) c[q] *= taper_[static_cast<size_t>(q)];
        std::fill(c + kernelLength_, c + m_, 0.0);
        fft_->forward(c, h);
        return false;
    }

    /** @brief The channel's frame times the filter (or one minus it), overlap-added. */
    void apply(int ch, bool unity) noexcept
    {
        double* s = spectra_[static_cast<size_t>(ch)].data();
        const double* h = kernel_.data();
        if (delta_)
        {
            if (unity) std::fill(s, s + m_ + 2, 0.0);
            else
                for (int k = 0; k < bins_; ++k)
                {
                    const double hr = 1.0 - h[2 * k], hi = -h[2 * k + 1];
                    const double xr = s[2 * k], xi = s[2 * k + 1];
                    s[2 * k] = xr * hr - xi * hi;
                    s[2 * k + 1] = xr * hi + xi * hr;
                }
        }
        else if (!unity)
        {
            for (int k = 0; k < bins_; ++k)
            {
                const double hr = h[2 * k], hi = h[2 * k + 1];
                const double xr = s[2 * k], xi = s[2 * k + 1];
                s[2 * k] = xr * hr - xi * hi;
                s[2 * k + 1] = xr * hi + xi * hr;
            }
        }
        double* y = frame_.data();
        fft_->inverse(s, y);
        auto& a = accum_[static_cast<size_t>(ch)];
        const int accMask = 2 * m_ - 1;
        for (int k = 0; k < m_; ++k) a[static_cast<size_t>((readPos_ + k) & accMask)] += norm_ * y[k];
    }

    double fs_ = 48000.0, norm_ = 0.5;
    int n_ = 2048, pad_ = 2, m_ = 4096, hop_ = 512, bins_ = 2049, kernelLength_ = 2049, channels_ = 0;
    int writePos_ = 0, readPos_ = 0, hopCounter_ = 0;
    bool prepared_ = false, linked_ = false, delta_ = false, keyActive_ = false;
    std::unique_ptr<FFTReal<double>> fft_;
    std::vector<double> window_, taper_, frame_, kernel_, keySpectrum_;
    std::vector<std::vector<double>> ring_, keyRing_, accum_, spectra_;
    std::vector<T> power_, gain_;
};

} // namespace dspark
