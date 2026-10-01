// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OnsetFeatures.h
 * @brief Shared internal spectrum and novelty kernels for onset analysis.
 *
 * Preparation allocates. Prepared spectrum/novelty evaluation and reset
 * allocate nothing and take no locks. Each instance has one stream owner; it is
 * never concurrently prepared, processed or inspected. No sample clock, peak
 * picker, source I/O or offline job policy is owned here. Dependencies:
 * DspMath.h, FFT.h, WindowFunctions.h, AudioBuffer.h and C++20.
 */

#include "../../Core/AudioBuffer.h"
#include "../../Core/DspMath.h"
#include "../../Core/FFT.h"
#include "../../Core/WindowFunctions.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace dspark::detail
{

/** @brief Allocation-free symmetric novelty peak picker with an explicit frame
 * clock. */
template <FloatType T> class OnsetPeakPicker final
{
  public:
    struct Windows
    {
        std::size_t preMaximum = 0, postMaximum = 0;
        std::size_t preAverage = 0, postAverage = 0;
        std::size_t wait = 0; ///< Selected indices must be more than wait frames apart.
    };
    /**
     * @brief Examines increasing indices in an immutable, finite novelty
     * sequence. The accessor returns T. Summation order and equal-maximum
     * handling match the original offline OnsetDetector picker. No source timing
     * is inferred.
     */
    template <typename Value>
    [[nodiscard]] bool accept(std::size_t index, std::size_t count, Value value, Windows windows,
                              T delta)
    {
        if (index >= count || (hasLast_ && (index <= last_ || index - last_ <= windows.wait)))
            return false;
        const T current = value(index);
        const auto fromMax = index - std::min(index, windows.preMaximum);
        const auto toMax = index + std::min(count - index - 1, windows.postMaximum);
        for (auto j = fromMax; j <= toMax; ++j)
            if (value(j) > current)
                return false;
        const auto fromAvg = index - std::min(index, windows.preAverage);
        const auto toAvg = index + std::min(count - index - 1, windows.postAverage);
        T sum = T(0);
        for (auto j = fromAvg; j <= toAvg; ++j)
            sum += value(j);
        const T mean = sum / static_cast<T>(toAvg - fromAvg + 1);
        if (current < mean + delta)
            return false;
        last_ = index;
        hasLast_ = true;
        return true;
    }

  private:
    std::size_t last_ = 0;
    bool hasLast_ = false;
};

/** @brief Reusable periodic-Hann real spectrum, without feature history. */
template <FloatType T> class OnsetSpectrum final
{
  public:
    [[nodiscard]] static int defaultFrameSize(double sampleRate) noexcept
    {
        int size = 512;
        const double target = sampleRate * (2048.0 / 48000.0);
        while (size < 16384 && static_cast<double>(size) < target)
            size <<= 1;
        return size;
    }
    /** @brief Conservative setup payload allowance for a fresh power-of-two
     * instance. */
    [[nodiscard]] static std::size_t preparationBytes(int fftSize) noexcept
    {
        // Five frontend arrays: 4*N+4 scalars. FFTReal owns fewer than 4*N
        // scalars plus its short Stockham pass plan. The 2048-byte allowance
        // covers cumulative plan-vector growth through N=65536 on the tested
        // standard libraries, including construction temporaries.
        return (8 * static_cast<std::size_t>(fftSize) + 4) * sizeof(T) + sizeof(FFTReal<T>) + 2048;
    }
    /** @brief Allocates a valid power-of-two frame; the caller owns frame timing.
     */
    void prepare(int fftSize)
    {
        fft_.reset();
        phaseValid_ = false;
        fftSize_ = fftSize;
        window_.assign(static_cast<size_t>(fftSize_), T(0));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        time_.assign(static_cast<size_t>(fftSize_), T(0));
        spectrum_.assign(static_cast<size_t>(fftSize_) + 2, T(0));
        magnitude_.assign(static_cast<size_t>(fftSize_ / 2 + 1), T(0));
        phase_.assign(magnitude_.size(), T(0));
        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize_));
    }
    /**
     * @brief Evaluates one contiguous frame, optionally including bin phases.
     * Input is finite and scaled by the caller to avoid overflow in the FFT and
     * squared magnitudes. Suppressing phases avoids unnecessary atan2 work for
     * a pooled magnitude spectrum, which has no unique complex-domain phase.
     */
    void compute(const T *input, bool withPhase = true) noexcept
    {
        if (!fft_)
            return;
        transform(input);
        for (size_t k = 0; k < magnitude_.size(); ++k)
        {
            const T re = spectrum_[2 * k], im = spectrum_[2 * k + 1];
            magnitude_[k] = binMagnitude(k);
            if (withPhase)
                phase_[k] = std::atan2(im, re);
        }
        phaseValid_ = withPhase;
    }
    /**
     * @brief Pools mono/stereo spectral energy without a phase-cancelling
     * downmix. The view must contain exactly one prepared frame. Invalid shape
     * returns false; callers must not consume old magnitudes after failure. Two
     * channels share one FFT instance and scratch, with one transform per
     * channel. Equal magnitudes (including dual mono and antiphase) retain their
     * exact value.
     */
    [[nodiscard]] bool computePooled(AudioBufferView<const T> frame) noexcept
    {
        phaseValid_ = false;
        if (!fft_ || frame.getNumSamples() != fftSize_ || frame.getNumChannels() < 1 ||
            frame.getNumChannels() > 2)
            return false;
        for (int c = 0; c < frame.getNumChannels(); ++c)
            if (!frame.getChannel(c))
                return false;
        compute(frame.getChannel(0), false);
        if (frame.getNumChannels() == 2)
        {
            transform(frame.getChannel(1));
            for (size_t k = 0; k < magnitude_.size(); ++k)
            {
                const T other = binMagnitude(k);
                if (magnitude_[k] != other)
                    magnitude_[k] = std::hypot(magnitude_[k], other) * invSqrt2<T>;
            }
        }
        return true;
    }
    /** @brief Borrowed scratch, valid until the next compute/prepare or
     * destruction. */
    [[nodiscard]] std::span<T> magnitudes() noexcept
    {
        return magnitude_;
    }
    /** @brief Empty when the last frame did not compute phases. Same scratch
     * lifetime. */
    [[nodiscard]] std::span<const T> phases() const noexcept
    {
        return phaseValid_ ? std::span<const T>(phase_) : std::span<const T>();
    }

  private:
    void transform(const T *input) noexcept
    {
        for (int k = 0; k < fftSize_; ++k)
            time_[static_cast<size_t>(k)] = input[k] * window_[static_cast<size_t>(k)];
        fft_->forward(time_.data(), spectrum_.data());
    }
    [[nodiscard]] T binMagnitude(size_t k) const noexcept
    {
        const T re = spectrum_[2 * k], im = spectrum_[2 * k + 1];
        return std::sqrt(re * re + im * im);
    }
    int fftSize_ = 0;
    bool phaseValid_ = false;
    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_, time_, spectrum_, magnitude_, phase_;
};

/** @brief Shared whitening, spectral/complex flux and SuperFlux feature
 * history. */
template <FloatType T> class OnsetNovelty final
{
  public:
    enum class Method
    {
        SpectralFlux,
        ComplexDomain,
        SuperFlux,
        BothFlux
    };
    static constexpr int kNumRegisters = 4;
    struct Frame
    {
        T value = T(0);        ///< Requested function; SuperFlux when BothFlux is requested.
        T spectralFlux = T(0); ///< Also returned by BothFlux, from the same spectrum.
        std::array<T, kNumRegisters> registers{};
        bool valid = false;
    };
    /** @brief Setup payload allowance when prepare() reserves the bounded
     * filterbank. */
    [[nodiscard]] static std::size_t preparationBytes(int fftSize) noexcept
    {
        return (3 * static_cast<std::size_t>(fftSize) + 4 + 4 * kBandCapacity) * sizeof(T) +
               5 * kBandCapacity * sizeof(int);
    }
    /** @brief Allocates histories and the canonical quarter-tone filterbank. */
    void prepare(double sampleRate, int fftSize, bool boundedWorkspace = false)
    {
        prepared_ = false;
        sampleRate_ = sampleRate;
        fftSize_ = fftSize;
        numBins_ = fftSize / 2 + 1;
        odfScale_ = static_cast<T>(kOdfRefFrame / static_cast<double>(fftSize_));
        prevPhase_.assign(static_cast<size_t>(numBins_), T(0));
        prevPhase2_.assign(static_cast<size_t>(numBins_), T(0));
        prevMag_.assign(static_cast<size_t>(numBins_), T(0));
        whitenPeak_.assign(static_cast<size_t>(numBins_), T(0));
        buildFilterBank(boundedWorkspace);
        bandCur_.assign(static_cast<size_t>(numBands_), T(0));
        bandPrev_.assign(static_cast<size_t>(numBands_), T(0));
        bandMaxPrev_.assign(static_cast<size_t>(numBands_), T(0));
        reset();
        prepared_ = true;
    }
    /** @brief Clears feature history without allocation or changing the
     * filterbank. */
    void reset() noexcept
    {
        std::fill(prevMag_.begin(), prevMag_.end(), T(0));
        std::fill(prevPhase_.begin(), prevPhase_.end(), T(0));
        std::fill(prevPhase2_.begin(), prevPhase2_.end(), T(0));
        std::fill(whitenPeak_.begin(), whitenPeak_.end(), T(0));
        std::fill(bandPrev_.begin(), bandPrev_.end(), T(0));
        std::fill(bandMaxPrev_.begin(), bandMaxPrev_.end(), T(0));
        curRegisters_.fill(T(0));
        phaseHistoryValid_ = true;
    }
    [[nodiscard]] int getNumBands() const noexcept
    {
        return numBands_;
    }
    /**
     * @brief Advances one frame; optional whitening modifies the supplied
     * magnitudes. Phase-less frames support SpectralFlux, SuperFlux and BothFlux.
     * ComplexDomain requires phases on every frame since reset;
     * missing/incompatible data returns valid=false without changing history. No
     * source clock is inferred here.
     */
    [[nodiscard]] Frame process(std::span<T> mag, std::span<const T> phase, Method method,
                                bool whiten = false) noexcept
    {
        Frame result;
        if (!prepared_ || mag.size() != static_cast<size_t>(numBins_) ||
            (!phase.empty() && phase.size() != mag.size()) ||
            (method == Method::ComplexDomain && (phase.empty() || !phaseHistoryValid_)))
            return result;
        if (whiten)
            applyWhitening(mag);
        curRegisters_.fill(T(0));
        switch (method)
        {
        case Method::SpectralFlux:
            result.value = result.spectralFlux = computeSpectralFlux(mag);
            break;
        case Method::ComplexDomain:
            result.value = computeComplexFlux(mag, phase);
            break;
        case Method::BothFlux:
            result.spectralFlux = computeSpectralFlux(mag);
            [[fallthrough]];
        case Method::SuperFlux:
            result.value = computeSuperFlux(mag, whiten);
            break;
        }
        if (!phase.empty())
        {
            std::copy(prevPhase_.begin(), prevPhase_.end(), prevPhase2_.begin());
            std::copy(phase.begin(), phase.end(), prevPhase_.begin());
        }
        else
            phaseHistoryValid_ = false;
        std::copy(mag.begin(), mag.end(), prevMag_.begin());
        result.registers = curRegisters_;
        result.valid = true;
        return result;
    }

  private:
    T computeSpectralFlux(std::span<const T> mag) const noexcept
    {
        T odf = T(0);
        for (int k = 0; k < numBins_; ++k)
        {
            const T d = mag[static_cast<size_t>(k)] - prevMag_[static_cast<size_t>(k)];
            if (d > T(0))
                odf += d;
        }
        odf /= static_cast<T>(numBins_);
        return odf;
    }
    T computeComplexFlux(std::span<const T> mag, std::span<const T> phase) const noexcept
    {
        T odf = T(0);
        // Rectified complex-domain deviation (Dixon 2006): phase-predict
        // each bin, sum |X - Xhat| where magnitude increased.
        for (int k = 0; k < numBins_; ++k)
        {
            const T target = princArg(T(2) * prevPhase_[static_cast<size_t>(k)] -
                                      prevPhase2_[static_cast<size_t>(k)]);
            const T pm = prevMag_[static_cast<size_t>(k)];
            const T cm = mag[static_cast<size_t>(k)];
            const T re = cm * std::cos(phase[static_cast<size_t>(k)]) - pm * std::cos(target);
            const T im = cm * std::sin(phase[static_cast<size_t>(k)]) - pm * std::sin(target);
            if (cm >= pm)
                odf += std::sqrt(re * re + im * im);
        }
        odf /= static_cast<T>(numBins_);
        return odf;
    }
    T computeSuperFlux(std::span<const T> mag, bool whiten) noexcept
    {
        T odf = T(0);
        // Log-filtered magnitude bands, flux to the mu-th previous
        // frame after a frequency maximum filter on the reference.
        // The frame-invariant magnitude scale (kOdfRefFrame/fftSize)
        // applies to the raw spectrum only: adaptive whitening
        // already divides each bin by its running peak, which
        // carries the same linear-in-N growth, so above kWhitenFloor
        // the whitened spectrum is dimensionless and scaling it
        // again would INVERT the rate dependence instead of removing
        // it. Below the floor the divisor is the absolute constant
        // kWhitenFloor, so those bins keep the linear-in-N growth --
        // the residual rate dependence documented at prepare().
        filterLogBands(mag, bandCur_, whiten ? T(1) : odfScale_);
        curRegisters_.fill(T(0));
        for (int b = 0; b < numBands_; ++b)
        {
            const T d = bandCur_[static_cast<size_t>(b)] - bandMaxPrev_[static_cast<size_t>(b)];
            if (d > T(0))
            {
                odf += d;
                if (b < static_cast<int>(bandRegister_.size()))
                    curRegisters_[static_cast<size_t>(bandRegister_[static_cast<size_t>(b)])] += d;
            }
        }
        odf /= static_cast<T>(numBands_);
        for (int g = 0; g < kNumRegisters; ++g)
            if (registerBands_[static_cast<size_t>(g)] > 0)
                curRegisters_[static_cast<size_t>(g)] /=
                    static_cast<T>(registerBands_[static_cast<size_t>(g)]);
        // Rotate: previous <- current, and rebuild the max-filtered
        // reference from the (new) previous frame.
        bandPrev_ = bandCur_;
        maxFilterFreq(bandPrev_, bandMaxPrev_);
        return odf;
    }
    void applyWhitening(std::span<T> mag) noexcept
    {
        for (int k = 0; k < numBins_; ++k)
        {
            T &pk = whitenPeak_[static_cast<size_t>(k)];
            const T decayed = pk * kWhitenDecay;
            const T m = mag[static_cast<size_t>(k)];
            pk = std::max({m, kWhitenFloor, decayed});
            mag[static_cast<size_t>(k)] = m / pk;
        }
    }

    void buildFilterBank(bool boundedWorkspace)
    {
        fbStart_.clear();
        fbWeights_.clear();
        fbOffset_.clear();
        bandRegister_.clear();
        registerBands_.fill(0);

        const double binHz = sampleRate_ / static_cast<double>(fftSize_);
        const double fMax = std::min(kFMaxHz, sampleRate_ * 0.5 * 0.999);

        // Quarter-tone centre bins, strictly increasing and unique.
        std::vector<int> centres;
        if (boundedWorkspace)
        {
            // At most 221 quarter-tone centres between 27.5 and 16000 Hz.
            // Consecutive triangles contain at most N + centres weights.
            // Reserving these bounds avoids data-dependent vector growth in
            // worker jobs whose allocation payload is checked before setup.
            centres.reserve(kBandCapacity);
            fbStart_.reserve(kBandCapacity);
            fbOffset_.reserve(kBandCapacity);
            fbCount_.reserve(kBandCapacity);
            bandRegister_.reserve(kBandCapacity);
            fbWeights_.reserve(static_cast<size_t>(fftSize_) + kBandCapacity);
        }
        for (int i = 0;; ++i)
        {
            const double f = kFMin * std::pow(2.0, static_cast<double>(i) /
                                                       static_cast<double>(kBandsPerOctave));
            if (f > fMax)
                break;
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
            if (!(lo < ce && ce < hi))
                continue;

            {
                const double fc = static_cast<double>(ce) * binHz;
                const int g = fc < 200.0 ? 0 : fc < 800.0 ? 1 : fc < 3200.0 ? 2 : 3;
                bandRegister_.push_back(g);
                ++registerBands_[static_cast<size_t>(g)];
            }
            fbStart_.push_back(lo);
            fbOffset_.push_back(static_cast<int>(fbWeights_.size()));
            for (int k = lo; k <= hi; ++k)
            {
                T wv;
                if (k <= ce)
                    wv = static_cast<T>(static_cast<double>(k - lo) / static_cast<double>(ce - lo));
                else
                    wv = static_cast<T>(static_cast<double>(hi - k) / static_cast<double>(hi - ce));
                fbWeights_.push_back(wv);
            }
            ++numBands_;
        }
        fbCount_.clear();
        for (int b = 0; b < numBands_; ++b)
        {
            const int off = fbOffset_[static_cast<size_t>(b)];
            const int nextOff = (b + 1 < numBands_) ? fbOffset_[static_cast<size_t>(b + 1)]
                                                    : static_cast<int>(fbWeights_.size());
            fbCount_.push_back(nextOff - off);
        }
        if (numBands_ < 1)
            numBands_ = 1; // degenerate guard (tiny fftSize)
    }

    void filterLogBands(std::span<const T> mag, std::vector<T> &out, T scale) noexcept
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
                    acc += mag[static_cast<size_t>(k)] * fbWeights_[static_cast<size_t>(off + i)];
            }
            out[static_cast<size_t>(b)] = std::log10(acc * scale + T(1));
        }
    }

    void maxFilterFreq(const std::vector<T> &in, std::vector<T> &out) const noexcept
    {
        for (int b = 0; b < numBands_; ++b)
        {
            T mx = in[static_cast<size_t>(b)];
            if (b > 0)
                mx = std::max(mx, in[static_cast<size_t>(b - 1)]);
            if (b + 1 < numBands_)
                mx = std::max(mx, in[static_cast<size_t>(b + 1)]);
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

    static constexpr double kOdfRefFrame = 2048.0;
    static constexpr double kFMin = 27.5;
    static constexpr double kFMaxHz = 16000.0;
    static constexpr int kBandsPerOctave = 24;
    static constexpr std::size_t kBandCapacity = 256;
    static constexpr T kWhitenDecay = T(0.9995), kWhitenFloor = T(1e-4);
    double sampleRate_ = 44100;
    int fftSize_ = 0, numBins_ = 0, numBands_ = 1;
    bool phaseHistoryValid_ = true;
    bool prepared_ = false;
    T odfScale_ = T(1);
    std::vector<T> prevPhase_, prevPhase2_, prevMag_, whitenPeak_;
    std::vector<int> fbStart_, fbOffset_, fbCount_, bandRegister_;
    std::vector<T> fbWeights_, bandCur_, bandPrev_, bandMaxPrev_;
    std::array<T, kNumRegisters> curRegisters_{};
    std::array<int, kNumRegisters> registerBands_{};
};

} // namespace dspark::detail
