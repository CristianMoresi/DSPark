// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file FFT.h
 * @brief Fast Fourier Transform (Cooley-Tukey radix-2) with SIMD acceleration.
 *
 * Provides forward and inverse FFT for both complex and real-valued signals.
 * Optimised for power-of-two sizes typical in audio.
 *
 * Performance features:
 * - **SIMD-accelerated butterfly**: SSE3 on x86-64, NEON on ARM64.
 * - **Aligned Memory Operations**: Assumes and enforces 16-byte aligned pointers for SIMD.
 * - **Zero Allocations**: All buffers pre-allocated in constructors.
 *
 * @note To fully leverage the 32-byte alignment goal of the DSPark framework, 
 * ensure the pointers passed to forward()/inverse() are aligned, and consider 
 * using a custom aligned allocator for internal std::vector containers.
 */

#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define DSPARK_FFT_SSE3 1
    #include <pmmintrin.h> // Includes SSE3 (_mm_addsub_ps)
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DSPARK_FFT_NEON 1
    #include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dspark {

// ============================================================================
// FFTComplex
// ============================================================================

/**
 * @class FFTComplex
 * @brief In-place Cooley-Tukey radix-2 DIT FFT for complex data.
 *
 * Data layout: interleaved [re0, im0, re1, im1, ...], total 2*N elements.
 * * @tparam T Sample type (float or double).
 */
template <typename T>
class FFTComplex
{
public:
    /**
     * @brief Constructs an FFT processor for the given size.
     * @param size Number of complex samples. Must be a power of two and >= 2.
     * @throw std::invalid_argument if size is invalid.
     */
    explicit FFTComplex(size_t size)
        : size_(size)
    {
        if (size < 2 || (size & (size - 1)) != 0)
            throw std::invalid_argument("FFTComplex size must be a power of two >= 2");
            
        computeTwiddles();
        computeBitReversalTable();
    }

    /**
     * @brief Returns the FFT size (number of complex points).
     * @return The size provided at construction.
     */
    [[nodiscard]] size_t getSize() const noexcept { return size_; }

    /**
     * @brief Performs a forward (time->frequency) FFT in-place.
     * @param data Interleaved complex data [re, im, ...], 2*N elements.
     *
     * The pointer does not need to be 16-byte aligned: the SIMD path uses
     * unaligned loads/stores (`_mm_loadu_ps`/`_mm_storeu_ps`), which on every
     * x86 CPU released in the last decade run at the same throughput as the
     * aligned variants when the data happens to be aligned.
     */
    void forward(T* data) noexcept
    {
        bitReverse(data);
        butterflyPass(data, false);
    }

    /**
     * @brief Performs an inverse (frequency->time) FFT in-place.
     * @param data Interleaved complex data, overwritten with time-domain result.
     * @note The output is automatically scaled by 1/N. Unaligned-safe.
     */
    void inverse(T* data) noexcept
    {
        bitReverse(data);
        butterflyPass(data, true);

        const T invN = T(1) / static_cast<T>(size_);
        const size_t total = size_ * 2;
        for (size_t i = 0; i < total; ++i)
            data[i] *= invN;
    }

private:
    void computeTwiddles()
    {
        twiddles_.clear();
        size_t numStages = 0;
        for (size_t s = size_; s > 1; s >>= 1) ++numStages;

        size_t stride = 2;
        for (size_t stage = 0; stage < numStages; ++stage)
        {
            size_t halfStride = stride / 2;
            for (size_t k = 0; k < halfStride; ++k)
            {
                double angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k)
                               / static_cast<double>(stride);
                twiddles_.push_back(static_cast<T>(std::cos(angle)));
                twiddles_.push_back(static_cast<T>(std::sin(angle)));
            }
            stride *= 2;
        }
    }

    void computeBitReversalTable()
    {
        bitrev_.resize(size_);
        size_t bits = 0;
        for (size_t s = size_; s > 1; s >>= 1) ++bits;

        for (size_t i = 0; i < size_; ++i)
        {
            size_t rev = 0;
            size_t val = i;
            for (size_t b = 0; b < bits; ++b)
            {
                rev = (rev << 1) | (val & 1);
                val >>= 1;
            }
            bitrev_[i] = rev;
        }
    }

    void bitReverse(T* data) const noexcept
    {
        for (size_t i = 0; i < size_; ++i)
        {
            size_t j = bitrev_[i];
            if (i < j)
            {
                std::swap(data[2 * i],     data[2 * j]);
                std::swap(data[2 * i + 1], data[2 * j + 1]);
            }
        }
    }

    void butterflyPass(T* data, bool isInverse) const noexcept
    {
        size_t twiddleOffset = 0;
        size_t stride = 2;

        while (stride <= size_)
        {
            size_t halfStride = stride / 2;

            for (size_t group = 0; group < size_; group += stride)
            {
                size_t k = 0;

                // --- SIMD path: float only, 2 butterflies at a time -----------
#ifdef DSPARK_FFT_SSE3
                if constexpr (std::is_same_v<T, float>)
                {
                    __m128 invTwMask = _mm_setzero_ps();
                    if (isInverse)
                    {
                        alignas(16) static constexpr float kInvTw[4] = { 0.0f, -0.0f, 0.0f, -0.0f };
                        invTwMask = _mm_load_ps(kInvTw);  // safe: alignas(16)
                    }

                    for (; k + 1 < halfStride; k += 2)
                    {
                        const size_t twIdx = twiddleOffset + k * 2;
                        const size_t eIdx  = 2 * (group + k);
                        const size_t oIdx  = 2 * (group + k + halfStride);

                        // Unaligned loads — `data` and `twiddles_` come from
                        // std::vector<float> whose data pointer is only
                        // 4-byte aligned for float. Using _mm_load_ps here
                        // produced a SIGSEGV on every FFT triggered by
                        // SpectrumAnalyzer. _mm_loadu_ps is the same speed
                        // as the aligned form on Sandy-Bridge and newer.
                        __m128 e = _mm_loadu_ps(&data[eIdx]);
                        __m128 o = _mm_loadu_ps(&data[oIdx]);
                        __m128 w = _mm_xor_ps(_mm_loadu_ps(&twiddles_[twIdx]), invTwMask);

                        __m128 o_re  = _mm_shuffle_ps(o, o, _MM_SHUFFLE(2,2,0,0));
                        __m128 o_im  = _mm_shuffle_ps(o, o, _MM_SHUFFLE(3,3,1,1));
                        __m128 w_sw  = _mm_shuffle_ps(w, w, _MM_SHUFFLE(2,3,0,1));

                        __m128 p1    = _mm_mul_ps(w, o_re);
                        __m128 p2    = _mm_mul_ps(w_sw, o_im);

                        // _mm_addsub_ps inherently processes the complex conjugate math:
                        // Even: p1 - p2 (Real part) | Odd: p1 + p2 (Imaginary part)
                        __m128 t     = _mm_addsub_ps(p1, p2);

                        _mm_storeu_ps(&data[eIdx], _mm_add_ps(e, t));
                        _mm_storeu_ps(&data[oIdx], _mm_sub_ps(e, t));
                    }
                }
#endif // DSPARK_FFT_SSE3

#ifdef DSPARK_FFT_NEON
                if constexpr (std::is_same_v<T, float>)
                {
                    static const uint32_t kAddSub[4] = { 0x80000000u, 0, 0x80000000u, 0 };
                    const uint32x4_t addsubMask = vld1q_u32(kAddSub);

                    static const uint32_t kInvTw[4] = { 0, 0x80000000u, 0, 0x80000000u };
                    const uint32x4_t invTwMask = isInverse ? vld1q_u32(kInvTw) : vdupq_n_u32(0);

                    for (; k + 1 < halfStride; k += 2)
                    {
                        const size_t twIdx = twiddleOffset + k * 2;
                        const size_t eIdx  = 2 * (group + k);
                        const size_t oIdx  = 2 * (group + k + halfStride);

                        float32x4_t e = vld1q_f32(&data[eIdx]);
                        float32x4_t o = vld1q_f32(&data[oIdx]);
                        float32x4_t w = vreinterpretq_f32_u32(veorq_u32(
                            vreinterpretq_u32_f32(vld1q_f32(&twiddles_[twIdx])), invTwMask));

                        float32x4_t o_re = vtrn1q_f32(o, o);
                        float32x4_t o_im = vtrn2q_f32(o, o);
                        float32x4_t w_sw = vrev64q_f32(w);
                        float32x4_t p1   = vmulq_f32(w, o_re);
                        float32x4_t p2   = vmulq_f32(w_sw, o_im);
                        float32x4_t t    = vaddq_f32(p1,
                            vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(p2), addsubMask)));

                        vst1q_f32(&data[eIdx], vaddq_f32(e, t));
                        vst1q_f32(&data[oIdx], vsubq_f32(e, t));
                    }
                }
#endif // DSPARK_FFT_NEON

                // --- Scalar path: remainder + double + non-SIMD platforms -----
                for (; k < halfStride; ++k)
                {
                    size_t idx = twiddleOffset + k * 2;
                    T wr = twiddles_[idx];
                    T wi = twiddles_[idx + 1];

                    if (isInverse) wi = -wi;

                    size_t evenIdx = 2 * (group + k);
                    size_t oddIdx  = 2 * (group + k + halfStride);

                    T tr = wr * data[oddIdx] - wi * data[oddIdx + 1];
                    T ti = wr * data[oddIdx + 1] + wi * data[oddIdx];

                    data[oddIdx]     = data[evenIdx]     - tr;
                    data[oddIdx + 1] = data[evenIdx + 1] - ti;
                    data[evenIdx]    += tr;
                    data[evenIdx + 1] += ti;
                }
            }

            twiddleOffset += halfStride * 2;
            stride *= 2;
        }
    }

    size_t size_;
    std::vector<T> twiddles_;
    std::vector<size_t> bitrev_;
};

// ============================================================================
// FFTReal
// ============================================================================

/**
 * @class FFTReal
 * @brief FFT optimised for real-valued input signals (the common audio case).
 *
 * Uses a half-size complex FFT internally, saving ~50% computation.
 * * **Frequency domain layout**: N+2 elements (interleaved complex).
 * - Bins 0 to N/2 inclusive -> (N/2 + 1) complex values -> (N + 2) floats.
 * - `data[2*k]` = real part of bin k.
 * - `data[2*k+1]` = imaginary part of bin k.
 * - Bin 0 = DC, bin N/2 = Nyquist.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FFTReal
{
public:
    /**
     * @brief Constructs a real FFT processor.
     * @param size Number of real samples. Must be a power of two and >= 4.
     * @throw std::invalid_argument if size is invalid.
     */
    explicit FFTReal(size_t size)
        : realSize_(validateSize(size))   // validates BEFORE complexFFT_ is built
        , halfSize_(size / 2)
        , complexFFT_(size / 2)
    {
        computePostTwiddles();
        workBuffer_.resize(size);
    }

    /** @brief Returns the number of real input samples (N). */
    [[nodiscard]] size_t getSize() const noexcept { return realSize_; }

    /** @brief Returns the frequency-domain buffer size in elements (N + 2). */
    [[nodiscard]] size_t getFrequencyDomainSize() const noexcept { return realSize_ + 2; }

    /** @brief Returns the number of frequency bins (N/2 + 1) including DC and Nyquist. */
    [[nodiscard]] size_t getNumBins() const noexcept { return halfSize_ + 1; }

    /**
     * @brief Forward transform: real time-domain -> complex frequency-domain.
     * @param timeData Input: N real samples.
     * @param freqData Output: N+2 elements (interleaved complex, N/2+1 bins).
     */
    void forward(const T* timeData, T* freqData) noexcept
    {
        T* work = workBuffer_.data();
        for (size_t i = 0; i < halfSize_; ++i)
        {
            work[2 * i]     = timeData[2 * i];
            work[2 * i + 1] = timeData[2 * i + 1];
        }

        complexFFT_.forward(work);
        unpackForward(work, freqData);
    }

    /**
     * @brief Inverse transform: complex frequency-domain -> real time-domain.
     * @param freqData Input: N+2 elements (interleaved complex, N/2+1 bins).
     * @param timeData Output: N real samples.
     */
    void inverse(const T* freqData, T* timeData) noexcept
    {
        T* work = workBuffer_.data();
        packInverse(freqData, work);
        complexFFT_.inverse(work);

        for (size_t i = 0; i < halfSize_; ++i)
        {
            timeData[2 * i]     = work[2 * i];
            timeData[2 * i + 1] = work[2 * i + 1];
        }
    }

    /**
     * @brief Computes the magnitude of each frequency bin.
     * @param freqData   Input: frequency-domain data (N+2 elements).
     * @param magnitudes Output: N/2+1 magnitude values. Must be pre-allocated.
     */
    void computeMagnitudes(const T* freqData, T* magnitudes) const noexcept
    {
        for (size_t k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            magnitudes[k] = std::sqrt(re * re + im * im);
        }
    }

    /**
     * @brief Computes the phase angle of each frequency bin.
     * @param freqData Input: frequency-domain data (N+2 elements).
     * @param phases   Output: N/2+1 phase values in radians (-pi to +pi).
     */
    void computePhases(const T* freqData, T* phases) const noexcept
    {
        for (size_t k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            phases[k] = std::atan2(im, re);
        }
    }

    /**
     * @brief Computes the power spectrum (magnitude squared) of each bin.
     * @param freqData Input: frequency-domain data (N+2 elements).
     * @param power    Output: N/2+1 power values.
     */
    void computePowerSpectrum(const T* freqData, T* power) const noexcept
    {
        for (size_t k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            power[k] = re * re + im * im;
        }
    }

    /**
     * @brief Returns the frequency in Hz corresponding to a given bin index.
     * @param binIndex   Bin index (0 to N/2).
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (N).
     * @return Center frequency of the requested bin.
     */
    [[nodiscard]] static T binToFrequency(size_t binIndex, double sampleRate, size_t fftSize) noexcept
    {
        return static_cast<T>(static_cast<double>(binIndex) * sampleRate / static_cast<double>(fftSize));
    }

    /**
     * @brief Returns the bin index closest to a given frequency.
     * @param frequency  Frequency in Hz.
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (N).
     * @return Integer index of the nearest bin.
     */
    [[nodiscard]] static size_t frequencyToBin(double frequency, double sampleRate, size_t fftSize) noexcept
    {
        return static_cast<size_t>(std::round(frequency * static_cast<double>(fftSize) / sampleRate));
    }

private:
    /** @brief Validates the real-FFT size and returns it (used in the init list
     *  so the FFTReal-specific message fires before the inner FFTComplex). */
    static size_t validateSize(size_t size)
    {
        if (size < 4 || (size & (size - 1)) != 0)
            throw std::invalid_argument("FFTReal size must be a power of two >= 4");
        return size;
    }

    void computePostTwiddles()
    {
        postTwiddles_.resize(halfSize_ * 2);
        for (size_t k = 0; k < halfSize_; ++k)
        {
            double angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k)
                         / static_cast<double>(realSize_);
            postTwiddles_[2 * k]     = static_cast<T>(std::cos(angle));
            postTwiddles_[2 * k + 1] = static_cast<T>(std::sin(angle));
        }
    }

    void unpackForward(const T* halfFFT, T* fullSpectrum) const noexcept
    {
        const size_t N2 = halfSize_;

        T dcRe = halfFFT[0] + halfFFT[1];
        T nyRe = halfFFT[0] - halfFFT[1];

        fullSpectrum[0] = dcRe;
        fullSpectrum[1] = T(0);
        fullSpectrum[2 * N2]     = nyRe;
        fullSpectrum[2 * N2 + 1] = T(0);

        for (size_t k = 1; k < N2; ++k)
        {
            size_t kConj = N2 - k;

            T hkRe = halfFFT[2 * k];
            T hkIm = halfFFT[2 * k + 1];
            T hcRe = halfFFT[2 * kConj];
            T hcIm = halfFFT[2 * kConj + 1];

            T xeRe = T(0.5) * (hkRe + hcRe);
            T xeIm = T(0.5) * (hkIm - hcIm);

            T xoRe = T(0.5) * (hkRe - hcRe);
            T xoIm = T(0.5) * (hkIm + hcIm);

            T wr = postTwiddles_[2 * k];
            T wi = postTwiddles_[2 * k + 1];

            T joRe = xoIm;
            T joIm = -xoRe;

            T twRe = wr * joRe - wi * joIm;
            T twIm = wr * joIm + wi * joRe;

            fullSpectrum[2 * k]     = xeRe + twRe;
            fullSpectrum[2 * k + 1] = xeIm + twIm;
        }
    }

    void packInverse(const T* fullSpectrum, T* halfFFT) const noexcept
    {
        const size_t N2 = halfSize_;

        T dcRe = fullSpectrum[0];
        T nyRe = fullSpectrum[2 * N2];

        halfFFT[0] = T(0.5) * (dcRe + nyRe);
        halfFFT[1] = T(0.5) * (dcRe - nyRe);

        for (size_t k = 1; k < N2; ++k)
        {
            size_t kConj = N2 - k;

            T xkRe = fullSpectrum[2 * k];
            T xkIm = fullSpectrum[2 * k + 1];
            T xcRe = fullSpectrum[2 * kConj];
            T xcIm = fullSpectrum[2 * kConj + 1];

            T xeRe = T(0.5) * (xkRe + xcRe);
            T xeIm = T(0.5) * (xkIm - xcIm);

            T diffRe = T(0.5) * (xkRe - xcRe);
            T diffIm = T(0.5) * (xkIm + xcIm);

            T wr =  postTwiddles_[2 * k];
            T wi = -postTwiddles_[2 * k + 1];

            T twRe = wr * diffRe - wi * diffIm;
            T twIm = wr * diffIm + wi * diffRe;

            T xoRe = -twIm;
            T xoIm =  twRe;

            halfFFT[2 * k]     = xeRe + xoRe;
            halfFFT[2 * k + 1] = xeIm + xoIm;
        }
    }

    size_t realSize_;
    size_t halfSize_;
    FFTComplex<T> complexFFT_;
    std::vector<T> postTwiddles_; 
    std::vector<T> workBuffer_; // Removed mutable. Internal state mutations disqualify const.
};

} // namespace dspark