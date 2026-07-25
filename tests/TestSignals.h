// DSPark Test Signals - Reusable signal generators and measurement utilities

#pragma once

#include "../Core/DspMath.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include <cmath>
#include <cstdint>
#include <numeric>

namespace dspark::test {

// ============================================================================
// Default spec for tests
// ============================================================================

inline constexpr double kSampleRate   = 44100.0;
inline constexpr int    kBlockSize    = 512;
inline constexpr int    kNumChannels  = 2;

inline AudioSpec defaultSpec()
{
    return { .sampleRate = kSampleRate, .maxBlockSize = kBlockSize, .numChannels = kNumChannels };
}

inline AudioSpec monoSpec()
{
    return { .sampleRate = kSampleRate, .maxBlockSize = kBlockSize, .numChannels = 1 };
}

inline AudioSpec spec(double sr, int block, int ch)
{
    return { .sampleRate = sr, .maxBlockSize = block, .numChannels = ch };
}

// ============================================================================
// Signal generators (write into raw float*)
// ============================================================================

inline void generateSine(float* buf, int n, float freq, float sampleRate, float amp = 1.0f)
{
    const float inc = dspark::twoPi<float> * freq / sampleRate;
    for (int i = 0; i < n; ++i)
        buf[i] = amp * std::sin(inc * static_cast<float>(i));
}

inline void generateSine(double* buf, int n, double freq, double sampleRate, double amp = 1.0)
{
    const double inc = dspark::twoPi<double> * freq / sampleRate;
    for (int i = 0; i < n; ++i)
        buf[i] = amp * std::sin(inc * static_cast<double>(i));
}

inline void generateImpulse(float* buf, int n, float amp = 1.0f)
{
    for (int i = 0; i < n; ++i)
        buf[i] = (i == 0) ? amp : 0.0f;
}

inline void generateDC(float* buf, int n, float level)
{
    for (int i = 0; i < n; ++i)
        buf[i] = level;
}

inline void generateSilence(float* buf, int n)
{
    for (int i = 0; i < n; ++i)
        buf[i] = 0.0f;
}

inline void generateWhiteNoise(float* buf, int n, float amp = 1.0f, uint32_t seed = 12345u)
{
    // Simple xorshift32 - deterministic for reproducible tests
    uint32_t s = seed;
    for (int i = 0; i < n; ++i)
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Map to [-1, 1]
        buf[i] = amp * (static_cast<float>(s) / static_cast<float>(0xFFFFFFFFu) * 2.0f - 1.0f);
    }
}

inline void generateSweep(float* buf, int n, float startFreq, float endFreq, float sampleRate)
{
    // Linear chirp
    float phase = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(n);
        float freq = startFreq + (endFreq - startFreq) * t;
        phase += dspark::twoPi<float> * freq / sampleRate;
        buf[i] = std::sin(phase);
    }
}

inline void generateSquare(float* buf, int n, float freq, float sampleRate, float amp = 1.0f)
{
    const float period = sampleRate / freq;
    for (int i = 0; i < n; ++i)
        buf[i] = amp * (std::fmod(static_cast<float>(i), period) < period * 0.5f ? 1.0f : -1.0f);
}

// ============================================================================
// Measurement utilities
// ============================================================================

inline float measureRMS(const float* buf, int n)
{
    if (n <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += static_cast<double>(buf[i]) * static_cast<double>(buf[i]);
    return static_cast<float>(std::sqrt(sum / n));
}

inline float measurePeak(const float* buf, int n)
{
    float peak = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        float v = std::abs(buf[i]);
        if (v > peak) peak = v;
    }
    return peak;
}

inline float measurePeakDb(const float* buf, int n)
{
    float peak = measurePeak(buf, n);
    return dspark::gainToDecibels(peak);
}

inline float measureRMSDb(const float* buf, int n)
{
    float rms = measureRMS(buf, n);
    return dspark::gainToDecibels(rms);
}

inline bool isSilent(const float* buf, int n, float threshold = 1e-6f)
{
    for (int i = 0; i < n; ++i)
        if (std::abs(buf[i]) > threshold) return false;
    return true;
}

inline bool hasNaN(const float* buf, int n)
{
    for (int i = 0; i < n; ++i)
        if (std::isnan(buf[i]) || std::isinf(buf[i])) return true;
    return false;
}

/**
 * Measure magnitude of a specific frequency using single-bin DFT (Goertzel-like).
 * Returns linear magnitude (not dB).
 */
inline float measureFrequencyMagnitude(const float* buf, int n, float targetFreq, float sampleRate)
{
    double w = dspark::twoPi<double> * static_cast<double>(targetFreq) / static_cast<double>(sampleRate);
    double realPart = 0.0, imagPart = 0.0;
    for (int i = 0; i < n; ++i)
    {
        realPart += static_cast<double>(buf[i]) * std::cos(w * i);
        imagPart += static_cast<double>(buf[i]) * std::sin(w * i);
    }
    return static_cast<float>(std::sqrt(realPart * realPart + imagPart * imagPart) * 2.0 / n);
}

/**
 * Compute correlation coefficient between two buffers. 1.0 = identical, 0.0 = uncorrelated.
 */
inline float correlation(const float* a, const float* b, int n)
{
    double sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
    for (int i = 0; i < n; ++i)
    {
        double da = a[i], db = b[i];
        sumA += da;
        sumB += db;
        sumAB += da * db;
        sumA2 += da * da;
        sumB2 += db * db;
    }
    double num = n * sumAB - sumA * sumB;
    double den = std::sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
    return den > 1e-12 ? static_cast<float>(num / den) : 0.0f;
}

// ============================================================================
// Buffer helpers - create ready-to-use AudioBuffer + fill channels
// ============================================================================

struct TestBuffer
{
    AudioBuffer<float> buffer;
    int numSamples;

    AudioBufferView<float> view() { return buffer.toView(); }

    float* ch(int c) { return buffer.getChannel(c); }

    void fillSine(float freq, float sr, float amp = 1.0f)
    {
        for (int c = 0; c < buffer.getNumChannels(); ++c)
            generateSine(buffer.getChannel(c), numSamples, freq, sr, amp);
    }

    void fillDC(float level)
    {
        for (int c = 0; c < buffer.getNumChannels(); ++c)
            generateDC(buffer.getChannel(c), numSamples, level);
    }

    void fillSilence()
    {
        buffer.clear();
    }

    void fillImpulse(float amp = 1.0f)
    {
        buffer.clear();
        for (int c = 0; c < buffer.getNumChannels(); ++c)
            buffer.getChannel(c)[0] = amp;
    }

    void fillNoise(float amp = 1.0f, uint32_t seed = 12345u)
    {
        for (int c = 0; c < buffer.getNumChannels(); ++c)
            generateWhiteNoise(buffer.getChannel(c), numSamples, amp, seed + static_cast<uint32_t>(c) * 7919u);
    }
};

inline TestBuffer makeBuffer(int numChannels, int numSamples)
{
    TestBuffer tb;
    tb.numSamples = numSamples;
    tb.buffer.resize(numChannels, numSamples);
    return tb;
}

inline TestBuffer makeStereoBuffer(int numSamples = kBlockSize)
{
    return makeBuffer(2, numSamples);
}

inline TestBuffer makeMonoBuffer(int numSamples = kBlockSize)
{
    return makeBuffer(1, numSamples);
}

} // namespace dspark::test
