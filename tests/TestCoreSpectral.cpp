// DSPark Tests - Core Spectral
// FFT, WindowFunctions, Convolver, Hilbert, Resampler

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"
#include "../Core/Convolver.h"
#include "../Core/ZeroLatencyConvolver.h"
#include "../Core/Hilbert.h"
#include "../Core/Resampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// FFT
// ============================================================================

DSPARK_TEST(FFTReal_forward_inverse_roundtrip)
{
    constexpr int N = 1024;
    FFTReal<float> fft(N);

    // Create a test signal
    std::vector<float> input(N);
    generateSine(input.data(), N, 440.0f, 44100.0f);

    // Save a copy
    std::vector<float> original(input.begin(), input.end());

    // Forward
    std::vector<float> freq(fft.getFrequencyDomainSize());
    fft.forward(input.data(), freq.data());

    // Inverse
    std::vector<float> output(N);
    fft.inverse(freq.data(), output.data());

    // Should match original
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(output[i], original[i], 1e-4f);
}

DSPARK_TEST(FFTReal_Parseval_energy_conservation)
{
    constexpr int N = 512;
    FFTReal<float> fft(N);

    std::vector<float> input(N);
    generateSine(input.data(), N, 1000.0f, 44100.0f);

    // Time-domain energy
    double timeEnergy = 0.0;
    for (int i = 0; i < N; ++i)
        timeEnergy += static_cast<double>(input[i]) * input[i];

    // Forward FFT
    std::vector<float> freq(fft.getFrequencyDomainSize());
    fft.forward(input.data(), freq.data());

    // Frequency-domain energy (Parseval)
    int numBins = N / 2 + 1;
    double freqEnergy = 0.0;
    for (int i = 0; i < numBins; ++i)
    {
        double re = freq[i * 2];
        double im = freq[i * 2 + 1];
        double binEnergy = re * re + im * im;
        // DC and Nyquist bins count once, others count twice
        if (i == 0 || i == N / 2)
            freqEnergy += binEnergy;
        else
            freqEnergy += 2.0 * binEnergy;
    }
    freqEnergy /= N;

    EXPECT_NEAR(static_cast<float>(timeEnergy), static_cast<float>(freqEnergy),
                static_cast<float>(timeEnergy) * 0.01f);
}

DSPARK_TEST(FFTReal_sine_peak_at_correct_bin)
{
    constexpr int N = 1024;
    FFTReal<float> fft(N);
    const float sr = 44100.0f;
    const float testFreq = 1000.0f;

    std::vector<float> input(N);
    generateSine(input.data(), N, testFreq, sr);

    std::vector<float> freq(fft.getFrequencyDomainSize());
    fft.forward(input.data(), freq.data());

    // Find peak bin
    int numBins = N / 2 + 1;
    int peakBin = 0;
    float peakMag = 0.0f;
    for (int i = 0; i < numBins; ++i)
    {
        float re = freq[i * 2];
        float im = freq[i * 2 + 1];
        float mag = std::sqrt(re * re + im * im);
        if (mag > peakMag) { peakMag = mag; peakBin = i; }
    }

    float expectedBin = testFreq * N / sr;
    EXPECT_NEAR(static_cast<float>(peakBin), expectedBin, 1.5f);
}

DSPARK_TEST(FFTReal_silence_is_zero)
{
    constexpr int N = 256;
    FFTReal<float> fft(N);

    std::vector<float> input(N, 0.0f);
    std::vector<float> freq(fft.getFrequencyDomainSize());
    fft.forward(input.data(), freq.data());

    for (size_t i = 0; i < fft.getFrequencyDomainSize(); ++i)
        EXPECT_NEAR(freq[i], 0.0f, 1e-10f);
}

DSPARK_TEST(FFTReal_double_template)
{
    constexpr int N = 256;
    FFTReal<double> fft(N);

    std::vector<double> input(N);
    generateSine(input.data(), N, 440.0, 44100.0);

    std::vector<double> freq(fft.getFrequencyDomainSize());
    fft.forward(input.data(), freq.data());

    std::vector<double> output(N);
    fft.inverse(freq.data(), output.data());

    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(output[i], input[i], 1e-10);
}

// Forward transform against an INDEPENDENT reference: a naive O(N^2) DFT in
// double. A round-trip test alone would pass even if forward and inverse
// shared a compensating error (wrong twiddle sign, swapped bins, bad scale);
// this pins every bin of the forward spectrum on its own.
DSPARK_TEST(FFTReal_matches_reference_DFT)
{
    constexpr int N = 256;
    FFTReal<float> fft(N);

    std::vector<float> input(N);
    unsigned int rng = 0x9E3779B9u;
    for (int i = 0; i < N; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        input[i] = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }

    std::vector<float> freq(fft.getFrequencyDomainSize());
    fft.forward(input.data(), freq.data());

    float maxErr = 0.0f, maxMag = 0.0f;
    for (int k = 0; k <= N / 2; ++k)
    {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; ++n)
        {
            const double a = -2.0 * 3.14159265358979323846 * k * n / N;
            re += static_cast<double>(input[n]) * std::cos(a);
            im += static_cast<double>(input[n]) * std::sin(a);
        }
        maxErr = std::max(maxErr, std::fabs(freq[2 * k]     - static_cast<float>(re)));
        maxErr = std::max(maxErr, std::fabs(freq[2 * k + 1] - static_cast<float>(im)));
        maxMag = std::max(maxMag, static_cast<float>(std::sqrt(re * re + im * im)));
    }

    EXPECT_GT(maxMag, 1.0f);              // sanity: the spectrum is not trivial
    EXPECT_LT(maxErr / maxMag, 1e-5f);    // every bin matches the reference
}

// FFTComplex is public API but was only exercised indirectly through FFTReal.
// A shifted impulse has an exact analytic spectrum, X[k] = e^(-2*pi*i*k*n0/N):
// unit magnitude and linear phase on every bin. Also checks the inverse
// (conjugate twiddles + 1/N scale) recovers the impulse exactly.
DSPARK_TEST(FFTComplex_shifted_impulse_exact_spectrum)
{
    constexpr int N = 64;
    constexpr int n0 = 5;
    FFTComplex<float> fft(N);

    std::vector<float> data(2 * N, 0.0f);
    data[2 * n0] = 1.0f;

    fft.forward(data.data());
    for (int k = 0; k < N; ++k)
    {
        const double a = -2.0 * 3.14159265358979323846 * k * n0 / N;
        EXPECT_NEAR(data[2 * k],     static_cast<float>(std::cos(a)), 1e-5f);
        EXPECT_NEAR(data[2 * k + 1], static_cast<float>(std::sin(a)), 1e-5f);
    }

    fft.inverse(data.data());
    for (int i = 0; i < N; ++i)
    {
        EXPECT_NEAR(data[2 * i],     (i == n0) ? 1.0f : 0.0f, 1e-6f);
        EXPECT_NEAR(data[2 * i + 1], 0.0f, 1e-6f);
    }
}

// ============================================================================
// WindowFunctions
// ============================================================================

DSPARK_TEST(Window_Hann_symmetric)
{
    constexpr int N = 256;
    std::vector<float> win(N);
    WindowFunctions<float>::hann(win.data(), N, false); // Symmetric (not periodic)

    for (int i = 0; i < N / 2; ++i)
        EXPECT_NEAR(win[i], win[N - 1 - i], 1e-6f);
}

DSPARK_TEST(Window_Hann_endpoints)
{
    constexpr int N = 256;
    std::vector<float> win(N);
    WindowFunctions<float>::hann(win.data(), N, false);

    // Endpoints should be near zero
    EXPECT_NEAR(win[0], 0.0f, 0.01f);
    EXPECT_NEAR(win[N - 1], 0.0f, 0.01f);
}

DSPARK_TEST(Window_Hann_center_near_one)
{
    constexpr int N = 256;
    std::vector<float> win(N);
    WindowFunctions<float>::hann(win.data(), N, false);

    // Center should be near 1.0
    EXPECT_GT(win[N / 2], 0.95f);
}

DSPARK_TEST(Window_Rectangular_all_ones)
{
    constexpr int N = 64;
    std::vector<float> win(N);
    WindowFunctions<float>::rectangular(win.data(), N);

    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(win[i], 1.0f, 1e-6f);
}

DSPARK_TEST(Window_Blackman_symmetric)
{
    constexpr int N = 128;
    std::vector<float> win(N);
    WindowFunctions<float>::blackman(win.data(), N, false);

    for (int i = 0; i < N / 2; ++i)
        EXPECT_NEAR(win[i], win[N - 1 - i], 1e-5f);
}

// The symmetric Kaiser window's endpoints equal 1/I0(beta) exactly (x = +-1
// makes the numerator I0(0) = 1) and its centre equals 1. Checking the
// endpoints against tabulated I0 values pins the whole Bessel series -- the
// engine behind every Kaiser FIR design in the framework (Oversampling,
// Resampler, FIRFilter). References: I0(1) = 1.2660658777520084,
// I0(2) = 2.2795853023360673, I0(5) = 27.239871823604450.
DSPARK_TEST(Window_Kaiser_endpoints_match_bessel_reference)
{
    struct Case { double beta; double i0; };
    const Case cases[] = {
        { 1.0, 1.2660658777520084 },
        { 2.0, 2.2795853023360673 },
        { 5.0, 27.239871823604450 },
    };

    constexpr int N = 33; // odd: exact centre sample
    for (const auto& c : cases)
    {
        std::vector<double> win(N);
        WindowFunctions<double>::kaiser(win.data(), N, c.beta, false);

        EXPECT_NEAR(win[0],     1.0 / c.i0, 1e-12);
        EXPECT_NEAR(win[N - 1], 1.0 / c.i0, 1e-12);
        EXPECT_NEAR(win[N / 2], 1.0, 1e-12);

        // float path goes through the same double engine
        std::vector<float> winF(N);
        WindowFunctions<float>::kaiser(winF.data(), N, static_cast<float>(c.beta), false);
        EXPECT_NEAR(winF[0], static_cast<float>(1.0 / c.i0), 1e-7f);
    }
}

// Periodic Hann is exactly COLA at 50% overlap: w[i] + w[i + N/2] == 1 for
// every i. This is the property STFT processors (SpectralProcessor,
// PitchShifter) build on; the symmetric variant does NOT satisfy it, which is
// why periodic is the default.
DSPARK_TEST(Window_Hann_periodic_is_COLA_at_half_overlap)
{
    constexpr int N = 512;
    std::vector<float> win(N);
    WindowFunctions<float>::hann(win.data(), N, true);

    for (int i = 0; i < N / 2; ++i)
        EXPECT_NEAR(win[i] + win[i + N / 2], 1.0f, 1e-6f);
}

// Periodic cosine-sum windows have ANALYTIC gains: the cosine terms average
// to zero over a full period, so coherent gain == a0 and the energy gain
// follows from the sum of squared coefficients (Hann: sqrt(0.25 + 0.125)).
DSPARK_TEST(Window_gains_match_analytic_values)
{
    constexpr int N = 1024;
    std::vector<float> win(N);
    WindowFunctions<float>::hann(win.data(), N, true);

    EXPECT_NEAR(WindowFunctions<float>::coherentGain(win.data(), N), 0.5f, 1e-6f);
    EXPECT_NEAR(WindowFunctions<float>::energyGain(win.data(), N),
                0.61237244f /* sqrt(3/8) */, 1e-6f);

    WindowFunctions<float>::blackman(win.data(), N, true);
    EXPECT_NEAR(WindowFunctions<float>::coherentGain(win.data(), N), 0.42f, 1e-6f);
}

// The flat-top window's reason to exist: a tone at the WORST-CASE frequency
// (exactly between two bins) still reads its true amplitude from the peak bin
// after coherent-gain calibration. Hann loses ~1.4 dB there (scallop loss);
// flat-top must stay within 0.02 dB.
DSPARK_TEST(Window_flattop_amplitude_accurate_off_bin)
{
    constexpr int N = 1024;
    FFTReal<float> fft(N);

    std::vector<float> win(N), sig(N), freq(fft.getFrequencyDomainSize());
    WindowFunctions<float>::flatTop(win.data(), N, true);
    const float cg = WindowFunctions<float>::coherentGain(win.data(), N);

    constexpr float amp = 0.5f;
    const float bin = 100.5f; // worst case: halfway between bins
    for (int i = 0; i < N; ++i)
        sig[i] = amp * std::cos(2.0f * 3.14159265f * bin * static_cast<float>(i) / N);

    WindowFunctions<float>::apply(sig.data(), win.data(), N);
    fft.forward(sig.data(), freq.data());

    float peak = 0.0f;
    for (int k = 1; k < N / 2; ++k)
    {
        const float re = freq[2 * k], im = freq[2 * k + 1];
        peak = std::max(peak, std::sqrt(re * re + im * im));
    }

    const float measured = peak * 2.0f / (static_cast<float>(N) * cg);
    const float errDb = 20.0f * std::log10(measured / amp);
    EXPECT_NEAR(errDb, 0.0f, 0.02f);
}

// ============================================================================
// Convolver
// ============================================================================

DSPARK_TEST(Convolver_identity_IR)
{
    // Convolving with a unit impulse [1, 0, 0, ...] should return the original signal
    // Note: overlap-save has blockSize latency - first block fills the pipeline
    constexpr int blockSize = 256;
    constexpr int irLen = 1;
    float ir[1] = { 1.0f };

    Convolver<float> conv;
    conv.prepare(blockSize, ir, irLen);

    std::vector<float> input(blockSize);
    generateSine(input.data(), blockSize, 440.0f, 44100.0f);

    std::vector<float> output(blockSize);

    // First block fills pipeline (output is latency)
    conv.process(input.data(), output.data(), blockSize);

    // Second block: feed same signal, now output should match the first input
    std::vector<float> input2(blockSize);
    generateSine(input2.data(), blockSize, 440.0f, 44100.0f, 1.0f);
    conv.process(input2.data(), output.data(), blockSize);

    // Output should match the first input (delayed by one block)
    float errSum = 0.0f;
    float sigSum = 0.0f;
    for (int i = 0; i < blockSize; ++i)
    {
        float err = output[i] - input[i];
        errSum += err * err;
        sigSum += input[i] * input[i];
    }
    float snr = -10.0f * std::log10(errSum / (sigSum + 1e-30f));
    EXPECT_GT(snr, 40.0f);
}

DSPARK_TEST(Convolver_silence_in_silence_out)
{
    constexpr int blockSize = 256;
    float ir[] = { 0.5f, 0.3f, 0.2f };

    Convolver<float> conv;
    conv.prepare(blockSize, ir, 3);

    std::vector<float> input(blockSize, 0.0f);
    std::vector<float> output(blockSize);
    conv.process(input.data(), output.data(), blockSize);

    EXPECT_SILENT(output.data(), blockSize, 1e-8f);
}

DSPARK_TEST(Convolver_impulse_reproduces_IR)
{
    // Input = impulse -> output should equal IR (delayed by blockSize due to overlap-save)
    constexpr int blockSize = 256;
    float ir[] = { 1.0f, 0.5f, 0.25f, 0.125f };
    constexpr int irLen = 4;

    Convolver<float> conv;
    conv.prepare(blockSize, ir, irLen);

    std::vector<float> input(blockSize, 0.0f);
    input[0] = 1.0f; // Impulse

    std::vector<float> output(blockSize);
    // First block: fills pipeline
    conv.process(input.data(), output.data(), blockSize);

    // Second block: silence input, get the delayed impulse response
    std::vector<float> silence(blockSize, 0.0f);
    conv.process(silence.data(), output.data(), blockSize);

    // The IR should appear at the start of this block
    for (int i = 0; i < irLen; ++i)
        EXPECT_NEAR(output[i], ir[i], 0.01f);
}

// Direct null test of the uniform partitioned convolver: a random 1000-sample
// IR (4 partitions at block 256) against a double-precision direct
// convolution, streamed with VARIABLE call lengths that cross block
// boundaries (including runs longer than the block size -- the documented
// "arbitrary lengths" contract, previously only exercised indirectly through
// ZeroLatencyConvolver).
DSPARK_TEST(Convolver_nulls_against_direct_convolution)
{
    constexpr int blockSize = 256;
    constexpr int irLen = 1000;
    constexpr int total = 6144;

    std::vector<float> ir(irLen), sig(total);
    unsigned int rng = 0x1234ABCDu;
    auto frand = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    };
    for (auto& v : ir)  v = frand() * 0.5f;
    for (auto& v : sig) v = frand();

    Convolver<float> conv;
    conv.prepare(blockSize, ir.data(), irLen);
    const int latency = conv.getLatency();

    std::vector<float> out(total, 0.0f);
    int pos = 0;
    while (pos < total)
    {
        rng = rng * 1664525u + 1013904223u;
        int bs = 1 + static_cast<int>(rng % 384u); // 1..384 (can exceed blockSize)
        if (bs > total - pos) bs = total - pos;
        conv.process(sig.data() + pos, out.data() + pos, bs);
        pos += bs;
    }

    double errSum = 0.0, refSum = 0.0;
    for (int n = 0; n + latency < total; ++n)
    {
        double ref = 0.0;
        const int kMax = std::min(irLen - 1, n);
        for (int k = 0; k <= kMax; ++k)
            ref += static_cast<double>(ir[static_cast<size_t>(k)])
                 * static_cast<double>(sig[static_cast<size_t>(n - k)]);
        const double e = static_cast<double>(out[static_cast<size_t>(n + latency)]) - ref;
        errSum += e * e;
        refSum += ref * ref;
    }
    const float snrDb = static_cast<float>(-10.0 * std::log10(errSum / (refSum + 1e-30)));
    EXPECT_GT(snrDb, 90.0f);

    // In-place contract: process(data, data, n) must be bit-identical to the
    // out-of-place path (same code, same order).
    Convolver<float> convA, convB;
    convA.prepare(blockSize, ir.data(), irLen);
    convB.prepare(blockSize, ir.data(), irLen);
    std::vector<float> inPlace(sig), outB(total, 0.0f);
    for (int p = 0; p < total; p += blockSize)
    {
        convA.processInPlace(inPlace.data() + p, blockSize);
        convB.process(sig.data() + p, outB.data() + p, blockSize);
    }
    float maxDiff = 0.0f;
    for (int n = 0; n < total; ++n)
        maxDiff = std::max(maxDiff, std::abs(inPlace[static_cast<size_t>(n)] - outB[static_cast<size_t>(n)]));
    EXPECT_EQ(maxDiff, 0.0f);
}

// ============================================================================
// Hilbert
// ============================================================================

DSPARK_TEST(Hilbert_envelope_constant_for_sine)
{
    Hilbert<float> h;
    h.prepare(48000.0);

    // Warmup: allpass filters need settling time
    for (int i = 0; i < 8192; ++i)
    {
        float in = std::sin(twoPi<float> * 1000.0f * static_cast<float>(i) / 48000.0f);
        (void)h.process(in);
    }

    // Measure envelope: sqrt(real^2 + imag^2) should be approximately constant
    float minEnv = 10.0f, maxEnv = 0.0f;
    for (int i = 8192; i < 48000; ++i)
    {
        float in = std::sin(twoPi<float> * 1000.0f * static_cast<float>(i) / 48000.0f);
        auto [r, im] = h.process(in);
        float env = std::sqrt(r * r + im * im);
        if (env < minEnv) minEnv = env;
        if (env > maxEnv) maxEnv = env;
    }

    // The 191-tap windowed-sinc FIR Hilbert is essentially flat at 1 kHz, so the
    // analytic envelope sqrt(real^2 + imag^2) should be near-constant.
    float ratio = maxEnv / (minEnv + 1e-10f);
    EXPECT_LT(ratio, 1.05f); // < 5% ripple (measured ~0.03% for the FIR design at 1 kHz)
    EXPECT_GT(minEnv, 0.9f);  // Close to unity for a unit-amplitude sine
}

DSPARK_TEST(Hilbert_no_NaN)
{
    Hilbert<float> h;
    h.prepare(44100.0);

    for (int i = 0; i < 8192; ++i)
    {
        float in = std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);
        auto [r, im] = h.process(in);
        EXPECT_FALSE(std::isnan(r));
        EXPECT_FALSE(std::isnan(im));
    }
}

DSPARK_TEST(Hilbert_real_branch_is_exact_delayed_input)
{
    // The alignment contract the Compressor's Hilbert-detector compensation
    // relies on: real[n] == x[n - getLatencySamples()] BIT-exactly.
    Hilbert<float> h;
    h.prepare(48000.0);
    const int lat = Hilbert<float>::getLatencySamples();

    std::vector<float> x(2048);
    uint32_t rng = 777u;
    for (auto& v : x)
    {
        rng = rng * 1664525u + 1013904223u;
        v = static_cast<float>(rng >> 8) / static_cast<float>(1u << 24) - 0.5f;
    }

    for (int n = 0; n < 2048; ++n)
    {
        auto [re, im] = h.process(x[static_cast<size_t>(n)]);
        (void)im;
        const float expected = (n >= lat) ? x[static_cast<size_t>(n - lat)] : 0.0f;
        EXPECT_TRUE(re == expected);
    }
}

DSPARK_TEST(Hilbert_block_matches_per_sample_and_dc_rejection)
{
    // processBlock must be bit-identical to the per-sample path.
    Hilbert<float> blk, ref;
    blk.prepare(48000.0);
    ref.prepare(48000.0);

    std::vector<float> in(1024), outR(1024), outI(1024);
    for (int i = 0; i < 1024; ++i)
        in[static_cast<size_t>(i)] =
            std::sin(0.29f * static_cast<float>(i)) * 0.7f;

    blk.processBlock(std::span<const float>(in),
                     std::span<float>(outR), std::span<float>(outI));
    for (int i = 0; i < 1024; ++i)
    {
        auto [re, im] = ref.process(in[static_cast<size_t>(i)]);
        EXPECT_TRUE(outR[static_cast<size_t>(i)] == re);
        EXPECT_TRUE(outI[static_cast<size_t>(i)] == im);
    }

    // Antisymmetric (Type III) kernel: DC input produces ~zero quadrature.
    Hilbert<float> dc;
    dc.prepare(48000.0);
    float maxImag = 0.0f;
    for (int i = 0; i < 1000; ++i)
    {
        auto [re, im] = dc.process(1.0f);
        (void)re;
        if (i > 400) maxImag = std::max(maxImag, std::abs(im)); // after fill
    }
    EXPECT_LT(maxImag, 1e-5f);
}

// ============================================================================
// Resampler
// ============================================================================

DSPARK_TEST(Resampler_44100_to_48000_roundtrip)
{
    constexpr int N = 4096;
    std::vector<float> input(N);
    generateSine(input.data(), N, 440.0f, 44100.0f);

    // Upsample 44100 -> 48000
    Resampler<float> up;
    up.prepare(44100.0, 48000.0, Resampler<float>::Quality::High);
    auto upsampled = up.process(input.data(), N);

    // Downsample 48000 -> 44100
    Resampler<float> down;
    down.prepare(48000.0, 44100.0, Resampler<float>::Quality::High);
    auto output = down.process(upsampled.data(), static_cast<int>(upsampled.size()));

    // The sinc filters introduce group delay. Use cross-correlation to find alignment.
    int outLen = static_cast<int>(output.size());
    int bestOffset = 0;
    float bestCorr = -1.0f;
    int compareLen = std::min(outLen, N) - 200;

    if (compareLen > 200)
    {
        for (int offset = 0; offset < 100; ++offset)
        {
            float c = correlation(input.data() + 100, output.data() + 100 + offset,
                                  compareLen - 100);
            if (c > bestCorr) { bestCorr = c; bestOffset = offset; }
        }

        // After alignment, measure SNR
        float errSum = 0.0f, sigSum = 0.0f;
        for (int i = 200; i < 200 + compareLen / 2; ++i)
        {
            int j = i + bestOffset;
            if (j >= outLen) break;
            float err = output[j] - input[i];
            errSum += err * err;
            sigSum += input[i] * input[i];
        }
        float snrDb = -10.0f * std::log10(errSum / (sigSum + 1e-30f));
        EXPECT_GT(snrDb, 20.0f); // At least 20 dB SNR after aligned round-trip
        EXPECT_GT(bestCorr, 0.9f); // High correlation
    }
}

DSPARK_TEST(Resampler_same_rate_passthrough)
{
    constexpr int N = 512;
    std::vector<float> input(N);
    generateSine(input.data(), N, 440.0f, 44100.0f);

    Resampler<float> r;
    r.prepare(44100.0, 44100.0, Resampler<float>::Quality::Normal);
    auto output = r.process(input.data(), N);

    // Same rate should produce nearly identical output (within filter latency)
    EXPECT_TRUE(static_cast<int>(output.size()) >= N - 10);
}

DSPARK_TEST(Resampler_multichannel_AudioBufferView)
{
    constexpr int N = 1024;
    auto s = spec(44100.0, N, 2);

    Resampler<float> r;
    r.prepare(s, 48000.0, Resampler<float>::Quality::Normal);

    auto inBuf = makeBuffer(2, N);
    generateSine(inBuf.ch(0), N, 440.0f, 44100.0f);
    generateSine(inBuf.ch(1), N, 880.0f, 44100.0f);

    int maxOut = r.getMaxOutputSamples(N);
    auto outBuf = makeBuffer(2, maxOut);

    int produced = r.processBlock(inBuf.view(), outBuf.view());

    EXPECT_GT(produced, 0);
    // Expected: ~1024 * (48000/44100) ~ 1115
    EXPECT_NEAR(produced, 1115, 5);

    // Check output has signal on both channels
    float e0 = 0.0f, e1 = 0.0f;
    for (int i = 64; i < produced; ++i)
    {
        e0 += outBuf.ch(0)[i] * outBuf.ch(0)[i];
        e1 += outBuf.ch(1)[i] * outBuf.ch(1)[i];
    }
    EXPECT_GT(e0, 10.0f);
    EXPECT_GT(e1, 10.0f);
}

namespace {

// Least-squares fit of a sin/cos pair at frequency f over x[n0..n0+len).
// Returns the amplitude; optionally subtracts the fitted component in place.
// Fit-and-subtract measures residuals far below any window's leakage floor.
double resamplerFitTone(std::vector<double>& x, int n0, int len,
                        double f, double fs, bool subtract)
{
    double sss = 0, scc = 0, ssc = 0, sxs = 0, sxc = 0;
    const double w = 2.0 * 3.14159265358979323846 * f / fs;
    for (int i = 0; i < len; ++i)
    {
        const double ph = w * (n0 + i);
        const double s = std::sin(ph), c = std::cos(ph);
        sss += s * s; scc += c * c; ssc += s * c;
        sxs += x[static_cast<size_t>(n0 + i)] * s;
        sxc += x[static_cast<size_t>(n0 + i)] * c;
    }
    const double det = sss * scc - ssc * ssc;
    if (std::abs(det) < 1e-12) return 0.0;
    const double a = (sxs * scc - sxc * ssc) / det;
    const double b = (sxc * sss - sxs * ssc) / det;
    if (subtract)
        for (int i = 0; i < len; ++i)
        {
            const double ph = w * (n0 + i);
            x[static_cast<size_t>(n0 + i)] -= a * std::sin(ph) + b * std::cos(ph);
        }
    return std::sqrt(a * a + b * b);
}

// Streams a tone through a fresh resampler, returns the output as double.
std::vector<double> resamplerStreamTone(double srcRate, double dstRate,
                                        Resampler<float>::Quality q,
                                        double toneHz, int n)
{
    std::vector<float> in(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        in[static_cast<size_t>(i)] = static_cast<float>(
            std::sin(2.0 * 3.14159265358979323846 * toneHz * i / srcRate));

    Resampler<float> r;
    r.prepare(srcRate, dstRate, q);
    std::vector<float> out(static_cast<size_t>(r.getMaxOutputSamples(n)), 0.0f);
    const int produced = r.processBlock(in.data(), n, out.data());
    return { out.begin(), out.begin() + produced };
}

} // namespace

// The streaming path delays the signal by exactly sincPoints/2 input samples;
// getLatency() must report it. At integer output ratios the impulse peak index
// is exact; at fractional ratios the peak may land one sample off the rounded
// getter (the true delay is fractional in output samples).
DSPARK_TEST(Resampler_streaming_latency_matches_getter)
{
    using Q = Resampler<float>::Quality;
    for (Q q : { Q::Draft, Q::Normal, Q::High, Q::Ultra })
    {
        struct Case { double src, dst; bool exact; };
        const Case cases[] = {
            { 48000.0, 48000.0, true },   // ratio 1: integer latency
            { 48000.0, 96000.0, true },   // ratio 2: integer latency
            { 44100.0, 48000.0, false },  // fractional latency
        };
        for (const auto& c : cases)
        {
            constexpr int n0 = 1024, N = 4096;
            std::vector<float> in(static_cast<size_t>(N), 0.0f);
            in[n0] = 1.0f;

            Resampler<float> r;
            r.prepare(c.src, c.dst, q);
            std::vector<float> out(static_cast<size_t>(r.getMaxOutputSamples(N)), 0.0f);
            const int produced = r.processBlock(in.data(), N, out.data());

            int kmax = 0; float vmax = 0.0f;
            for (int k = 0; k < produced; ++k)
                if (std::abs(out[static_cast<size_t>(k)]) > vmax)
                {
                    vmax = std::abs(out[static_cast<size_t>(k)]);
                    kmax = k;
                }

            const double ratio = c.dst / c.src;
            const double measured = kmax - n0 * ratio;
            if (c.exact)
            {
                EXPECT_NEAR(measured, static_cast<double>(r.getLatency()), 1e-9);
                EXPECT_NEAR(vmax, 1.0f, 1e-5f); // on-grid impulse passes intact
            }
            else
            {
                EXPECT_NEAR(measured, static_cast<double>(r.getLatency()), 1.0);
            }
        }
    }
}

// At equal rates the kernel's phase-0 branch is an exact discrete delta, so
// streaming must be a pure delay of getLatency() samples (also fixes the
// one-output-per-input cadence).
DSPARK_TEST(Resampler_ratio_one_is_pure_delay)
{
    constexpr int N = 8192;
    std::vector<float> in(static_cast<size_t>(N));
    uint32_t rng = 12345u;
    for (auto& v : in)
    {
        rng = rng * 1664525u + 1013904223u;
        v = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }

    Resampler<float> r;
    r.prepare(48000.0, 48000.0, Resampler<float>::Quality::Normal);
    std::vector<float> out(static_cast<size_t>(r.getMaxOutputSamples(N)), 0.0f);
    const int produced = r.processBlock(in.data(), N, out.data());
    EXPECT_EQ(produced, N);

    const int lat = r.getLatency();
    float maxErr = 0.0f;
    for (int k = lat; k < produced; ++k)
        maxErr = std::max(maxErr,
                          std::abs(out[static_cast<size_t>(k)] -
                                   in[static_cast<size_t>(k - lat)]));
    EXPECT_LT(maxErr, 1e-9f);
}

// Offline (time-aligned) and streaming (causal) share the same kernels: at
// ratio 2 the phase sequence is exact in both paths, so the streaming output
// must equal the offline output shifted by the latency, bit for bit.
DSPARK_TEST(Resampler_offline_matches_streaming_shifted)
{
    constexpr int N = 8192;
    std::vector<float> in(static_cast<size_t>(N));
    uint32_t rng = 777u;
    for (auto& v : in)
    {
        rng = rng * 1664525u + 1013904223u;
        v = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }

    Resampler<float> r;
    r.prepare(48000.0, 96000.0, Resampler<float>::Quality::Normal);
    auto offline = r.process(in.data(), N);

    Resampler<float> rs;
    rs.prepare(48000.0, 96000.0, Resampler<float>::Quality::Normal);
    std::vector<float> streamed(static_cast<size_t>(rs.getMaxOutputSamples(N)), 0.0f);
    const int produced = rs.processBlock(in.data(), N, streamed.data());

    const int lat = rs.getLatency(); // 2 * halfSinc at ratio 2
    float maxDiff = 0.0f;
    int compared = 0;
    for (int k = 512; k < produced; ++k)
    {
        const int j = k - lat;
        if (j < 0 || j >= static_cast<int>(offline.size())) break;
        maxDiff = std::max(maxDiff,
                           std::abs(streamed[static_cast<size_t>(k)] -
                                    offline[static_cast<size_t>(j)]));
        ++compared;
    }
    EXPECT_GT(compared, 4096);
    EXPECT_TRUE(maxDiff == 0.0f);
}

// Upsampling image rejection per quality tier (the numbers documented in the
// class table, with margin). A 20 kHz tone at 48->96 kHz leaves its image at
// 28 kHz; fit-and-subtract of the fundamental exposes everything else.
DSPARK_TEST(Resampler_quality_tiers_reject_upsampling_images)
{
    using Q = Resampler<float>::Quality;
    struct Tier { Q q; double maxResidualDb; };
    const Tier tiers[] = {
        { Q::Draft,  -9.0 },   // measured -12 dB
        { Q::Normal, -50.0 },  // measured -56 dB
        { Q::High,   -120.0 }, // measured -142 dB
        { Q::Ultra,  -120.0 }, // measured -139 dB
    };

    for (const auto& t : tiers)
    {
        constexpr int N = 1 << 15;
        auto out = resamplerStreamTone(48000.0, 96000.0, t.q, 20000.0, N);
        const int n0 = 4096;
        const int len = static_cast<int>(out.size()) - 8192;

        const double fund = resamplerFitTone(out, n0, len, 20000.0, 96000.0, true);
        double rms = 0.0;
        for (int i = 0; i < len; ++i)
            rms += out[static_cast<size_t>(n0 + i)] * out[static_cast<size_t>(n0 + i)];
        rms = std::sqrt(rms / len);

        const double residualDb = 20.0 * std::log10(rms / (fund / std::sqrt(2.0)) + 1e-30);
        EXPECT_LT(residualDb, t.maxResidualDb);

        // High and Ultra must also keep the 20 kHz passband intact.
        if (t.q == Q::High || t.q == Q::Ultra)
            EXPECT_NEAR(fund, 1.0, 0.01);
    }
}

// Downsampling: a 26 kHz tone at 96->44.1 kHz sits above the target Nyquist
// and must be rejected (it would alias to 18.1 kHz), while an in-band 10 kHz
// tone must pass at unity. The 26 kHz case probes the transition band right
// at the band edge: only Ultra's kernel keeps it in the deep stopband.
DSPARK_TEST(Resampler_downsampling_rejects_aliases)
{
    using Q = Resampler<float>::Quality;
    struct Tier { Q q; double maxAliasDb; };
    const Tier tiers[] = {
        { Q::Normal, -20.0 },  // measured -27 dB
        { Q::High,   -55.0 },  // measured -69 dB (inside its transition band)
        { Q::Ultra,  -120.0 }, // measured -143 dB
    };

    for (const auto& t : tiers)
    {
        constexpr int N = 1 << 16;
        auto out = resamplerStreamTone(96000.0, 44100.0, t.q, 26000.0, N);
        const int n0 = 4096;
        const int len = static_cast<int>(out.size()) - 8192;

        const double alias = resamplerFitTone(out, n0, len, 44100.0 - 26000.0, 44100.0, false);
        EXPECT_LT(20.0 * std::log10(alias + 1e-30), t.maxAliasDb);

        auto pass = resamplerStreamTone(96000.0, 44100.0, t.q, 10000.0, N);
        const int lenP = static_cast<int>(pass.size()) - 8192;
        const double level = resamplerFitTone(pass, n0, lenP, 10000.0, 44100.0, false);
        EXPECT_NEAR(level, 1.0, 0.01);
    }
}


// ============================================================================
// ZeroLatencyConvolver (Gardner non-uniform partitioning)
// ============================================================================

namespace {

// Direct convolution in double as ground truth for null tests.
std::vector<double> zlcDirectConv(const std::vector<float>& x, const std::vector<float>& h)
{
    std::vector<double> y(x.size(), 0.0);
    for (size_t n = 0; n < x.size(); ++n)
    {
        double acc = 0.0;
        const size_t kMax = std::min(n + 1, h.size());
        for (size_t k = 0; k < kMax; ++k)
            acc += static_cast<double>(h[k]) * static_cast<double>(x[n - k]);
        y[n] = acc;
    }
    return y;
}

// Residual (dB) of the ZL convolver against direct convolution for a random
// IR of `irLen`, streaming with the given block size (0 = random sizes).
double zlcResidualDb(int irLen, int blockSize, int headSize = 128)
{
    uint32_t rng = 0x13572468u;
    auto frand = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    };

    std::vector<float> ir(static_cast<size_t>(irLen));
    for (auto& v : ir) v = frand();
    const int total = 24000;
    std::vector<float> x(static_cast<size_t>(total));
    for (auto& v : x) v = 0.5f * frand();

    const auto ref = zlcDirectConv(x, ir);

    ZeroLatencyConvolver<float> conv;
    conv.prepare(ir.data(), irLen, headSize);

    std::vector<float> y = x;
    int i = 0;
    uint32_t br = 7u;
    while (i < total)
    {
        int n = blockSize;
        if (n <= 0)
        {
            br = br * 1103515245u + 12345u;
            n = 1 + static_cast<int>((br >> 16) % 700);
        }
        n = std::min(n, total - i);
        conv.processInPlace(y.data() + i, n);
        i += n;
    }

    double err = 0, refPow = 0;
    for (int k = 0; k < total; ++k)
    {
        const double e = static_cast<double>(y[static_cast<size_t>(k)]) - ref[static_cast<size_t>(k)];
        err += e * e;
        refPow += ref[static_cast<size_t>(k)] * ref[static_cast<size_t>(k)];
    }
    return 10.0 * std::log10((err + 1e-30) / (refPow + 1e-30));
}

} // namespace

DSPARK_TEST(ZeroLatencyConvolver_nulls_against_direct_all_levels)
{
    // 6000-tap IR exercises head + uniform mid + time-distributed tail.
    EXPECT_LT(zlcResidualDb(6000, 512), -100.0);
}

DSPARK_TEST(ZeroLatencyConvolver_head_and_mid_only)
{
    EXPECT_LT(zlcResidualDb(100, 512), -100.0);    // head only
    EXPECT_LT(zlcResidualDb(1500, 512), -100.0);   // head + mid
}

DSPARK_TEST(ZeroLatencyConvolver_arbitrary_block_sizes)
{
    EXPECT_LT(zlcResidualDb(6000, 32), -100.0);    // tiny
    EXPECT_LT(zlcResidualDb(6000, 480), -100.0);   // non-power-of-two
    EXPECT_LT(zlcResidualDb(6000, 0), -100.0);     // random 1..700 per call
}

DSPARK_TEST(ZeroLatencyConvolver_custom_head_sizes)
{
    // Both clamp corners, plus requests that must round (100 -> 128) and
    // clamp (4096 -> 512). The mid level's block follows the head size, so
    // this re-validates the whole level alignment at each resolved value.
    EXPECT_LT(zlcResidualDb(6000, 512, 32),   -100.0);
    EXPECT_LT(zlcResidualDb(6000, 512, 512),  -100.0);
    EXPECT_LT(zlcResidualDb(6000, 512, 100),  -100.0);
    EXPECT_LT(zlcResidualDb(6000, 512, 4096), -100.0);

    ZeroLatencyConvolver<float> conv;
    std::vector<float> ir(3000, 0.01f);
    conv.prepare(ir.data(), 3000, 100);
    EXPECT_EQ(conv.getHeadSize(), 128);
    conv.prepare(ir.data(), 3000, 4096);
    EXPECT_EQ(conv.getHeadSize(), 512);
}

// reset() in the middle of a time-distributed tail task must discard the
// half-done task and leave the engine bit-identical to a freshly prepared
// one (same IR, same subsequent input).
DSPARK_TEST(ZeroLatencyConvolver_reset_mid_task_matches_fresh)
{
    uint32_t rng = 0xBEEF1234u;
    auto frand = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    };

    std::vector<float> ir(6000);
    for (auto& v : ir) v = frand();

    ZeroLatencyConvolver<float> used, fresh;
    used.prepare(ir.data(), static_cast<int>(ir.size()));
    fresh.prepare(ir.data(), static_cast<int>(ir.size()));

    // Drive 1500 samples: the first tail task launches at 1024 and is only
    // ~46% through its unit quota here.
    std::vector<float> warm(1500);
    for (auto& v : warm) v = frand();
    used.processInPlace(warm.data(), 1500);

    used.reset();

    std::vector<float> a(4096), b;
    for (auto& v : a) v = frand();
    b = a;
    used.processInPlace(a.data(), static_cast<int>(a.size()));
    fresh.processInPlace(b.data(), static_cast<int>(b.size()));

    float maxDiff = 0.0f;
    for (size_t n = 0; n < a.size(); ++n)
        maxDiff = std::max(maxDiff, std::abs(a[n] - b[n]));
    EXPECT_EQ(maxDiff, 0.0f);
}

DSPARK_TEST(ZeroLatencyConvolver_is_zero_latency)
{
    // ir[0] must land on the very first output sample.
    std::vector<float> ir(6000, 0.0f);
    ir[0] = 0.8f;
    ir[1] = -0.25f;

    ZeroLatencyConvolver<float> conv;
    conv.prepare(ir.data(), static_cast<int>(ir.size()));
    EXPECT_EQ(conv.getLatency(), 0);

    std::vector<float> x(64, 0.0f);
    x[0] = 1.0f;
    conv.processInPlace(x.data(), 64);
    EXPECT_NEAR(x[0], 0.8f, 1e-6f);
    EXPECT_NEAR(x[1], -0.25f, 1e-6f);
}

DSPARK_TEST(ZeroLatencyConvolver_cpu_is_flat)
{
    // Acceptance criterion: no per-block spike above 2x the mean. The work
    // pattern is deterministic, so run the same block sequence several times
    // and keep the per-block MINIMUM - that filters OS scheduling noise out
    // of the wall-clock measurement while preserving any algorithmic spike.
    std::vector<float> ir(48000);
    uint32_t rng = 99u;
    for (auto& v : ir)
    {
        rng = rng * 1664525u + 1013904223u;
        v = 0.01f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
    }

    ZeroLatencyConvolver<float> conv;
    conv.prepare(ir.data(), static_cast<int>(ir.size()));

    constexpr int kBlock = 512;
    constexpr int kRuns = 5;
    std::vector<float> buf(kBlock, 0.1f);
    const int blocks = 48000 * 2 / kBlock;
    std::vector<double> best(static_cast<size_t>(blocks), 1e30);

    for (int run = 0; run < kRuns; ++run)
    {
        conv.reset();
        for (int b = 0; b < blocks; ++b)
        {
            const auto t0 = std::chrono::steady_clock::now();
            conv.processInPlace(buf.data(), kBlock);
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            best[static_cast<size_t>(b)] = std::min(best[static_cast<size_t>(b)], us);
        }
    }

    double mean = 0, mx = 0;
    const int start = 16;
    for (int b = start; b < blocks; ++b)
    {
        mean += best[static_cast<size_t>(b)];
        mx = std::max(mx, best[static_cast<size_t>(b)]);
    }
    mean /= (blocks - start);
    EXPECT_LT(mx / mean, 2.0);
}
