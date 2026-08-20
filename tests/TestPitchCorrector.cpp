// DSPark - PitchCorrector contract and acceptance tests
//
// Everything the corrected pitch is measured with in this file is written from
// scratch: a time-domain autocorrelation f0 estimator, a heterodyne
// fundamental tracker and a cepstral spectral-envelope centroid. None of them
// share code with the framework's own pitch detector or FFT, so a defect in
// the detector cannot hide behind a test that asks the same detector whether
// the output is in tune. Both oracles are pinned against synthetic signals
// whose answer is known in closed form before they are used to judge anything.

#include "dspark_test.h"
#include "../Effects/PitchCorrector.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace dspark;

namespace {

// -- Independent f0 oracle ----------------------------------------------------
// Normalized autocorrelation over a fixed span. A periodic signal correlates
// just as well at every multiple of its period, so the coarse pass takes the
// SMALLEST local maximum that reaches most of the global peak; the refinement
// then measures the highest multiple of that period which still fits the span,
// dividing the parabolic interpolation error by that multiple.

double acfAt(const std::vector<double>& x, int lag, int span)
{
    double num = 0.0, e0 = 0.0, e1 = 0.0;
    for (int n = 0; n < span; ++n)
    {
        const double a = x[static_cast<std::size_t>(n)];
        const double b = x[static_cast<std::size_t>(n + lag)];
        num += a * b;
        e0 += a * a;
        e1 += b * b;
    }
    const double den = std::sqrt(e0 * e1);
    return den > 0.0 ? num / den : 0.0;
}

double estimateF0(const std::vector<double>& input, double sampleRate,
                  double fMin, double fMax)
{
    const int n = static_cast<int>(input.size());
    std::vector<double> x(input.begin(), input.end());
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(n);
    for (double& v : x) v -= mean;

    const int minLag = std::max(2, static_cast<int>(std::floor(sampleRate / fMax)));
    const int maxLag = static_cast<int>(std::ceil(sampleRate / fMin));
    if (minLag >= maxLag || maxLag * 2 >= n) return 0.0;
    const int span = n - maxLag;

    std::vector<double> r(static_cast<std::size_t>(maxLag + 2), 0.0);
    double globalMax = -2.0;
    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        r[static_cast<std::size_t>(lag)] = acfAt(x, lag, span);
        globalMax = std::max(globalMax, r[static_cast<std::size_t>(lag)]);
    }
    int coarse = -1;
    for (int lag = minLag + 1; lag < maxLag; ++lag)
    {
        const double v = r[static_cast<std::size_t>(lag)];
        if (v < 0.93 * globalMax) continue;
        if (v >= r[static_cast<std::size_t>(lag - 1)]
            && v >= r[static_cast<std::size_t>(lag + 1)])
        { coarse = lag; break; }
    }
    if (coarse < 0) return 0.0;

    int multiple = std::max(1, (maxLag - 1) / coarse);
    int centre = multiple * coarse;
    const int radius = std::max(2, coarse / 4);
    while (centre + radius + 1 >= n - span && multiple > 1)
    {
        --multiple;
        centre = multiple * coarse;
    }
    int peak = centre;
    double peakValue = -2.0;
    for (int lag = centre - radius; lag <= centre + radius; ++lag)
    {
        if (lag < minLag || lag + span >= n) continue;
        const double v = acfAt(x, lag, span);
        if (v > peakValue) { peakValue = v; peak = lag; }
    }
    const double ym = acfAt(x, peak - 1, span);
    const double y0 = acfAt(x, peak, span);
    const double yp = acfAt(x, peak + 1, span);
    const double denom = ym - 2.0 * y0 + yp;
    const double delta = denom != 0.0 ? 0.5 * (ym - yp) / denom : 0.0;
    const double refined = static_cast<double>(peak)
                         + (std::abs(delta) < 1.0 ? delta : 0.0);
    return refined > 0.0 ? sampleRate * static_cast<double>(multiple) / refined : 0.0;
}

double centsBetween(double hz, double referenceHz)
{
    if (hz <= 0.0 || referenceHz <= 0.0) return 1.0e9;
    return 1200.0 * std::log2(hz / referenceHz);
}

double midiToHz(double midi) { return 440.0 * std::exp2((midi - 69.0) / 12.0); }

// -- Independent time-resolved fundamental tracker ----------------------------
// Heterodyne to baseband at a reference frequency and average over exactly one
// reference period: that boxcar has spectral nulls at every multiple of the
// reference, which is where the harmonics and the negative-frequency image
// land, so the fundamental survives alone. The instantaneous frequency is then
// the phase difference across one more period, i.e. the average frequency
// between the two window centres. The reported value at index i describes the
// signal at i + D/2 - (N-1)/2 and is shifted back by that offset here; what
// remains is a symmetric smoothing over D samples, so a frequency step reads
// as complete D/2 samples late - the conservative direction for a
// time-to-target measurement.

std::vector<double> trackFundamental(const std::vector<double>& signal,
                                     double sampleRate, double referenceHz)
{
    const int n = static_cast<int>(signal.size());
    const int period = std::max(4, static_cast<int>(std::lround(sampleRate / referenceHz)));
    std::vector<double> hz(static_cast<std::size_t>(n), 0.0);
    if (n < 4 * period) return hz;

    std::vector<std::complex<double>> smoothed(static_cast<std::size_t>(n),
                                               std::complex<double>(0.0, 0.0));
    std::complex<double> running(0.0, 0.0);
    std::vector<std::complex<double>> mixed(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double w = -twoPi<double> * referenceHz * static_cast<double>(i) / sampleRate;
        mixed[static_cast<std::size_t>(i)] = signal[static_cast<std::size_t>(i)]
            * std::complex<double>(std::cos(w), std::sin(w));
    }
    for (int i = 0; i < n; ++i)
    {
        running += mixed[static_cast<std::size_t>(i)];
        if (i >= period) running -= mixed[static_cast<std::size_t>(i - period)];
        if (i >= period - 1)
            smoothed[static_cast<std::size_t>(i)] = running / static_cast<double>(period);
    }
    const int diff = period;
    std::vector<double> raw(static_cast<std::size_t>(n), 0.0);
    for (int i = period - 1; i + diff < n; ++i)
    {
        const std::complex<double> a = smoothed[static_cast<std::size_t>(i)];
        const std::complex<double> b = smoothed[static_cast<std::size_t>(i + diff)];
        if (std::abs(a) <= 0.0 || std::abs(b) <= 0.0) continue;
        raw[static_cast<std::size_t>(i)] = referenceHz
            + sampleRate * std::arg(b / a) / (twoPi<double> * static_cast<double>(diff));
    }
    const int shift = static_cast<int>(std::lround(0.5 * static_cast<double>(diff)
                                                   - 0.5 * static_cast<double>(period - 1)));
    for (int i = std::max(0, shift); i < n; ++i)
        hz[static_cast<std::size_t>(i)] = raw[static_cast<std::size_t>(i - shift)];
    return hz;
}

// First sample from `from` onward after which the tracked pitch never again
// leaves `toleranceCents` of the target, searched backwards so the cost is
// linear in the measurement window.
int arrivalSample(const std::vector<double>& hz, int from, int to, double targetHz,
                  double toleranceCents)
{
    int arrival = from;
    for (int i = to - 1; i >= from; --i)
    {
        if (std::abs(centsBetween(hz[static_cast<std::size_t>(i)], targetHz))
            > toleranceCents)
        { arrival = i + 1; break; }
    }
    return arrival < to ? arrival : -1;
}

// -- Independent spectrum helpers ---------------------------------------------

void fftRadix2(std::vector<std::complex<double>>& a)
{
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i)
    {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1)
    {
        const double angle = -twoPi<double> / static_cast<double>(len);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len)
        {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k)
            {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= step;
            }
        }
    }
}

// Centroid of the magnitude spectrum, and of the cepstrally smoothed spectral
// ENVELOPE, averaged over consecutive frames. The envelope form follows the
// formant structure instead of the harmonic comb, which is what a formant
// claim is about; the raw form is reported as well because it is the coarser
// and more easily reproduced measure of the same drift.
struct Centroids { double raw = 0.0; double envelope = 0.0; };

Centroids spectralCentroids(const std::vector<double>& signal, double sampleRate,
                            int frameSize, double loHz, double hiHz)
{
    const int frames = static_cast<int>(signal.size()) / frameSize;
    Centroids out;
    if (frames <= 0) return out;
    const int lift = std::max(2, static_cast<int>(0.001 * sampleRate));   // 1 ms
    double rawTotal = 0.0, envTotal = 0.0;
    int counted = 0;
    for (int f = 0; f < frames; ++f)
    {
        std::vector<std::complex<double>> spec(static_cast<std::size_t>(frameSize));
        for (int i = 0; i < frameSize; ++i)
        {
            const double w = 0.5 - 0.5 * std::cos(twoPi<double> * static_cast<double>(i)
                                                  / static_cast<double>(frameSize));
            spec[static_cast<std::size_t>(i)] =
                signal[static_cast<std::size_t>(f * frameSize + i)] * w;
        }
        fftRadix2(spec);
        std::vector<std::complex<double>> logMag(static_cast<std::size_t>(frameSize));
        for (int k = 0; k < frameSize; ++k)
            logMag[static_cast<std::size_t>(k)] =
                std::log(std::max(std::abs(spec[static_cast<std::size_t>(k)]), 1.0e-12));
        fftRadix2(logMag);
        for (int q = lift; q <= frameSize - lift; ++q)
            logMag[static_cast<std::size_t>(q)] = 0.0;
        fftRadix2(logMag);

        double rawNum = 0.0, rawDen = 0.0, envNum = 0.0, envDen = 0.0;
        for (int k = 1; k < frameSize / 2; ++k)
        {
            const double hz = sampleRate * static_cast<double>(k)
                            / static_cast<double>(frameSize);
            if (hz < loHz || hz > hiHz) continue;
            const double mag = std::abs(spec[static_cast<std::size_t>(k)]);
            rawNum += hz * mag;
            rawDen += mag;
            const double env = std::exp(logMag[static_cast<std::size_t>(k)].real()
                                        / static_cast<double>(frameSize));
            envNum += hz * env;
            envDen += env;
        }
        if (rawDen > 0.0 && envDen > 0.0)
        {
            rawTotal += rawNum / rawDen;
            envTotal += envNum / envDen;
            ++counted;
        }
    }
    if (counted > 0)
    {
        out.raw = rawTotal / static_cast<double>(counted);
        out.envelope = envTotal / static_cast<double>(counted);
    }
    return out;
}

// -- Test signals -------------------------------------------------------------

std::vector<double> harmonicTone(double f0, double sampleRate, int n,
                                 double amplitude = 0.35)
{
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const int maxHarmonic = static_cast<int>(0.45 * sampleRate / f0);
    for (int k = 1; k <= maxHarmonic; ++k)
    {
        const double a = amplitude / static_cast<double>(k);
        const double w = twoPi<double> * f0 * static_cast<double>(k) / sampleRate;
        const double phase = 0.37 * static_cast<double>(k);
        for (int i = 0; i < n; ++i)
            out[static_cast<std::size_t>(i)] += a * std::sin(w * static_cast<double>(i) + phase);
    }
    return out;
}

// Source-filter vowel: the harmonic source through three fixed formant
// resonators, so the spectral envelope does not follow f0.
std::vector<double> vowelTone(double f0, double sampleRate, int n)
{
    const std::vector<double> source = harmonicTone(f0, sampleRate, n, 0.25);
    const double formantHz[3] = { 730.0, 1090.0, 2440.0 };
    const double bandwidthHz[3] = { 80.0, 90.0, 120.0 };
    const double gain[3] = { 1.0, 0.55, 0.25 };
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int b = 0; b < 3; ++b)
    {
        const double r = std::exp(-pi<double> * bandwidthHz[b] / sampleRate);
        const double theta = twoPi<double> * formantHz[b] / sampleRate;
        const double a1 = -2.0 * r * std::cos(theta);
        const double a2 = r * r;
        double z1 = 0.0, z2 = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double y = source[static_cast<std::size_t>(i)] - a1 * z1 - a2 * z2;
            z2 = z1;
            z1 = y;
            out[static_cast<std::size_t>(i)] += gain[b] * (1.0 - r) * y;
        }
    }
    return out;
}

// -- Harmonic integrity: the stimulus, and an instrument that proves itself ---
//
// A harmonic-to-residual ratio in decibels is a property of the instrument that
// reads it as much as of the audio it reads. A probe counts whatever its window
// leaks outside the capture band as residual, even on a signal that has none,
// so a probe with a true ratio R and its own leakage floor F reports
// 1/M = 1/R + 1/F: near its floor it is measuring itself and not the product,
// and a threshold quoted without its instrument is not a threshold but a
// reading. The two calibrations below are therefore run before any number from
// this instrument is allowed to decide anything.

// Additive source-filter vowel. Because it is a sum of exact harmonics of f0,
// its residual is zero by construction, which is what lets the same signal
// serve as the instrument's calibration reference. The harmonic phases come
// from a stated integer recurrence rather than from a standard-library
// distribution, whose mapping onto a real interval is not specified across
// implementations, so this stimulus is identical on every platform.
std::vector<double> vowelStimulus(double f0, double sampleRate, int n)
{
    const double formantHz[5]   = { 730.0, 1090.0, 2440.0, 3400.0, 4500.0 };
    const double bandwidthHz[5] = {  90.0,  110.0,  140.0,  180.0,  220.0 };
    const double gain[5]        = {   1.0,    0.7,    0.4,    0.2,    0.1 };

    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const int harmonics = static_cast<int>(
        std::floor(std::min(8000.0, 0.45 * sampleRate) / f0));
    std::uint32_t state = 12345u;
    for (int k = 1; k <= harmonics; ++k)
    {
        const double f = static_cast<double>(k) * f0;
        double envelope = 0.0;
        for (int j = 0; j < 5; ++j)
        {
            const double detune = f * f - formantHz[j] * formantHz[j];
            const double magnitude = std::sqrt(
                detune * detune + 4.0 * f * f * bandwidthHz[j] * bandwidthHz[j]);
            envelope += gain[j] * formantHz[j] * bandwidthHz[j]
                      / std::max(magnitude, 1.0e-9);
        }
        const double amplitude = envelope / (1.0 + (f / 300.0) * (f / 300.0));
        state = 1664525u * state + 1013904223u;
        const double phase = twoPi<double> * static_cast<double>(state) / 4294967296.0;
        const double omega = twoPi<double> * f / sampleRate;
        for (int i = 0; i < n; ++i)
            out[static_cast<std::size_t>(i)] +=
                amplitude * std::sin(omega * static_cast<double>(i) + phase);
    }
    double peak = 0.0;
    for (double v : out) peak = std::max(peak, std::abs(v));
    if (peak > 0.0) for (double& v : out) v *= 0.5 / peak;
    return out;
}

// The analysis segment: the largest power of two whose duration does not exceed
// 0.75 s. Stated in TIME, so the resolution in Hz is the same at every rate; a
// rule that rounds UP to a power of two gives two adjacent supported rates a
// factor of two in resolution and makes them disagree about the same audio.
int analysisSegment(double sampleRate)
{
    int n = 1;
    while (2.0 * static_cast<double>(n) <= 0.75 * sampleRate) n <<= 1;
    return n;
}

// Harmonic-to-residual ratio in dB over one segment: one window over the whole
// segment and one DFT of exactly that many points, with NO zero padding, which
// would widen the window's main lobe in bins while the capture stayed where it
// was. The capture is four bins around each harmonic because the four-term
// Blackman-Harris window nulls at four bins: capturing less than its main lobe
// counts the skirt as residual for every harmonic and drops the instrument's
// own floor from 89 dB to 45 dB, which is below the values it has to judge.
// `f0` is the MEASURED fundamental of this segment, never the nominal target:
// on a nominal grid a pitch error walks the grid off the harmonics and this
// measurement double-counts a failure the accuracy tests already own.
double harmonicResidualDb(const std::vector<double>& x, double sampleRate, double f0)
{
    const int n = static_cast<int>(x.size());
    if (n <= 0 || (n & (n - 1)) != 0 || f0 <= 0.0) return -300.0;
    std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const double w = 0.35875 - 0.48829 * std::cos(twoPi<double> * t)
                       + 0.14128 * std::cos(2.0 * twoPi<double> * t)
                       - 0.01168 * std::cos(3.0 * twoPi<double> * t);
        spectrum[static_cast<std::size_t>(i)] = { x[static_cast<std::size_t>(i)] * w, 0.0 };
    }
    fftRadix2(spectrum);

    const double binHz = sampleRate / static_cast<double>(n);
    const int lowest = 2;                      // DC leakage, out of BOTH sums
    const int top = std::min(n / 2, static_cast<int>(
        std::floor(std::min(20000.0, 0.45 * sampleRate) / binHz)));
    std::vector<char> isHarmonic(static_cast<std::size_t>(top) + 1, 0);
    for (int k = 1; static_cast<double>(k) * f0 <= 8000.0; ++k)
    {
        const double centre = static_cast<double>(k) * f0 / binHz;
        const int lo = static_cast<int>(std::ceil(centre - 4.0));
        const int hi = static_cast<int>(std::floor(centre + 4.0));
        for (int b = std::max(lo, lowest); b <= std::min(hi, top); ++b)
            isHarmonic[static_cast<std::size_t>(b)] = 1;
    }
    double harmonic = 0.0, residual = 0.0;
    for (int b = lowest; b <= top; ++b)
    {
        const double power = std::norm(spectrum[static_cast<std::size_t>(b)]);
        if (isHarmonic[static_cast<std::size_t>(b)]) harmonic += power;
        else                                        residual += power;
    }
    if (harmonic <= 0.0) return -300.0;
    return 10.0 * std::log10(harmonic / std::max(residual, 1.0e-300));
}

// `clean` plus white Gaussian noise band-limited to exactly the bins the
// instrument sums as residual, scaled so the in-band ratio of the sum is
// `snrDb`. Band-limiting is the point: noise spread to Nyquist, read by a probe
// whose residual band stops short of it, measures a bookkeeping mismatch
// instead of the probe. The deviates come from a stated recurrence rather than
// from <random>, whose engines' mapping onto doubles is implementation-defined.
std::vector<double> plantNoise(const std::vector<double>& clean, double sampleRate,
                               double snrDb, std::uint64_t seed)
{
    const int n = static_cast<int>(clean.size());
    if (n <= 0 || (n & (n - 1)) != 0) return clean;

    std::uint64_t state = seed;
    auto uniform = [&state]() {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    };
    std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i += 2)
    {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = twoPi<double> * uniform();
        spectrum[static_cast<std::size_t>(i)] = { radius * std::cos(angle), 0.0 };
        if (i + 1 < n)
            spectrum[static_cast<std::size_t>(i + 1)] = { radius * std::sin(angle), 0.0 };
    }
    fftRadix2(spectrum);

    const double binHz = sampleRate / static_cast<double>(n);
    const int top = std::min(n / 2, static_cast<int>(
        std::floor(std::min(20000.0, 0.45 * sampleRate) / binHz)));
    for (int b = 0; b <= n / 2; ++b)
    {
        if (b >= 2 && b <= top) continue;
        spectrum[static_cast<std::size_t>(b)] = { 0.0, 0.0 };
        if (b > 0 && b < n / 2) spectrum[static_cast<std::size_t>(n - b)] = { 0.0, 0.0 };
    }
    for (auto& value : spectrum) value = std::conj(value);   // inverse through
    fftRadix2(spectrum);                                     // the forward one

    double noisePower = 0.0, signalPower = 0.0;
    std::vector<double> noise(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
    {
        noise[static_cast<std::size_t>(i)] =
            spectrum[static_cast<std::size_t>(i)].real() / static_cast<double>(n);
        noisePower += noise[static_cast<std::size_t>(i)] * noise[static_cast<std::size_t>(i)];
        signalPower += clean[static_cast<std::size_t>(i)] * clean[static_cast<std::size_t>(i)];
    }
    if (noisePower <= 0.0) return clean;
    const double scale = std::sqrt(signalPower * std::pow(10.0, -snrDb / 10.0) / noisePower);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = clean[static_cast<std::size_t>(i)]
                                         + scale * noise[static_cast<std::size_t>(i)];
    return out;
}

// -- Corrector driver ---------------------------------------------------------

constexpr std::uint16_t kChromatic = 0x0FFF;
constexpr std::uint16_t kMajor = (1u << 0) | (1u << 2) | (1u << 4) | (1u << 5)
                               | (1u << 7) | (1u << 9) | (1u << 11);
constexpr std::uint16_t kMinorPentatonic = (1u << 0) | (1u << 3) | (1u << 5)
                                         | (1u << 7) | (1u << 10);

struct Run
{
    double sampleRate = 48000.0;
    int block = 256;
    std::uint16_t mask = kChromatic;
    int root = 0;
    double retuneMs = 0.0;
    bool formant = false;
    int channels = 1;
    int frame = 0;                    ///< 0 = the rate's automatic analysis frame
    int scaleAt = -1;                 ///< absolute sample of a second setScale
    std::uint16_t secondMask = 0;
    int secondRoot = 0;
};

// Renders `mono` through one corrector on every channel and returns channel 0.
template <typename T>
std::vector<std::vector<T>> renderAll(const std::vector<double>& mono, const Run& run)
{
    const int n = static_cast<int>(mono.size());
    std::vector<std::vector<T>> channels(static_cast<std::size_t>(run.channels),
                                         std::vector<T>(static_cast<std::size_t>(n), T(0)));
    for (auto& ch : channels)
        for (int i = 0; i < n; ++i)
            ch[static_cast<std::size_t>(i)] = static_cast<T>(mono[static_cast<std::size_t>(i)]);

    AudioSpec spec;
    spec.sampleRate = run.sampleRate;
    spec.maxBlockSize = std::max(run.block, 1);
    spec.numChannels = run.channels;

    PitchCorrector<T> corrector;
    corrector.setScale(run.mask, run.root);
    corrector.setRetuneSpeedMs(static_cast<T>(run.retuneMs));
    corrector.setFormantPreserve(run.formant);
    corrector.prepare(spec, run.frame);

    int position = 0;
    while (position < n)
    {
        int length = std::min(run.block, n - position);
        if (run.scaleAt > position && run.scaleAt < position + length)
            length = run.scaleAt - position;
        if (position == run.scaleAt) corrector.setScale(run.secondMask, run.secondRoot);
        // Sized like the view's own channel limit, so a case that asks for more
        // channels than the array holds cannot exist.
        T* pointers[16] = {};
        for (int ch = 0; ch < run.channels && ch < 16; ++ch)
            pointers[ch] = channels[static_cast<std::size_t>(ch)].data() + position;
        corrector.processBlock(AudioBufferView<T>(pointers, run.channels, length));
        position += length;
    }
    return channels;
}

std::vector<double> render(const std::vector<double>& mono, const Run& run)
{
    const auto channels = renderAll<float>(mono, run);
    const auto& first = channels.front();
    return std::vector<double>(first.begin(), first.end());
}

std::vector<double> slice(const std::vector<double>& x, int from, int count)
{
    return std::vector<double>(x.begin() + from, x.begin() + from + count);
}

// The note the corrector must snap to: the nearest member of the rotated mask,
// ties to the lower note, computed here independently of the header.
int expectedTarget(int nearMidi, std::uint16_t mask, int root)
{
    const auto absolute = static_cast<std::uint32_t>(harmony::scaleAtRoot(mask, root));
    int best = nearMidi;
    double bestDistance = 1.0e9;
    for (int note = nearMidi - 12; note <= nearMidi + 12; ++note)
    {
        const int pitchClass = ((note % 12) + 12) % 12;
        if ((absolute & (1u << pitchClass)) == 0u) continue;
        const double distance = std::abs(static_cast<double>(note - nearMidi));
        if (distance < bestDistance) { bestDistance = distance; best = note; }
    }
    return best;
}

} // namespace

// ============================================================================
// Oracles first: nothing below is trustworthy unless these hold.
// ============================================================================

DSPARK_TEST(PitchCorrector_oracles_recover_known_signals)
{
    for (double hz : { 82.407, 220.0, 440.0, 1046.502 })
    {
        const auto x = harmonicTone(hz, 48000.0, 24000);
        const double measured = estimateF0(x, 48000.0, 60.0, 1400.0);
        EXPECT_LT(std::abs(centsBetween(measured, hz)), 0.5);
    }

    // A tone whose frequency steps by one semitone at a known sample: the
    // tracker must report both plateaus exactly and place the step within a
    // few milliseconds of where it was written.
    const double fs = 48000.0;
    const int n = 24000;
    const int stepAt = 12000;
    const double target = 220.0 * std::exp2(1.0 / 12.0);
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    double phase = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double f = (i < stepAt) ? 220.0 : target;
        x[static_cast<std::size_t>(i)] = 0.5 * std::sin(phase) + 0.2 * std::sin(2.0 * phase);
        phase += twoPi<double> * f / fs;
    }
    const auto hz = trackFundamental(x, fs, target);
    EXPECT_LT(std::abs(centsBetween(hz[static_cast<std::size_t>(n - 3000)], target)), 0.5);
    const int arrival = arrivalSample(hz, stepAt, n - 2000, target, 10.0);
    EXPECT_TRUE(arrival >= stepAt);
    EXPECT_LT(1000.0 * static_cast<double>(arrival - stepAt) / fs, 5.0);
}

// ============================================================================
// Contract
// ============================================================================

DSPARK_TEST(PitchCorrector_public_api_float_double)
{
    PitchCorrector<float> f;
    PitchCorrector<double> d;

    // Unprepared: no latency, and audio passes through untouched.
    EXPECT_EQ(f.getLatency(), 0);
    std::vector<float> data(64, 0.25f);
    float* pointer = data.data();
    f.processBlock(AudioBufferView<float>(&pointer, 1, 64));
    for (float v : data) EXPECT_EQ(v, 0.25f);

    // An invalid spec is ignored and leaves the instance unprepared.
    AudioSpec bad;
    bad.sampleRate = 0.0;
    bad.maxBlockSize = 0;
    bad.numChannels = 0;
    f.prepare(bad);
    EXPECT_EQ(f.getLatency(), 0);

    AudioSpec spec;
    spec.sampleRate = 48000.0;
    spec.maxBlockSize = 512;
    spec.numChannels = 2;
    f.prepare(spec);
    d.prepare(spec);
    EXPECT_EQ(f.getLatency(), 4096);   // two frames of the internal 2048 shifter
    EXPECT_EQ(d.getLatency(), 4096);

    // Defaults: chromatic scale at C, hard snap, formants free to move.
    EXPECT_EQ(static_cast<int>(f.getScaleMask()), 0x0FFF);
    EXPECT_EQ(f.getRootPitchClass(), 0);
    EXPECT_EQ(f.getRetuneSpeedMs(), 0.0f);
    EXPECT_FALSE(f.getFormantPreserve());

    // Scale packing: the mask is kept as given, the root folds into 0..11 and
    // bits above the low twelve are dropped.
    f.setScale(static_cast<std::uint16_t>(kMajor | 0xF000u), -13);
    EXPECT_EQ(static_cast<int>(f.getScaleMask()), static_cast<int>(kMajor));
    EXPECT_EQ(f.getRootPitchClass(), 11);
    f.setScale(kMinorPentatonic, 21);
    EXPECT_EQ(static_cast<int>(f.getScaleMask()), static_cast<int>(kMinorPentatonic));
    EXPECT_EQ(f.getRootPitchClass(), 9);

    // Retune speed: clamped, and non-finite values leave it alone.
    f.setRetuneSpeedMs(-5.0f);
    EXPECT_EQ(f.getRetuneSpeedMs(), 0.0f);
    f.setRetuneSpeedMs(1.0e9f);
    EXPECT_EQ(f.getRetuneSpeedMs(), 10000.0f);
    f.setRetuneSpeedMs(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(f.getRetuneSpeedMs(), 10000.0f);
    f.setRetuneSpeedMs(40.0f);
    EXPECT_EQ(f.getRetuneSpeedMs(), 40.0f);

    f.setFormantPreserve(true);
    EXPECT_TRUE(f.getFormantPreserve());
    f.setFormantPreserve(false);
    EXPECT_FALSE(f.getFormantPreserve());

    // A zero-length and a channel-less block are both no-ops, not crashes.
    float* none = nullptr;
    f.processBlock(AudioBufferView<float>(&none, 0, 0));
    std::vector<float> two(128, 0.0f);
    float* stereo[2] = { two.data(), two.data() + 64 };
    f.processBlock(AudioBufferView<float>(stereo, 2, 0));
}

DSPARK_TEST(PitchCorrector_invalid_rates_preserve_the_prepared_stream)
{
    auto check = []<typename T>() {
        AudioSpec valid { 48000.0, 64, 1 };
        PitchCorrector<T> subject;
        PitchCorrector<T> reference;
        subject.prepare(valid);
        reference.prepare(valid);

        const int frame = subject.getFrameSize();
        const int latency = subject.getLatency();
        double phase = 0.0;
        auto processEqualBlock = [&]() {
            std::vector<T> subjectSamples(64);
            std::vector<T> referenceSamples(64);
            for (int sample = 0; sample < 64; ++sample)
            {
                const T value = static_cast<T>(0.2 * std::sin(phase));
                phase += twoPi<double> * 220.0 / valid.sampleRate;
                subjectSamples[static_cast<std::size_t>(sample)] = value;
                referenceSamples[static_cast<std::size_t>(sample)] = value;
            }
            T* subjectChannel = subjectSamples.data();
            T* referenceChannel = referenceSamples.data();
            subject.processBlock(AudioBufferView<T>(&subjectChannel, 1, 64));
            reference.processBlock(AudioBufferView<T>(&referenceChannel, 1, 64));
            for (int sample = 0; sample < 64; ++sample)
                EXPECT_EQ(subjectSamples[static_cast<std::size_t>(sample)],
                          referenceSamples[static_cast<std::size_t>(sample)]);
        };

        for (int block = 0; block < 8; ++block)
            processEqualBlock();

        const double invalidRates[] = {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN(),
            -std::numeric_limits<double>::infinity(),
            0.0,
            -48000.0
        };
        for (double rate : invalidRates)
        {
            AudioSpec invalid = valid;
            invalid.sampleRate = rate;
            subject.prepare(invalid);
            EXPECT_EQ(subject.getFrameSize(), frame);
            EXPECT_EQ(subject.getLatency(), latency);
            processEqualBlock();
        }
    };

    check.template operator()<float>();
    check.template operator()<double>();
}

// ============================================================================
// Steady-state accuracy, against the independent autocorrelation oracle
// ============================================================================

namespace {

// Runs one steady tone detuned from an in-scale note and returns the error of
// the corrected output, in cents from that note.
double steadyErrorCents(double sampleRate, std::uint16_t mask, int root,
                        int note, double detuneCents)
{
    const int target = expectedTarget(note, mask, root);
    const double f0 = midiToHz(static_cast<double>(target) + detuneCents / 100.0);
    const int n = static_cast<int>(sampleRate * 0.45);
    const auto input = harmonicTone(f0, sampleRate, n);
    Run run;
    run.sampleRate = sampleRate;
    run.mask = mask;
    run.root = root;
    const auto output = render(input, run);
    const int measured = static_cast<int>(sampleRate * 0.2);
    const auto segment = slice(output, n - measured, measured);
    return centsBetween(estimateF0(segment, sampleRate, 60.0, 1400.0), midiToHz(target));
}

} // namespace

DSPARK_TEST(PitchCorrector_steady_state_within_ten_cents_48k)
{
    struct Case { std::uint16_t mask; int root; };
    const Case scales[] = { { kChromatic, 0 }, { kMajor, 0 }, { kMinorPentatonic, 9 } };
    const int notes[] = { 40, 57, 69, 84 };            // E2, A3, A4, C6
    const double detunes[] = { -45.0, 45.0 };
    for (const auto& scale : scales)
        for (int note : notes)
            for (double detune : detunes)
                EXPECT_LT(std::abs(steadyErrorCents(48000.0, scale.mask, scale.root,
                                                    note, detune)), 10.0);
}

DSPARK_TEST(PitchCorrector_steady_state_within_ten_cents_44k)
{
    const int notes[] = { 45, 60, 76 };                // A2, C4, E5
    for (int note : notes)
        for (double detune : { -20.0, 20.0 })
            EXPECT_LT(std::abs(steadyErrorCents(44100.0, kMajor, 0, note, detune)), 10.0);
}

DSPARK_TEST(PitchCorrector_empty_scale_leaves_the_pitch_alone)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 0.45);
    const double f0 = midiToHz(69.0 - 0.40);           // 40 cents flat of A4
    const auto input = harmonicTone(f0, fs, n);
    Run run;
    run.mask = 0;                                      // nothing to snap to
    const auto output = render(input, run);
    const auto segment = slice(output, n - 9600, 9600);
    EXPECT_LT(std::abs(centsBetween(estimateF0(segment, fs, 60.0, 1400.0), f0)), 10.0);
}

DSPARK_TEST(PitchCorrector_quantizer_boundary_picks_the_nearer_note)
{
    // C major has no C#, so everything between C4 and D4 is decided at their
    // midpoint. Ten cents either side of it must land on opposite notes. The
    // midpoint ITSELF is not tested end to end: the decision is taken on the
    // detected fundamental, which never lands on it exactly, and the header
    // documents that measure-zero case rather than pretending to control it.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 0.45);
    struct Case { double midi; double expected; };
    for (const Case& c : { Case { 60.90, 60.0 }, Case { 61.10, 62.0 } })
    {
        const auto input = harmonicTone(midiToHz(c.midi), fs, n);
        Run run;
        run.mask = kMajor;
        const auto output = render(input, run);
        const auto segment = slice(output, n - 9600, 9600);
        const double measured = estimateF0(segment, fs, 60.0, 1400.0);
        EXPECT_LT(std::abs(centsBetween(measured, midiToHz(c.expected))), 10.0);
    }
}

// ============================================================================
// Retune timing
// ============================================================================

namespace {

// Time in milliseconds from a scale change until the output settles within
// 10 cents of the new note and stays there. `phase` offsets the change inside
// the vocoder's analysis-hop period, and `firstMask` selects the correction it
// starts from, so the SIZE of the change is what the caller chooses.
double timeToTargetMs(double sampleRate, double retuneMs, std::uint16_t secondMask,
                      double targetMidi, int phase = 0,
                      std::uint16_t firstMask = kChromatic,
                      double inputMidi = 69.0)
{
    const int n = static_cast<int>(sampleRate * 3.0);
    const int stepAt = static_cast<int>(sampleRate * 0.6) + phase;
    const auto input = harmonicTone(midiToHz(inputMidi), sampleRate, n);
    Run run;
    run.sampleRate = sampleRate;
    run.retuneMs = retuneMs;
    run.mask = firstMask;
    run.scaleAt = stepAt;
    run.secondMask = secondMask;
    const auto output = render(input, run);
    const double target = midiToHz(targetMidi);
    const auto hz = trackFundamental(output, sampleRate, target);
    const int arrival = arrivalSample(hz, stepAt, n - 4000, target, 10.0);
    if (arrival < 0) return 1.0e9;
    return 1000.0 * static_cast<double>(arrival - stepAt) / sampleRate;
}

// One-note scales that force an exactly known correction on A4: A# puts it a
// semitone up. Both sizes that matter follow from the widest gap of the mask -
// the largest correction is half that gap and the largest CHANGE between two
// corrections is the whole gap, taken when the sung pitch crosses its middle.
// The chromatic default therefore reaches a change of one semitone, a diatonic
// scale two, harmonic minor and the pentatonics three, Hirajoshi four, and a
// one-note mask twelve.
constexpr std::uint16_t kOnlyASharp = static_cast<std::uint16_t>(1u << 10);

// The change lands at an absolute sample the caller chooses, so its phase
// against the vocoder's analysis hop is set rather than lucky: the settling
// time varies about two-fold across that period.
constexpr int kNearHopStart = 0;
constexpr int kMidHop = 256;

std::uint16_t noteMask(int midi)
{
    return static_cast<std::uint16_t>(1u << (((midi % 12) + 12) % 12));
}

// How long the phase vocoder needs to traverse a correction change, from its
// own published slew law: at most half a semitone per analysis hop, and the
// hop is fftSize/4 input samples divided by the active ratio, so it dilates
// while the active shift is negative.
//
//   T_slew(D, s0) = (fftSize/4)/fs * SUM(i = 0..k-1) 2^(-s_i/12)
//   k = ceil((|D| - 0.1) / 0.5)
//
// The settling bound is max(30 ms, 1.10 * T_slew): a wall-clock promise where
// the pipeline's own latency can hide the transition, and the engine's law
// above it. The 10% covers the estimator window of the measurement.
double slewLawMs(double semitoneChange, double startShift, double sampleRate,
                 int frameSize = 2048)
{
    const int steps = static_cast<int>(std::ceil((std::abs(semitoneChange) - 0.1) / 0.5));
    const double target = startShift + semitoneChange;
    const double direction = semitoneChange >= 0.0 ? 1.0 : -1.0;
    double shift = startShift;
    double sum = 0.0;
    for (int i = 0; i < steps; ++i)
    {
        sum += std::exp2(-shift / 12.0);
        shift += 0.5 * direction;
        shift = direction > 0.0 ? std::min(shift, target) : std::max(shift, target);
    }
    return 1000.0 * (static_cast<double>(frameSize) / 4.0) / sampleRate * sum;
}

double ruledBoundMs(double semitoneChange, double startShift, double sampleRate)
{
    return std::max(30.0, 1.10 * slewLawMs(semitoneChange, startShift, sampleRate));
}

// One gap of a published scale, crossed at its middle: the input sits at the
// midpoint and the two one-note masks are the legal targets on either side, so
// the correction steps by exactly the gap width - the largest change that
// scale can ask for.
void checkGapCrossing(int lowerMidi, int upperMidi)
{
    const double size = static_cast<double>(upperMidi - lowerMidi);
    const double inputMidi = 0.5 * static_cast<double>(lowerMidi + upperMidi);
    for (double rate : { 44100.0, 48000.0 })
    {
        for (int phase : { kNearHopStart, kMidHop })
        {
            for (int direction = 0; direction < 2; ++direction)
            {
                const bool up = direction == 0;
                const double change = up ? size : -size;
                const double startShift = up ? -0.5 * size : 0.5 * size;
                const double measured = timeToTargetMs(
                    rate, 0.0, noteMask(up ? upperMidi : lowerMidi),
                    up ? upperMidi : lowerMidi, phase,
                    noteMask(up ? lowerMidi : upperMidi), inputMidi);
                const double law = slewLawMs(change, startShift, rate);
                EXPECT_LT(measured, ruledBoundMs(change, startShift, rate));
                // It cannot be much faster than the engine it drives either,
                // which is what would happen if the measurement were seeing
                // something other than the traversal. A change landing just
                // before a hop boundary saves at most one hop out of the k the
                // law counts, and k is 4 here at the smallest, so 0.7 of the
                // law is below every phase and above an instant jump.
                EXPECT_GT(measured, 0.7 * law);
            }
        }
    }
}

} // namespace

DSPARK_TEST(PitchCorrector_hard_snap_reaches_the_target_within_30ms)
{
    // A one-semitone change of correction, worst of two control phases at each
    // rate. The engine moves the shift by at most half a semitone per analysis
    // hop, so this is two hops plus up to one hop of adoption delay.
    for (int phase : { kNearHopStart, kMidHop })
    {
        EXPECT_LT(timeToTargetMs(48000.0, 0.0, kOnlyASharp, 70.0, phase), 30.0);
        EXPECT_LT(timeToTargetMs(44100.0, 0.0, kOnlyASharp, 70.0, phase), 30.0);
    }
}

DSPARK_TEST(PitchCorrector_two_semitone_hard_snap_is_bounded_by_the_slew_rate)
{
    // A whole-tone gap - G to A in the major scale, and in twenty other
    // published scales. Crossing its middle changes the correction by two
    // semitones, which is four slew hops, so this is where the 30 ms floor
    // stops being reachable and the engine's own law takes over. The bound is
    // computed here from that law rather than pinned as a constant, so the
    // case tracks the frame instead of trailing it.
    checkGapCrossing(67, 69);
}

DSPARK_TEST(PitchCorrector_four_semitone_hard_snap_at_the_widest_published_gap)
{
    // D# to G in Hirajoshi: four semitones, the widest gap in the framework's
    // published scale table (InSen, Balinese, Chinese and Japanese reach it
    // too). Eight slew hops. An arbitrary one-note mask can ask for twelve,
    // which the class documents; this pins the widest case a published scale
    // can produce.
    checkGapCrossing(63, 67);
}

DSPARK_TEST(PitchCorrector_time_to_target_is_monotonic_in_retune_speed)
{
    // Strictly increasing ABOVE the shifter's slew floor: once the glide is
    // slower than half a semitone per hop, the glide is what decides.
    const double speeds[] = { 20.0, 50.0, 100.0, 200.0 };
    double previous = -1.0;
    for (double speed : speeds)
    {
        const double ms = timeToTargetMs(48000.0, speed, kOnlyASharp, 70.0);
        EXPECT_LT(ms, 1.0e8);           // it must arrive at all
        EXPECT_GT(ms, previous);        // strictly slower than the speed below
        previous = ms;
    }

    // BELOW that floor the slew binds and the response is flat, so the claim
    // there is non-decreasing, not strictly increasing. Pinning it keeps the
    // documentation and the test saying the same thing.
    double lastFast = -1.0;
    for (double speed : { 0.0, 1.0, 2.0, 4.0 })
    {
        const double ms = timeToTargetMs(48000.0, speed, kOnlyASharp, 70.0);
        EXPECT_TRUE(ms >= lastFast);
        EXPECT_LT(ms, 30.0);            // still the hard-snap floor
        lastFast = ms;
    }
    EXPECT_GT(previous, lastFast);      // the glides above the floor are slower
}

DSPARK_TEST(PitchCorrector_holds_the_correction_through_an_unvoiced_gap)
{
    // A held note 40 cents flat, corrected up to A4, interrupted by 60 ms of
    // noise. The detector reports nothing for about 75 ms (the noise plus its
    // analysis window), and the correction must survive that untouched: a
    // release to zero would be a 40-cent dip in the middle of the note.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 1.6);
    const int gapFrom = static_cast<int>(fs * 0.8);
    const int gapLength = static_cast<int>(fs * 0.06);
    auto input = harmonicTone(midiToHz(69.0 - 0.40), fs, n);
    std::uint32_t state = 12345u;
    for (int i = gapFrom; i < gapFrom + gapLength; ++i)
    {
        state = state * 1664525u + 1013904223u;
        input[static_cast<std::size_t>(i)] =
            0.2 * (static_cast<double>(state >> 8) / 8388608.0 - 1.0);
    }
    Run run;
    const auto output = render(input, run);
    const auto hz = trackFundamental(output, fs, 440.0);
    // Start past the gap, its 4096-sample signal latency and one settling hop.
    const int from = gapFrom + gapLength + 4096 + static_cast<int>(fs * 0.03);
    double worst = 0.0;
    for (int i = from; i < n - 4000; ++i)
        worst = std::max(worst, std::abs(centsBetween(hz[static_cast<std::size_t>(i)], 440.0)));
    EXPECT_LT(worst, 10.0);
}

DSPARK_TEST(PitchCorrector_new_note_emerges_already_corrected)
{
    // The correction decision trails the input by the detector's analysis span
    // (about 53 ms), which is shorter than the 4096-sample signal latency, so
    // the first audible moment of a new note is already in tune.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 1.4);
    const int changeAt = static_cast<int>(fs * 0.7);
    const double first = midiToHz(64.0 - 0.40);        // E4, 40 cents flat
    const double second = midiToHz(69.0 + 0.40);       // A4, 40 cents sharp
    std::vector<double> input(static_cast<std::size_t>(n), 0.0);
    {
        const auto a = harmonicTone(first, fs, changeAt);
        const auto b = harmonicTone(second, fs, n - changeAt);
        std::copy(a.begin(), a.end(), input.begin());
        std::copy(b.begin(), b.end(), input.begin() + changeAt);
    }
    Run run;
    const auto output = render(input, run);
    const auto hz = trackFundamental(output, fs, 440.0);
    // The new note leaves the corrector one signal latency after it arrived;
    // allow one analysis hop for the vocoder to render it.
    const int from = changeAt + 4096 + 1024;
    double worst = 0.0;
    for (int i = from; i < n - 4000; ++i)
        worst = std::max(worst, std::abs(centsBetween(hz[static_cast<std::size_t>(i)], 440.0)));
    EXPECT_LT(worst, 15.0);
}

// ============================================================================
// Formant preservation
// ============================================================================

DSPARK_TEST(PitchCorrector_formant_preserve_holds_the_envelope_at_unity)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 0.9);
    const auto input = vowelTone(220.0, fs, n);        // A3: in tune already
    const int from = 4096 + static_cast<int>(fs * 0.15);
    const int count = 16384;
    const auto reference = spectralCentroids(slice(input, from - 4096, count),
                                             fs, 4096, 50.0, 8000.0);
    EXPECT_GT(reference.envelope, 0.0);

    Run run;
    run.formant = true;
    const auto output = render(input, run);
    const auto measured = spectralCentroids(slice(output, from, count),
                                            fs, 4096, 50.0, 8000.0);
    EXPECT_LT(100.0 * std::abs(measured.envelope - reference.envelope) / reference.envelope, 5.0);
    EXPECT_LT(100.0 * std::abs(measured.raw - reference.raw) / reference.raw, 5.0);
}

DSPARK_TEST(PitchCorrector_formant_preserve_matters_at_a_large_correction)
{
    // G3 with a one-note C scale is corrected five semitones up. Without the
    // envelope pre-warp the formants ride the pitch; with it they stay put.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 0.9);
    const auto input = vowelTone(midiToHz(55.0), fs, n);
    const int from = 4096 + static_cast<int>(fs * 0.15);
    const int count = 16384;
    const auto reference = spectralCentroids(slice(input, from - 4096, count),
                                             fs, 4096, 50.0, 8000.0);
    Run run;
    run.mask = static_cast<std::uint16_t>(1u << 0);
    const auto free = spectralCentroids(slice(render(input, run), from, count),
                                        fs, 4096, 50.0, 8000.0);
    run.formant = true;
    const auto held = spectralCentroids(slice(render(input, run), from, count),
                                        fs, 4096, 50.0, 8000.0);
    const double freeDrift = 100.0 * std::abs(free.envelope - reference.envelope)
                           / reference.envelope;
    const double heldDrift = 100.0 * std::abs(held.envelope - reference.envelope)
                           / reference.envelope;
    EXPECT_GT(freeDrift, 20.0);
    EXPECT_LT(heldDrift, 0.5 * freeDrift);
}

// ============================================================================
// Determinism, channels and hostile input
// ============================================================================

DSPARK_TEST(PitchCorrector_output_is_bit_exact_across_block_sizes)
{
    const double fs = 48000.0;
    const int n = 40000;
    const auto input = harmonicTone(261.0, fs, n);
    Run run;
    run.mask = kMajor;
    run.retuneMs = 35.0;
    run.block = 512;
    const auto reference = render(input, run);
    for (int block : { 1, 7, 64, 100, 333, 1024 })
    {
        run.block = block;
        const auto other = render(input, run);
        for (int i = 0; i < n; ++i)
            EXPECT_EQ(other[static_cast<std::size_t>(i)],
                      reference[static_cast<std::size_t>(i)]);
    }
}

DSPARK_TEST(PitchCorrector_reset_restores_a_fresh_stream)
{
    const double fs = 48000.0;
    const int n = 24000;
    const auto input = harmonicTone(midiToHz(69.0 - 0.4), fs, n);
    Run run;
    run.retuneMs = 25.0;
    const auto reference = render(input, run);

    AudioSpec spec;
    spec.sampleRate = fs;
    spec.maxBlockSize = 256;
    spec.numChannels = 1;
    PitchCorrector<float> corrector;
    corrector.setRetuneSpeedMs(25.0f);
    corrector.prepare(spec);

    std::vector<float> scratch(static_cast<std::size_t>(n), 0.0f);
    for (int pass = 0; pass < 2; ++pass)
    {
        for (int i = 0; i < n; ++i)
            scratch[static_cast<std::size_t>(i)] =
                static_cast<float>(input[static_cast<std::size_t>(i)]);
        if (pass == 1) corrector.reset();
        int position = 0;
        while (position < n)
        {
            const int length = std::min(256, n - position);
            float* pointer = scratch.data() + position;
            corrector.processBlock(AudioBufferView<float>(&pointer, 1, length));
            position += length;
        }
    }
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(scratch[static_cast<std::size_t>(i)],
                  static_cast<float>(reference[static_cast<std::size_t>(i)]));
}

DSPARK_TEST(PitchCorrector_channels_share_one_correction)
{
    const double fs = 48000.0;
    const int n = 24000;
    const auto input = harmonicTone(midiToHz(69.0 - 0.4), fs, n);
    Run run;
    run.channels = 2;
    const auto channels = renderAll<float>(input, run);
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(channels[0][static_cast<std::size_t>(i)],
                  channels[1][static_cast<std::size_t>(i)]);

    // A third channel on a stereo-prepared instance is left untouched, exactly
    // as the internal shifter documents.
    AudioSpec spec;
    spec.sampleRate = fs;
    spec.maxBlockSize = 512;
    spec.numChannels = 2;
    PitchCorrector<float> corrector;
    corrector.prepare(spec);
    std::vector<std::vector<float>> three(3, std::vector<float>(512, 0.0f));
    for (int ch = 0; ch < 3; ++ch)
        for (int i = 0; i < 512; ++i)
            three[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)] =
                static_cast<float>(input[static_cast<std::size_t>(i)]);
    float* pointers[3] = { three[0].data(), three[1].data(), three[2].data() };
    corrector.processBlock(AudioBufferView<float>(pointers, 3, 512));
    for (int i = 0; i < 512; ++i)
        EXPECT_EQ(three[2][static_cast<std::size_t>(i)],
                  static_cast<float>(input[static_cast<std::size_t>(i)]));
}

DSPARK_TEST(PitchCorrector_survives_nonfinite_and_silent_input)
{
    const double fs = 48000.0;
    AudioSpec spec;
    spec.sampleRate = fs;
    spec.maxBlockSize = 512;
    spec.numChannels = 1;
    PitchCorrector<float> corrector;
    corrector.prepare(spec);

    // Silence in, silence out.
    std::vector<float> block(512, 0.0f);
    for (int b = 0; b < 20; ++b)
    {
        float* pointer = block.data();
        corrector.processBlock(AudioBufferView<float>(&pointer, 1, 512));
        EXPECT_SILENT(block.data(), 512, 1.0e-6f);
    }

    // One block of NaN and infinity, and then NO reset: the stream has to
    // recover on its own, which is the stronger property and the one the
    // detector's own contract promises (it holds no recursive state, so the
    // bad samples flush out of the analysis window). A reset here would hide a
    // regression that made a non-finite burst permanent.
    for (int i = 0; i < 512; ++i)
        block[static_cast<std::size_t>(i)] = (i % 2 == 0)
            ? std::numeric_limits<float>::quiet_NaN()
            : std::numeric_limits<float>::infinity();
    {
        float* pointer = block.data();
        corrector.processBlock(AudioBufferView<float>(&pointer, 1, 512));
    }
    const auto tone = harmonicTone(440.0, fs, 24000);
    std::vector<float> stream(24000, 0.0f);
    for (int i = 0; i < 24000; ++i)
        stream[static_cast<std::size_t>(i)] = static_cast<float>(tone[static_cast<std::size_t>(i)]);
    int position = 0;
    while (position < 24000)
    {
        const int length = std::min(512, 24000 - position);
        float* pointer = stream.data() + position;
        corrector.processBlock(AudioBufferView<float>(&pointer, 1, length));
        position += length;
    }
    // Recovery is bounded, not merely eventual: the last non-finite sample
    // must be inside 200 ms of the burst (measured 135 ms, which is the signal
    // latency plus the analysis window flushing out), and the correction must
    // be right again afterwards.
    int lastBad = -1;
    for (int i = 0; i < 24000; ++i)
        if (!std::isfinite(stream[static_cast<std::size_t>(i)])) lastBad = i;
    EXPECT_LT(1000.0 * static_cast<double>(lastBad + 1) / fs, 200.0);
    for (int i = 12000; i < 24000; ++i)
    {
        EXPECT_TRUE(std::isfinite(stream[static_cast<std::size_t>(i)]));
        EXPECT_LT(std::abs(stream[static_cast<std::size_t>(i)]), 4.0f);
    }
    const std::vector<double> recovered(stream.begin() + 14400, stream.end());
    EXPECT_LT(std::abs(centsBetween(estimateF0(recovered, fs, 60.0, 1400.0), 440.0)),
              10.0);
}

DSPARK_TEST(PitchCorrector_unvoiced_material_is_never_transposed)
{
    // The detector's first analysis hop after prepare() reports half the
    // sample rate at full confidence on any non-silent input, because its
    // window is still three quarters zeros. The hold policy would otherwise
    // latch the correction that comes out of it and transpose material that
    // has no pitch at all, so a warm-up gate ignores reports until one full
    // window has been fed. The proof: noise corrected under a scale must be
    // bit-identical to the same noise with correction switched off.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    std::vector<double> noise(static_cast<std::size_t>(n), 0.0);
    std::uint32_t state = 24601u;
    for (int i = 0; i < n; ++i)
    {
        state = state * 1664525u + 1013904223u;
        noise[static_cast<std::size_t>(i)] =
            0.25 * (static_cast<double>(state >> 8) / 8388608.0 - 1.0);
    }
    Run disabled;
    disabled.mask = 0;                       // correction switched off
    const auto reference = render(noise, disabled);
    for (std::uint16_t mask : { kChromatic, kMajor })
    {
        Run run;
        run.mask = mask;
        const auto corrected = render(noise, run);
        for (int i = 0; i < n; ++i)
            EXPECT_EQ(corrected[static_cast<std::size_t>(i)],
                      reference[static_cast<std::size_t>(i)]);
    }

    // The same gate must not weaken the hold: after real voiced input, an
    // unvoiced stretch still keeps the correction (pinned separately in
    // PitchCorrector_holds_the_correction_through_an_unvoiced_gap), and a
    // fundamental this effect cannot act on - above a quarter of the sample
    // rate, where not even the second harmonic fits under Nyquist - is refused
    // rather than snapped.
    const auto tone = harmonicTone(midiToHz(69.0 - 0.4), fs, n);
    Run voiced;
    const auto corrected = render(tone, voiced);
    const auto segment = slice(corrected, n - 19200, 19200);
    EXPECT_LT(std::abs(centsBetween(estimateF0(segment, fs, 60.0, 1400.0), 440.0)),
              10.0);
}

DSPARK_TEST(PitchCorrector_holds_its_analysis_span_across_sample_rates)
{
    // The frame decides two things and both are properties of TIME: whether a
    // bass voice's harmonics are resolvable, and how long a correction takes to
    // settle. Pinned in samples it would halve its own span at every doubling
    // of the rate, so it is chosen from the rate instead - the same policy the
    // detector inside this class already applies to its own window.
    struct Expected { double rate; int frame; };
    for (const Expected& e : { Expected { 44100.0, 2048 }, Expected { 48000.0, 2048 },
                               Expected { 88200.0, 4096 }, Expected { 96000.0, 4096 },
                               Expected { 176400.0, 8192 }, Expected { 192000.0, 8192 } })
    {
        AudioSpec spec;
        spec.sampleRate = e.rate;
        spec.maxBlockSize = 512;
        spec.numChannels = 1;
        PitchCorrector<float> corrector;
        corrector.prepare(spec);
        EXPECT_EQ(corrector.getFrameSize(), e.frame);
        EXPECT_EQ(corrector.getLatency(), 2 * e.frame);
        // The span stays inside a 40-50 ms window at every one of them.
        const double spanMs = 1000.0 * static_cast<double>(e.frame) / e.rate;
        EXPECT_GT(spanMs, 40.0);
        EXPECT_LT(spanMs, 50.0);
    }

    // An explicit frame overrides the policy and is rounded to a power of two
    // in the range the shifter accepts, because the shifter ignores a malformed
    // one and would leave the stream silently unprepared.
    AudioSpec spec;
    spec.sampleRate = 48000.0;
    spec.maxBlockSize = 512;
    spec.numChannels = 1;
    PitchCorrector<float> explicitFrame;
    explicitFrame.prepare(spec, 1000);
    EXPECT_EQ(explicitFrame.getFrameSize(), 1024);
    EXPECT_EQ(explicitFrame.getLatency(), 2048);
    PitchCorrector<float> tiny;
    tiny.prepare(spec, 1);
    EXPECT_EQ(tiny.getFrameSize(), 256);
}

DSPARK_TEST(PitchCorrector_keeps_the_low_register_resolved_at_its_analysis_span)
{
    // What the analysis span is FOR, measured rather than asserted. A bass
    // voice's harmonics are resolvable only in a long enough span, so this
    // reads the harmonic-to-residual ratio of a corrected vowel at F2, A2 and
    // A3 and requires 25 dB; and it reads the same thing again with the frame
    // pinned to half the policy span, where the low notes must collapse. The
    // control is expressed as half of whatever the policy chooses at this rate,
    // never as a sample count: at a high rate a fixed count is a quarter of the
    // span and would be testing something else. A3 is measured for the same
    // floor as a guard against a global collapse, but it is NOT required to
    // fail at half the span, and it does not - which is exactly why it takes a
    // bass voice to pin the frame from below.
    const double fs = 48000.0;
    const int segment = analysisSegment(fs);
    const int n = static_cast<int>(2.6 * fs);
    const int offset = static_cast<int>(0.9 * fs);

    AudioSpec spec;
    spec.sampleRate = fs;
    spec.maxBlockSize = 512;
    spec.numChannels = 1;
    PitchCorrector<float> policy;
    policy.prepare(spec);
    const int policyFrame = policy.getFrameSize();
    const int policyLatency = policy.getLatency();
    PitchCorrector<float> halved;
    halved.prepare(spec, policyFrame / 2);
    const int halvedLatency = halved.getLatency();
    EXPECT_EQ(halved.getFrameSize(), policyFrame / 2);

    struct Note { double f0; int midi; };
    const Note notes[3] = { { 87.31, 41 }, { 110.0, 45 }, { 220.0, 57 } };

    for (int index = 0; index < 3; ++index)
    {
        const auto stimulus = vowelStimulus(notes[index].f0, fs, n);
        const double target = notes[index].f0 * std::exp2(1.0 / 12.0);

        Run run;
        run.sampleRate = fs;
        run.block = 512;
        run.mask = noteMask(notes[index].midi + 1);      // forces exactly +1 st
        const auto corrected = render(stimulus, run);
        const auto tail = slice(corrected, policyLatency + offset, segment);
        const double measured = estimateF0(tail, fs, 40.0, 700.0);
        // The grid sits on the measured f0, so it has to BE the corrected note:
        // otherwise this measurement would be re-reporting a tuning error.
        EXPECT_LT(std::abs(centsBetween(measured, target)), 10.0);
        const double reading = harmonicResidualDb(tail, fs, measured);

        // The instrument, pointed at the same stimulus synthesized at the
        // corrected pitch and never passed through the class: that signal's
        // residual is zero by construction, so whatever it reports there is its
        // own leakage floor. Ten decibels of headroom over the value it is
        // about to judge bounds that value's underestimate at 0.41 dB.
        const auto reference = vowelStimulus(target, fs, n);
        const auto referenceTail = slice(reference, offset, segment);
        const double referenceF0 = estimateF0(referenceTail, fs, 40.0, 700.0);
        const double instrumentFloor = harmonicResidualDb(referenceTail, fs, referenceF0);
        EXPECT_GT(instrumentFloor - reading, 10.0);

        if (index == 0)
        {
            // And it returns an answer planted in advance. A capture narrower
            // than the window's main lobe fails this by leakage, a much wider
            // one by absorbing the planted noise, so the one check closes both
            // ends of the instrument space.
            for (double planted : { 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 })
            {
                const auto noisy = plantNoise(referenceTail, fs, planted, 20260816u);
                const double noisyF0 = estimateF0(noisy, fs, 40.0, 700.0);
                EXPECT_LT(std::abs(harmonicResidualDb(noisy, fs, noisyF0) - planted), 1.0);
            }
        }

        EXPECT_GT(reading, 25.0);

        if (index < 2)
        {
            Run control = run;
            control.frame = policyFrame / 2;
            const auto collapsed = render(stimulus, control);
            const auto controlTail = slice(collapsed, halvedLatency + offset, segment);
            const double controlF0 = estimateF0(controlTail, fs, 40.0, 700.0);
            const double controlDb = harmonicResidualDb(controlTail, fs, controlF0);
            // The control must be shown firing: a control that passes the floor
            // proves nothing about what the span buys.
            EXPECT_LT(controlDb, 25.0);
            EXPECT_GT(reading - controlDb, 15.0);
        }
    }
}

DSPARK_TEST(PitchCorrector_corrects_accurately_at_every_supported_rate)
{
    // Criterion 1 at the professional session rates, not only at 48 kHz: a low
    // voice 45 cents flat, and a large correction, both read by the
    // independent autocorrelation oracle. Before the frame followed the rate,
    // 88.2 and 96 kHz missed by tens of cents here.
    for (double rate : { 44100.0, 48000.0, 88200.0, 96000.0, 192000.0 })
    {
        const int n = static_cast<int>(rate * 0.6);
        const int measured = static_cast<int>(rate * 0.25);

        const auto flat = harmonicTone(midiToHz(40.0 - 0.45), rate, n);   // E2
        Run run;
        run.sampleRate = rate;
        const auto corrected = render(flat, run);
        EXPECT_LT(std::abs(centsBetween(
                      estimateF0(slice(corrected, n - measured, measured), rate,
                                 60.0, 1400.0), midiToHz(40.0))), 10.0);

        // Three semitones up from A2, forced by a one-note scale.
        const auto tone = harmonicTone(midiToHz(45.0), rate, n);
        Run big;
        big.sampleRate = rate;
        big.mask = noteMask(48);
        const auto shifted = render(tone, big);
        EXPECT_LT(std::abs(centsBetween(
                      estimateF0(slice(shifted, n - measured, measured), rate,
                                 40.0, 2000.0), midiToHz(48.0))), 10.0);
    }
}

DSPARK_TEST(PitchCorrector_double_precision_tracks_the_same_note)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 0.45);
    const auto input = harmonicTone(midiToHz(69.0 - 0.45), fs, n);
    Run run;
    const auto channels = renderAll<double>(input, run);
    const auto output = channels.front();
    const auto segment = slice(std::vector<double>(output.begin(), output.end()),
                               n - 9600, 9600);
    EXPECT_LT(std::abs(centsBetween(estimateF0(segment, fs, 60.0, 1400.0),
                                    midiToHz(69.0))), 10.0);
}
