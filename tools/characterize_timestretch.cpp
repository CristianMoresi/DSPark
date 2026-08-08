// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

// Offline characterisation harness for Effects/TimeStretch.h.
//
// Measures the acceptance criteria of the time stretcher and writes one
// machine-generated artifact per criterion plus a summary table. It is a
// measurement tool, not part of the library: the WSOLA reference and the
// consistency analyser below exist only so the phase vocoder can be compared
// against something independent of itself.
//
//   g++ -std=c++20 -O2 -Wall -Wextra -I. tools/characterize_timestretch.cpp -o ts
//   ./ts <output-directory>
//
// Criterion tags used in the artifacts (defined here, used here):
//   C1: stationary fidelity (log-spectral distance, spectral convergence)
//   C2: spurious / alias floor
//   C3: transient preservation against a WSOLA reference
//   C4: vertical coherence (STFT consistency)
//   C5: stereo integrity (level difference, coherence)
//   C6: exact ratio and drift
//   C7: unity passthrough
//   C8: determinism under block chopping

#include "../Core/FFT.h"
#include "../Effects/TimeStretch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kRate = 48000.0;
constexpr int kBlock = 512;

using Signal = std::vector<std::vector<double>>;   // [channel][sample]

// -- deterministic sources -----------------------------------------------------

struct Lcg
{
    uint32_t s = 987654321u;
    double next() { s = s * 1664525u + 1013904223u; return static_cast<double>(s >> 8) / 8388608.0 - 1.0; }
};

/// 12-partial harmonic bed, f0 = 110 Hz, 1/k amplitudes, fixed phases.
Signal harmonicBed(double seconds, int channels)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    for (int k = 1; k <= 12; ++k)
    {
        const double f = 110.0 * k;
        const double a = 0.35 / k;
        const double ph = 0.7 * k;
        for (size_t i = 0; i < n; ++i)
        {
            const double v = a * std::sin(2.0 * kPi * f * static_cast<double>(i) / kRate + ph);
            for (auto& ch : sig) ch[i] += v;
        }
    }
    return sig;
}

/// The same bed with 3 Hz / 30 cent vibrato on every partial. Phase locking
/// only has something to do when the bins of one partial disagree, which a
/// perfectly stationary partial never makes them do.
Signal vibratoBed(double seconds, int channels)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    for (int k = 1; k <= 12; ++k)
    {
        const double a = 0.35 / k;
        double phase = 0.7 * k;
        for (size_t i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(i) / kRate;
            const double f = 110.0 * k * std::pow(2.0, 0.30 / 12.0 * std::sin(2.0 * kPi * 3.0 * t));
            phase += 2.0 * kPi * f / kRate;
            const double v = a * std::sin(phase);
            for (auto& ch : sig) ch[i] += v;
        }
    }
    return sig;
}

/// Sine at an exact analysis bin, so the analyser contributes no leakage.
Signal sine(double freq, double seconds, int channels, double amp = 0.5)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
        const double v = amp * std::sin(2.0 * kPi * freq * static_cast<double>(i) / kRate);
        for (auto& ch : sig) ch[i] = v;
    }
    return sig;
}

/// Pink noise by the Voss-McCartney octave sum (deterministic).
Signal pinkNoise(double seconds, int channels)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    Lcg rng;
    double rows[16] = { 0 };
    for (size_t i = 0; i < n; ++i)
    {
        uint32_t counter = static_cast<uint32_t>(i);
        for (int b = 0; b < 16; ++b)
            if (((counter >> b) & 1u) == 0u) { rows[b] = rng.next(); break; }
        double sum = 0.0;
        for (double r : rows) sum += r;
        const double v = 0.08 * sum;
        for (auto& ch : sig) ch[i] = v;
    }
    return sig;
}

/// Band-limited click: a windowed sinc, so it has no energy above the cutoff.
void addClick(std::vector<double>& dst, size_t at, double cutoffHz, double amp)
{
    const int half = 96;
    for (int k = -half; k <= half; ++k)
    {
        const auto idx = static_cast<long long>(at) + k;
        if (idx < 0 || idx >= static_cast<long long>(dst.size())) continue;
        const double x = 2.0 * cutoffHz * static_cast<double>(k) / kRate;
        const double s = (k == 0) ? 1.0 : std::sin(kPi * x) / (kPi * x);
        const double w = 0.5 + 0.5 * std::cos(kPi * static_cast<double>(k) / half);
        dst[static_cast<size_t>(idx)] += amp * s * w;
    }
}

/// Click train at `bpm`, band-limited to 16 kHz.
Signal clickTrain(double seconds, double bpm, int channels, std::vector<size_t>& onsets)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    onsets.clear();
    const double period = 60.0 / bpm * kRate;
    for (double t = 4800.0; t < static_cast<double>(n) - 4800.0; t += period)
    {
        const auto at = static_cast<size_t>(t);
        onsets.push_back(at);
        for (auto& ch : sig) addClick(ch, at, 16000.0, 0.9);
    }
    return sig;
}

// -- analysis helpers ----------------------------------------------------------

std::vector<double> hannWindow(int n)
{
    std::vector<double> w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        w[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n);
    return w;
}

/// Frame-averaged magnitude spectrum (Hann, 75% overlap).
std::vector<double> averageSpectrum(const std::vector<double>& x, int n, int hop)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    const auto w = hannWindow(n);
    std::vector<double> frame(static_cast<size_t>(n)), spec(static_cast<size_t>(n + 2));
    const int bins = n / 2 + 1;
    std::vector<double> acc(static_cast<size_t>(bins), 0.0);
    int frames = 0;
    for (size_t start = 0; start + static_cast<size_t>(n) <= x.size();
         start += static_cast<size_t>(hop))
    {
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        for (int k = 0; k < bins; ++k)
        {
            const double re = spec[static_cast<size_t>(2 * k)];
            const double im = spec[static_cast<size_t>(2 * k + 1)];
            acc[static_cast<size_t>(k)] += re * re + im * im;
        }
        ++frames;
    }
    if (frames == 0) frames = 1;
    for (auto& a : acc) a = std::sqrt(a / frames);
    return acc;
}

/// Rectangular-window magnitude spectrum of one segment (coherent sampling).
std::vector<double> segmentSpectrum(const std::vector<double>& x, size_t start, int n)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    std::vector<double> frame(static_cast<size_t>(n), 0.0), spec(static_cast<size_t>(n + 2));
    for (int i = 0; i < n; ++i)
        if (start + static_cast<size_t>(i) < x.size())
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)];
    fft.forward(frame.data(), spec.data());
    const int bins = n / 2 + 1;
    std::vector<double> mag(static_cast<size_t>(bins));
    for (int k = 0; k < bins; ++k)
    {
        const double re = spec[static_cast<size_t>(2 * k)];
        const double im = spec[static_cast<size_t>(2 * k + 1)];
        mag[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
    }
    return mag;
}

double rms(const std::vector<double>& x, size_t from, size_t to)
{
    if (to > x.size()) to = x.size();
    if (from >= to) return 0.0;
    double s = 0.0;
    for (size_t i = from; i < to; ++i) s += x[i] * x[i];
    return std::sqrt(s / static_cast<double>(to - from));
}

double db(double v) { return 20.0 * std::log10(std::max(v, 1e-300)); }

// -- STFT consistency (Laroche-Dolson) -----------------------------------------

/// Consistency ratio: energy of the STFT over the energy it loses when it is
/// resynthesised and analysed again. Higher means the frames agree with each
/// other, which is the property phase locking is there to preserve.
double stftConsistency(const std::vector<double>& x, int n, int hop)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    const auto w = hannWindow(n);
    const int bins = n / 2 + 1;
    const size_t nFrames = (x.size() < static_cast<size_t>(n))
                         ? 0 : (x.size() - static_cast<size_t>(n)) / static_cast<size_t>(hop) + 1;
    if (nFrames < 4) return 0.0;

    std::vector<double> Z(nFrames * static_cast<size_t>(n + 2));
    std::vector<double> frame(static_cast<size_t>(n)), spec(static_cast<size_t>(n + 2));
    for (size_t f = 0; f < nFrames; ++f)
    {
        const size_t start = f * static_cast<size_t>(hop);
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        std::copy(spec.begin(), spec.end(), Z.begin() + static_cast<long>(f) * (n + 2));
    }

    // Inverse: weighted overlap-add, normalised by the summed squared window.
    const size_t len = (nFrames - 1) * static_cast<size_t>(hop) + static_cast<size_t>(n);
    std::vector<double> y(len, 0.0), norm(len, 0.0), timeBuf(static_cast<size_t>(n));
    for (size_t f = 0; f < nFrames; ++f)
    {
        std::copy(Z.begin() + static_cast<long>(f) * (n + 2),
                  Z.begin() + static_cast<long>(f + 1) * (n + 2), spec.begin());
        fft.inverse(spec.data(), timeBuf.data());
        const size_t start = f * static_cast<size_t>(hop);
        for (int i = 0; i < n; ++i)
        {
            y[start + static_cast<size_t>(i)] += timeBuf[static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
            norm[start + static_cast<size_t>(i)] += w[static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        }
    }
    for (size_t i = 0; i < len; ++i)
        if (norm[i] > 1e-12) y[i] /= norm[i];

    // Re-analyse and compare, skipping the first and last frame where the
    // overlap-add normalisation is incomplete.
    double num = 0.0, den = 0.0;
    for (size_t f = 1; f + 1 < nFrames; ++f)
    {
        const size_t start = f * static_cast<size_t>(hop);
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = y[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        for (int k = 0; k < bins; ++k)
        {
            const double zr = Z[static_cast<size_t>(f) * (n + 2) + static_cast<size_t>(2 * k)];
            const double zi = Z[static_cast<size_t>(f) * (n + 2) + static_cast<size_t>(2 * k + 1)];
            const double xr = spec[static_cast<size_t>(2 * k)];
            const double xi = spec[static_cast<size_t>(2 * k + 1)];
            num += zr * zr + zi * zi;
            den += (zr - xr) * (zr - xr) + (zi - xi) * (zi - xi);
        }
    }
    return num / std::max(den, 1e-300);
}

/**
 * Vertical phase coherence around spectral peaks.
 *
 * A single sinusoid seen through a Hann window puts a fixed phase relation
 * between a peak bin and its neighbours: the window kernel's first side lobes
 * are negative, so the neighbours sit half a turn away from the peak. Identity
 * phase locking exists precisely to keep that relation; a plain vocoder
 * advances each bin on its own and loses it, which is what "phasiness" is.
 * The measure is the energy-weighted mean of that relation over every peak of
 * every frame: 1.0 is perfect coherence, 0.0 none.
 *
 * This is reported beside the round-trip consistency ratio because the
 * round-trip ratio cannot separate the two cases: the transform pair used
 * there inverts exactly on any real signal, so the spectrogram of ANY output
 * is self-consistent and the ratio is pinned at its numerical ceiling.
 */
double verticalCoherence(const std::vector<double>& x, int n, int hop)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    const auto w = hannWindow(n);
    const int bins = n / 2 + 1;
    std::vector<double> frame(static_cast<size_t>(n)), spec(static_cast<size_t>(n + 2));
    double sum = 0.0, weight = 0.0;
    for (size_t start = 0; start + static_cast<size_t>(n) <= x.size();
         start += static_cast<size_t>(hop))
    {
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        std::vector<double> mag(static_cast<size_t>(bins));
        double peak = 0.0;
        for (int k = 0; k < bins; ++k)
        {
            const double re = spec[static_cast<size_t>(2 * k)];
            const double im = spec[static_cast<size_t>(2 * k + 1)];
            mag[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
            peak = std::max(peak, mag[static_cast<size_t>(k)]);
        }
        if (peak < 1e-9) continue;
        const double floorMag = peak * 1e-3;
        for (int k = 2; k < bins - 2; ++k)
        {
            const double m = mag[static_cast<size_t>(k)];
            if (m < floorMag) continue;
            if (!(m > mag[static_cast<size_t>(k - 1)] && m >= mag[static_cast<size_t>(k + 1)]
                  && m > mag[static_cast<size_t>(k - 2)] && m >= mag[static_cast<size_t>(k + 2)]))
                continue;
            const double pr = spec[static_cast<size_t>(2 * k)];
            const double pi = spec[static_cast<size_t>(2 * k + 1)];
            for (int d : { -1, 1 })
            {
                const double nr = spec[static_cast<size_t>(2 * (k + d))];
                const double ni = spec[static_cast<size_t>(2 * (k + d) + 1)];
                const double cr = nr * pr + ni * pi;     // Re(neighbour * conj(peak))
                const double ci = ni * pr - nr * pi;
                const double amp = std::sqrt(cr * cr + ci * ci);
                if (amp < 1e-30) continue;
                sum += m * m * (-cr / amp);
                weight += m * m;
            }
        }
    }
    return (weight > 0.0) ? sum / weight : 0.0;
}

// -- WSOLA reference -----------------------------------------------------------

/// Independent time-domain time-scaler used only as the transient reference.
/// Overlap-add with a search for the segment that continues the output best,
/// which is what keeps a strike intact without any spectral processing.
std::vector<double> wsola(const std::vector<double>& x, double ratio, int frame, int seek)
{
    const int hopS = frame / 2;
    const auto hopA = static_cast<double>(hopS) / ratio;
    const auto outLen = static_cast<size_t>(std::lround(static_cast<double>(x.size()) * ratio));
    std::vector<double> y(outLen + static_cast<size_t>(frame), 0.0);
    std::vector<double> norm(y.size(), 0.0);
    const auto w = hannWindow(frame);

    double aPos = 0.0;
    size_t sPos = 0;
    std::vector<double> tail(static_cast<size_t>(hopS), 0.0);   // wanted continuation
    while (sPos + static_cast<size_t>(frame) < y.size()
           && static_cast<size_t>(aPos) + static_cast<size_t>(frame + seek) < x.size())
    {
        long best = 0;
        double bestScore = -1e300;
        const auto base = static_cast<long>(aPos);
        for (long d = -seek; d <= seek; ++d)
        {
            const long s = base + d;
            if (s < 0 || s + frame >= static_cast<long>(x.size())) continue;
            double num = 0.0, en = 1e-12;
            for (int i = 0; i < hopS; ++i)
            {
                const double v = x[static_cast<size_t>(s + i)];
                num += v * tail[static_cast<size_t>(i)];
                en += v * v;
            }
            const double score = num / std::sqrt(en);
            if (score > bestScore) { bestScore = score; best = s; }
        }
        for (int i = 0; i < frame; ++i)
        {
            const auto si = static_cast<size_t>(best + i);
            if (si >= x.size()) break;
            y[sPos + static_cast<size_t>(i)] += x[si] * w[static_cast<size_t>(i)];
            norm[sPos + static_cast<size_t>(i)] += w[static_cast<size_t>(i)];
        }
        for (int i = 0; i < hopS; ++i)
        {
            const auto si = static_cast<size_t>(best + hopS + i);
            tail[static_cast<size_t>(i)] = (si < x.size()) ? x[si] : 0.0;
        }
        sPos += static_cast<size_t>(hopS);
        aPos += hopA;
    }
    for (size_t i = 0; i < y.size(); ++i)
        if (norm[i] > 1e-9) y[i] /= norm[i];
    y.resize(outLen);
    return y;
}

// -- device under test ---------------------------------------------------------

struct Options
{
    double ratio = 1.0;
    bool phaseLock = true;
    bool transient = true;
    int fftSize = 2048;
};

Signal stretchOffline(const Signal& in, const Options& opt)
{
    const int nCh = static_cast<int>(in.size());
    const auto nS = static_cast<int>(in[0].size());

    dspark::AudioBuffer<float> src, dst;
    src.resize(nCh, nS);
    for (int ch = 0; ch < nCh; ++ch)
        for (int i = 0; i < nS; ++i)
            src.getChannel(ch)[i] = static_cast<float>(in[static_cast<size_t>(ch)][static_cast<size_t>(i)]);

    dspark::TimeStretch<float> ts;
    ts.prepare({ kRate, kBlock, nCh }, opt.fftSize);
    ts.setTimeRatio(static_cast<float>(opt.ratio));
    ts.setPhaseLock(opt.phaseLock);
    ts.setTransientPreserve(opt.transient);
    ts.process(src.toView(), dst);

    Signal out(static_cast<size_t>(nCh),
               std::vector<double>(static_cast<size_t>(dst.getNumSamples()), 0.0));
    for (int ch = 0; ch < nCh; ++ch)
        for (int i = 0; i < dst.getNumSamples(); ++i)
            out[static_cast<size_t>(ch)][static_cast<size_t>(i)] = dst.getChannel(ch)[i];
    return out;
}

/// Streaming run with a caller-chosen chopping pattern.
std::vector<double> stretchStreaming(const std::vector<double>& in, const Options& opt,
                                     const std::vector<int>& pattern)
{
    dspark::TimeStretch<float> ts;
    ts.prepare({ kRate, 4096, 1 }, opt.fftSize);
    ts.setTimeRatio(static_cast<float>(opt.ratio));

    std::vector<double> out;
    out.reserve(in.size());
    std::vector<float> scratch(4096);
    size_t pos = 0, p = 0;
    while (pos < in.size())
    {
        const auto want = static_cast<size_t>(pattern[p % pattern.size()]);
        const auto n = static_cast<int>(std::min(want, in.size() - pos));
        for (int i = 0; i < n; ++i)
            scratch[static_cast<size_t>(i)] = static_cast<float>(in[pos + static_cast<size_t>(i)]);
        float* ptrs[1] = { scratch.data() };
        dspark::AudioBufferView<float> view(ptrs, 1, n);
        ts.processBlock(view);
        for (int i = 0; i < n; ++i) out.push_back(scratch[static_cast<size_t>(i)]);
        pos += static_cast<size_t>(n);
        ++p;
    }
    return out;
}

// -- reporting -----------------------------------------------------------------

struct Row
{
    std::string label;
    double ratio = 1.0;
    double lsd = 0.0, sc = 0.0, spurious = 0.0, transientKeep = 0.0, attack = 0.0;
    double consistencyIn = 0.0, consistencyOut = 0.0, consistencyPlain = 0.0;
    double vcohIn = 0.0, vcohLocked = 0.0, vcohPlain = 0.0;
    double ild = 0.0, coherence = 0.0;
    double ratioError = 0.0;
    long long outLen = 0, wantLen = 0;
};

std::string fmt(double v, int prec = 3)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", prec, v);
    return buf;
}

}   // namespace

int main(int argc, char** argv)
{
    const std::string dir = (argc > 1) ? argv[1] : ".";

    // The tone frequency is placed exactly on an analysis bin of the 32768
    // spectrum used for C2/C7, so a rectangular window leaks nothing and the
    // measured floor is the device's, not the analyser's.
    constexpr int kToneFft = 32768;
    const double toneHz = 683.0 * kRate / kToneFft;   // 1000.195 Hz

    struct Target { const char* label; double ratio; double lsdLimit; };
    const Target targets[] = {
        { "r=0.926 (-8%)",       0.926,               1.5 },
        { "r=0.950",             0.95,                1.0 },
        { "r=1.000",             1.0,                 1.0 },
        { "r=1.050",             1.05,                1.0 },
        { "r=1.081 (+8%)",       1.081,               1.5 },
        { "120->110 BPM",        120.0 / 110.0,       1.5 },
        { "120->130 BPM",        120.0 / 130.0,       1.5 },
    };

    std::vector<Row> rows;

    // ---------------------------------------------------------------- C6, C7
    // The regression tripwires run first, before any aesthetic measurement.
    {
        std::ofstream f(dir + "/c6-exact-ratio-drift.csv");
        f << "target_ratio,input_samples,output_samples,expected_samples,length_error_samples,"
             "one_hop,realised_ratio_from_onsets,ratio_error_percent,onsets_used,"
             "worst_onset_error_samples\n";
        for (const auto& t : targets)
        {
            // 32 s of clicks: the realised ratio is measured from where the
            // clicks land, not from the buffer length the class allocates.
            std::vector<size_t> onsets;
            Signal in = clickTrain(32.0, 120.0, 1, onsets);
            Options opt; opt.ratio = t.ratio;
            Signal out = stretchOffline(in, opt);

            const auto n = static_cast<long long>(in[0].size());
            const auto m = static_cast<long long>(out[0].size());
            const long long want = std::llround(static_cast<double>(n) * t.ratio);

            // Locate each click in the output by peak search around where it
            // should be, then fit a straight line through the pairs.
            double sxy = 0.0, sxx = 0.0;
            double worst = 0.0;
            int used = 0;
            for (size_t o : onsets)
            {
                const auto expect = static_cast<long long>(std::llround(static_cast<double>(o) * t.ratio));
                const long long lo = std::max(0LL, expect - 2000);
                const long long hi = std::min(m, expect + 2000);
                long long peak = -1;
                double best = 0.0;
                for (long long i = lo; i < hi; ++i)
                {
                    const double v = std::fabs(out[0][static_cast<size_t>(i)]);
                    if (v > best) { best = v; peak = i; }
                }
                if (peak < 0 || best < 0.05) continue;
                sxy += static_cast<double>(o) * static_cast<double>(peak);
                sxx += static_cast<double>(o) * static_cast<double>(o);
                worst = std::max(worst, std::fabs(static_cast<double>(peak - expect)));
                ++used;
            }
            const double realised = (sxx > 0.0) ? sxy / sxx : 0.0;
            const double err = (realised - t.ratio) / t.ratio * 100.0;

            f << fmt(t.ratio, 6) << ',' << n << ',' << m << ',' << want << ','
              << (m - want) << ',' << (opt.fftSize / 4) << ','
              << fmt(realised, 8) << ',' << fmt(err, 6) << ',' << used << ','
              << fmt(worst, 1) << '\n';

            Row r;
            r.label = t.label; r.ratio = t.ratio;
            r.outLen = m; r.wantLen = want; r.ratioError = err;
            rows.push_back(r);
        }
    }

    {
        std::ofstream f(dir + "/c7-unity-passthrough.csv");
        f << "signal,path,residual_dbfs,thd_n_dbfs,peak_dbfs\n";
        Options unity; unity.ratio = 1.0;

        auto measure = [&](const char* name, const Signal& in, bool streaming)
        {
            std::vector<double> out;
            int latency = 0;
            if (streaming)
            {
                dspark::TimeStretch<float> probe;
                probe.prepare({ kRate, kBlock, 1 }, unity.fftSize);
                latency = probe.getLatency();
                out = stretchStreaming(in[0], unity, { kBlock });
            }
            else
            {
                out = stretchOffline(in, unity)[0];
            }

            // Residual against the aligned input over the settled region.
            const size_t start = static_cast<size_t>(latency) + 24000;
            double num = 0.0;
            size_t count = 0;
            for (size_t i = start; i < out.size() && i - static_cast<size_t>(latency) < in[0].size(); ++i)
            {
                const double d = out[i] - in[0][i - static_cast<size_t>(latency)];
                num += d * d; ++count;
            }
            const double residual = db(std::sqrt(num / std::max<size_t>(count, 1)));

            // THD+N of the tone: everything outside the fundamental bin group.
            // On a bed that is not a single tone the quantity is not defined,
            // and the field is written `n/a`. Writing 0.00 there would read as
            // a measurement of zero dB - a 60 dB failure - to anything checking
            // this file with a machine rather than an eye.
            bool thdnApplies = false;
            double thdn = 0.0;
            if (std::strcmp(name, "1 kHz tone") == 0)
            {
                thdnApplies = true;
                const auto mag = segmentSpectrum(out, start, kToneFft);
                double fund = 0.0, rest = 0.0;
                for (size_t k = 0; k < mag.size(); ++k)
                {
                    const double p = mag[k] * mag[k];
                    if (std::llabs(static_cast<long long>(k) - 683LL) <= 2) fund += p;
                    else rest += p;
                }
                thdn = 10.0 * std::log10(rest / std::max(fund, 1e-300));
            }
            double peak = 0.0;
            for (double v : out) peak = std::max(peak, std::fabs(v));
            f << name << ',' << (streaming ? "streaming" : "offline") << ','
              << fmt(residual, 2) << ',' << (thdnApplies ? fmt(thdn, 2) : std::string("n/a"))
              << ',' << fmt(db(peak), 2) << '\n';
        };

        const Signal tone = sine(toneHz, 4.0, 1);
        const Signal pink = pinkNoise(4.0, 1);
        measure("1 kHz tone", tone, false);
        measure("1 kHz tone", tone, true);
        measure("pink noise", pink, false);
        measure("pink noise", pink, true);
    }

    // ------------------------------------------------------------------- C8
    {
        std::ofstream f(dir + "/c8-chopping-determinism.csv");
        f << "ratio,pattern,samples_compared,differing_samples,first_divergence,"
             "max_abs_difference\n";
        const std::vector<std::vector<int>> patterns = {
            { 512 }, { 64 }, { 1, 7, 63, 512, 129, 4096 }, { 4096 }, { 333 }
        };
        Signal bed = harmonicBed(6.0, 1);
        for (double r : { 0.926, 1.0, 1.05, 1.081 })
        {
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = r;
                const auto ref = stretchStreaming(bed[0], opt, patterns[0]);
                for (size_t p = 1; p < patterns.size(); ++p)
                {
                    const auto got = stretchStreaming(bed[0], opt, patterns[p]);
                    const size_t n = std::min(ref.size(), got.size());
                    size_t diff = 0;
                    long long first = -1;
                    double worst = 0.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        const double d = std::fabs(ref[i] - got[i]);
                        if (d != 0.0)
                        {
                            ++diff;
                            if (first < 0) first = static_cast<long long>(i);
                            worst = std::max(worst, d);
                        }
                    }
                    std::string desc;
                    for (int b : patterns[p]) desc += std::to_string(b) + "|";
                    f << fmt(r, 3) << ',' << desc << ','
                      << n << ',' << diff << ',' << first << ',' << fmt(worst, 12) << '\n';
                }
            }
        }
    }

    // --------------------------------------------------------------- C1, C4
    {
        std::ofstream f1(dir + "/c1-stationary-fidelity.csv");
        f1 << "# LSD is computed on both averaged spectra clipped to 80 dB below the\n"
              "# reference's own peak. The bed is 12 partials, so most bins of the\n"
              "# 20 Hz-16 kHz range hold nothing but the analyser's numerical residue\n"
              "# in BOTH spectra; without the clip their ratio is the dominant term\n"
              "# and the metric reports the analyser instead of the device. The\n"
              "# unclipped value is carried alongside so the difference is visible.\n";
        f1 << "ratio,lsd_db,lsd_limit_db,lsd_db_unclipped,spectral_convergence_db\n";
        std::ofstream f4(dir + "/c4-vertical-coherence.csv");
        f4 << "# consistency_* is the round-trip ratio as specified. The transform\n"
              "# pair inverts exactly on any real signal, so every one of these is\n"
              "# pinned at the numerical ceiling and the comparison between them is\n"
              "# noise; vcoh_* is the discriminating measure (peak-to-neighbour phase\n"
              "# relation, 1.0 = fully coherent).\n";
        f4 << "ratio,bed,consistency_in,consistency_out_locked,consistency_out_plain,"
              "ratio_out_over_in,vcoh_in,vcoh_locked,vcoh_plain,locked_better_than_plain\n";

        const Signal bed = harmonicBed(5.0, 1);
        const auto sref = averageSpectrum(bed[0], 4096, 1024);
        const double cIn = stftConsistency(bed[0], 2048, 512);
        const double vIn = verticalCoherence(bed[0], 2048, 512);
        const Signal vib = vibratoBed(5.0, 1);
        const double cVibIn = stftConsistency(vib[0], 2048, 512);
        const double vVibIn = verticalCoherence(vib[0], 2048, 512);
        const int loBin = static_cast<int>(std::ceil(20.0 * 4096.0 / kRate));
        const int hiBin = static_cast<int>(std::floor(16000.0 * 4096.0 / kRate));
        double srefPeak = 0.0;
        for (double v : sref) srefPeak = std::max(srefPeak, v);
        const double clip = srefPeak * 1e-4;   // 80 dB below the reference peak

        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            const auto& t = targets[ti];
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = t.ratio;
                const auto out = stretchOffline(bed, opt)[0];
                const auto sout = averageSpectrum(out, 4096, 1024);

                double acc = 0.0, accRaw = 0.0, dn = 0.0, dd = 0.0;
                int used = 0;
                for (int k = loBin; k <= hiBin && k < static_cast<int>(sref.size()); ++k)
                {
                    const double a = sref[static_cast<size_t>(k)];
                    const double b = sout[static_cast<size_t>(k)];
                    const double d = 20.0 * std::log10(std::max(a, clip) / std::max(b, clip));
                    acc += d * d;
                    const double dRaw = 20.0 * std::log10(std::max(a, 1e-12) / std::max(b, 1e-12));
                    accRaw += dRaw * dRaw;
                    ++used;
                    const double e = a - b;
                    dn += e * e; dd += a * a;
                }
                const double lsd = std::sqrt(acc / std::max(used, 1));
                const double lsdRaw = std::sqrt(accRaw / std::max(used, 1));
                const double sc = 10.0 * std::log10(dn / std::max(dd, 1e-300));
                f1 << fmt(t.ratio, 6) << ','
                   << fmt(lsd, 4) << ',' << fmt(t.lsdLimit, 1) << ',' << fmt(lsdRaw, 2) << ','
                   << fmt(sc, 2) << '\n';
                { rows[ti].lsd = lsd; rows[ti].sc = sc; }
            }

            // Phase locking versus the plain vocoder on the same bed.
            Options locked; locked.ratio = t.ratio; locked.phaseLock = true;
            Options plain;  plain.ratio = t.ratio;  plain.phaseLock = false;
            const auto lockedOut = stretchOffline(bed, locked)[0];
            const auto plainOut  = stretchOffline(bed, plain)[0];
            const double cLocked = stftConsistency(lockedOut, 2048, 512);
            const double cPlain  = stftConsistency(plainOut, 2048, 512);
            const double vLocked = verticalCoherence(lockedOut, 2048, 512);
            const double vPlain  = verticalCoherence(plainOut, 2048, 512);
            f4 << fmt(t.ratio, 6) << ",stationary," << fmt(cIn, 3) << ',' << fmt(cLocked, 3) << ','
               << fmt(cPlain, 3) << ',' << fmt(cLocked / std::max(cIn, 1e-30), 4) << ','
               << fmt(vIn, 5) << ',' << fmt(vLocked, 5) << ',' << fmt(vPlain, 5) << ','
               << (vLocked > vPlain ? "yes" : "NO") << '\n';

            const auto vibLocked = stretchOffline(vib, locked)[0];
            const auto vibPlain  = stretchOffline(vib, plain)[0];
            const double cvLocked = stftConsistency(vibLocked, 2048, 512);
            const double cvPlain  = stftConsistency(vibPlain, 2048, 512);
            const double vvLocked = verticalCoherence(vibLocked, 2048, 512);
            const double vvPlain  = verticalCoherence(vibPlain, 2048, 512);
            f4 << fmt(t.ratio, 6) << ",vibrato," << fmt(cVibIn, 3) << ',' << fmt(cvLocked, 3) << ','
               << fmt(cvPlain, 3) << ',' << fmt(cvLocked / std::max(cVibIn, 1e-30), 4) << ','
               << fmt(vVibIn, 5) << ',' << fmt(vvLocked, 5) << ',' << fmt(vvPlain, 5) << ','
               << (vvLocked > vvPlain ? "yes" : "NO") << '\n';

            rows[ti].consistencyIn = cIn;
            rows[ti].consistencyOut = cLocked;
            rows[ti].consistencyPlain = cPlain;
            rows[ti].vcohIn = vVibIn;
            rows[ti].vcohLocked = vvLocked;
            rows[ti].vcohPlain = vvPlain;
        }
    }

    // ------------------------------------------------------------------- C2
    {
        std::ofstream f(dir + "/c2-spurious-floor.csv");
        f << "ratio,worst_spurious_db_re_fundamental,worst_bin,worst_hz\n";
        const Signal tone = sine(toneHz, 4.0, 1);
        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = targets[ti].ratio;
                const auto out = stretchOffline(tone, opt)[0];
                const auto mag = segmentSpectrum(out, 48000, kToneFft);
                double fund = 0.0;
                for (long long k = 681; k <= 685; ++k)
                    fund = std::max(fund, mag[static_cast<size_t>(k)]);
                double worst = 0.0;
                long long worstBin = 0;
                for (size_t k = 1; k < mag.size(); ++k)
                {
                    bool harmonic = false;
                    for (long long h = 1; h * 683 < static_cast<long long>(mag.size()); ++h)
                        if (std::llabs(static_cast<long long>(k) - h * 683) <= 2) { harmonic = true; break; }
                    if (harmonic) continue;
                    if (mag[k] > worst) { worst = mag[k]; worstBin = static_cast<long long>(k); }
                }
                const double rel = db(worst / std::max(fund, 1e-300));
                f << fmt(targets[ti].ratio, 6) << ','
                  << fmt(rel, 2) << ',' << worstBin << ','
                  << fmt(static_cast<double>(worstBin) * kRate / kToneFft, 1) << '\n';
                rows[ti].spurious = rel;
            }
        }
    }

    // ------------------------------------------------------------------- C3
    {
        std::ofstream f(dir + "/c3-transient-preservation.csv");
        f << "ratio,onsets,mean_energy_ratio_vs_wsola,worst_energy_ratio_vs_wsola,"
             "source_attack_ms,output_attack_ms,attack_expansion_percent\n";

        std::vector<size_t> onsets;
        const Signal clicks = clickTrain(5.0, 120.0, 1, onsets);

        // Short-time envelope, 1 ms window, used for both attack measurements.
        auto envelope = [](const std::vector<double>& x)
        {
            const int w = 48;   // 1 ms at 48 kHz
            std::vector<double> e(x.size(), 0.0);
            double acc = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                acc += x[i] * x[i];
                if (i >= static_cast<size_t>(w)) acc -= x[i - static_cast<size_t>(w)] * x[i - static_cast<size_t>(w)];
                e[i] = std::sqrt(acc / w);
            }
            return e;
        };
        auto attackMs = [](const std::vector<double>& env, size_t peakAt)
        {
            const double peak = env[peakAt];
            const size_t lo = (peakAt > 2400) ? peakAt - 2400 : 0;
            size_t t10 = peakAt, t90 = peakAt;
            for (size_t i = peakAt; i > lo; --i) { if (env[i] < 0.9 * peak) { t90 = i; break; } }
            for (size_t i = t90; i > lo; --i)   { if (env[i] < 0.1 * peak) { t10 = i; break; } }
            return static_cast<double>(t90 - t10) / kRate * 1000.0;
        };

        const auto srcEnv = envelope(clicks[0]);
        double srcAttack = 0.0;
        {
            double s = 0.0;
            int c = 0;
            for (size_t o : onsets)
            {
                size_t peak = o;
                double best = 0.0;
                for (size_t i = (o > 200 ? o - 200 : 0); i < o + 200 && i < srcEnv.size(); ++i)
                    if (srcEnv[i] > best) { best = srcEnv[i]; peak = i; }
                s += attackMs(srcEnv, peak); ++c;
            }
            srcAttack = s / std::max(c, 1);
        }

        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            const double r = targets[ti].ratio;
            const auto ref = wsola(clicks[0], r, 2048, 512);
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = r;
                const auto out = stretchOffline(clicks, opt)[0];
                const auto outEnv = envelope(out);

                // Each method is allowed its own onset timing: a WSOLA
                // reference splices at correlation maxima, so its clicks land
                // up to a frame away from where the ratio alone would put
                // them, and comparing the two at the SAME absolute sample
                // would credit the vocoder for the reference's timing rather
                // than measure either one's concentration. Both windows are
                // therefore centred on the onset each output actually
                // produced, found within +-40 ms of the mapped time.
                const auto win = static_cast<size_t>(0.005 * kRate);   // +-5 ms
                const auto search = static_cast<size_t>(0.040 * kRate);
                auto locate = [&](const std::vector<double>& sig, size_t around) -> size_t
                {
                    const size_t lo = (around > search) ? around - search : 0;
                    const size_t hi = std::min(sig.size(), around + search);
                    size_t at = lo;
                    double best = 0.0;
                    for (size_t i = lo; i < hi; ++i)
                        if (std::fabs(sig[i]) > best) { best = std::fabs(sig[i]); at = i; }
                    return at;
                };
                auto energyAround = [&](const std::vector<double>& sig, size_t at)
                {
                    double e = 0.0;
                    const size_t lo = (at > win) ? at - win : 0;
                    const size_t hi = std::min(sig.size(), at + win);
                    for (size_t i = lo; i < hi; ++i) e += sig[i] * sig[i];
                    return e;
                };

                double sumRatio = 0.0, worstRatio = 1e300, sumAttack = 0.0;
                int used = 0;
                for (size_t o : onsets)
                {
                    const auto mapped = static_cast<size_t>(std::llround(static_cast<double>(o) * r));
                    if (mapped + search >= out.size() || mapped + search >= ref.size()
                        || mapped < search) continue;
                    const size_t pOut = locate(out, mapped);
                    const size_t pRef = locate(ref, mapped);
                    const double eOut = energyAround(out, pOut);
                    const double eRef = energyAround(ref, pRef);
                    if (eRef < 1e-12) continue;
                    const double q = eOut / eRef;
                    sumRatio += q;
                    worstRatio = std::min(worstRatio, q);
                    sumAttack += attackMs(outEnv, locate(outEnv, mapped));
                    ++used;
                }
                const double meanRatio = sumRatio / std::max(used, 1);
                const double outAttack = sumAttack / std::max(used, 1);
                const double expansion = (outAttack / std::max(srcAttack * r, 1e-9) - 1.0) * 100.0;
                f << fmt(r, 6) << ',' << used << ','
                  << fmt(meanRatio, 4) << ',' << fmt(worstRatio, 4) << ','
                  << fmt(srcAttack * r, 4) << ',' << fmt(outAttack, 4) << ','
                  << fmt(expansion, 2) << '\n';
                { rows[ti].transientKeep = meanRatio; rows[ti].attack = expansion; }
            }
        }
    }

    // ------------------------------------------------------------------- C5
    {
        std::ofstream f(dir + "/c5-stereo-integrity.csv");
        f << "ratio,ild_in_db,ild_out_db,ild_delta_db,coherence_in,coherence_out\n";

        // Correlated stereo: the same bed in both channels, right delayed 3 ms.
        // The bed is pink noise rather than the 12 partials of the fidelity
        // test, because the criterion averages coherence over every bin from
        // 20 Hz to 16 kHz: a sparse spectrum leaves most of those bins empty
        // in both channels, and their coherence is then the ratio of two
        // numerical residues, which is what the average would end up
        // reporting.
        Signal st = pinkNoise(5.0, 2);
        const auto delay = static_cast<size_t>(0.003 * kRate);
        {
            std::vector<double> r(st[1].size(), 0.0);
            for (size_t i = delay; i < r.size(); ++i) r[i] = st[1][i - delay] * 0.85;
            st[1] = r;
        }

        auto coherence = [](const std::vector<double>& a, const std::vector<double>& b)
        {
            // Long segments: the two channels are 3 ms apart, and a window
            // only a few times that long makes the estimator report the
            // window mismatch rather than the signals' relation.
            constexpr int n = 8192;
            dspark::FFTReal<double> fft(n);
            const auto w = hannWindow(n);
            const int bins = n / 2 + 1;
            std::vector<double> saa(static_cast<size_t>(bins), 0.0), sbb(static_cast<size_t>(bins), 0.0);
            std::vector<double> sabRe(static_cast<size_t>(bins), 0.0), sabIm(static_cast<size_t>(bins), 0.0);
            std::vector<double> fa(n), fb(n), za(n + 2), zb(n + 2);
            const size_t len = std::min(a.size(), b.size());
            for (size_t s = 0; s + n <= len; s += n / 2)
            {
                for (int i = 0; i < n; ++i)
                {
                    fa[static_cast<size_t>(i)] = a[s + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
                    fb[static_cast<size_t>(i)] = b[s + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
                }
                fft.forward(fa.data(), za.data());
                fft.forward(fb.data(), zb.data());
                for (int k = 0; k < bins; ++k)
                {
                    const double ar = za[static_cast<size_t>(2 * k)], ai = za[static_cast<size_t>(2 * k + 1)];
                    const double br = zb[static_cast<size_t>(2 * k)], bi = zb[static_cast<size_t>(2 * k + 1)];
                    saa[static_cast<size_t>(k)] += ar * ar + ai * ai;
                    sbb[static_cast<size_t>(k)] += br * br + bi * bi;
                    sabRe[static_cast<size_t>(k)] += ar * br + ai * bi;
                    sabIm[static_cast<size_t>(k)] += ai * br - ar * bi;
                }
            }
            const int lo = static_cast<int>(std::ceil(20.0 * n / kRate));
            const int hi = static_cast<int>(std::floor(16000.0 * n / kRate));
            double sum = 0.0;
            int used = 0;
            for (int k = lo; k <= hi && k < bins; ++k)
            {
                const double den = saa[static_cast<size_t>(k)] * sbb[static_cast<size_t>(k)];
                if (den < 1e-24) continue;
                const double num = sabRe[static_cast<size_t>(k)] * sabRe[static_cast<size_t>(k)]
                                 + sabIm[static_cast<size_t>(k)] * sabIm[static_cast<size_t>(k)];
                sum += num / den; ++used;
            }
            return sum / std::max(used, 1);
        };

        const double ildIn = db(rms(st[0], 0, st[0].size())) - db(rms(st[1], 0, st[1].size()));
        const double cohIn = coherence(st[0], st[1]);
        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = targets[ti].ratio;
                const auto out = stretchOffline(st, opt);
                const size_t skip = 8192;
                const double ildOut = db(rms(out[0], skip, out[0].size() - skip))
                                    - db(rms(out[1], skip, out[1].size() - skip));
                const double cohOut = coherence(out[0], out[1]);
                f << fmt(targets[ti].ratio, 6) << ','
                  << fmt(ildIn, 4) << ',' << fmt(ildOut, 4) << ','
                  << fmt(ildOut - ildIn, 4) << ',' << fmt(cohIn, 5) << ',' << fmt(cohOut, 5) << '\n';
                { rows[ti].ild = ildOut - ildIn; rows[ti].coherence = cohOut; }
            }
        }
    }

    // -------------------------------------------------------------- summary
    {
        std::ofstream f(dir + "/timestretch-metrics.md");
        f << "# TimeStretch characterisation (48 kHz, 2048 frame)\n\n"
             "Machine-generated by `tools/characterize_timestretch.cpp`. Every number\n"
             "here is at 48 kHz with the default 2048-sample frame; the CSV files\n"
             "beside this one carry the plain-vocoder control as well.\n\n"
             "| Target | out/in length | ratio error % | LSD dB | SC dB | spurious dB |"
             " transient vs WSOLA | attack exp % | vertical coherence | dILD dB | stereo coherence |\n"
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& r : rows)
        {
            f << "| " << r.label << " | " << r.outLen << "/" << r.wantLen << " | "
              << fmt(r.ratioError, 4) << " | " << fmt(r.lsd, 3) << " | " << fmt(r.sc, 1) << " | "
              << fmt(r.spurious, 1) << " | " << fmt(r.transientKeep, 3) << " | "
              << fmt(r.attack, 1) << " | "
              << fmt(r.vcohLocked, 4) << " | "
              << fmt(r.ild, 3) << " | " << fmt(r.coherence, 4) << " |\n";
        }
        f << "\nPhase locking against the plain vocoder on the same bed. The\n"
             "round-trip consistency ratio is reported because it is the stated\n"
             "measure, and the peak-to-neighbour phase coherence beside it because\n"
             "the round-trip is pinned at its numerical ceiling for every real\n"
             "signal and cannot tell the two apart.\n\n"
             "| Target | consistency out/in | vertical coherence locked | plain | locked better |\n"
             "|---|---:|---:|---:|---|\n";
        for (const auto& r : rows)
            f << "| " << r.label << " | "
              << fmt(r.consistencyOut / std::max(r.consistencyIn, 1e-30), 4) << " | "
              << fmt(r.vcohLocked, 5) << " | " << fmt(r.vcohPlain, 5) << " | "
              << (r.vcohLocked > r.vcohPlain ? "yes" : "NO") << " |\n";
    }

    std::printf("characterisation written to %s\n", dir.c_str());
    return 0;
}
