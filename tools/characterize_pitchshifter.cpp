// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License
//
// Offline characterisation harness for Effects/PitchShifter.h.
//
// Measures, from the outside and with no knowledge of the implementation:
//   - realised pitch ratio (sine peak, parabolic interpolation) per
//     configuration (sample rate x channels x fftSize x semitones),
//   - measured latency (white-noise cross-correlation against the input,
//     FFT-based, full lag search) versus the reported getLatency(),
//   - unity-ratio transparency: THD+N and the time-aligned residual,
//   - onset delay (first sample of audible output),
//   - an FNV-1a 64 hash of every output stream, so two builds can be
//     compared for bit-identity from the artifact files alone,
//   - magnitude spectra as plain-text artifacts.
//
// Extra probe modes:
//   chop   output equality under different host block choppings
//   drift  realised ratio stability over a long run (fractional-hop drift)
//   nan    recovery after non-finite and denormal input
//   rt     allocation count across processBlock under parameter churn
//
// Usage:
//   characterize_pitchshifter <outdir> [sweep|chop|drift|nan|rt|all]
//
// Everything is deterministic: fixed seeds, fixed block sizes, no wall-clock
// dependence. Build (from the repository root):
//   g++ -std=c++20 -O2 -Wall -Wextra -I. tools/characterize_pitchshifter.cpp
//       -o characterize_pitchshifter   (one line)

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/FFT.h"
#include "../Effects/PitchShifter.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Global allocation counters (used by the `rt` mode; passive otherwise).
// ---------------------------------------------------------------------------
namespace {
std::atomic<uint64_t> gAllocCount{0};
std::atomic<uint64_t> gFreeCount{0};
} // namespace

namespace {
void* countedAlloc(std::size_t n)
{
    gAllocCount.fetch_add(1, std::memory_order_relaxed);
    void* p = std::malloc(n ? n : 1);
    if (!p) std::abort();
    return p;
}
void countedFree(void* p) noexcept
{
    gFreeCount.fetch_add(1, std::memory_order_relaxed);
    std::free(p);
}
} // namespace

void* operator new(std::size_t n) { return countedAlloc(n); }
void* operator new[](std::size_t n) { return countedAlloc(n); }
void operator delete(void* p) noexcept { countedFree(p); }
void operator delete[](void* p) noexcept { countedFree(p); }
void operator delete(void* p, std::size_t) noexcept { countedFree(p); }
void operator delete[](void* p, std::size_t) noexcept { countedFree(p); }

namespace {

constexpr double kPi = std::numbers::pi;

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------
uint64_t fnv1a64(const void* data, size_t bytes, uint64_t h = 1469598103934665603ull)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i)
    {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t hashChannels(const std::vector<std::vector<float>>& chans)
{
    uint64_t h = 1469598103934665603ull;
    for (const auto& c : chans)
        h = fnv1a64(c.data(), c.size() * sizeof(float), h);
    return h;
}

struct Lcg
{
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<uint32_t>(s >> 33); }
    // Uniform in [-0.5, 0.5).
    float noise() { return (static_cast<float>(next()) / 4294967296.0f) * 2.0f - 0.5f; }
};

std::vector<std::vector<float>> makeSine(double rate, int ch, size_t len, double f0, double a = 0.5)
{
    std::vector<std::vector<float>> out(static_cast<size_t>(ch));
    for (int c = 0; c < ch; ++c)
    {
        const double f = (c == 0) ? f0 : f0 * 620.5 / 440.0;  // second channel off-harmonic
        out[static_cast<size_t>(c)].resize(len);
        double ph = 0.0;
        const double w = 2.0 * kPi * f / rate;
        for (size_t n = 0; n < len; ++n)
        {
            out[static_cast<size_t>(c)][n] = static_cast<float>(a * std::sin(ph));
            ph += w;
            if (ph > 2.0 * kPi) ph -= 2.0 * kPi;
        }
    }
    return out;
}

std::vector<std::vector<float>> makeNoise(int ch, size_t len, uint64_t seed)
{
    std::vector<std::vector<float>> out(static_cast<size_t>(ch));
    for (int c = 0; c < ch; ++c)
    {
        Lcg r(seed + static_cast<uint64_t>(c) * 7919u);
        out[static_cast<size_t>(c)].resize(len);
        for (size_t n = 0; n < len; ++n)
            out[static_cast<size_t>(c)][n] = 0.5f * r.noise();
    }
    return out;
}

// Runs the shifter over `input` in place with the given fixed block size (or a
// caller-supplied chopping) and returns the processed channels.
std::vector<std::vector<float>> processAll(dspark::PitchShifter<float>& ps,
                                           std::vector<std::vector<float>> input,
                                           const std::vector<int>& chopping)
{
    const int ch = static_cast<int>(input.size());
    const size_t len = input[0].size();
    std::vector<float*> ptrs(static_cast<size_t>(ch));
    size_t pos = 0;
    size_t chopIdx = 0;
    while (pos < len)
    {
        const int want = chopping[chopIdx % chopping.size()];
        ++chopIdx;
        const int blk = static_cast<int>(std::min<size_t>(static_cast<size_t>(want), len - pos));
        for (int c = 0; c < ch; ++c)
            ptrs[static_cast<size_t>(c)] = input[static_cast<size_t>(c)].data() + pos;
        dspark::AudioBufferView<float> view(ptrs.data(), ch, blk);
        ps.processBlock(view);
        pos += static_cast<size_t>(blk);
    }
    return input;
}

// ---------------------------------------------------------------------------
// Spectral measurement (double precision, Hann window, parabolic peak)
// ---------------------------------------------------------------------------
struct SpectrumResult
{
    double peakHz = 0.0;
    double thdnDb = 0.0;           // 10*log10((total - fundamental)/fundamental)
    std::vector<double> magDb;     // full magnitude spectrum of the window
    double binHz = 0.0;
};

SpectrumResult measureSpectrum(const float* x, int W, double rate)
{
    SpectrumResult r;
    dspark::FFTReal<double> fft(static_cast<size_t>(W));
    std::vector<double> t(static_cast<size_t>(W));
    std::vector<double> f(static_cast<size_t>(W + 2));
    for (int n = 0; n < W; ++n)
    {
        const double w = 0.5 - 0.5 * std::cos(2.0 * kPi * n / W);
        t[static_cast<size_t>(n)] = static_cast<double>(x[n]) * w;
    }
    fft.forward(t.data(), f.data());
    const int bins = W / 2 + 1;
    std::vector<double> p(static_cast<size_t>(bins));
    for (int k = 0; k < bins; ++k)
    {
        const double re = f[static_cast<size_t>(2 * k)];
        const double im = f[static_cast<size_t>(2 * k + 1)];
        p[static_cast<size_t>(k)] = re * re + im * im;
    }
    r.binHz = rate / W;
    // Peak (skip DC region).
    int kPk = 2;
    for (int k = 3; k < bins - 1; ++k)
        if (p[static_cast<size_t>(k)] > p[static_cast<size_t>(kPk)]) kPk = k;
    // Parabolic interpolation on log magnitude.
    const double eps = 1e-300;
    const double la = 0.5 * std::log(p[static_cast<size_t>(kPk - 1)] + eps);
    const double lb = 0.5 * std::log(p[static_cast<size_t>(kPk)] + eps);
    const double lc = 0.5 * std::log(p[static_cast<size_t>(kPk + 1)] + eps);
    const double den = la - 2.0 * lb + lc;
    const double d = (std::abs(den) > 1e-12) ? 0.5 * (la - lc) / den : 0.0;
    r.peakHz = (static_cast<double>(kPk) + d) * r.binHz;
    // THD+N: everything outside fundamental +-3 bins, 20 Hz .. min(20 kHz, Nyquist).
    double pTot = 0.0, pFund = 0.0;
    const int kLo = std::max(1, static_cast<int>(20.0 / r.binHz));
    const int kHi = std::min(bins - 1, static_cast<int>(std::min(20000.0, rate * 0.5) / r.binHz));
    for (int k = kLo; k <= kHi; ++k) pTot += p[static_cast<size_t>(k)];
    for (int k = std::max(kLo, kPk - 3); k <= std::min(kHi, kPk + 3); ++k)
        pFund += p[static_cast<size_t>(k)];
    r.thdnDb = 10.0 * std::log10(std::max(pTot - pFund, 1e-300) / std::max(pFund, 1e-300));
    r.magDb.resize(static_cast<size_t>(bins));
    for (int k = 0; k < bins; ++k)
        r.magDb[static_cast<size_t>(k)] = 10.0 * std::log10(p[static_cast<size_t>(k)] + 1e-300);
    return r;
}

// ---------------------------------------------------------------------------
// FFT cross-correlation: best alignment of a slice of `out` against `in`.
// Returns lag (out delayed by `lag` samples relative to in) and the
// normalized peak correlation.
// ---------------------------------------------------------------------------
int64_t bestLag(const std::vector<float>& in, const std::vector<float>& out,
                size_t slicePos, int sliceLen, double& peakNorm)
{
    const size_t N = in.size();
    size_t M = 1;
    while (M < N + static_cast<size_t>(sliceLen)) M <<= 1;
    dspark::FFTReal<float> fft(M);
    std::vector<float> a(M, 0.0f), b(M, 0.0f);
    std::vector<float> fa(M + 2), fb(M + 2);
    std::copy(in.begin(), in.end(), a.begin());
    for (int k = 0; k < sliceLen; ++k)
        b[static_cast<size_t>(k)] = out[slicePos + static_cast<size_t>(k)];
    fft.forward(a.data(), fa.data());
    fft.forward(b.data(), fb.data());
    // r = IFFT( FFT(in) * conj(FFT(slice)) ):  r[m] = sum_k in[m+k] * slice[k]
    const size_t bins = M / 2 + 1;
    for (size_t k = 0; k < bins; ++k)
    {
        const float ar = fa[2 * k], ai = fa[2 * k + 1];
        const float br = fb[2 * k], bi = fb[2 * k + 1];
        fa[2 * k]     = ar * br + ai * bi;
        fa[2 * k + 1] = ai * br - ar * bi;
    }
    fft.inverse(fa.data(), a.data());
    size_t mBest = 0;
    float rBest = -1.0f;
    for (size_t m = 0; m + static_cast<size_t>(sliceLen) <= N; ++m)
        if (std::abs(a[m]) > rBest) { rBest = std::abs(a[m]); mBest = m; }
    // Normalization for the peak value.
    double eIn = 0.0, eSl = 0.0;
    for (int k = 0; k < sliceLen; ++k)
    {
        const double vi = in[mBest + static_cast<size_t>(k)];
        const double vs = out[slicePos + static_cast<size_t>(k)];
        eIn += vi * vi;
        eSl += vs * vs;
    }
    peakNorm = (eIn > 0.0 && eSl > 0.0) ? static_cast<double>(rBest) / std::sqrt(eIn * eSl) : 0.0;
    return static_cast<int64_t>(slicePos) - static_cast<int64_t>(mBest);
}

// ---------------------------------------------------------------------------
// Configured run description
// ---------------------------------------------------------------------------
struct RunCfg
{
    double rate;
    int ch;
    int fft;
    double st;         // semitones
};

std::string cfgId(const RunCfg& c)
{
    char buf[96];
    std::snprintf(buf, sizeof(buf), "s%dc%df%dst%+05.1f",
                  static_cast<int>(c.rate), c.ch, c.fft, c.st);
    return std::string(buf);
}

std::unique_ptr<dspark::PitchShifter<float>> makeShifter(const RunCfg& c, float mix = 1.0f)
{
    auto ps = std::make_unique<dspark::PitchShifter<float>>();
    ps->setSemitones(static_cast<float>(c.st));
    ps->setMix(mix);
    dspark::AudioSpec spec;
    spec.sampleRate = c.rate;
    spec.maxBlockSize = 512;
    spec.numChannels = c.ch;
    ps->prepare(spec, c.fft);   // reset() inside snaps the active ratio: no glide
    return ps;
}

FILE* gCsv = nullptr;
std::string gOutDir;

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------
void sweepOne(const RunCfg& c, bool writeSpectrum)
{
    const std::string id = cfgId(c);
    const double targetRatio = std::exp2(c.st / 12.0);
    auto ps = makeShifter(c);
    const int reported = ps->getLatency();

    const int W = 131072;
    const size_t skip = static_cast<size_t>(reported) + 4u * static_cast<size_t>(c.fft) + 8192u;
    const size_t len = skip + static_cast<size_t>(W) + 4096u;

    auto in = makeSine(c.rate, c.ch, len, 440.0);
    auto inCopy = in;   // for residual
    auto out = processAll(*ps, std::move(in), {512});
    const uint64_t h = hashChannels(out);

    // Onset delay: first output sample above 5% of nominal amplitude.
    size_t onset = len;
    for (size_t n = 0; n < len; ++n)
        if (std::abs(out[0][n]) > 0.025f) { onset = n; break; }

    // Realised frequency, both channels.
    const auto sp0 = measureSpectrum(out[0].data() + skip, W, c.rate);
    double f1 = 0.0;
    if (c.ch > 1)
        f1 = measureSpectrum(out[1].data() + skip, W, c.rate).peakHz;
    const double realized = sp0.peakHz / 440.0;
    const double centsErr = 1200.0 * std::log2(realized / targetRatio);

    // Unity-only: THD+N and residual at the REPORTED latency.
    double thdn = std::numeric_limits<double>::quiet_NaN();
    double residDb = std::numeric_limits<double>::quiet_NaN();
    if (c.st == 0.0)
    {
        thdn = sp0.thdnDb;
        double eR = 0.0, eI = 0.0;
        for (size_t n = skip; n < skip + static_cast<size_t>(W); ++n)
        {
            const double d = static_cast<double>(out[0][n])
                           - static_cast<double>(inCopy[0][n - static_cast<size_t>(reported)]);
            const double v = inCopy[0][n - static_cast<size_t>(reported)];
            eR += d * d;
            eI += v * v;
        }
        residDb = 10.0 * std::log10(std::max(eR, 1e-300) / std::max(eI, 1e-300));
    }

    std::fprintf(gCsv,
        "sine,%s,%.0f,%d,%d,%+.1f,%.10f,%zu,%zu,%d,%zu,%.6f,%.6f,%.10f,%+.4f,%.2f,%.2f,%016" PRIx64 "\n",
        id.c_str(), c.rate, c.ch, c.fft, c.st, targetRatio, len, len, reported,
        onset, sp0.peakHz, f1, realized, centsErr, thdn, residDb, h);
    std::fflush(gCsv);

    if (writeSpectrum)
    {
        const std::string path = gOutDir + "/spectrum_" + id + ".txt";
        if (FILE* f = std::fopen(path.c_str(), "w"))
        {
            std::fprintf(f, "# hz db  (%s, steady window %d)\n", id.c_str(), W);
            const int bins = static_cast<int>(sp0.magDb.size());
            for (int k = 0; k < bins; ++k)
            {
                const double hz = k * sp0.binHz;
                if (hz > 20000.0) break;
                std::fprintf(f, "%.3f %.2f\n", hz, sp0.magDb[static_cast<size_t>(k)]);
            }
            std::fclose(f);
        }
    }
}

void latencyOne(const RunCfg& c, float mix)
{
    const std::string id = cfgId(c) + (mix == 0.0f ? "-dry" : "-wet");
    auto ps = makeShifter(c, mix);
    const int reported = ps->getLatency();
    const int W = 65536;
    const size_t skip = static_cast<size_t>(reported) + 4u * static_cast<size_t>(c.fft);
    const size_t len = skip + static_cast<size_t>(W) + 8192u;
    auto in = makeNoise(c.ch, len, 0x5eed0001u);
    auto inCopy = in;
    auto out = processAll(*ps, std::move(in), {512});
    const uint64_t h = hashChannels(out);

    double pk0 = 0.0, pk1 = 0.0;
    const int64_t lag0 = bestLag(inCopy[0], out[0], skip, W, pk0);
    int64_t lag1 = 0;
    if (c.ch > 1) lag1 = bestLag(inCopy[1], out[1], skip, W, pk1);

    // Dry-path check: with mix 0 the output should be an EXACT delayed copy.
    size_t exact = 0, total = 0;
    if (mix == 0.0f)
    {
        for (size_t n = static_cast<size_t>(reported); n < len; ++n)
        {
            total++;
            if (out[0][n] == inCopy[0][n - static_cast<size_t>(reported)]) exact++;
        }
    }

    std::fprintf(gCsv,
        "latency,%s,%.0f,%d,%d,%+.1f,%d,%" PRId64 ",%" PRId64 ",%.4f,%.4f,%" PRId64 ",%zu,%zu,%016" PRIx64 "\n",
        id.c_str(), c.rate, c.ch, c.fft, c.st, reported, lag0, lag1, pk0, pk1,
        lag0 - reported, exact, total, h);
    std::fflush(gCsv);
}

void runSweep()
{
    std::fprintf(gCsv, "# sine,id,rate,ch,fft,st,target_ratio,in_len,out_len,reported_latency,"
                       "onset,f_out_ch0,f_out_ch1,realized_ratio,cents_err,thdn_db,resid_db,hash\n");
    std::fprintf(gCsv, "# latency,id,rate,ch,fft,st,reported_latency,lag_ch0,lag_ch1,peak_ch0,"
                       "peak_ch1,lag_minus_reported,dry_exact,dry_total,hash\n");

    const double stSet[] = { -12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0 };
    for (double rate : { 44100.0, 96000.0 })
        for (int ch : { 1, 2 })
            for (double st : stSet)
                sweepOne({ rate, ch, 2048, st },
                         rate == 44100.0 && ch == 2);   // spectra for the 44.1 stereo family

    for (int fft : { 256, 1024, 8192, 65536 })
        for (double st : { -12.0, 0.0, 12.0 })
            sweepOne({ 44100.0, 2, fft, st }, st != 0.0 || fft == 256 || fft == 65536);

    // The upper extreme of the accepted range, unity, mono (long run).
    sweepOne({ 44100.0, 1, 1 << 20, 0.0 }, true);

    // Measured latency at unity across the fftSize range, plus rate/channel spread.
    for (int fft : { 256, 1024, 2048, 8192, 65536 })
        latencyOne({ 44100.0, 1, fft, 0.0 }, 1.0f);
    latencyOne({ 96000.0, 2, 2048, 0.0 }, 1.0f);
    latencyOne({ 96000.0, 1, 256, 0.0 }, 1.0f);
    latencyOne({ 44100.0, 1, 1 << 20, 0.0 }, 1.0f);
    latencyOne({ 44100.0, 1, 2048, 0.0 }, 0.0f);   // dry path
}

// ---------------------------------------------------------------------------
// chop: bit-identity under different host block choppings
// ---------------------------------------------------------------------------
void runChop()
{
    std::fprintf(gCsv, "# chop,id,st,chopping,hash,max_abs_diff_vs_512\n");
    for (double st : { 0.0, 7.0 })
    {
        const RunCfg c { 44100.0, 2, 2048, st };
        const size_t len = 44100 * 3;
        auto base = makeSine(c.rate, c.ch, len, 440.0);
        // Add deterministic noise so the signal is not analytically simple.
        Lcg r(0xC0FFEEu);
        for (auto& chv : base)
            for (auto& v : chv) v += 0.05f * r.noise();

        struct Chop { const char* name; std::vector<int> blocks; };
        std::vector<Chop> chops;
        chops.push_back({ "b512", {512} });
        chops.push_back({ "b64",  {64} });
        chops.push_back({ "b1",   {1} });
        chops.push_back({ "b371", {371} });
        {
            std::vector<int> mixed;
            Lcg m(0xB10C5u);
            for (int i = 0; i < 64; ++i) mixed.push_back(1 + static_cast<int>(m.next() % 512u));
            chops.push_back({ "mixed", mixed });
        }

        std::vector<std::vector<float>> ref;
        for (const auto& chop : chops)
        {
            auto ps = makeShifter(c);
            auto out = processAll(*ps, base, chop.blocks);
            const uint64_t h = hashChannels(out);
            double maxDiff = 0.0;
            if (chop.name == std::string("b512")) ref = out;
            else
                for (size_t cc = 0; cc < out.size(); ++cc)
                    for (size_t n = 0; n < out[cc].size(); ++n)
                        maxDiff = std::max(maxDiff,
                            std::abs(static_cast<double>(out[cc][n]) - static_cast<double>(ref[cc][n])));
            std::fprintf(gCsv, "chop,%s,%+.1f,%s,%016" PRIx64 ",%.3e\n",
                         cfgId(c).c_str(), st, chop.name, h, maxDiff);
            std::fflush(gCsv);
        }
    }
}

// ---------------------------------------------------------------------------
// drift: realised ratio stability over a long run
// ---------------------------------------------------------------------------
void runDrift()
{
    std::fprintf(gCsv, "# drift,id,st,window_index,t_sec,f_out,realized_ratio,cents_err\n");
    for (double st : { 7.0, -7.0 })
    {
        const RunCfg c { 44100.0, 1, 2048, st };
        const double targetRatio = std::exp2(st / 12.0);
        const size_t len = static_cast<size_t>(44100) * 120;   // 2 minutes
        auto ps = makeShifter(c);
        auto out = processAll(*ps, makeSine(c.rate, c.ch, len, 440.0), {512});

        const int W = 131072;
        const size_t start = static_cast<size_t>(ps->getLatency()) + 8u * 2048u;
        int idx = 0;
        double cMin = 1e9, cMax = -1e9;
        for (size_t p = start; p + static_cast<size_t>(W) < len; p += static_cast<size_t>(W))
        {
            const auto sp = measureSpectrum(out[0].data() + p, W, c.rate);
            const double realized = sp.peakHz / 440.0;
            const double cents = 1200.0 * std::log2(realized / targetRatio);
            cMin = std::min(cMin, cents);
            cMax = std::max(cMax, cents);
            std::fprintf(gCsv, "drift,%s,%+.1f,%d,%.2f,%.6f,%.10f,%+.5f\n",
                         cfgId(c).c_str(), st, idx, p / c.rate, sp.peakHz, realized, cents);
            ++idx;
        }
        std::fprintf(gCsv, "# drift summary st=%+.1f: cents min %+.5f max %+.5f spread %.5f\n",
                     st, cMin, cMax, cMax - cMin);
        std::fflush(gCsv);
    }
}

// ---------------------------------------------------------------------------
// nan: recovery after non-finite and denormal input
// ---------------------------------------------------------------------------
void runNan()
{
    std::fprintf(gCsv, "# nan,event,inject_begin,inject_end,last_nonfinite_out,recovery_samples\n");
    const RunCfg c { 44100.0, 1, 2048, 0.0 };
    const size_t len = 44100 * 6;
    auto in = makeSine(c.rate, c.ch, len, 440.0);

    struct Inject { const char* name; size_t begin; size_t end; };
    const size_t sr = 44100;
    std::vector<Inject> inj = {
        { "nan",      sr,     sr + 256 },
        { "inf",      2 * sr, 2 * sr + 256 },
        { "denormal", 3 * sr, 3 * sr + static_cast<size_t>(sr) / 4 },
    };
    for (size_t n = inj[0].begin; n < inj[0].end; ++n)
        in[0][n] = std::numeric_limits<float>::quiet_NaN();
    for (size_t n = inj[1].begin; n < inj[1].end; ++n)
        in[0][n] = std::numeric_limits<float>::infinity();
    for (size_t n = inj[2].begin; n < inj[2].end; ++n)
        in[0][n] = 1.0e-38f * ((n & 1) ? 1.0f : -1.0f);

    auto ps = makeShifter(c);
    auto out = processAll(*ps, std::move(in), {512});

    for (size_t ei = 0; ei < inj.size(); ++ei)
    {
        const auto& e = inj[ei];
        size_t last = 0;
        bool any = false;
        // Scan up to the next injection (or the end), so events do not overlap.
        const size_t scanEnd = (ei + 1 < inj.size()) ? inj[ei + 1].begin : len;
        for (size_t n = e.begin; n < scanEnd; ++n)
            if (!std::isfinite(out[0][n])) { last = n; any = true; }
        const long long rec = any ? static_cast<long long>(last) - static_cast<long long>(e.end) : -1;
        std::fprintf(gCsv, "nan,%s,%zu,%zu,%zu,%lld\n", e.name, e.begin, e.end,
                     any ? last : 0, rec);
    }
    // Post-recovery sanity: amplitude back in range in the final second.
    float maxTail = 0.0f;
    for (size_t n = len - sr; n < len; ++n) maxTail = std::max(maxTail, std::abs(out[0][n]));
    std::fprintf(gCsv, "# nan tail max abs (last second): %.4f (expected ~0.5..0.7)\n", maxTail);
    std::fflush(gCsv);
}

// ---------------------------------------------------------------------------
// rt: allocation count across processBlock under parameter churn
// ---------------------------------------------------------------------------
void runRt()
{
    const RunCfg c { 44100.0, 2, 2048, 0.0 };
    auto ps = makeShifter(c);
    const int blk = 512;
    std::vector<std::vector<float>> buf(2, std::vector<float>(static_cast<size_t>(blk)));
    auto sig = makeSine(c.rate, 2, static_cast<size_t>(blk), 440.0);
    std::vector<float*> ptrs = { buf[0].data(), buf[1].data() };

    const uint64_t a0 = gAllocCount.load(std::memory_order_relaxed);
    const uint64_t f0 = gFreeCount.load(std::memory_order_relaxed);
    for (int i = 0; i < 4000; ++i)
    {
        // Parameter churn (single-threaded here; the setters are atomic stores).
        ps->setSemitones(static_cast<float>((i % 25) - 12));
        ps->setMix(static_cast<float>(i % 3) * 0.5f);
        ps->setTransientPreserve((i & 1) != 0);
        ps->setFormantPreserve((i & 2) != 0);
        for (int ch = 0; ch < 2; ++ch)
            std::memcpy(buf[static_cast<size_t>(ch)].data(), sig[static_cast<size_t>(ch)].data(),
                        static_cast<size_t>(blk) * sizeof(float));
        dspark::AudioBufferView<float> view(ptrs.data(), 2, blk);
        ps->processBlock(view);
    }
    const uint64_t a1 = gAllocCount.load(std::memory_order_relaxed);
    const uint64_t f1 = gFreeCount.load(std::memory_order_relaxed);
    std::fprintf(gCsv, "# rt,alloc_delta,%" PRIu64 ",free_delta,%" PRIu64 ",blocks,4000\n",
                 a1 - a0, f1 - f0);
    std::printf("rt: allocations during 4000 processBlock calls with parameter churn: "
                "%" PRIu64 " new / %" PRIu64 " delete (must be 0 / 0)\n", a1 - a0, f1 - f0);
    std::fflush(gCsv);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <outdir> [sweep|chop|drift|nan|rt|all]\n", argv[0]);
        return 2;
    }
    gOutDir = argv[1];
    const std::string mode = (argc > 2) ? argv[2] : "sweep";

    const std::string csvPath = gOutDir + "/characterization_" + mode + ".csv";
    gCsv = std::fopen(csvPath.c_str(), "w");
    if (!gCsv)
    {
        std::fprintf(stderr, "cannot open %s\n", csvPath.c_str());
        return 2;
    }

    if (mode == "sweep" || mode == "all") runSweep();
    if (mode == "chop"  || mode == "all") runChop();
    if (mode == "drift" || mode == "all") runDrift();
    if (mode == "nan"   || mode == "all") runNan();
    if (mode == "rt"    || mode == "all") runRt();

    std::fclose(gCsv);
    std::printf("wrote %s\n", csvPath.c_str());
    return 0;
}
