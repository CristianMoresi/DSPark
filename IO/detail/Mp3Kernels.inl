// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// Mp3Kernels.inl - the transforms of the MPEG audio decoder in IO/Mp3File.h,
// written on the vector abstraction of Core/detail/SimdVec.inl and included
// once per instruction set into its own namespace (no include guard on
// purpose): the build's baseline into dspark::detail::mp3 and, in x86-64
// builds below AVX, again into dspark::detail::mp3::avx2 inside an AVX2+FMA
// target region, called only after the run-time CPU check. The includer
// defines, in the enclosing namespace, `template <typename T, int W> using
// Vec` and `inline constexpr int kWide` (lanes of a double vector). This file
// includes nothing; <array> and <cmath> come from IO/Mp3File.h.
//
// Every output is summed in the same order as the plain formula, so the
// baseline copy without FMA reproduces the scalar arithmetic bit for bit.

// ============================================================================
// Polyphase synthesis (ISO 11172-3 Annex A.2)
// ============================================================================

/// N vectors of W lanes held by value. A struct of its own rather than a
/// standard array, so that no vector type becomes a template argument (GCC
/// drops a vector type's attributes there and warns).
template <int N, int W>
struct DctBlock
{
    typename Vec<double, W>::V v[N];
};

/// One level of the 32-point DCT-II split of B. G. Lee (IEEE Trans. ASSP
/// 32(6), 1984) on W independent lanes, unrolled at compile time. Each level
/// takes and returns its values by value: with no aliasing between the
/// levels the compiler keeps the small ones in registers instead of storing
/// every butterfly to memory and reading it back.
template <int N, int W>
inline DctBlock<N, W> dctValues(const DctBlock<N, W>& x, const double* twiddle) noexcept
{
    using VD = Vec<double, W>;
    if constexpr (N == 1) return x;
    else
    {
        constexpr int h = N / 2;
        DctBlock<h, W> lo, hi;
        for (int i = 0; i < h; ++i)
        {
            const typename VD::V a = x.v[i], b = x.v[N - 1 - i];
            lo.v[i] = VD::add(a, b);
            hi.v[i] = VD::mul(VD::sub(a, b), VD::set1(twiddle[h + i]));
        }
        const DctBlock<h, W> L = dctValues<h, W>(lo, twiddle);
        const DctBlock<h, W> H = dctValues<h, W>(hi, twiddle);
        DctBlock<N, W> y;
        for (int i = 0; i < h - 1; ++i)
        {
            y.v[2 * i] = L.v[i];
            y.v[2 * i + 1] = VD::add(H.v[i], H.v[i + 1]);
        }
        y.v[2 * h - 2] = L.v[h - 1];
        y.v[2 * h - 1] = H.v[h - 1];
        return y;
    }
}

/// Twiddles of the split: 0.5 / cos(pi (2i + 1) / (2n)) at index n / 2 + i.
inline const double* dctTwiddles() noexcept
{
    static const auto table = [] {
        std::array<double, 32> t {};
        constexpr double kPi = 3.14159265358979323846;
        for (int n = 2; n <= 32; n *= 2)
            for (int i = 0; i < n / 2; ++i)
                t[static_cast<size_t>(n / 2 + i)] = 0.5 / std::cos(kPi * (2 * i + 1) / (2.0 * n));
        return t;
    }();
    return table.data();
}

/// Values in the ring of matrixed slots: 32 vectors of 64, the 16 the window
/// reads plus room to store a whole group of new slots before windowing it.
inline constexpr int kSynthRing = 2048;

/// The windowing of S slots that are already in the ring at offsets[s]:
/// pcm[32 s + j] = sum_{i<16} U_s[32 i + j] D[32 i + j], with U_s taken from
/// the 16 stored vectors (U[64 i + j] = V_i[j] and U[64 i + 32 + j] =
/// V_i[96 + j]). Both pieces are 32 contiguous values that never wrap
/// (offsets are multiples of 64), so each term is one contiguous multiply-add
/// over the outputs, summed in the order i = 0..15. The S slots share every
/// load of the window, which is what bounds the cost of a single slot; eight
/// accumulators in all cover the multiply-add latency. Outputs are clamped
/// to [-1, 1] and converted to float.
template <int S>
inline void windowSlots(const double* buf, const int* offsets, const double* window, float* pcm) noexcept
{
    using VD = Vec<double, kWide>;
    constexpr int kAcc = (S == 1 && kWide < 4) ? 4 : 8;
    constexpr int kVecs = (kAcc / S > 0) ? kAcc / S : 1;
    constexpr int kChunk = kVecs * kWide;
    constexpr int kMask = kSynthRing - 1;
    static_assert(32 % kChunk == 0, "the outputs must fill whole chunks");
    for (int jb = 0; jb < 32; jb += kChunk)
    {
        typename VD::V a[S][kVecs];
        for (int s = 0; s < S; ++s)
            for (int v = 0; v < kVecs; ++v) a[s][v] = VD::set1(0.0);
        for (int i = 0; i < 8; ++i)
        {
            const double* da = window + i * 64 + jb;
            const double* db = da + 32;
            for (int v = 0; v < kVecs; ++v)
            {
                const typename VD::V w = VD::load(da + v * kWide);
                for (int s = 0; s < S; ++s)
                    a[s][v] = VD::madd(VD::load(buf + ((offsets[s] + i * 128) & kMask) + jb + v * kWide),
                                       w, a[s][v]);
            }
            for (int v = 0; v < kVecs; ++v)
            {
                const typename VD::V w = VD::load(db + v * kWide);
                for (int s = 0; s < S; ++s)
                    a[s][v] = VD::madd(VD::load(buf + ((offsets[s] + i * 128 + 96) & kMask) + jb + v * kWide),
                                       w, a[s][v]);
            }
        }
        for (int s = 0; s < S; ++s)
        {
            double acc[kChunk];
            for (int v = 0; v < kVecs; ++v) VD::store(acc + v * kWide, a[s][v]);
            float* out = pcm + 32 * s + jb;
            for (int j = 0; j < kChunk; ++j)
            {
                // [-1, 1] in the form min/max instructions take (a NaN would
                // pass through, as from the comparisons it replaces).
                double x = acc[j];
                x = (-1.0 > x) ? -1.0 : x;
                x = (1.0 < x) ? 1.0 : x;
                out[j] = static_cast<float>(x);
            }
        }
    }
}

/// Stores the quad q[0..3] at dst, optionally reversed and negated, with
/// W-lane vectors. The negation is a multiplication by -1, which gives -x
/// bit for bit for every finite x, zeros included.
template <int W, bool kReverse, bool kNegate>
inline void putQuad(double* dst, const double* q) noexcept
{
    using VD = Vec<double, W>;
    if constexpr (W == 4)
    {
        typename VD::V v = VD::load(q);
        if constexpr (kReverse) v = VD::reverse(v);
        if constexpr (kNegate) v = VD::mul(v, VD::set1(-1.0));
        VD::store(dst, v);
    }
    else if constexpr (W == 2)
    {
        typename VD::V lo = VD::load(q), hi = VD::load(q + 2);
        if constexpr (kReverse)
        {
            const typename VD::V t = VD::reverse(hi);
            hi = VD::reverse(lo);
            lo = t;
        }
        if constexpr (kNegate)
        {
            lo = VD::mul(lo, VD::set1(-1.0));
            hi = VD::mul(hi, VD::set1(-1.0));
        }
        VD::store(dst, lo);
        VD::store(dst + 2, hi);
    }
    else
    {
        for (int m = 0; m < 4; ++m)
        {
            const double x = q[kReverse ? 3 - m : m];
            dst[m] = kNegate ? -x : x;
        }
    }
}

/**
 * Synthesizes W consecutive time slots starting at slot ss, from subband
 * samples stored subband by subband (input[sb * stride + slot]): the samples
 * of neighbouring slots are adjacent in memory, so the W slots go through the
 * DCT together, one per lane; then each slot's vector is stored in the
 * kSynthRing-value ring buf (offset steps back by 64 per slot), and the W
 * slots are windowed together into W * 32 samples of pcm.
 */
template <int W>
inline void synthesizeSlots(const double* input, int stride, int ss, double* buf, int& offset,
                            const double* window, float* pcm) noexcept
{
    using VD = Vec<double, W>;
    DctBlock<32, W> X;
    for (int sb = 0; sb < 32; ++sb) X.v[sb] = VD::load(input + sb * stride + ss);
    const DctBlock<32, W> S = dctValues<32, W>(X, dctTwiddles());
    // The DCT outputs C[k] of each lane, four at a time: T[4 W b + 4 lane + m]
    // = C[4 b + m] of that lane.
    double T[32 * W];
    for (int b = 0; b < 8; ++b)
        VD::storeInterleave4(T + 4 * W * b, S.v[4 * b], S.v[4 * b + 1], S.v[4 * b + 2], S.v[4 * b + 3]);

    int offsets[W];
    for (int lane = 0; lane < W; ++lane)
    {
        // Matrixing V[i] = sum_k S[k] cos((16 + i)(2k + 1) pi / 64) needs only
        // the 32 DCT-II outputs C[m]: V[i] = C[16 + i] for i < 16, V[16] = 0,
        // V[16 + i] = -C[32 - i], V[48 - i] = -C[i] and V[48 + i] = V[48 - i]
        // (so V[48 + i] = -C[i] for i = 0..15).
        offset = (offset - 64) & (kSynthRing - 1);
        offsets[lane] = offset;
        double* V = buf + offset;   // a multiple of 64: no wrap
        const double* q = T + 4 * lane;   // quad b at q + 4 W b
        for (int b = 0; b < 4; ++b)
        {
            putQuad<W, false, false>(V + 4 * b, q + 4 * W * (4 + b));
            putQuad<W, true, true>(V + 17 + 4 * b, q + 4 * W * (7 - b));
            putQuad<W, true, true>(V + 33 + 4 * b, q + 4 * W * (3 - b));
        }
        for (int b = 0; b < 4; ++b) putQuad<W, false, true>(V + 48 + 4 * b, q + 4 * W * b);
        V[16] = 0.0;
    }
    windowSlots<W>(buf, offsets, window, pcm + ss * 32);
}

// ============================================================================
// IMDCT (ISO 11172-3 2.4.3.4.10)
// ============================================================================

/// out[k] = sum_n in[n] * table[n * S + k] for the S outputs of a column
/// (S a whole number of vectors), each output summed in the order n = 0..N-1.
template <int N, int S>
inline void macColumns(const double* in, const double* table, double* out) noexcept
{
    using VD = Vec<double, kWide>;
    static_assert(S % kWide == 0, "the column stride must be a whole number of vectors");
    constexpr int kVecs = S / kWide;
    typename VD::V acc[kVecs];
    for (int v = 0; v < kVecs; ++v) acc[v] = VD::set1(0.0);
    for (int n = 0; n < N; ++n)
    {
        const typename VD::V x = VD::set1(in[n]);
        const double* c = table + n * S;
        for (int v = 0; v < kVecs; ++v) acc[v] = VD::madd(x, VD::load(c + v * kWide), acc[v]);
    }
    for (int v = 0; v < kVecs; ++v) VD::store(out + v * kWide, acc[v]);
}

/// The 18 columns of the IMDCT outputs rounded up to whole vectors.
inline constexpr int kImdctStride = (18 + kWide - 1) / kWide * kWide;

/// Cosine table of imdct36(), 18 columns of kImdctStride values.
///
/// The 36 outputs have 18 independent values: out[17 - k] = -out[k] and
/// out[35 - k] = out[18 + k] for k = 0..8. Column n of the table holds the
/// cosines of in[n] for those 18 outputs (zero past them).
inline const double* imdctTable36() noexcept
{
    static const auto table = [] {
        std::array<double, 18 * kImdctStride> t {};
        constexpr double kPi = 3.14159265358979323846;
        for (int n = 0; n < 18; ++n)
            for (int k = 0; k < 9; ++k)
            {
                t[static_cast<size_t>(n * kImdctStride + k)]
                    = std::cos(kPi / 72.0 * (2.0 * k + 19.0) * (2.0 * n + 1.0));
                t[static_cast<size_t>(n * kImdctStride + 9 + k)]
                    = std::cos(kPi / 72.0 * (2.0 * (k + 18) + 19.0) * (2.0 * n + 1.0));
            }
        return t;
    }();
    return table.data();
}

/// 36-point IMDCT, out[k] = sum_n in[n] cos(pi / 72 (2k + 19)(2n + 1)).
inline void imdct36(const double in[18], double out[36]) noexcept
{
    double acc[kImdctStride];
    macColumns<18, kImdctStride>(in, imdctTable36(), acc);
    for (int k = 0; k < 9; ++k)
    {
        out[k] = acc[k];          out[17 - k] = -acc[k];
        out[k + 18] = acc[9 + k]; out[35 - k] = acc[9 + k];
    }
}

/// Two 36-point IMDCTs at once (see imdct36()): with vectors of four or more
/// lanes one transform has too few independent accumulators to cover the
/// multiply-add latency, and two interleaved ones fill the gap. Each output is
/// summed exactly as in imdct36().
inline void imdct36Pair(const double inA[18], const double inB[18], double outA[36], double outB[36]) noexcept
{
    using VD = Vec<double, kWide>;
    constexpr int kVecs = kImdctStride / kWide;
    const double* table = imdctTable36();
    typename VD::V a[kVecs], b[kVecs];
    for (int v = 0; v < kVecs; ++v) { a[v] = VD::set1(0.0); b[v] = VD::set1(0.0); }
    for (int n = 0; n < 18; ++n)
    {
        const typename VD::V xa = VD::set1(inA[n]);
        const typename VD::V xb = VD::set1(inB[n]);
        const double* c = table + n * kImdctStride;
        for (int v = 0; v < kVecs; ++v)
        {
            const typename VD::V cv = VD::load(c + v * kWide);
            a[v] = VD::madd(xa, cv, a[v]);
            b[v] = VD::madd(xb, cv, b[v]);
        }
    }
    double accA[kImdctStride], accB[kImdctStride];
    for (int v = 0; v < kVecs; ++v)
    {
        VD::store(accA + v * kWide, a[v]);
        VD::store(accB + v * kWide, b[v]);
    }
    for (int k = 0; k < 9; ++k)
    {
        outA[k] = accA[k];          outA[17 - k] = -accA[k];
        outA[k + 18] = accA[9 + k]; outA[35 - k] = accA[9 + k];
        outB[k] = accB[k];          outB[17 - k] = -accB[k];
        outB[k + 18] = accB[9 + k]; outB[35 - k] = accB[9 + k];
    }
}

/// 12-point IMDCT, out[k] = sum_n in[n] cos(pi / 24 (2k + 7)(2n + 1)).
inline void imdct12(const double in[6], double out[12]) noexcept
{
    static_assert(12 % kWide == 0, "12 outputs must fill whole vectors");
    static const auto table = [] {
        std::array<double, 6 * 12> t {};
        constexpr double kPi = 3.14159265358979323846;
        for (int n = 0; n < 6; ++n)
            for (int k = 0; k < 12; ++k)
                t[static_cast<size_t>(n * 12 + k)]
                    = std::cos(kPi / 24.0 * (2.0 * k + 7.0) * (2.0 * n + 1.0));
        return t;
    }();
    macColumns<6, 12>(in, table.data(), out);
}
