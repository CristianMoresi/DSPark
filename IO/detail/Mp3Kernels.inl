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

/// One level of the 32-point DCT-II split of B. G. Lee (IEEE Trans. ASSP
/// 32(6), 1984) on W independent lanes, unrolled at compile time.
template <int N, int W>
inline void dctLevel(typename Vec<double, W>::V* x, typename Vec<double, W>::V* t,
                     const double* twiddle) noexcept
{
    using VD = Vec<double, W>;
    if constexpr (N > 1)
    {
        constexpr int h = N / 2;
        for (int i = 0; i < h; ++i)
        {
            const typename VD::V a = x[i], b = x[N - 1 - i];
            t[i] = VD::add(a, b);
            t[h + i] = VD::mul(VD::sub(a, b), VD::set1(twiddle[h + i]));
        }
        dctLevel<h, W>(t, x, twiddle);
        dctLevel<h, W>(t + h, x + h, twiddle);
        for (int i = 0; i < h - 1; ++i)
        {
            x[2 * i] = t[i];
            x[2 * i + 1] = VD::add(t[h + i], t[h + i + 1]);
        }
        x[2 * h - 2] = t[h - 1];
        x[2 * h - 1] = t[2 * h - 1];
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

/// The windowing of one slot: pcm[j] = sum_{i<16} U[32 i + j] D[32 i + j],
/// with U taken from the 16 stored vectors (U[64 i + j] = V_i[j] and
/// U[64 i + 32 + j] = V_i[96 + j]). Both pieces are 32 contiguous values that
/// never wrap (offsets are multiples of 32), so each term is one contiguous
/// multiply-add over the outputs, summed in the order i = 0..15. Several
/// vectors of outputs are held in accumulators across all sixteen terms, as
/// many independent chains as the register file holds (eight with four-lane
/// vectors, four with two).
inline void windowSlot(const double* buf, int offset, const double* window, double acc[32]) noexcept
{
    using VD = Vec<double, kWide>;
    constexpr int kVecs = (kWide >= 4) ? 8 : 4;
    constexpr int kChunk = kVecs * kWide;
    static_assert(32 % kChunk == 0, "the outputs must fill whole chunks");
    for (int jb = 0; jb < 32; jb += kChunk)
    {
        typename VD::V a[kVecs];
        for (int v = 0; v < kVecs; ++v) a[v] = VD::set1(0.0);
        for (int i = 0; i < 8; ++i)
        {
            const double* va = buf + ((offset + i * 128) & 1023) + jb;
            const double* vb = buf + ((offset + i * 128 + 96) & 1023) + jb;
            const double* da = window + i * 64 + jb;
            const double* db = da + 32;
            for (int v = 0; v < kVecs; ++v)
                a[v] = VD::madd(VD::load(va + v * kWide), VD::load(da + v * kWide), a[v]);
            for (int v = 0; v < kVecs; ++v)
                a[v] = VD::madd(VD::load(vb + v * kWide), VD::load(db + v * kWide), a[v]);
        }
        for (int v = 0; v < kVecs; ++v) VD::store(acc + jb + v * kWide, a[v]);
    }
}

/**
 * Synthesizes W consecutive time slots starting at slot ss, from subband
 * samples stored subband by subband (input[sb * stride + slot]): the samples
 * of neighbouring slots are adjacent in memory, so the W slots go through the
 * DCT together, one per lane; then each slot's vector is stored in the
 * 1024-value ring buf (offset steps back by 64 per slot) and windowed into 32
 * samples of pcm.
 */
template <int W>
inline void synthesizeSlots(const double* input, int stride, int ss, double* buf, int& offset,
                            const double* window, float* pcm) noexcept
{
    using VD = Vec<double, W>;
    typename VD::V S[32];
    for (int sb = 0; sb < 32; ++sb) S[sb] = VD::load(input + sb * stride + ss);
    typename VD::V scratch[32];
    dctLevel<32, W>(S, scratch, dctTwiddles());
    double C[32][W];
    for (int k = 0; k < 32; ++k) VD::store(C[k], S[k]);

    for (int lane = 0; lane < W; ++lane)
    {
        // Matrixing V[i] = sum_k S[k] cos((16 + i)(2k + 1) pi / 64) needs only
        // the 32 DCT-II outputs C[m]: V[i] = C[16 + i] for i < 16, V[16] = 0,
        // V[16 + i] = -C[32 - i], V[48 - i] = -C[i] and V[48 + i] = V[48 - i].
        offset = (offset - 64) & 1023;
        double* V = buf + offset;   // a multiple of 64: no wrap
        for (int i = 0; i < 16; ++i) V[i] = C[16 + i][lane];
        V[16] = 0.0;
        for (int i = 1; i <= 16; ++i) V[16 + i] = -C[32 - i][lane];
        for (int i = 0; i < 16; ++i) V[48 - i] = -C[i][lane];
        for (int i = 1; i < 16; ++i) V[48 + i] = V[48 - i];

        double acc[32];
        windowSlot(buf, offset, window, acc);
        float* out = pcm + (ss + lane) * 32;
        for (int j = 0; j < 32; ++j)
        {
            const double s = acc[j] < -1.0 ? -1.0 : (acc[j] > 1.0 ? 1.0 : acc[j]);
            out[j] = static_cast<float>(s);
        }
    }
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
