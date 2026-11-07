// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// WavetableKernels.inl - the block kernels of WavetableBank::readBlock(),
// written once and included once per instruction set into its own namespace
// (no include guard on purpose). WavetableBank.h includes it into
// detail::wavetable_narrow (SSE2 or NEON) and, in x86-64 builds, again into
// detail::wavetable_avx2 inside SimdOps.h's exact AVX2 region (AVX2 without
// FMA, so nothing is contracted). The includer first defines two structs,
// FloatLanes and DoubleLanes, holding the instruction set's lane operations:
//
//   S, V, N                 sample type, vector of N lanes of S, lane count
//   add, sub, mul, set1     lane arithmetic
//   store(S*, V)            unaligned store of N samples
//   inRange(ph)             all N phases lie in [0, 1) (NaN fails)
//   grid(ph, size, idx)     table indices (truncated) and fractions of N phases
//   frames(pos, maxPos, f0, t)  frame rows and blends of N positions, clamped
//   Rows                    the N lanes' samples at offsets -2..+3
//   gather<Blend>(r, at, step, t)  lane l: the samples at at(l), blended
//                           toward those step(l) further by t[l] when Blend
//   gatherMix<Blend>(r, atA, stepA, atB, stepB, t, w)  the same for two
//                           levels, then b + w (a - b), lane by lane
//   columns(Rows, y)        the six columns y[0..5] (offset -2..+3)
//
// Every operation is the one WavetableBank::read() performs on the same
// values, in the same order, so the kernels return read()'s results bit for
// bit. This file includes nothing.

/// WavetableBank's interpolator (Niemitalo's z-form) on N lanes.
template <class L>
inline typename L::V interpolateLanes(const typename L::V* y, typename L::V x) noexcept
{
    using S = typename L::S;
    using V = typename L::V;
    const auto& k = kWavetableNiemitalo;
    const auto c = [&](size_t i) noexcept { return L::set1(static_cast<S>(k[i])); };
    const V z = L::sub(x, L::set1(S(0.5)));
    const V even1 = L::add(y[3], y[2]), odd1 = L::sub(y[3], y[2]);
    const V even2 = L::add(y[4], y[1]), odd2 = L::sub(y[4], y[1]);
    const V even3 = L::add(y[5], y[0]), odd3 = L::sub(y[5], y[0]);
    const auto sum3 = [&](V p, V q, V r, size_t i) noexcept {
        return L::add(L::add(L::mul(p, c(i)), L::mul(q, c(i + 1))), L::mul(r, c(i + 2)));
    };
    V r = L::add(L::mul(sum3(odd1, odd2, odd3, 15), z), sum3(even1, even2, even3, 12));
    r = L::add(L::mul(r, z), sum3(odd1, odd2, odd3, 9));
    r = L::add(L::mul(r, z), sum3(even1, even2, even3, 6));
    r = L::add(L::mul(r, z), sum3(odd1, odd2, odd3, 3));
    return L::add(L::mul(r, z), sum3(even1, even2, even3, 0));
}

/**
 * @brief Samples N at a time for one level layout while the phases stay in
 *        [0, 1); returns how many were written (a multiple of N), stopping
 *        before the first group holding an out-of-range phase.
 * @tparam Layout    kWavetableOneLevel, kWavetableSameGrid or kWavetableTwoGrids.
 * @tparam PerSample A frame position per sample (else `fixed` for the block).
 * @tparam Blend     The fixed position blends two frames (t not 0).
 */
template <class L, int Layout, bool PerSample, bool Blend>
int readLanes(const WavetableRows& s, const double* phases, const double* positions,
              const WavetableFrame& fixed, typename L::S w, typename L::S* out, int n) noexcept
{
    using S = typename L::S;
    using V = typename L::V;
    constexpr int N = L::N;
    const V wv = L::set1(w);
    const double sizeA = s.sizeA, sizeB = s.sizeB;
    const size_t strideA = s.strideA, strideB = s.strideB;
    const float* const baseA = s.baseA;
    const float* const baseB = s.baseB;
    const float* const fixedA = baseA + fixed.f0 * strideA;
    const float* const fixedB = baseB + fixed.f0 * strideB;
    const size_t fixedStepA = fixed.step * strideA, fixedStepB = fixed.step * strideB;
    [[maybe_unused]] const double maxPos = static_cast<double>(s.numFrames - 1);
    [[maybe_unused]] const int32_t lastFrame = s.numFrames - 1;
    [[maybe_unused]] alignas(32) S ta[N];
    const S fixedT = static_cast<S>(fixed.t);
    // The frame blend of lane l: one value for the block, or one per lane.
    const auto tAt = [&](int l) noexcept -> S {
        if constexpr (PerSample) return ta[l];
        else { (void)l; return fixedT; }
    };
    int j = 0;
    for (; j + N <= n; j += N)
    {
        if (!L::inRange(phases + j)) return j;
        alignas(32) uint32_t ia[N];
        [[maybe_unused]] alignas(32) uint32_t ib[N];
        [[maybe_unused]] alignas(32) int32_t fa[N];
        const V xa = L::grid(phases + j, sizeA, ia);
        V xb = xa;
        if constexpr (Layout == kWavetableTwoGrids) xb = L::grid(phases + j, sizeB, ib);
        if constexpr (PerSample) L::frames(positions + j, maxPos, fa, ta);

        // Lane l's six samples start at at*(l), its next frame's step*(l)
        // samples later.
        const auto atA = [&](int l) noexcept -> const float* {
            if constexpr (PerSample) return baseA + static_cast<size_t>(fa[l]) * strideA + ia[l];
            else return fixedA + ia[l];
        };
        const auto atB = [&](int l) noexcept -> const float* {
            uint32_t i = ia[l];
            if constexpr (Layout == kWavetableTwoGrids) i = ib[l];
            if constexpr (PerSample) return baseB + static_cast<size_t>(fa[l]) * strideB + i;
            else return fixedB + i;
        };
        const auto stepA = [&](int l) noexcept -> size_t {
            if constexpr (PerSample) return fa[l] < lastFrame ? strideA : 0;
            else { (void)l; return fixedStepA; }
        };
        const auto stepB = [&](int l) noexcept -> size_t {
            if constexpr (PerSample) return fa[l] < lastFrame ? strideB : 0;
            else { (void)l; return fixedStepB; }
        };
        // Frame blends with t = 0 read their own samples back exactly: the
        // tables hold no negative zeros (see WavetableBank::build()).
        constexpr bool kLerp = Blend || PerSample;
        typename L::Rows ra;
        V y[6];
        if constexpr (Layout == kWavetableOneLevel)
        {
            L::template gather<kLerp>(ra, atA, stepA, tAt);
            L::columns(ra, y);
            L::store(out + j, interpolateLanes<L>(y, xa));
        }
        else if constexpr (Layout == kWavetableSameGrid)
        {
            // Same grid: blend the samples, interpolate once.
            L::template gatherMix<kLerp>(ra, atA, stepA, atB, stepB, tAt, wv);
            L::columns(ra, y);
            L::store(out + j, interpolateLanes<L>(y, xa));
        }
        else
        {
            typename L::Rows rb;
            L::template gather<kLerp>(ra, atA, stepA, tAt);
            L::template gather<kLerp>(rb, atB, stepB, tAt);
            L::columns(ra, y);
            const V sa = interpolateLanes<L>(y, xa);
            L::columns(rb, y);
            const V sb = interpolateLanes<L>(y, xb);
            L::store(out + j, L::add(sb, L::mul(wv, L::sub(sa, sb))));
        }
    }
    return j;
}

/**
 * @brief readLanes() for the block's layout, frame mode and blend: the loop
 *        specialised for each combination.
 * @return Samples written (a multiple of the lane count).
 */
template <class L>
int readBlockLanes(const WavetableRows& s, int layout, const double* phases, const double* positions,
                   const WavetableFrame& fixed, typename L::S w, typename L::S* out, int n) noexcept
{
    if (positions != nullptr)
    {
        if (layout == kWavetableOneLevel) return readLanes<L, kWavetableOneLevel, true, true>(s, phases, positions, fixed, w, out, n);
        if (layout == kWavetableSameGrid) return readLanes<L, kWavetableSameGrid, true, true>(s, phases, positions, fixed, w, out, n);
        return readLanes<L, kWavetableTwoGrids, true, true>(s, phases, positions, fixed, w, out, n);
    }
    if (static_cast<typename L::S>(fixed.t) != typename L::S(0))
    {
        if (layout == kWavetableOneLevel) return readLanes<L, kWavetableOneLevel, false, true>(s, phases, positions, fixed, w, out, n);
        if (layout == kWavetableSameGrid) return readLanes<L, kWavetableSameGrid, false, true>(s, phases, positions, fixed, w, out, n);
        return readLanes<L, kWavetableTwoGrids, false, true>(s, phases, positions, fixed, w, out, n);
    }
    if (layout == kWavetableOneLevel) return readLanes<L, kWavetableOneLevel, false, false>(s, phases, positions, fixed, w, out, n);
    if (layout == kWavetableSameGrid) return readLanes<L, kWavetableSameGrid, false, false>(s, phases, positions, fixed, w, out, n);
    return readLanes<L, kWavetableTwoGrids, false, false>(s, phases, positions, fixed, w, out, n);
}

/** @brief The block kernels for float or double output. */
inline int readBlockKernel(const WavetableRows& s, int layout, const double* phases, const double* positions,
                           const WavetableFrame& fixed, float w, float* out, int n) noexcept
{
    return readBlockLanes<FloatLanes>(s, layout, phases, positions, fixed, w, out, n);
}

/** @brief The double form. */
inline int readBlockKernel(const WavetableRows& s, int layout, const double* phases, const double* positions,
                           const WavetableFrame& fixed, double w, double* out, int n) noexcept
{
    return readBlockLanes<DoubleLanes>(s, layout, phases, positions, fixed, w, out, n);
}
