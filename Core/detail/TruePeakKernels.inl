// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// TruePeakKernels.inl - the reading loops of the true peak estimators on the
// vector abstraction, written once and included once per instruction set into
// its own namespace (no include guard on purpose). TruePeakDetector.h includes
// it into dspark::detail::truepeak for the build's baseline and, in x86-64
// builds below AVX, again into dspark::detail::truepeak::avx2 inside an
// AVX2+FMA target region, run after the run-time CPU check. The includer
// provides Vec, kWide and kNarrow (the copy's widest vector and the next
// width down) and, from the enclosing namespace, the scalar forms
// refineVertex(), gridReading(), eighthCrests() and raiseToEighthSamples().
//
// Every lane performs the scalar form's operations in the same order (no
// multiply feeds an add, so nothing is contracted either), so the kernels
// return the scalar results bit for bit on every copy. Each loop runs whole
// vectors at the wide width, then at the narrow one, and the positions left
// in the scalar form. This file includes nothing.

/// Lanes of the loops over an 8x grid: the copy's vectors, except two lanes
/// on NEON, which run in the scalar form. There a vector division costs more
/// than the scalar loop spends on the few crests of so fine a grid (measured
/// on an arm64 server core: 2 to 11 % slower in double).
#if defined(DSPARK_SIMD_NEON)
template <typename T> inline constexpr int kEighthWide = kWide<T> > 2 ? kWide<T> : 1;
template <typename T> inline constexpr int kEighthNarrow = kNarrow<T> > 2 ? kNarrow<T> : 1;
#else
template <typename T> inline constexpr int kEighthWide = kWide<T>;
template <typename T> inline constexpr int kEighthNarrow = kNarrow<T>;
#endif

/// refineVertex() on W lanes, its branch replaced by a selection: the vertex
/// is computed for every lane and kept where it applies. Where d is 0 the
/// quotient is infinite or not a number and is not selected (b * d is then
/// not negative).
template <typename T, int W>
inline typename Vec<T, W>::V refineVertexLanes(typename Vec<T, W>::V a, typename Vec<T, W>::V b,
                                               typename Vec<T, W>::V c, typename Vec<T, W>::V m) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const V ab = O::abs(b);
    const V d = O::sub(O::add(a, c), O::add(b, b));
    const V q = O::sub(c, a);
    const V v = O::abs(O::sub(b, O::div(O::mul(q, q), O::mul(O::set1(T(8)), d))));
    const typename O::M crest = O::both(O::both(O::le(O::abs(a), ab), O::le(O::abs(c), ab)),
                                        O::both(O::lt(O::mul(b, d), O::set1(T(0))), O::lt(m, v)));
    return O::select(crest, v, m);
}

/// gridReading() for the W positions from i on.
template <typename T, int W>
inline typename Vec<T, W>::V gridReadingLanes(const T* s, const T* q, const T* h, const T* t, int i) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const V a0 = O::load(s + i), a4 = O::load(s + i + 1);
    const V y0 = O::load(t + i), y1 = O::load(q + i), y2 = O::load(h + i), y3 = O::load(t + i + 1);
    V r = O::maxOf(O::maxOf(O::abs(a0), O::abs(a4)), O::maxOf(O::abs(y2), O::maxOf(O::abs(y1), O::abs(y3))));
    r = refineVertexLanes<T, W>(y0, a0, y1, r);
    r = refineVertexLanes<T, W>(a0, y1, y2, r);
    r = refineVertexLanes<T, W>(y1, y2, y3, r);
    r = refineVertexLanes<T, W>(y2, y3, a4, r);
    return r;
}

/// eighthCrests() for the W points of an 8x grid from u on.
template <typename T, int W>
inline typename Vec<T, W>::V eighthCrestsLanes(const T* a, const T* e, int u, typename Vec<T, W>::V m) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const V before = O::load(a + u - 1), a0 = O::load(a + u), e0 = O::load(e + u), e1 = O::load(e + u + 1);
    return refineVertexLanes<T, W>(e0, a0, e1, refineVertexLanes<T, W>(before, e0, a0, m));
}

/// How far above the grid point it is centred on a refined crest can stand:
/// a quarter of it (see refineVertex()), and the rounding of the vertex's own
/// arithmetic, which stays orders of magnitude below the margin kept here.
template <typename T> inline constexpr T kCrestReach = T(1.2501);

/// Where a crest centred on v could stand above limit (false where v is not
/// a number: no crest is refined there).
template <typename T, int W>
inline typename Vec<T, W>::M reaches(typename Vec<T, W>::V v, typename Vec<T, W>::V limit) noexcept
{
    using O = Vec<T, W>;
    return O::lt(limit, O::mul(O::abs(v), O::set1(kCrestReach<T>)));
}

/// The largest lane of best and result.
template <typename T, int W>
inline T largestLane(typename Vec<T, W>::V best, T result) noexcept
{
    T lanes[W];
    Vec<T, W>::store(lanes, best);
    for (int j = 0; j < W; ++j) result = std::max(result, lanes[j]);
    return result;
}

/// A running maximum of finite values on W lanes, kept in two vectors so
/// that consecutive vectors of a loop do not wait for each other. A value
/// that is not finite counts as zero, which the maximum is at least (it
/// starts at a finite value that is not negative), so raising it needs no
/// condition of its own.
template <typename T, int W>
struct FiniteMaximum
{
    using O = Vec<T, W>;
    using V = typename O::V;
    V finite, zero, even, odd;
    explicit FiniteMaximum(T start) noexcept
        : finite(O::set1(std::numeric_limits<T>::max())), zero(O::set1(T(0))), even(O::set1(start)), odd(even)
    {
    }
    V counted(V v) const noexcept { return O::select(O::le(v, finite), v, zero); }
    void raise(V first, V second) noexcept
    {
        even = O::maxNum(even, counted(first));
        odd = O::maxNum(odd, counted(second));
    }
    void raise(V v) noexcept { even = O::maxNum(even, counted(v)); }
    T largest(T result) const noexcept { return largestLane<T, W>(O::maxNum(even, odd), result); }
};

// ---------------------------------------------------------------------------
// raiseToReadings
// ---------------------------------------------------------------------------

/// raiseToReadings() over the whole vectors of W positions from i on;
/// returns the first position left.
template <typename T, int W>
inline int raiseToReadingsAt(const T* s, const T* q, const T* h, const T* t, T* DSPARK_RESTRICT peak, int i,
                             int n) noexcept
{
    using O = Vec<T, W>;
    for (; i + W <= n; i += W)
        O::store(peak + i, O::maxOf(O::load(peak + i), gridReadingLanes<T, W>(s, q, h, t, i)));
    return i;
}

/// Raises peak[i] to the reading of position i (gridReading()) for i in
/// [0, n). A reading that is not a number is ignored.
template <typename T>
void raiseToReadings(const T* s, const T* q, const T* h, const T* t, T* DSPARK_RESTRICT peak, int n) noexcept
{
    int i = 0;
    if constexpr (kWide<T> > 1)
        i = raiseToReadingsAt<T, kWide<T>>(s, q, h, t, peak, i, n);
    if constexpr (kNarrow<T> > 1 && kNarrow<T> < kWide<T>)
        i = raiseToReadingsAt<T, kNarrow<T>>(s, q, h, t, peak, i, n);
    for (; i < n; ++i)
        peak[i] = std::max(peak[i], gridReading(s, q, h, t, i));
}

// ---------------------------------------------------------------------------
// largestFiniteReading
// ---------------------------------------------------------------------------

/// largestFiniteReading() over the whole vectors of W positions from i on:
/// raises result and returns the first position left. A reading stands at
/// most kCrestReach above the largest of its five grid points, so a vector
/// with no grid point that reaches the result it started from cannot raise
/// it and is not read: in a running maximum that is most of them.
template <typename T, int W>
inline int largestFiniteReadingAt(const T* s, const T* q, const T* h, const T* t, int i, int n, T& result) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    FiniteMaximum<T, W> best(result);
    const V limit = O::set1(result);
    for (; i + W <= n; i += W)
    {
        const typename O::M samples = O::either(reaches<T, W>(O::load(s + i), limit),
                                                reaches<T, W>(O::load(s + i + 1), limit));
        const typename O::M between = O::either(O::either(reaches<T, W>(O::load(q + i), limit),
                                                          reaches<T, W>(O::load(h + i), limit)),
                                                reaches<T, W>(O::load(t + i + 1), limit));
        if (O::any(O::either(samples, between)))
            best.raise(gridReadingLanes<T, W>(s, q, h, t, i));
    }
    result = best.largest(result);
    return i;
}

/// The largest finite reading (gridReading()) among the positions [0, n) and
/// start, which is finite and not negative.
template <typename T>
[[nodiscard]] T largestFiniteReading(const T* s, const T* q, const T* h, const T* t, int n, T start) noexcept
{
    constexpr T kFinite = std::numeric_limits<T>::max();
    T result = start;
    int i = 0;
    if constexpr (kWide<T> > 1)
        i = largestFiniteReadingAt<T, kWide<T>>(s, q, h, t, i, n, result);
    if constexpr (kNarrow<T> > 1 && kNarrow<T> < kWide<T>)
        i = largestFiniteReadingAt<T, kNarrow<T>>(s, q, h, t, i, n, result);
    for (; i < n; ++i)
    {
        const T r = gridReading(s, q, h, t, i);
        if (r > result && r <= kFinite) result = r;
    }
    return result;
}

// ---------------------------------------------------------------------------
// raiseToMagnitudes
// ---------------------------------------------------------------------------

/// raiseToMagnitudes() over the whole vectors of W values from i on; returns
/// the first position left.
template <typename T, int W>
inline int raiseToMagnitudesAt(T* DSPARK_RESTRICT dst, const T* src, int i, int n) noexcept
{
    using O = Vec<T, W>;
    for (; i + W <= n; i += W)
        O::store(dst + i, O::maxOf(O::load(dst + i), O::abs(O::load(src + i))));
    return i;
}

/// Raises dst[i] to |src[i]| for i in [0, n). A value that is not a number
/// is ignored.
template <typename T>
void raiseToMagnitudes(T* DSPARK_RESTRICT dst, const T* src, int n) noexcept
{
    int i = 0;
    if constexpr (kWide<T> > 1)
        i = raiseToMagnitudesAt<T, kWide<T>>(dst, src, i, n);
    if constexpr (kNarrow<T> > 1 && kNarrow<T> < kWide<T>)
        i = raiseToMagnitudesAt<T, kNarrow<T>>(dst, src, i, n);
    for (; i < n; ++i)
        dst[i] = std::max(dst[i], std::abs(src[i]));
}

// ---------------------------------------------------------------------------
// largestFiniteMagnitude
// ---------------------------------------------------------------------------

/// largestFiniteMagnitude() over the whole vectors of W values from i on:
/// raises result and returns the first position left.
template <typename T, int W>
inline int largestFiniteMagnitudeAt(const T* v, int i, int count, T& result) noexcept
{
    using O = Vec<T, W>;
    FiniteMaximum<T, W> best(result);
    for (; i + 2 * W <= count; i += 2 * W)
        best.raise(O::abs(O::load(v + i)), O::abs(O::load(v + i + W)));
    for (; i + W <= count; i += W)
        best.raise(O::abs(O::load(v + i)));
    result = best.largest(result);
    return i;
}

/// The largest finite magnitude among count values and start, which is
/// finite and not negative.
template <typename T>
[[nodiscard]] T largestFiniteMagnitude(const T* v, int count, T start) noexcept
{
    constexpr T kFinite = std::numeric_limits<T>::max();
    T result = start;
    int i = 0;
    if constexpr (kWide<T> > 1)
        i = largestFiniteMagnitudeAt<T, kWide<T>>(v, i, count, result);
    if constexpr (kNarrow<T> > 1 && kNarrow<T> < kWide<T>)
        i = largestFiniteMagnitudeAt<T, kNarrow<T>>(v, i, count, result);
    for (; i < count; ++i)
    {
        const T a = std::abs(v[i]);
        if (a > result && a <= kFinite) result = a;
    }
    return result;
}

// ---------------------------------------------------------------------------
// raiseToEighthReadings
// ---------------------------------------------------------------------------

/// The readings of the points [u, points) of an 8x grid at whole vectors of W
/// into r: for point u, the larger magnitude of its pair (a[u] and e[u + 1])
/// raised to the crests the pair completes (eighthCrests()). Never a value
/// that is not a number. Returns the first point left.
template <typename T, int W>
inline int eighthPointReadingsAt(const T* a, const T* e, T* DSPARK_RESTRICT r, int u, int points) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const V zero = O::set1(T(0));
    for (; u + W <= points; u += W)
    {
        const V m = O::maxOf(O::maxOf(zero, O::abs(O::load(a + u))), O::abs(O::load(e + u + 1)));
        O::store(r + u, eighthCrestsLanes<T, W>(a, e, u, m));
    }
    return u;
}

/// Raises peak[i] to the reading of sample i of an 8x grid (see
/// raiseToEighthSamples(): the four pairs of points 4 i .. 4 i + 3) for i in
/// [0, n). Magnitudes that are not numbers are ignored. On vectors, the
/// readings of the points are taken first, one each, and the four of a sample
/// joined after: the largest of the same values.
template <typename T>
void raiseToEighthReadings(const T* a, const T* e, T* DSPARK_RESTRICT peak, int n) noexcept
{
    int i = 0;
    if constexpr (kEighthWide<T> > 1)
    {
        constexpr int kSamples = 32; // per round: 128 points on the stack
        T r[4 * kSamples];
        while (i < n)
        {
            const int count = n - i < kSamples ? n - i : kSamples;
            const T* const a0 = a + 4 * i;
            const T* const e0 = e + 4 * i;
            int u = eighthPointReadingsAt<T, kEighthWide<T>>(a0, e0, r, 0, 4 * count);
            if constexpr (kEighthNarrow<T> > 1 && kEighthNarrow<T> < kEighthWide<T>)
                u = eighthPointReadingsAt<T, kEighthNarrow<T>>(a0, e0, r, u, 4 * count);
            for (; u < 4 * count; ++u)
                r[u] = eighthCrests(a0, e0, u, std::max(std::max(T(0), std::abs(a0[u])), std::abs(e0[u + 1])));
            for (int k = 0; k < count; ++k)
            {
                const T* const p = r + 4 * k;
                peak[i + k] = std::max(peak[i + k], std::max(std::max(p[0], p[1]), std::max(p[2], p[3])));
            }
            i += count;
        }
    }
    raiseToEighthSamples(a, e, peak, i, n);
}

// ---------------------------------------------------------------------------
// largestFiniteEighthCrest
// ---------------------------------------------------------------------------

/// largestFiniteEighthCrest() over the whole vectors of W points from u on:
/// raises result and returns the first point left. The two crests of a point
/// are centred on e[u] and a[u]; a vector where neither reaches the result
/// it started from is not refined (see largestFiniteReadingAt()).
template <typename T, int W>
inline int largestFiniteEighthCrestAt(const T* a, const T* e, int u, int points, T& result) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    FiniteMaximum<T, W> best(result);
    const V limit = O::set1(result);
    for (; u + W <= points; u += W)
    {
        if (O::any(O::either(reaches<T, W>(O::load(e + u), limit), reaches<T, W>(O::load(a + u), limit))))
            best.raise(eighthCrestsLanes<T, W>(a, e, u, best.zero));
    }
    result = best.largest(result);
    return u;
}

/// The largest finite crest (eighthCrests() above zero) among the points
/// [0, points) of an 8x grid and start, which is finite and not negative.
template <typename T>
[[nodiscard]] T largestFiniteEighthCrest(const T* a, const T* e, int points, T start) noexcept
{
    constexpr T kFinite = std::numeric_limits<T>::max();
    T result = start;
    int u = 0;
    if constexpr (kEighthWide<T> > 1)
        u = largestFiniteEighthCrestAt<T, kEighthWide<T>>(a, e, u, points, result);
    if constexpr (kEighthNarrow<T> > 1 && kEighthNarrow<T> < kEighthWide<T>)
        u = largestFiniteEighthCrestAt<T, kEighthNarrow<T>>(a, e, u, points, result);
    for (; u < points; ++u)
    {
        const T m = eighthCrests(a, e, u, T(0));
        if (m > result && m <= kFinite) result = m;
    }
    return result;
}
