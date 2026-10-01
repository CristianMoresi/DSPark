// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TubePreamp.h
 * @brief Tube preamp: Koren 12AX7 stages, WDF tone circuit, supply sag.
 *
 * Circuit-level preamp modelling, not a waveshaper:
 *
 * - **Triode stages** (1 or 2) use a tabulated implicit plate-current
 *   solution with bicubic interpolation; an analytic Newton solve handles
 *   voltages outside the table. The tube follows Koren's SPICE model (Koren
 *   1996; published 12AX7 parameters MU=100, EX=1.4, KG1=1060, KP=600,
 *   KVB=300) in a classic common-cathode stage: 300 V supply, 100 k ohm plate
 *   load, 1.5 k ohm cathode resistor with its 22 uF bypass capacitor
 *   integrated trapezoidally - the capacitor state is folded into the
 *   load-line equation, so the table retains the capacitor feedback. Grid
 *   conduction is approximated by a soft clamp toward +0.7 V (full blocking
 *   distortion needs the input-coupling state and is left for a later pass).
 * - **Supply sag**: the effective B+ droops with smoothed plate current
 *   (one-pole, ~70 ms) times a sag resistance - drive into the stage and
 *   the headroom breathes back, the classic touch response.
 * - **Tone stack**: the full Fender '59 Bassman FMV treble/bass/middle
 *   network, solved exactly as a 12-port WDF R-type adaptor
 *   (wdf::ToneStackFMV - verified sample-exact against the symbolic
 *   transfer function of Yeh & Smith, DAFx-06). The stack sits between the
 *   stages and uses a fixed 38 k ohm source-impedance approximation; its
 *   load is not fed back into the triode solve. Controls interact
 *   non-orthogonally - that is the circuit, not a bug.
 * - Output level: the circuit's program response at the reference tone
 *   setting is measured once in prepare() (settled channel, pink-weighted
 *   multitone). A 3-section EQ designed from that measurement flattens the
 *   FMV stack's fixed ~10 dB mid-scoop envelope - neutral knobs sound
 *   neutral, the tone controls act relative to flat, and the triode's
 *   nonlinear harmonic generation is retained. Loudness divides out the program gain
 *   MEASURED AT EACH DRIVE (prepare-time sweep LUT, so the link tracks the
 *   circuit's real compression) plus a +0.25 dB/dB residual slope: backing
 *   off is audible, pushing raises density at a gently rising level - the
 *   same contract as TapeMachine/TransformerModel. Use setOutput() for
 *   static trim; gain changes are ramped (~30 ms) so drags stay click-free.
 *
 * The nonlinear core runs at 2x oversampling by DEFAULT, and the factor is
 * configurable via setOversampling(int) (setup thread only, since it
 * reallocates and re-calibrates like prepare()): 1 = OFF (no internal
 * resampling or ADAA, zero added latency, but the triode/grid nonlinearity then
 * aliases in-band unless you oversample the surrounding chain yourself), 2 =
 * default (group-delay latency reported by getLatency()/getLatencySamples()),
 * 4/8/16 = progressively lower alias floor with higher processing cost.
 * At factors >= 2, each stage antialiases the grid clamp and the implicit
 * plate-current interpolant with antiderivative antialiasing: the grid uses
 * the second-order divided difference (Bilbao et al., IEEE SPL 2017), and
 * the current uses the exact first integral of its bicubic surface.
 * Supply and cathode history are held fixed across each current
 * integral; the averaged current advances the cathode and sag states.
 * Out-of-table voltages use the analytic midpoint solve. This is a discrete
 * circuit approximation, not exact integration of all coupled state motion.
 *
 * A bounded linear-phase Core FIR compensates the ADAA small-signal droop:
 * the minimax inverse of (1-u)*(1-4u/3)^stages, u = sin(w/2)^2, includes
 * the one-stage path's extra half-sample alignment filter. It avoids the
 * unit-circle poles of an exact inverse. FIR residual error is <0.05 dB
 * over the entire base-rate Nyquist band at factors >= 2. Stage-count
 * padding and whole-sample padding keep wet and dry
 * aligned. Total latency for 1/2/4/8/16x is 0/73/98/113/121 base samples,
 * independent of setStages(). At 1x the original point circuit is retained:
 * inverting an ADAA null at the audio Nyquist would require unbounded gain.
 * Use 1x only when the surrounding host chain supplies a suitable high rate.
 * getLatency() always reflects the ACTIVE
 * factor (0 at 1x) so hosts get correct PDC. Measured at 48 kHz over tones
 * from 1 to 20 kHz in 1 kHz steps, plus 12.75/15.25/19.25 kHz, at -6 and
 * -18 dBFS: each tone is moved to the nearest odd bin of a 16384-point FFT.
 * After settling, coherent rectangular DFT measurements exclude only DC
 * and exact physical harmonic bins. In particular, adjacent folded lines
 * are retained. The worst component below 20 kHz, relative to the tone, is:
 *
 *   stages, drive     2x (default)   4x          8x
 *   1, -12 dB        -102.9         -137.9       -135.6
 *   1, 0 dB           -77.3         -118.4       -115.9
 *   1, +12 dB         -56.4          -89.3       -102.3
 *   1, +24 dB         -42.5          -56.2        -79.3
 *   1, +36 dB         -22.5          -38.9        -48.5
 *   2, -12 dB         -96.2         -129.7       -123.0
 *   2, 0 dB           -71.2         -106.2       -104.7
 *   2, +12 dB         -53.6          -78.0        -95.0
 *   2, +24 dB         -36.2          -50.4        -70.4
 *   2, +36 dB         -19.2          -34.2        -44.9
 *
 * A high-gain triode turns a loud top-octave sine into a nearly square wave
 * whose harmonics fall slowly. The worst frequency varies with the setting;
 * testing only 15 kHz misses worse components at other frequencies.
 * High-drive aliasing remains a limitation even at the higher factors:
 * choose a factor from measurements at the intended drive and sample rate.
 *
 * The current/primitive table is 235016 bytes shared by all channels in an
 * instance. Preparation builds the table and calibrates the circuit;
 * processing and stage changes allocate no memory. The antialiased circuit
 * has a different waveform from the former point-sampled solver. The
 * small-signal transfer and THD signature are verified in the
 * suite: single-stage distortion is 2nd-harmonic dominant (asymmetric
 * triode), DC operating point matches an independent high-precision solve
 * of the same circuit equations (the check SPICE would perform).
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored; changes are published with
 * a release store and consumed at the next block). prepare() is setup-thread
 * only (allocates and runs the reference calibration; invalid specs are
 * ignored and an unprepared instance passes audio through). reset() belongs
 * to the stream owner. getState()/setState() are setup/UI threads.
 * getSupplyVoltage() is a metering-style read. The dry/wet mix ramps at a
 * rate of at most full scale per 20 ms, whatever the block size (a per-block
 * ramp landed in 0.7 ms with 32-sample blocks and clicked). Channels beyond the prepared count pass through
 * untouched.
 *
 * Dependencies: Core/WDF.h, Core/Oversampling.h, Core/Biquad.h, Core/FIRFilter.h,
 * Core/AudioSpec.h, Core/AudioBuffer.h, Core/DspMath.h, Core/DenormalGuard.h,
 * Core/SmoothedValue.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FIRFilter.h"
#include "../Core/Oversampling.h"
#include "../Core/SmoothedValue.h"
#include "../Core/StateBlob.h"
#include "../Core/WDF.h"
#include "../Core/detail/LogCosh.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

namespace dspark {

/// @cond DSPARK_INTERNAL
namespace detail {
    struct TubePreampCompensation
    {
        std::array<std::vector<double>, 2> taps;
        int latency;

        explicit TubePreampCompensation(int factor)
        {
            // 1x retains the explicitly unresampled point circuit. Inverting
            // the ADAA null at that same Nyquist would be unbounded; the
            // compensated ADAA path requires the selected factor to be >= 2.
            if (factor == 1)
            {
                latency = 0;
                return;
            }
            // Bounded minimax inverse of (1-u)*(1-4u/3)^stages.
            // Banks store center-out symmetric coefficients. Unity DC is
            // constrained; passband relative error is <0.05 dB. The maximum
            // correction over all internal frequencies is <=12.001/36.001
            // at 2x (one/two stages), <=4.001 at higher factors. No pole or
            // gain singularity is introduced outside the audio passband.
            // The extra half-sample average in the one-stage path makes
            // its phase equal to three internal samples after integer pad.
            const int order = factor == 2 ? 14 : (factor == 4 ? 4 : 2);
            latency = (order + 3 + factor - 1) / factor;
            const int pad = latency * factor - (order + 3);
            std::array<std::vector<double>, 2> banks;
            switch (factor)
            {
            case 2:
                banks[0] = {
                    1.9235629963970406, 0.44648231230742969, -1.5580934002318299,
                    0.27439449777428138, 0.95396435999096119, -0.27851755188567112,
                    -1.547981227472137, 2.6229978502006257, -2.1323567229872782,
                    0.81377578397761241, 0.21025863734304612, -0.50457523628053735,
                    0.34120770592379895, -0.12455765609373881, 0.021219149234916305
                };
                banks[1] = {
                    4.1949350794813807, 0.44755626226910095, -3.5513262161664114,
                    -0.059683685251895137, 3.4543837933883861, -1.0019140998285381,
                    -3.4140408147555226, 3.1993666390142548, 1.9031788564870777,
                    -6.6025711322194587, 7.1102262654542958, -4.4409762973601534,
                    1.6511735090435748, -0.29177768662331721, -0.0010629331920828475
                };
                break;
            case 4:
                banks[0] = {
                    1.6916530526685123, 0.22926948453401663, -1.0131494471944595,
                    0.53918417640406124, -0.10113074007787473
                };
                banks[1] = {
                    1.2261507380985748, 0.83046321660400191, -1.3670690928381859,
                    0.41953678339599815, 0.0039937237888984208
                };
                break;
            case 8:
                banks[0] = {
                    2.431527180594093, -0.75, 0.034236409702953496
                };
                banks[1] = {
                    2.6249087461926157, -0.75, -0.062454373096307858
                };
                break;
            case 16:
                banks[0] = {
                    2.4201719089187108, -0.75, 0.039914045540644594
                };
                banks[1] = {
                    2.5931625075732305, -0.75, -0.046581253786615262
                };
                break;
            default: break;
            }
            for (int stages = 1; stages <= 2; ++stages)
            {
                auto& out = taps[static_cast<size_t>(stages - 1)];
                const auto& bank = banks[static_cast<size_t>(stages - 1)];
                out.assign(static_cast<size_t>(2 * order + 1 + pad + (stages == 1 ? 2 : 0)), 0.0);
                for (int j = 0; j <= 2 * order; ++j)
                {
                    const double v = bank[static_cast<size_t>(std::abs(j - order))];
                    const auto index = static_cast<size_t>(j + pad + (stages == 1 ? 1 : 0));
                    out[index] += stages == 1 ? 0.5 * v : v;
                    if (stages == 1) out[index + 1] += 0.5 * v;
                }
            }
        }
    };

    struct TubePreampGridClamp
    {
        static double value(double v) noexcept
        {
            return v > 0.0 ? 0.7 * std::tanh(v / 0.7) : v;
        }

        static double primitive(double v) noexcept
        {
            return v <= 0.0 ? 0.5 * v * v : 0.49 * logCosh(v / 0.7);
        }

        static double primitive2(double v) noexcept
        {
            return v <= 0.0 ? v * v * v / 6.0 : 0.343 * integralLogCosh(v / 0.7);
        }

        // Twice the symmetric second divided difference (Bilbao et al.,
        // IEEE SPL 2017). Sorting keeps the outer denominator well separated.
        static double divided(double a, double b, double c, double ga, double gb, double gc) noexcept
        {
            if (a > b) { std::swap(a, b); std::swap(ga, gb); }
            if (b > c) { std::swap(b, c); std::swap(gb, gc); }
            if (a > b) { std::swap(a, b); std::swap(ga, gb); }
            if (c <= 0.0) return (a + b + c) / 3.0;
            const double center = (a + b + c) / 3.0;
            const double span = c - a;
            if (span < 0.001 && a < 0.0)
            {
                // Across zero, the cubic correction exists only on the
                // positive branch. Integrate it against the exact simplex
                // density. Three Gauss nodes integrate this quartic exactly;
                // the omitted fifth-power correction is below 6e-16 V.
                constexpr double t[] = {0.1127016653792583, 0.5, 0.8872983346207417};
                constexpr double w[] = {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
                double correction = 0.0;
                for (int half = 0; half < 2; ++half)
                {
                    const double lo = half == 0 ? a : b;
                    const double hi = half == 0 ? b : c;
                    const double start = std::max(lo, 0.0);
                    if (hi <= start) continue;
                    for (int k = 0; k < 3; ++k)
                    {
                        const double x = start + t[k] * (hi - start);
                        const double density = 2.0 * (half == 0 ? x - a : c - x)
                            / ((hi - lo) * span);
                        correction -= (hi - start) * w[k] * density * x * x * x / 1.47;
                    }
                }
                return center + correction;
            }
            if (span < 0.001 && a >= 0.0)
            {
                // Moments of the uniform simplex give a stable local limit,
                // including curvature instead of flattening tiny variations.
                const double t = std::tanh(center / 0.7), q = 1.0 - t * t;
                const double da = a - center, db = b - center, dc = c - center;
                const double p2 = da * da + db * db + dc * dc;
                const double p3 = da * da * da + db * db * db + dc * dc * dc;
                const double f2 = -2.0 * t * q / 0.7;
                const double f3 = -2.0 * q * (1.0 - 3.0 * t * t) / 0.49;
                const double f4 = 8.0 * t * q * (2.0 - 3.0 * t * t) / 0.343;
                return 0.7 * t + f2 * p2 / 24.0 + f3 * p3 / 180.0
                    + f4 * p2 * p2 / 1440.0;
            }
            const double tolerance = 1e-6 * std::max({1.0, std::abs(a), std::abs(c)});
            if (b - a < tolerance)
            {
                const double mid = 0.5 * (a + b), d = c - mid;
                return 2.0 * ((gc - primitive2(mid)) / d - primitive(mid)) / d;
            }
            if (c - b < tolerance)
            {
                const double mid = 0.5 * (b + c), d = mid - a;
                return 2.0 * (primitive(mid) - (primitive2(mid) - ga) / d) / d;
            }
            return 2.0 * ((gc - gb) / (c - b) - (gb - ga) / (b - a)) / (c - a);
        }
        static double average(double a, double b, double fa, double fb) noexcept
        {
            const double mid = 0.5 * (a + b);
            if ((a <= 0.0 && b <= 0.0)
                || std::abs(b - a) < 1e-6 * std::max({1.0, std::abs(a), std::abs(b)}))
                return value(mid);
            return (fb - fa) / (b - a);
        }
    };

    struct TubePreampCurrentTable
    {
        static constexpr double kMu = 100.0, kEx = 1.4, kKg1 = 1060.0;
        static constexpr double kKp = 600.0, kKvb = 300.0;
        static constexpr double kRL = 100e3, kRk = 1.5e3, kCk = 22e-6;

        /** @brief Koren plate current and its partial derivatives. */
        static void koren(double vpk, double vgk, double& ip,
                          double& dIpdVpk, double& dIpdVgk) noexcept
        {
            vpk = std::max(vpk, 0.0);
            const double s = std::sqrt(kKvb + vpk * vpk);
            const double u = kKp * (1.0 / kMu + vgk / s);

            double sp = 0.0, sig = 0.0;                  // softplus(u), logistic(u)
            if (u > 30.0)       { sp = u; sig = 1.0; }
            else if (u < -30.0) { sp = std::exp(u); sig = sp; }
            else
            {
                const double eu = std::exp(u);
                sp = std::log1p(eu);
                sig = eu / (1.0 + eu);
            }

            const double e1 = (vpk / kKp) * sp;
            if (e1 <= 0.0)
            {
                ip = 0.0;
                dIpdVpk = 0.0;
                dIpdVgk = 0.0;
                return;
            }
            const double e1ex1 = std::pow(e1, kEx - 1.0);
            ip = 2.0 * e1ex1 * e1 / kKg1;
            const double dIpdE1 = 2.0 * kEx * e1ex1 / kKg1;

            const double dUdVpk = -kKp * vgk * vpk / (s * s * s);
            const double dE1dVpk = sp / kKp + (vpk / kKp) * sig * dUdVpk;
            const double dE1dVgk = vpk * sig / s;
            dIpdVpk = dIpdE1 * dE1dVpk;
            dIpdVgk = dIpdE1 * dE1dVgk;
        }

        // With S = B+ - A and G = Vgrid - A, the trapezoidal cathode
        // equation Vk = A + B*Ip leaves a TWO-dimensional implicit load line:
        // Ip = Koren(S - (RL+B)*Ip, G - B*Ip). B is fixed by prepare's rate.
        // Use R = G/S to align the cutoff knee across supply voltages. This
        // needs 235016 bytes including its primitive, with <5e-10 A error
        // in the independent load-line check.
        // Tabulate the root and implicit derivatives without quantizing the
        // capacitor or dropping its feedback. Bicubic Hermite interpolation
        // keeps current and both first derivatives continuous at cell edges.
        // Every channel and both calibration passes share this instance.
        struct Node
        {
            double y, ds, dg, dsg;
            double integral = 0.0, integralS = 0.0;
        };
        static constexpr int kNS = 18, kNG = 272;
        std::array<Node, kNS * kNG> nodes;
        double kb;

        explicit TubePreampCurrentTable(double rate)
            : kb((0.5 / (rate * kCk)) / (1.0 + 0.5 / (rate * kCk * kRk)))
        {
            for (int si = 0; si < kNS; ++si)
                for (int gi = 0; gi < kNG; ++gi)
                {
                    const double s = si <= 6 ? 80.0 + 8.0 * si
                                             : 128.0 + 16.0 * (si - 6);
                    // Tail cells must still resolve the exponential enough
                    // to keep the cubic nonnegative; integrating a cubic
                    // and then clipping its point evaluation would disagree.
                    const double r = gi <= 16 ? -0.08 + gi * 0.0025
                        : (gi <= 31 ? -0.04 + (gi - 16) * 0.001
                                    : -0.025 + (gi - 31) / 6400.0);
                    Node n = solve(s, s * r);
                    n.ds += r * n.dg;             // dI/dS with R held fixed
                    n.dg *= s;                   // dI/dR with S held fixed
                    // Differentiate the implicit dI/dR, not sampled currents;
                    // 0.001 V is much smaller than the 8/16 V supply cells.
                    n.dsg = ((s + 0.001) * solve(s + 0.001, (s + 0.001) * r).dg
                           - (s - 0.001) * solve(s - 0.001, (s - 0.001) * r).dg) / 0.002;
                    nodes[static_cast<size_t>(si * kNG + gi)] = n;
                }
            // Hermite interpolation is linear in the node data. Accumulate
            // the exact R-integral and its S derivative along each row;
            // the same S interpolation then integrates the entire surface.
            for (int si = 0; si < kNS; ++si)
                for (int gi = 1; gi < kNG; ++gi)
                {
                    const auto& a = nodes[static_cast<size_t>(si * kNG + gi - 1)];
                    auto& b = nodes[static_cast<size_t>(si * kNG + gi)];
                    const double h = gi <= 16 ? 0.0025 : (gi <= 31 ? 0.001 : 1.0 / 6400.0);
                    b.integral = a.integral + h * (0.5 * (a.y + b.y) + h * (a.dg - b.dg) / 12.0);
                    b.integralS = a.integralS + h * (0.5 * (a.ds + b.ds) + h * (a.dsg - b.dsg) / 12.0);
                }
        }

        [[nodiscard]] Node solve(double supply, double grid) const noexcept
        {
            // Positive current is bounded by the plate load. Safeguard Newton
            // with that bracket: setup converges faster than pure bisection
            // without accepting a root on an unphysical branch.
            double low = 0.0, high = supply / (kRL + kb);
            double y = 0.5 * high, p = 0.0, dp = 0.0, dg = 0.0;
            for (int k = 0; k < 60; ++k)
            {
                koren(supply - (kRL + kb) * y, grid - kb * y, p, dp, dg);
                const double f = y - p;
                // An absolute 1e-15 A stop leaves arbitrary residuals in
                // cutoff nodes whose true current is much smaller. Those
                // inconsistent values/slopes can make their cubic negative.
                if (std::abs(f) <= std::max(1e-30, 1e-13 * std::max(y, p))) break;
                if (f > 0.0) high = y;
                else low = y;
                const double slope = 1.0 + (kRL + kb) * dp + kb * dg;
                const double next = y - f / slope;
                y = next >= low && next <= high ? next : 0.5 * (low + high);
            }
            koren(supply - (kRL + kb) * y, grid - kb * y, p, dp, dg);
            const double den = 1.0 + (kRL + kb) * dp + kb * dg;
            return {y, dp / den, dg / den, 0.0};
        }

        static double cubic(double a, double b, double da, double db, double t) noexcept
        {
            const double diff = b - a;
            return (((da + db - 2.0 * diff) * t + (3.0 * diff - 2.0 * da - db)) * t + da) * t + a;
        }

        [[nodiscard]] bool covers(double s, double g) const noexcept
        {
            return s >= 80.0 && s < 304.0 && std::isfinite(g) && g < 1.0;
        }

        struct Cell
        {
            int si, gi;
            double st, gt, step, rstep;
        };

        static Cell locate(double s, double g) noexcept
        {
            const double step = s < 128.0 ? 8.0 : 16.0;
            const double r = g / s;
            const double rstep = r < -0.04 ? 0.0025 : (r < -0.025 ? 0.001 : 1.0 / 6400.0);
            const double sp = s < 128.0 ? (s - 80.0) / 8.0 : 6.0 + (s - 128.0) / 16.0;
            const double gp = r < -0.04 ? (r + 0.08) * 400.0
                : (r < -0.025 ? 16.0 + (r + 0.04) * 1000.0
                              : 31.0 + (r + 0.025) * 6400.0);
            // Adding the grid offset can round a value just below the upper
            // edge onto that edge. Use the final CELL, with t = 1, there.
            const int si = std::min(static_cast<int>(sp), kNS - 2);
            const int gi = std::min(static_cast<int>(gp), kNG - 2);
            return {si, gi, sp - si, gp - gi, step, rstep};
        }

        // Coefficients in ascending powers of the normalized R coordinate.
        [[nodiscard]] std::array<double, 4> polynomial(const Cell& q) const noexcept
        {
            const Node& a = nodes[static_cast<size_t>(q.si * kNG + q.gi)];
            const Node& b = nodes[static_cast<size_t>((q.si + 1) * kNG + q.gi)];
            const Node& c = nodes[static_cast<size_t>(q.si * kNG + q.gi + 1)];
            const Node& d = nodes[static_cast<size_t>((q.si + 1) * kNG + q.gi + 1)];
            const double p0 = cubic(a.y, b.y, a.ds * q.step, b.ds * q.step, q.st);
            const double p1 = cubic(c.y, d.y, c.ds * q.step, d.ds * q.step, q.st);
            const double m0 = q.rstep * cubic(a.dg, b.dg, a.dsg * q.step, b.dsg * q.step, q.st);
            const double m1 = q.rstep * cubic(c.dg, d.dg, c.dsg * q.step, d.dsg * q.step, q.st);
            const double diff = p1 - p0;
            return {p0, m0, 3.0 * diff - 2.0 * m0 - m1, m0 + m1 - 2.0 * diff};
        }

        [[nodiscard]] double eval(double s, double g) const noexcept
        {
            // Below this cutoff the unloaded Koren current is < 1e-27 A.
            if (g <= -0.08 * s) return 0.0;
            const auto q = locate(s, g);
            const auto p = polynomial(q);
            return std::max(0.0, ((p[3] * q.gt + p[2]) * q.gt + p[1]) * q.gt + p[0]);
        }

        /** Integral along G at fixed S, with zero in deep cutoff. */
        [[nodiscard]] double primitive(double s, double g) const noexcept
        {
            if (g <= -0.08 * s) return 0.0;
            return primitive(s, locate(s, g));
        }

        [[nodiscard]] double primitive(double s, const Cell& q) const noexcept
        {
            const auto p = polynomial(q);
            const auto& a = nodes[static_cast<size_t>(q.si * kNG + q.gi)];
            const auto& b = nodes[static_cast<size_t>((q.si + 1) * kNG + q.gi)];
            const double base = cubic(a.integral, b.integral,
                a.integralS * q.step, b.integralS * q.step, q.st);
            const double t = q.gt;
            const double part = (((p[3] * 0.25 * t + p[2] / 3.0) * t + p[1] * 0.5) * t + p[0]) * t;
            return s * (base + q.rstep * part);
        }

        /** Exact cell average; a local expression avoids primitive cancellation. */
        [[nodiscard]] double average(double s, double g0, double g1) const noexcept
        {
            if (std::max(g0, g1) <= -0.08 * s) return 0.0;
            if (std::min(g0, g1) > -0.08 * s)
            {
                const auto a = locate(s, g0), b = locate(s, g1);
                if (a.gi == b.gi)
                {
                    const auto p = polynomial(a);
                    const double m = 0.5 * (a.gt + b.gt), d = b.gt - a.gt;
                    return std::max(0.0, ((p[3] * m + p[2]) * m + p[1]) * m + p[0]
                        + (3.0 * p[3] * m + p[2]) * d * d / 12.0);
                }
                if (std::abs(g1 - g0) < 1e-6) return eval(s, 0.5 * (g0 + g1));
                return std::max(0.0, (primitive(s, b) - primitive(s, a)) / (g1 - g0));
            }
            if (std::abs(g1 - g0) < 1e-6) return eval(s, 0.5 * (g0 + g1));
            return std::max(0.0, (primitive(s, g1) - primitive(s, g0)) / (g1 - g0));
        }
    };
} // namespace detail
/// @endcond

/**
 * @class TubePreamp
 * @brief One/two 12AX7 stages with sag and a WDF tone circuit.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TubePreamp
{
public:
    // -- Lifecycle ---------------------------------------------------------------

    /** @brief Allocates the chain (one circuit instance per channel) and runs
     *  the reference calibration. Invalid specs (non-positive or non-finite
     *  rate, block size or channel count) are ignored: the previous state is
     *  kept and an unprepared instance stays pass-through. */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        spec_ = spec;
        sampleRate_ = spec.sampleRate;
        mixMaxStep_ = static_cast<T>(1.0 / std::max(1.0, sampleRate_ * 0.02));
        // Internal processing rate = active oversampling factor x base rate
        // (the factor is configurable; 1 = off, no resampling).
        fs2_ = static_cast<double>(osFactor_) * sampleRate_;
        numChannels_ = spec.numChannels;
        maxBlock_ = std::max(spec.maxBlockSize, 1);
        driveLogSmoother_.prepare(fs2_, 30.0);
        outputLogSmoother_.prepare(fs2_, 30.0);
        for (auto& ramp : gainRamps_)
            ramp.resize(static_cast<size_t>(maxBlock_) * static_cast<size_t>(osFactor_));
        compensationScratch_.resize(static_cast<size_t>(maxBlock_) * static_cast<size_t>(osFactor_));

        if (osFactor_ > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(
                osFactor_, Oversampling<T>::Quality::High);
            oversampler_->prepare(spec);
        }
        else
        {
            oversampler_.reset();
        }

        loadTable_ = std::make_unique<LoadTable>(fs2_);
        compensation_ = std::make_unique<detail::TubePreampCompensation>(osFactor_);
        channels_.clear();
        channels_.resize(static_cast<size_t>(numChannels_));
        for (auto& ch : channels_)
            ch = std::make_unique<ChannelState>(fs2_, loadTable_.get(), compensation_.get());

        latency_ = (oversampler_ ? oversampler_->getLatency() : 0) + compensation_->latency;
        drySize_ = 1;
        while (drySize_ < latency_ + maxBlock_ + 1) drySize_ <<= 1;
        dryRing_.assign(static_cast<size_t>(numChannels_),
                        std::vector<T>(static_cast<size_t>(drySize_), T(0)));
        dryPos_ = 0;

        calibrateReference();

        prepared_.store(true, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
        reset();
    }

    /** @brief Re-settles every stage at its DC operating point. RT-safe. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        const double sagR = static_cast<double>(sag_.load(std::memory_order_relaxed)) * 40e3;
        const int stages = stages_.load(std::memory_order_relaxed);
        for (auto& ch : channels_)
            ch->reset(sagR, stages);
        for (auto& d : dryRing_)
            std::fill(d.begin(), d.end(), T(0));
        dryPos_ = 0;
        if (oversampler_) oversampler_->reset();
        // Seed the anti-zipper ramps at their targets: no fade-in on start.
        gainsInitialized_ = false;
        currentMix_ = mix_.load(std::memory_order_relaxed);
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Input drive in dB [-12, +36]; level-compensated. Non-finite
     *  values are ignored. */
    void setDrive(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        driveDb_.store(std::clamp(db, T(-12), T(36)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Treble control of the FMV stack [0, 1]. Non-finite values are ignored. */
    void setTreble(T treble) noexcept
    {
        if (!std::isfinite(treble)) return;
        treble_.store(std::clamp(treble, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Bass control of the FMV stack [0, 1] (log-taper, like the original).
     *  Non-finite values are ignored. */
    void setBass(T bass) noexcept
    {
        if (!std::isfinite(bass)) return;
        bass_.store(std::clamp(bass, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Middle control of the FMV stack [0, 1]. Non-finite values are ignored. */
    void setMiddle(T middle) noexcept
    {
        if (!std::isfinite(middle)) return;
        middle_.store(std::clamp(middle, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Supply sag depth [0, 1] (0 = stiff supply). Non-finite values are ignored. */
    void setSag(T sag) noexcept
    {
        if (!std::isfinite(sag)) return;
        sag_.store(std::clamp(sag, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Number of triode stages (1 = clean/edge, 2 = high gain).
     *  RT-safe; the prepared latency does not change. */
    void setStages(int stages) noexcept
    {
        stages_.store(std::clamp(stages, 1, 2), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Configures internal oversampling of the nonlinear circuit
     *  (visible, tunable and switchable off). SETUP THREAD ONLY - it reallocates
     *  the polyphase filters and re-runs the reference calibration exactly like
     *  prepare(); never call it concurrently with processBlock().
     *
     * @param factor Power-of-two multiplier in {1,2,4,8,16}. 1 = OFF (no
     *  internal resampling, zero added latency; the triode/grid nonlinearity
     *  then aliases in-band unless the surrounding chain is oversampled). 2 is
     *  the default. Higher factors lower the alias floor at ~linearly higher
     *  CPU. Invalid or non-power-of-two values are ignored. getLatency()
     *  reflects the new factor after this call (0 at 1x).
     */
    void setOversampling(int factor)
    {
        if (factor < 1 || factor > 16 || (factor & (factor - 1)) != 0) return;
        if (factor == osFactor_) return;
        osFactor_ = factor;
        // Rebuild the whole chain at the new internal rate if already prepared
        // (fs2_, oversampler, per-channel circuits and calibration all depend
        // on the factor). Same setup-thread cost as prepare().
        if (prepared_.load(std::memory_order_relaxed))
            prepare(spec_);
    }

    /** @brief Active oversampling factor (1 = off, 2 = default). */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return osFactor_; }

    /** @brief Static output trim in dB [-24, +12]. Non-finite values are ignored. */
    void setOutput(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        outputDb_.store(std::clamp(db, T(-24), T(12)), std::memory_order_relaxed);
    }

    /** @brief Dry/wet mix [0, 1]; dry is latency-compensated and the mix is
     *  ramped over at least 20 ms. Non-finite values are ignored. */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getDrive() const noexcept { return driveDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTreble() const noexcept { return treble_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBass() const noexcept { return bass_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMiddle() const noexcept { return middle_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSag() const noexcept { return sag_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getStages() const noexcept { return stages_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getOutput() const noexcept { return outputDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Latency in samples the active oversampler adds (0 at 1x = off);
     *  reflects the current factor, so hosts can compensate (PDC). */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

    /** @brief Compatibility alias of getLatency(), in prepared-rate samples. */
    [[nodiscard]] int getLatencySamples() const noexcept { return getLatency(); }

    /** @brief Effective B+ supply voltage of channel 0 (sag meter readout). */
    [[nodiscard]] T getSupplyVoltage() const noexcept
    {
        return supplyNow_.load(std::memory_order_relaxed);
    }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TUBE"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("drive", static_cast<float>(driveDb_.load(std::memory_order_relaxed)));
        w.write("treble", static_cast<float>(treble_.load(std::memory_order_relaxed)));
        w.write("bass", static_cast<float>(bass_.load(std::memory_order_relaxed)));
        w.write("middle", static_cast<float>(middle_.load(std::memory_order_relaxed)));
        w.write("sag", static_cast<float>(sag_.load(std::memory_order_relaxed)));
        w.write("stages", stages_.load(std::memory_order_relaxed));
        w.write("output", static_cast<float>(outputDb_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("oversampling", osFactor_);
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TUBE")) return false;
        setDrive(static_cast<T>(r.read("drive", 0.0f)));
        setTreble(static_cast<T>(r.read("treble", 0.5f)));
        setBass(static_cast<T>(r.read("bass", 0.5f)));
        setMiddle(static_cast<T>(r.read("middle", 0.5f)));
        setSag(static_cast<T>(r.read("sag", 0.3f)));
        setStages(r.read("stages", 1));
        setOutput(static_cast<T>(r.read("output", 0.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        // Default 2 = the historical fixed factor, so blobs written before
        // it was configurable restore the 2x behaviour they were captured with.
        setOversampling(r.read("oversampling", 2));
        return true;
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Processes a block in-place. Pass-through until prepare() succeeds. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        if (nCh == 0 || nS == 0) return;

        // Front-door non-finite guard: a single NaN/Inf input sample would
        // poison the recursive triode Newton-Raphson / WDF tone stack / supply
        // sag / output DC-blocker / flatten-EQ state PERMANENTLY (only reset()
        // clears it, not clean input). Replace bad samples with silence before
        // they reach any state, so a transient upstream glitch cannot corrupt
        // the channel for the rest of the stream.
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                if (!std::isfinite(d[i])) d[i] = T(0);
        }

        // Acquire pairs with the setters' release stores so the recompute
        // always sees the values published before the flag.
        if (dirty_.load(std::memory_order_relaxed)
            && dirty_.exchange(false, std::memory_order_acquire))
            recompute();

        // Rate-limited mix ramp (moveTowards, exact landing; settled it
        // reduces to the constant, bit-identically).
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;
        const double outGain = mScale_
            * std::pow(10.0, static_cast<double>(outputDb_.load(std::memory_order_relaxed)) / 20.0);

        // Dry snapshot.
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* in = buffer.getChannel(ch);
            auto& dry = dryRing_[static_cast<size_t>(ch)];
            int dp = dryPos_;
            for (int i = 0; i < nS; ++i)
            {
                dry[static_cast<size_t>(dp)] = in[i];
                dp = (dp + 1) & (drySize_ - 1);
            }
        }

        // Nonlinear circuit at the active oversampling factor (1x = process the
        // base-rate buffer in place, no resampling).
        {
            const bool osOn = (oversampler_ != nullptr);
            auto osView = osOn ? oversampler_->upsample(buffer) : buffer;
            const int osN = osView.getNumSamples();

            // Smooth the compensation pair in logarithmic gain at the
            // internal sample clock. Interpolating between block endpoints
            // changes the trajectory when a host changes its block size.
            // Core's sample-exact one-pole retains the 30 ms time constant.
            const double driveLog = std::log(hScale_);
            const double outputLog = std::log(outGain);
            if (!gainsInitialized_)
            {
                driveLogSmoother_.reset(driveLog);
                outputLogSmoother_.reset(outputLog);
                gainsInitialized_ = true;
            }
            driveLogSmoother_.setTargetValue(driveLog);
            outputLogSmoother_.setTargetValue(outputLog);
            const bool driveRamping = driveLogSmoother_.isSmoothing();
            const bool outputRamping = outputLogSmoother_.isSmoothing();
            if (driveRamping)
            {
                driveLogSmoother_.processBlock(
                    std::span<double>(gainRamps_[0].data(), static_cast<size_t>(osN)));
                for (int i = 0; i < osN; ++i)
                {
                    auto& value = gainRamps_[0][static_cast<size_t>(i)];
                    value = value == driveLog ? hScale_ : std::exp(value);
                }
            }
            if (outputRamping)
            {
                outputLogSmoother_.processBlock(
                    std::span<double>(gainRamps_[1].data(), static_cast<size_t>(osN)));
                for (int i = 0; i < osN; ++i)
                {
                    auto& value = gainRamps_[1][static_cast<size_t>(i)];
                    value = value == outputLog ? outGain : std::exp(value);
                }
            }

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = osView.getChannel(ch);
                auto& state = *channels_[static_cast<size_t>(ch)];
                for (int i = 0; i < osN; ++i)
                {
                    const double h = driveRamping ? gainRamps_[0][static_cast<size_t>(i)] : hScale_;
                    const double value = state.template processSample<false>(
                        h * static_cast<double>(d[i]), numStagesActive_, sagR_);
                    if (osOn)
                        compensationScratch_[static_cast<size_t>(i)] = value;
                    else
                    {
                        const double g = outputRamping ? gainRamps_[1][static_cast<size_t>(i)] : outGain;
                        d[i] = static_cast<T>(g * value);
                    }
                }
                if (osOn)
                {
                    state.compensateBlock(compensationScratch_.data(), osN, numStagesActive_);
                    for (int i = 0; i < osN; ++i)
                    {
                        const double g = outputRamping ? gainRamps_[1][static_cast<size_t>(i)] : outGain;
                        d[i] = static_cast<T>(g * compensationScratch_[static_cast<size_t>(i)]);
                    }
                }
            }
            if (osOn) oversampler_->downsample(buffer);
            supplyNow_.store(static_cast<T>(kBplus - sagR_ * channels_[0]->ipLP),
                             std::memory_order_relaxed);
        }

        // Latency-compensated mix (ramped: a hard flip on the distorted wet
        // stream clicked at 1.6x the steady-state sample delta).
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = buffer.getChannel(ch);
            const auto& dry = dryRing_[static_cast<size_t>(ch)];
            for (int i = 0; i < nS; ++i)
            {
                const int idx = (dryPos_ + i - latency_) & (drySize_ - 1);
                const T drySample = dry[static_cast<size_t>(idx)];
                const T mixVal = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(i + 1));
                d[i] = drySample + (d[i] - drySample) * mixVal;
            }
        }
        currentMix_ = moveTowards(mixStart, mixTarget, mixMaxStep_ * static_cast<T>(nS));
        dryPos_ = (dryPos_ + nS) & (drySize_ - 1);
    }

private:
    // -- Circuit constants (classic 12AX7 common-cathode stage) -------------------
    static constexpr double kBplus = 300.0;
    static constexpr double kRL = detail::TubePreampCurrentTable::kRL;
    static constexpr double kRk = detail::TubePreampCurrentTable::kRk;
    static constexpr double kCk = detail::TubePreampCurrentTable::kCk;
    static constexpr double kInterstage = 0.12;   ///< Divider into stage 2.

    using LoadTable = detail::TubePreampCurrentTable;

    static void koren(double vpk, double vgk, double& ip,
                      double& dIpdVpk, double& dIpdVgk) noexcept
    {
        LoadTable::koren(vpk, vgk, ip, dIpdVpk, dIpdVgk);
    }

    /** @brief One common-cathode stage with trapezoidal cathode bypass. */
    struct TriodeStage
    {
        const LoadTable* table = nullptr;
        double fs2 = 96000.0;
        double ip = 8e-4;          ///< Plate current state / NR seed.
        double vk = 1.2;           ///< Cathode voltage (bypass cap state).
        double fPrev = 0.0;        ///< Previous net capacitor current (A).
        double vpDC = 200.0;       ///< Plate voltage at the operating point.
        double previousGrid = 0.0, previousClamped = 0.0;
        double previousPrimitive2 = 0.0, olderGrid = 0.0, olderPrimitive2 = 0.0;
        bool antialias = true;

        void settleDC(double bplus) noexcept
        {
            // Static operating point: Vk = Ip*Rk (capacitor fully charged).
            double i = 8e-4;
            for (int it = 0; it < 60; ++it)
            {
                const double vkS = i * kRk;
                const double vpk = bplus - i * kRL - vkS;
                double ipK = 0.0, dVpk = 0.0, dVgk = 0.0;
                koren(vpk, -vkS, ipK, dVpk, dVgk);
                const double f = i - ipK;
                const double fp = 1.0 - (dVpk * (-(kRL + kRk)) + dVgk * (-kRk));
                const double di = f / fp;
                i -= di;
                i = std::clamp(i, 0.0, bplus / (kRL + kRk));
                if (std::abs(di) < 1e-15) break;
            }
            ip = i;
            vk = i * kRk;
            fPrev = 0.0;
            vpDC = bplus - i * kRL;
            previousGrid = previousClamped = 0.0;
            previousPrimitive2 = olderGrid = olderPrimitive2 = 0.0;
        }

        /** @brief Processes one grid-volt sample, returns AC plate voltage. */
        [[nodiscard]] double processSample(double vg, double bplusEff) noexcept
        {
            // Antialias BOTH nonlinearities. Averaging only the triode cannot
            // remove aliases that were already generated by the grid clamp.
            double clamped = 0.0, oldClamped = 0.0;
            if (antialias)
            {
                const double primitive2 = detail::TubePreampGridClamp::primitive2(vg);
                clamped = detail::TubePreampGridClamp::divided(olderGrid, previousGrid, vg,
                    olderPrimitive2, previousPrimitive2, primitive2);
                olderGrid = previousGrid;
                olderPrimitive2 = previousPrimitive2;
                previousGrid = vg;
                previousPrimitive2 = primitive2;
                oldClamped = previousClamped;
                previousClamped = clamped;
                vg = 0.5 * (oldClamped + clamped);
            }
            else
                vg = oldClamped = clamped = detail::TubePreampGridClamp::value(vg);

            // Trapezoidal cathode bypass: Ck dVk/dt = Ip - Vk/Rk, with fPrev
            // holding the previous NET CURRENT (Ip - Vk/Rk), so the update is
            //   Vk_n = Vk_{n-1} + (T/2Ck)(I_n + I_{n-1}) = kA + kB * Ip_n.
            const double h2c = 0.5 / (fs2 * kCk);
            const double denom = 1.0 + h2c / kRk;
            const double kA = (vk + h2c * fPrev) / denom;
            const double kB = h2c / denom;

            const double iMax = bplusEff / kRL + 1e-3;
            double i = std::clamp(ip, 0.0, iMax);

            // Outside the prepared supply/grid range, retain the analytic
            // circuit solve. The table includes a bounded deep-cutoff limit.
            if (table->covers(bplusEff - kA, oldClamped - kA)
                && table->covers(bplusEff - kA, clamped - kA))
                i = antialias ? table->average(bplusEff - kA, oldClamped - kA, clamped - kA)
                              : table->eval(bplusEff - kA, vg - kA);
            else for (int it = 0; it < 8; ++it)
            {
                const double vkN = kA + kB * i;
                const double vpk = bplusEff - i * kRL - vkN;
                const double vgk = vg - vkN;
                double ipK = 0.0, dVpk = 0.0, dVgk = 0.0;
                koren(vpk, vgk, ipK, dVpk, dVgk);

                const double f = i - ipK;
                const double fp = 1.0 - (dVpk * (-kRL - kB) + dVgk * (-kB));
                const double di = f / std::max(fp, 1e-6);
                i -= di;
                i = std::clamp(i, 0.0, iMax);
                if (std::abs(di) < 1e-12) break;
            }

            const double vkN = kA + kB * i;
            fPrev = i - vkN / kRk;      // net capacitor current for the trapezoid
            vk = vkN;
            ip = i;
            return (bplusEff - i * kRL) - vpDC;     // AC component
        }
    };

    /** @brief Full per-channel circuit: two stages + FMV tone stack + sag. */
    struct ChannelState
    {
        explicit ChannelState(double fs2In, const LoadTable* table,
                              const detail::TubePreampCompensation* compensationIn)
            : compensation(compensationIn), fmv(38e3, 1e6) // fixed source impedance
        {
            stage1.table = table;
            stage2.table = table;
            stage1.fs2 = fs2In;
            stage2.fs2 = fs2In;
            stage1.antialias = stage2.antialias = compensation->latency > 0;
            fs2 = fs2In;
            fmv.prepare(fs2In);
            if (compensation->latency > 0)
                antiAliasEq.prepare(static_cast<int>(compensation->taps[0].size()), 1);
        }

        /** Settles the DC operating point CONSISTENTLY with the sagged
         *  supply: fixed point on B+ = kBplus - sagR*(Ip1+Ip2). Settling at
         *  the stiff kBplus while processSample() immediately applies the
         *  sag drop produced a ~19 V supply step at sag 0.3 - an audible
         *  activation thump, and worse: it sat inside the old 10 ms
         *  calibration window, inflating outSq and burying the whole wet
         *  path ~20 dB under unity. */
        void reset(double sagR, int numStages) noexcept
        {
            // Fixed point over the ACTIVE stage count: processSample only
            // draws current from the stages in use, so seeding ipLP with both
            // stages' current at 1-stage settings left a ~70 ms sag transient
            // (audible activation drift, and it polluted the noise floor).
            double bp = kBplus;
            double iTotal = 0.0;
            for (int it = 0; it < 8; ++it)
            {
                stage1.settleDC(bp);
                stage2.settleDC(bp);
                iTotal = stage1.ip + (numStages > 1 ? stage2.ip : 0.0);
                const double bpNew = kBplus - sagR * iTotal;
                if (std::abs(bpNew - bp) < 1e-9) break;
                bp = bpNew;
            }
            ipLP = iTotal;
            outHpX = outHpY = 0.0;
            fmv.reset();
            for (auto& set : flatten)
                for (auto& f : set)
                    f.reset();
            antiAliasEq.reset();
            compensationStages = 0;
        }

        /** Installs the reference-flattening EQ for one stage count. */
        void setFlattenCoeffs(int stageCount,
                              const std::array<BiquadCoeffs, 3>& c) noexcept
        {
            auto& set = flatten[static_cast<size_t>(stageCount - 1)];
            for (int k = 0; k < 3; ++k)
                set[static_cast<size_t>(k)].setCoeffs(c[static_cast<size_t>(k)]);
        }

        void setToneControls(double t, double b, double m) noexcept
        {
            fmv.setControls(t, b, m);   // rebuilds the R-type scattering
        }

        template <bool Compensate = true>
        [[nodiscard]] double processSample(double vgIn, int numStages, double sagR) noexcept
        {
            // Supply sag: B+ droops with smoothed total plate current.
            const double sagAlpha = 1.0 - std::exp(-1.0 / (0.07 * fs2));
            const double iTotal = stage1.ip + (numStages > 1 ? stage2.ip : 0.0);
            ipLP += sagAlpha * (iTotal - ipLP);
            const double bplusEff = kBplus - sagR * ipLP;

            // Stage 1 -> FMV tone stack -> (stage 2) -> output high-pass.
            double v = stage1.processSample(vgIn, bplusEff);
            v = fmv.processSample(v);
            if (numStages > 1)
                v = stage2.processSample(v * kInterstage, bplusEff);

            // Output coupling high-pass (~8 Hz, removes residual sag drift).
            const double a = 1.0 - 2.0 * std::numbers::pi * 8.0 / fs2;
            const double y = a * (outHpY + v - outHpX);
            outHpX = v;
            outHpY = y;
            // Restore absolute polarity for single-stage use.
            double out = (numStages > 1) ? y : -y;

            // Reference-flattening EQ: undoes the FMV stack's fixed envelope
            // at the neutral tone setting (designed in calibrateReference
            // from the measured response), so neutral knobs sound neutral
            // and the tone controls act RELATIVE to flat. The triode's
            // waveform is filtered here; this linear stage adds no harmonics.
            auto& fl = flatten[numStages > 1 ? 1 : 0];
            out = fl[0].processSample(out, 0);
            out = fl[1].processSample(out, 0);
            out = fl[2].processSample(out, 0);
            if constexpr (!Compensate) return out;
            if (compensation->latency == 0) return out;
            selectCompensation(numStages);
            return antiAliasEq.processSample(out, 0);
        }

        void selectCompensation(int numStages) noexcept
        {
            if (compensationStages != numStages)
            {
                // Fixed prepared coefficients; no allocation on stage changes.
                antiAliasEq.setCoefficients(compensation->taps[static_cast<size_t>(numStages - 1)]);
                compensationStages = numStages;
            }
        }

        void compensateBlock(double* data, int count, int numStages) noexcept
        {
            selectCompensation(numStages);
            // Core block processing hoists coefficient publication/atomics
            // out of the inner loop. The calibration uses the same FIR.
            antiAliasEq.processBlock(AudioBufferView<double>(&data, 1, count));
        }

        double fs2 = 96000.0;
        TriodeStage stage1, stage2;
        double ipLP = 1.6e-3;
        double outHpX = 0.0, outHpY = 0.0;
        std::array<std::array<Biquad<double, 1>, 3>, 2> flatten;   ///< Per stage count.
        const detail::TubePreampCompensation* compensation;
        FIRFilter<double> antiAliasEq;
        int compensationStages = 0;

        wdf::ToneStackFMV<double> fmv;   ///< Exact Bassman stack (R-type WDF).
    };

    /** @brief Applies parameter changes: stack controls, sag R, calibration. */
    void recompute() noexcept
    {
        const double drive = std::pow(10.0, static_cast<double>(
            driveDb_.load(std::memory_order_relaxed)) / 20.0);
        const double t = static_cast<double>(treble_.load(std::memory_order_relaxed));
        const double b = static_cast<double>(bass_.load(std::memory_order_relaxed));
        const double m = static_cast<double>(middle_.load(std::memory_order_relaxed));
        const double sag = static_cast<double>(sag_.load(std::memory_order_relaxed));
        numStagesActive_ = stages_.load(std::memory_order_relaxed);

        hScale_ = drive;                       // 0 dBFS -> 1 V grid at drive 0
        // Note on physics: a class-A preamp draws near-constant average
        // current, so supply sag shifts the operating point rather than
        // pumping like a push-pull power amp. The audible touch response of
        // this model comes from the cathode-bypass bias shift (modelled in
        // TriodeStage); the sag control changes voicing, not loudness.
        sagR_ = sag * 40e3;
        for (auto& ch : channels_)
            ch->setToneControls(t, b, m);

        // Loudness: divide out the circuit's program gain at the REFERENCE
        // tone setting (measured once in prepare on a settled channel) and
        // MOST of the drive factor. The link is partial (drive^0.75, i.e. a
        // residual +0.25 dB/dB slope): an exact 1/drive link leaves the knob
        // audibly dead below 0 dB (the circuit is still clean there) and
        // turns it into a pure attenuator above (compression eats level
        // faster than the link returns it). With the residual slope, -12 dB
        // drive sits ~3 dB lower and clean, high drive holds level while the
        // density grows. The tone knobs stay fully audible: their deviation
        // from the 0.5/0.5/0.5 reference is part of the tone, not the level.
        // Cheap by construction (no scratch processing on the audio thread).
        //
        // The divisor is the circuit's MEASURED program gain at this drive
        // (prepare-time LUT, log-interpolated) - same contract as
        // TapeMachine/TransformerModel, whose per-drive scratch calibration
        // is what kept their knobs healthy. The previous analytic
        // 1/drive^0.75 broke at high drive: once the triode pins at its
        // ceiling, output stops growing with drive while the divisor keeps
        // rising, so the level FELL hard instead of holding.
        const double driveDbNow = static_cast<double>(driveDb_.load(std::memory_order_relaxed));
        mScale_ = std::pow(drive, 0.25)
                / std::max(programGainAt(driveDbNow, numStagesActive_), 1e-9);
    }

    /** @brief Program gain at a drive setting (log-interpolated prepare LUT). */
    [[nodiscard]] double programGainAt(double driveDb, int stages) const noexcept
    {
        const auto& lut = gProgLut_[static_cast<size_t>(stages - 1)];
        // Ordered min/max instead of clamp: a NaN input resolves to 0 here
        // (defence in depth; the setter already rejects non-finite values, and
        // an unguarded NaN would UB-cast into a wild LUT index).
        const double pos = std::min(std::max(0.0, (driveDb - kDriveLutMinDb) / kDriveLutStepDb),
                                    static_cast<double>(kDriveLutN - 1));
        const int idx = std::min(static_cast<int>(pos), kDriveLutN - 2);
        const double frac = pos - static_cast<double>(idx);
        const double a = std::max(lut[static_cast<size_t>(idx)], 1e-9);
        const double b = std::max(lut[static_cast<size_t>(idx) + 1], 1e-9);
        return a * std::pow(b / a, frac);   // gains are smooth in dB
    }

    /**
     * Measures the per-stage-count reference response of a settled channel
     * at the reference tone (0.5/0.5/0.5), reference sag voicing and drive
     * 0 dB, using a pink-weighted multitone (one tone per octave,
     * 100..6400 Hz) at program level (~-15 dBFS RMS). From the per-tone
     * gains it designs a 3-section flattening EQ (low shelf / mid peak /
     * high shelf) that removes the FMV stack's fixed envelope - the raw
     * stack at neutral knobs is a ~10 dB mid scoop, far too much colour
     * for an insert effect - and then sweeps the drive grid through a
     * chained scratch channel (flattener installed) to fill the program
     * gain LUT that recompute() interpolates. A single 1 kHz tone is the
     * wrong probe for any of this (the old 1 kHz/10 ms calibration also
     * measured INSIDE the supply-sag settling step, burying the wet path
     * ~20 dB).
     */
    void calibrateReference() noexcept
    {
        constexpr double kRefTones[] = { 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0 };
        constexpr double kSagRef = 0.3 * 40e3;
        const int settle = static_cast<int>(0.060 * fs2_);
        const int meas   = static_cast<int>(0.050 * fs2_);
        constexpr double kAmpPerTone = 0.095 / 2.6457513;

        for (int st = 1; st <= 2; ++st)
        {
            ChannelState cal(fs2_, loadTable_.get(), compensation_.get());
            cal.setToneControls(0.5, 0.5, 0.5);
            cal.reset(kSagRef, st);

            // Per-tone Goertzel over the measured tail.
            std::array<double, 7> gs1 {}, gs2 {}, gc {};
            for (int k = 0; k < 7; ++k)
                gc[static_cast<size_t>(k)] =
                    2.0 * std::cos(2.0 * std::numbers::pi * kRefTones[k] / fs2_);
            for (int i = 0; i < settle + meas; ++i)
            {
                double x = 0.0;
                for (int k = 0; k < 7; ++k)
                    x += std::sin(2.0 * std::numbers::pi * kRefTones[k] * i / fs2_ + k * 1.7);
                x *= kAmpPerTone;
                const double y = cal.processSample(x, st, kSagRef);
                if (i >= settle)
                    for (int k = 0; k < 7; ++k)
                    {
                        auto& s1 = gs1[static_cast<size_t>(k)];
                        auto& s2 = gs2[static_cast<size_t>(k)];
                        const double s0 = y + gc[static_cast<size_t>(k)] * s1 - s2;
                        s2 = s1; s1 = s0;
                    }
            }
            std::array<double, 7> gainDb {};
            for (int k = 0; k < 7; ++k)
            {
                const double c = gc[static_cast<size_t>(k)] * 0.5;
                const double s1 = gs1[static_cast<size_t>(k)], s2 = gs2[static_cast<size_t>(k)];
                const double mag = std::sqrt(std::max(s1 * s1 + s2 * s2 - 2.0 * c * s1 * s2, 0.0))
                                 * 2.0 / meas;
                gainDb[static_cast<size_t>(k)] =
                    20.0 * std::log10(std::max(mag / kAmpPerTone, 1e-9));
            }

            // Flattening EQ from the band means, relative to the overall mean.
            double mean = 0.0;
            for (double g : gainDb) mean += g / 7.0;
            const double lo = 0.5 * (gainDb[0] + gainDb[1]) - mean;
            const double mid = (gainDb[2] + gainDb[3] + gainDb[4]) / 3.0 - mean;
            const double hi = 0.5 * (gainDb[5] + gainDb[6]) - mean;

            std::array<BiquadCoeffs, 3> fc = {
                BiquadCoeffs::makeLowShelf(fs2_, 180.0, -lo),
                BiquadCoeffs::makePeak(fs2_, 800.0, 0.55, -mid),
                BiquadCoeffs::makeHighShelf(fs2_, 4500.0, -hi)
            };
            for (auto& ch : channels_)
                ch->setFlattenCoeffs(st, fc);

            // Phase 2: program gain vs DRIVE, on a chained scratch channel
            // with the flattener installed (the chain as it really sounds).
            // Each grid point gets a re-settle (bias/sag adapt to the new
            // level) before its measurement window. recompute() interpolates
            // this LUT, so the loudness link tracks the circuit's actual
            // compression - the fix for the level falling at high drive.
            ChannelState sweep(fs2_, loadTable_.get(), compensation_.get());
            sweep.setToneControls(0.5, 0.5, 0.5);
            sweep.setFlattenCoeffs(st, fc);
            sweep.reset(kSagRef, st);
            const int settle2 = static_cast<int>(0.030 * fs2_);
            const int meas2   = static_cast<int>(0.030 * fs2_);
            int t = 0;   // continuous tone phase across the whole sweep
            for (int kd = 0; kd < kDriveLutN; ++kd)
            {
                const double d = std::pow(10.0,
                    (kDriveLutMinDb + kd * kDriveLutStepDb) / 20.0);
                double inSq = 0.0, outSq = 0.0;
                for (int i = 0; i < settle2 + meas2; ++i, ++t)
                {
                    double x = 0.0;
                    for (int k = 0; k < 7; ++k)
                        x += std::sin(2.0 * std::numbers::pi * kRefTones[k] * t / fs2_ + k * 1.7);
                    x *= kAmpPerTone;
                    const double y = sweep.processSample(d * x, st, kSagRef);
                    if (i >= settle2)
                    {
                        inSq += x * x;
                        outSq += y * y;
                    }
                }
                gProgLut_[static_cast<size_t>(st - 1)][static_cast<size_t>(kd)] =
                    (outSq > 0.0 && inSq > 0.0) ? std::sqrt(outSq / inSq) : 1.0;
            }
        }
    }

    // -- Members --------------------------------------------------------------------
    AudioSpec spec_ {};                ///< Last valid spec, for setOversampling re-prepare.
    double sampleRate_ = 48000.0;
    double fs2_ = 96000.0;
    int numChannels_ = 0;
    int maxBlock_ = 0;
    std::atomic<bool> prepared_ { false };
    int latency_ = 0;
    int drySize_ = 1;
    int osFactor_ = 2;                  ///< Oversampling factor (setup thread; 1 = off, 2 default).

    std::unique_ptr<LoadTable> loadTable_;
    std::unique_ptr<detail::TubePreampCompensation> compensation_;
    std::unique_ptr<Oversampling<T>> oversampler_;
    std::vector<std::unique_ptr<ChannelState>> channels_;

    std::vector<std::vector<T>> dryRing_;
    int dryPos_ = 0;

    static constexpr int    kDriveLutN = 13;       ///< -12..+36 dB in 4 dB steps.
    static constexpr double kDriveLutMinDb = -12.0;
    static constexpr double kDriveLutStepDb = 4.0;

    double hScale_ = 1.0;
    double mScale_ = 1.0;
    double sagR_ = 0.0;
    int numStagesActive_ = 1;
    /// Program gain vs drive, per stage count (prepare-time sweep).
    std::array<std::array<double, kDriveLutN>, 2> gProgLut_ {
        { { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
          { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } } };
    SmoothedValue<double> driveLogSmoother_, outputLogSmoother_;
    std::array<std::vector<double>, 2> gainRamps_;
    std::vector<double> compensationScratch_;
    bool gainsInitialized_ = false;
    T currentMix_ = T(1);                       ///< Audio-thread mix ramp state.
    T mixMaxStep_ = T(1.0 / 960.0);             ///< Mix ramp rate: full scale per 20 ms.

    std::atomic<T> driveDb_ { T(0) };
    std::atomic<T> treble_ { T(0.5) };
    std::atomic<T> bass_ { T(0.5) };
    std::atomic<T> middle_ { T(0.5) };
    std::atomic<T> sag_ { T(0.3) };
    std::atomic<int> stages_ { 1 };
    std::atomic<T> outputDb_ { T(0) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<T> supplyNow_ { T(300) };
    std::atomic<bool> dirty_ { true };
};

} // namespace dspark
