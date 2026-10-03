// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TubePreamp.h
 * @brief Tube preamp: Koren 12AX7 stages, Bassman FMV tone circuit, supply sag.
 *
 * Circuit-level preamp modelling, not a waveshaper:
 *
 * - **Triode stages** (1 or 2) use a tabulated implicit plate-current
 *   solution with bicubic interpolation; an analytic Newton solve handles
 *   voltages outside the table. The tube follows Koren's SPICE model (Koren
 *   1996; published 12AX7 parameters MU=100, EX=1.4, KG1=1060, KP=600,
 *   KVB=300) in a classic common-cathode stage: 300 V supply, 100 k ohm plate
 *   load, 1.5 k ohm cathode resistor with its 22 uF bypass capacitor. Grid
 *   conduction is approximated by a soft clamp toward +0.7 V (full blocking
 *   distortion needs the input-coupling state and is left for a later pass).
 * - **Supply sag**: the effective B+ droops with smoothed plate current
 *   (one-pole, ~70 ms) times a sag resistance - drive into the stage and
 *   the headroom breathes back, the classic touch response.
 * - **Tone stack**: the full Fender '59 Bassman FMV treble/bass/middle
 *   network (wdf::ToneStackFMV, verified against the symbolic transfer
 *   function of Yeh & Smith, DAFx-06). It sits between the stages with a
 *   fixed 38 k ohm source-impedance approximation; its load is not fed back
 *   into the triode solve. Controls interact non-orthogonally - that is the
 *   circuit, not a bug.
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
 * resampling, zero added latency, the original point-sampled circuit; the
 * triode/grid nonlinearity then aliases in-band unless you oversample the
 * surrounding chain yourself), 2 = default, 4/8/16 = lower residual error at
 * higher processing cost. getLatency()/getLatencySamples() report the active
 * total.
 *
 * At factors >= 2 the circuit is solved in continuous time inside each
 * internal sample interval, and the analog plate voltage is band-limited
 * ONCE, at the output:
 * - The stage-1 grid input between internal samples is a least-squares
 *   Farrow polynomial fitted to the oversampled band (reconstruction error
 *   about -83 dB at the top of the 2x band).
 * - Supply and cathode states are held at their predicted interval midpoint
 *   (their time constants are tens of milliseconds) and advanced exactly
 *   with the interval-average current.
 * - Each interval is split where the grid crosses the triode's cutoff,
 *   conduction and grid-clamp knees; saturated parts are integrated in
 *   closed form and the rest with Gauss-Legendre nodes, so the cost follows
 *   the signal rather than a fixed sub-step grid.
 * - The tone circuit is propagated exactly in its analog modal form
 *   (wdf::ToneStackFMV::analogStateSpace), including the stiff mode that
 *   appears near bass 0 / middle 1. The interstage voltage seen by stage 2
 *   is therefore the continuous circuit trajectory, not an interpolation of
 *   samples. Stage 2 is split at its own knees in the same way.
 * - The output projection is an order-6 B-spline kernel, applied through
 *   per-interval moments, followed by a fixed linear-phase FIR that
 *   compensates its passband droop. Because the kernel acts only on the
 *   final output, the compensation is exact at every signal level.
 *
 * Total latency for 1/2/4/8/16x is 0/71/99/113/121 base samples,
 * independent of setStages(); the dry path includes the same delay.
 * getLatency() always reflects the ACTIVE factor (0 at 1x) so hosts get
 * correct PDC. At 1x the original point circuit is retained: use 1x only
 * when the surrounding host chain supplies a suitable high rate.
 *
 * Measured at 48 kHz over tones from 1 to 20 kHz in 1 kHz steps, plus
 * 12.75/15.25/19.25 kHz, at -6 and -18 dBFS: each tone is moved to the
 * nearest odd bin of a 16384-point FFT. After settling, coherent
 * rectangular DFT measurements exclude only DC and exact physical harmonic
 * bins. In particular, adjacent folded lines are retained. The worst
 * component below 20 kHz, relative to the tone, is:
 *
 *   stages, drive     2x (default)   4x          8x
 *   1, -12 dB        -121.5         -134.3       -134.3
 *   1, 0 dB           -96.7         -115.0       -115.0
 *   1, +12 dB         -87.7         -101.4       -101.4
 *   1, +24 dB         -80.6          -96.4        -96.8
 *   1, +36 dB         -80.6          -96.4        -96.6
 *   2, -12 dB        -104.7         -120.8       -120.8
 *   2, 0 dB           -86.9         -101.5       -103.1
 *   2, +12 dB         -83.0          -93.9        -96.9
 *   2, +24 dB         -83.0          -93.9        -96.4
 *   2, +36 dB         -84.8          -96.4        -96.4
 *
 * On the same sweep the former antiderivative-antialiased core reached
 * -19.2 dBc (two stages, +36 dB, 2x). At 2x the worst value over all
 * drives is -81.8 dBc at 44.1 kHz and -96.1 dBc at 96 kHz.
 *
 * Against independent dense solutions of the same circuit equations (the
 * point circuit at 512x and 1024x the base rate, 78 settings of drive,
 * level, tone controls and frequency up to 19.9 kHz, including +36 dB),
 * the 2x waveform error is at most -73.5 dB (median -84.7 dB) with
 * fundamental errors below 0.002 dB; 4x/8x/16x reach -83.4/-88.8/-93.7 dB.
 * These figures concern the mathematical circuit model, not measurements of
 * a physical amplifier.
 *
 * Cost follows the signal: intervals are split only where a stage crosses
 * a knee. For the whole effect, stereo at 48 kHz and 2x, the development
 * machine measured 3.4-4.6 % of one core without drive or with low-level
 * or low-frequency material, and 9-11 % for high-frequency or broadband
 * material at +12..+36 dB: 1.1-2.9x the former core at 2x, and less than
 * that core needed at 8x. At 4x, 8x and 16x the cost is 1.2-2.1x,
 * 1.2-1.6x and about 1.4x the former core at the same factor.
 *
 * The plate-current table is 156672 bytes shared by all channels in an
 * instance. Preparation builds the table, designs the modal tone circuit
 * and calibrates the circuit; processing, tone changes and stage changes
 * allocate no memory. The suite checks the 2x output against the point
 * circuit run at 32x the base rate, the 2nd-harmonic dominant single-stage
 * signature, and the DC operating point against an independent
 * high-precision solve of the same circuit equations (the check SPICE
 * would perform).
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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace dspark {

/// @cond DSPARK_INTERNAL
namespace detail {
    struct TubePreampGridClamp
    {
        /** Identity below 0 V, 0.7*tanh(v/0.7) above (soft clamp toward +0.7 V). */
        static double value(double v) noexcept
        {
            if (v <= 0.0) return v;
            if (v >= 13.0) return 0.7;   // 0.7*(1 - 2e^-37) rounds to 0.7
            const double e = std::expm1(v * (-2.0 / 0.7));
            return -0.7 * e / (2.0 + e);
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
        // Ip = Koren(S - (RL+B)*Ip, G - B*Ip). B is fixed by the rate given
        // to the constructor: the 1x point circuit passes its sample rate,
        // the continuous core passes infinity (B = 0) and freezes A per
        // interval. Use R = G/S to align the cutoff knee across supply
        // voltages. The 156672-byte table has <5e-10 A error in the
        // independent load-line check. Bicubic Hermite interpolation keeps
        // current and both first derivatives continuous at cell edges.
        // Every channel and both calibration passes share this instance.
        struct Node
        {
            double y, ds, dg, dsg;
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
                    // to keep the cubic nonnegative.
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

        [[nodiscard]] double eval(double s, double g) const noexcept
        {
            // Below this cutoff the unloaded Koren current is < 1e-27 A.
            if (g <= -0.08 * s) return 0.0;
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
            const double st = sp - si, gt = gp - gi;
            const Node& a = nodes[static_cast<size_t>(si * kNG + gi)];
            const Node& b = nodes[static_cast<size_t>((si + 1) * kNG + gi)];
            const Node& c = nodes[static_cast<size_t>(si * kNG + gi + 1)];
            const Node& d = nodes[static_cast<size_t>((si + 1) * kNG + gi + 1)];
            const double p0 = cubic(a.y, b.y, a.ds * step, b.ds * step, st);
            const double p1 = cubic(c.y, d.y, c.ds * step, d.ds * step, st);
            const double m0 = rstep * cubic(a.dg, b.dg, a.dsg * step, b.dsg * step, st);
            const double m1 = rstep * cubic(c.dg, d.dg, c.dsg * step, d.dsg * step, st);
            return std::max(0.0, cubic(p0, p1, m0, m1, gt));
        }

        /** @brief The surface restricted to one supply S: the S blend weights
         *  are computed once, so repeated grid evaluations at a frozen supply
         *  cost one R cell each. Bit-for-bit the same value as eval(). */
        struct Slice
        {
            const Node* row0 = nullptr;
            const Node* row1 = nullptr;
            double s = 0.0, invS = 0.0, cut = 0.0;
            double h00 = 0.0, h01 = 0.0, h10 = 0.0, h11 = 0.0;
            bool inside = false;

            void set(const TubePreampCurrentTable& table, double supply) noexcept
            {
                s = supply;
                inside = supply >= 80.0 && supply < 304.0;
                if (!inside) return;
                invS = 1.0 / supply;
                cut = -0.08 * supply;
                const double step = supply < 128.0 ? 8.0 : 16.0;
                const double sp = supply < 128.0 ? (supply - 80.0) / 8.0
                                                 : 6.0 + (supply - 128.0) / 16.0;
                const int si = std::min(static_cast<int>(sp), kNS - 2);
                const double t = sp - si;
                h00 = (2.0 * t - 3.0) * t * t + 1.0;
                h01 = (3.0 - 2.0 * t) * t * t;
                h10 = step * ((t - 2.0) * t + 1.0) * t;
                h11 = step * (t - 1.0) * t * t;
                row0 = table.nodes.data() + si * kNG;
                row1 = row0 + kNG;
            }

            /** @pre inside and g < 1. */
            [[nodiscard]] double eval(double g) const noexcept
            {
                if (g <= cut) return 0.0;
                const double r = g * invS;
                double gp, rstep;
                if (r < -0.04) { gp = (r + 0.08) * 400.0; rstep = 0.0025; }
                else if (r < -0.025) { gp = 16.0 + (r + 0.04) * 1000.0; rstep = 0.001; }
                else { gp = 31.0 + (r + 0.025) * 6400.0; rstep = 1.0 / 6400.0; }
                const int gi = std::min(static_cast<int>(gp), kNG - 2);
                const double gt = gp - gi;
                const Node& a = row0[gi];
                const Node& c = row0[gi + 1];
                const Node& b = row1[gi];
                const Node& d = row1[gi + 1];
                const double p0 = h00 * a.y + h01 * b.y + h10 * a.ds + h11 * b.ds;
                const double p1 = h00 * c.y + h01 * d.y + h10 * c.ds + h11 * d.ds;
                const double m0 = rstep * (h00 * a.dg + h01 * b.dg + h10 * a.dsg + h11 * b.dsg);
                const double m1 = rstep * (h00 * c.dg + h01 * d.dg + h10 * c.dsg + h11 * d.dsg);
                const double diff = p1 - p0;
                const double v = (((m0 + m1 - 2.0 * diff) * gt + (3.0 * diff - 2.0 * m0 - m1)) * gt
                                  + m0) * gt + p0;
                return v > 0.0 ? v : 0.0;
            }
        };
    };

    /** Fixed coefficients of the continuous-time core for one factor. */
    struct TubePreampCoreDesign
    {
        static constexpr int kKernelOrder = 6;   ///< Order of the output B-spline.
        int taps = 0;                ///< Farrow taps: x[n-taps/2+1] .. x[n+taps/2].
        int degree = 0;              ///< Polynomial degree in the interval coordinate.
        const double* farrow = nullptr;          ///< taps x (degree+1), monomial in tau.
        /// B-spline piece for output m = n + k, k = -2..3, in powers of (tau - 1/2).
        std::array<std::array<double, kKernelOrder>, kKernelOrder> kernel {};
        std::vector<double> compensation;       ///< Linear-phase droop FIR plus padding.
        int latency = 0;             ///< Base-rate samples added by the core.

        explicit TubePreampCoreDesign(int factor)
        {
            // Least-squares Farrow fits over 0..22 kHz at 48 kHz (scaled with
            // the base rate), interpolating the two interval samples. Max
            // error vs the ideal band-limited interpolant: -82.8 dB (2x),
            // -104.2 dB (4x), -108.2 dB (8x), -104.6 dB (16x). The droop
            // compensation inverts sinc(f)^6 over the same band to better
            // than 2e-6 dB.
            static constexpr std::array<double, 100> kFarrow2 = {
                4.2321905086886495e-10, 0.0059466381516058153, -0.002278523941500805,
                -0.0054952548957599556, -0.014968825413515578, 0.054109227099701757,
                -0.078269013079604377, 0.062385435811858708, -0.025290994449854602,
                0.0038613094605228345, -2.2707239900454428e-09, -0.04509673531374573,
                0.02130560198974633, 0.044927139987920424, 0.068139628956554815,
                -0.29034355992063943, 0.41888697256651169, -0.33093026606533477,
                0.13329150667830314, -0.020180282199047366, 6.3151437033992178e-09,
                0.19516443759783259, -0.125629932201473, -0.18510777189666724,
                -0.11518179249706405, 0.79144728674092368, -1.1628931982523558,
                0.91250354516673049, -0.36515236023089487, 0.0548497671327925,
                -1.1679462053512646e-08, -0.73810415625364523, 0.8470174584244009,
                0.17008113400299396, 0.090159156301663379, -1.4018192300222059,
                2.141392397438906, -1.6757774481811469, 0.66635226461915309,
                -0.099301542452499336, 1.0000000156560307, -0.15092152990607394,
                -1.4813400355254824, 0.32501007016593614, -0.10665904644708368,
                1.7796410825385522, -2.8452200456412218, 2.2313784198552291,
                -0.8823072013373332, 0.13041824116835923, -1.5700830890361179e-08,
                1.0030991668852851, 0.84865934877928584, -0.74002126985840833,
                0.25444017376234029, -1.7071024411426592, 2.8159173504397854,
                -2.2212010942574856, 0.87438979546714801, -0.12818098509269454,
                1.1781332260547313e-08, -0.37253030413198274, -0.12785657025701674,
                0.56505142242672013, -0.33848444021117219, 1.2561039702068575,
                -2.0800938843434165, 1.6517199081920575, -0.64812744525532984,
                0.094217309828563259, -6.4097566291092411e-09, 0.13220572975363162,
                0.022984358585025653, -0.23015119647358298, 0.23722601584356839,
                -0.68523256280710365, 1.1138842977913868, -0.88863883259602172,
                0.3478529827090574, -0.050130774674655959, 2.3207754129408065e-09,
                -0.034616695799025515, -0.0030424597428906587, 0.06570936592805314,
                -0.092489611102646821, 0.25062568572871546, -0.39729342975322812,
                0.31732189085987816, -0.12390188239748591, 0.017687129761438363,
                -4.3623536317367461e-10, 0.0049101708741372896, 0.00017308418349044246,
                -0.010091070767974901, 0.017824294982185494, -0.04733411530281835,
                0.073532096156291082, -0.058588400350969086, 0.022788475313836876,
                -0.0032145338738844057
            };
            static constexpr std::array<double, 8> kComp2 = {
                2.71867910879388, -1.2024111681479335, 0.46738582076148377,
                -0.16331116573163179, 0.048832017855856398, -0.011604018931486495,
                0.0019406397776269202, -0.00017174220997273029
            };
            static constexpr std::array<double, 64> kFarrow4 = {
                -1.2900536553478881e-06, -0.0080151506262985533, -0.17777590204833338,
                1.3139119710287313, -4.0413832196153869, 6.3642884759250222,
                -4.992867908276553, 1.5418432994555842, 7.9871493323757734e-06,
                0.088046779796333074, 1.0604467092541447, -8.1687752563172111,
                25.099313210118467, -39.488286729375986, 30.975170253454831,
                -9.5659246558470752, -2.2105953216776039e-05, -0.55815096062851943,
                -2.3938696401524284, 22.40917634184877, -69.6019865941457,
                109.51192256668676, -85.894556644533552, 26.527491731519852,
                1.0000353884966167, -0.33265191629960283, 3.674166593014272,
                -35.4328783727806, 111.52063276854375, -175.64384089126642,
                137.76206099998669, -42.547532060929456, -3.5359293796327108e-05,
                1.098889851519979, -4.2836042287146743, 35.135653508220415,
                -111.44779322067839, 175.80749233768327, -137.90312082065205,
                42.592525392586815, 2.205090483678061e-05, -0.37161767694349446,
                3.0655130552579171, -21.923092860130158, 69.500034073335797,
                -109.8183373403734, 86.15979318327652, -26.612319122360798,
                -7.9535630817034395e-06, 0.095783904254697808, -1.1276037484567922,
                7.9530205187897351, -25.075497255063709, 39.67283956818612,
                -31.136230566192772, 9.6176971986688393, 1.2822935194890068e-06,
                -0.012285737644740915, 0.18272380080694042, -1.2870126749889277,
                4.0467156665242037, -6.4061717578660602, 5.0298415014987468,
                -1.5538123482944795
            };
            static constexpr std::array<double, 7> kComp4 = {
                0.77932253944179963, 0.48450011964418033, -0.63543552895607291,
                0.36687547570613865, -0.13057476517676106, 0.027680644656114826,
                -0.0027072253534879587
            };
            static constexpr std::array<double, 36> kFarrow8 = {
                2.1428912092597181e-06, -0.068053152547346507, 0.82615315111595389,
                -2.25344764450442, 2.3863455617383647, -0.89100287142393453,
                -1.0400687109947751e-05, 0.076025067275552194, -3.5474940938342541,
                10.692630341379365, -11.548180887960177, 4.3270436164777122,
                1.0000204893461682, -1.474312011333621, 7.0561524197541718,
                -20.727998495189027, 22.674059266177732, -8.5279485269151927,
                -2.0475134054884214e-05, 2.1464836357794721, -7.6380358894712534,
                20.54461472771839, -22.577478060052183, 8.5244628827979447,
                1.0378974319923476e-05, -0.83433580274857921, 4.1702488442657986,
                -10.417228839352022, 11.403009710027204, -4.3217178780150958,
                -2.1353801282180795e-06, 0.15419278048117258, -0.8670223096885199,
                2.1614225669978211, -2.3377483222244333, 0.88916021323615935
            };
            static constexpr std::array<double, 4> kComp8 = {
                2.0670375203804419, -0.6482179311307088, 0.12681513715972098,
                -0.012115968043045313
            };
            static constexpr std::array<double, 16> kFarrow16 = {
                -4.7274945894591747e-08, -0.33475471468652312, 0.50168301501965318,
                -0.16692822036410673, 1.0000001518786812, -0.49831699527905698,
                -1.0024720850935107, 0.50078884218257014, -1.6291311300334343e-07,
                1.0008871433692914, 0.4999030230846026, -0.50078992841108072,
                5.8345140851491506e-08, -0.16782377216653666, 0.00089437399553421448,
                0.16692931835075694
            };
            static constexpr std::array<double, 4> kComp16 = {
                2.2958105734077243, -0.82005508291669238, 0.19585990127822545,
                -0.023710110331518937
            };
            switch (factor)
            {
            case 2:  taps = 10; farrow = kFarrow2.data();  setCompensation(kComp2, 0); break;
            case 4:  taps = 8;  farrow = kFarrow4.data();  setCompensation(kComp4, 0); break;
            case 8:  taps = 6;  farrow = kFarrow8.data();  setCompensation(kComp8, 0); break;
            default: taps = 4;  farrow = kFarrow16.data(); setCompensation(kComp16, 9); break;
            }
            degree = taps - 1;
            // Interval n is processed when x[n + taps/2] arrives; its output
            // moments complete z[n - 2]; the FIR centre adds its half length.
            const int delay = taps / 2 + 2 + compensationHalf_ + padding_;
            latency = delay / factor;
            // Order-6 B-spline: B(u) = sum_j (-1)^j C(6,j) (u + 3 - j)_+^5 / 120.
            // For output m = n + k and tau = 1/2 + d, u = k - 1/2 - d; every
            // truncated power keeps one sign over the interval, so each piece
            // is an exact polynomial in d.
            constexpr double binom6[7] = {1, 6, 15, 20, 15, 6, 1};
            constexpr double binom5[6] = {1, 5, 10, 10, 5, 1};
            for (int ko = 0; ko < kKernelOrder; ++ko)
            {
                const int k = ko - 2;
                for (int j = 0; j <= 6; ++j)
                {
                    const double c = k + 2.5 - j;
                    if (c <= 0.0) continue;
                    const double sign = (j & 1) ? -1.0 : 1.0;
                    for (int p = 0; p <= 5; ++p)
                    {
                        const double term = sign * binom6[j] * binom5[p] * std::pow(c, 5 - p)
                                          * ((p & 1) ? -1.0 : 1.0) / 120.0;
                        kernel[static_cast<size_t>(ko)][static_cast<size_t>(p)] += term;
                    }
                }
            }
        }

    private:
        int padding_ = 0, compensationHalf_ = 0;

        template <size_t N>
        void setCompensation(const std::array<double, N>& half, int padding)
        {
            // half[0] is the centre tap; mirror, then delay by `padding`.
            const int h = static_cast<int>(N) - 1;
            padding_ = padding;
            compensationHalf_ = h;
            compensation.assign(static_cast<size_t>(2 * h + 1 + padding), 0.0);
            for (int q = 0; q <= h; ++q)
            {
                compensation[static_cast<size_t>(padding + h + q)] = half[static_cast<size_t>(q)];
                compensation[static_cast<size_t>(padding + h - q)] = half[static_cast<size_t>(q)];
            }
        }
    };

    /** Modal form of the FMV analog state space (symmetric similarity). */
    struct TubePreampToneModes
    {
        double lambda[3] {};      ///< Poles per internal sample (lambda * T).
        double beta[3] {};        ///< Input gain per internal sample.
        double gamma[3] {};       ///< Output gain.
        double direct = 0.0;      ///< Feedthrough.
        double toModal[3][3] {};
        double toPhysical[3][3] {};

        void design(const wdf::ToneStackFMV<double>::AnalogStateSpace& ss,
                    double sampleRate) noexcept
        {
            // a = -C^-1 G with symmetric G: S = C^(1/2) a C^(-1/2) is symmetric.
            double r[3], ri[3];
            for (int i = 0; i < 3; ++i)
            {
                r[i] = std::sqrt(ss.capacitance[i]);
                ri[i] = 1.0 / r[i];
            }
            double s[3][3], q[3][3] {};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    s[i][j] = r[i] * ss.a[i][j] * ri[j];
            for (int i = 0; i < 3; ++i)
                for (int j = i + 1; j < 3; ++j)
                    s[i][j] = s[j][i] = 0.5 * (s[i][j] + s[j][i]);
            for (int i = 0; i < 3; ++i) q[i][i] = 1.0;
            // Cyclic Jacobi rotations; robust also for coincident eigenvalues.
            for (int sweep = 0; sweep < 32; ++sweep)
            {
                double off = 0.0, norm = 0.0;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                    {
                        norm += s[i][j] * s[i][j];
                        if (i != j) off += s[i][j] * s[i][j];
                    }
                if (off <= 1e-30 * norm) break;
                for (int p = 0; p < 2; ++p)
                    for (int k = p + 1; k < 3; ++k)
                    {
                        if (s[p][k] == 0.0) continue;
                        const double theta = 0.5 * (s[k][k] - s[p][p]) / s[p][k];
                        const double t = (theta >= 0.0 ? 1.0 : -1.0)
                            / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
                        for (int i = 0; i < 3; ++i)
                        {
                            const double sip = s[i][p], sik = s[i][k];
                            s[i][p] = c * sip - sn * sik;
                            s[i][k] = sn * sip + c * sik;
                        }
                        for (int i = 0; i < 3; ++i)
                        {
                            const double spi = s[p][i], ski = s[k][i];
                            s[p][i] = c * spi - sn * ski;
                            s[k][i] = sn * spi + c * ski;
                        }
                        for (int i = 0; i < 3; ++i)
                        {
                            const double qip = q[i][p], qik = q[i][k];
                            q[i][p] = c * qip - sn * qik;
                            q[i][k] = sn * qip + c * qik;
                        }
                    }
            }
            const double period = 1.0 / sampleRate;
            for (int m = 0; m < 3; ++m)
            {
                lambda[m] = s[m][m] * period;
                double bm = 0.0, gm = 0.0;
                for (int i = 0; i < 3; ++i)
                {
                    toModal[m][i] = q[i][m] * r[i];      // Q^T C^(1/2)
                    toPhysical[i][m] = ri[i] * q[i][m];  // C^(-1/2) Q
                    bm += toModal[m][i] * ss.b[i];
                    gm += ss.c[i] * toPhysical[i][m];
                }
                beta[m] = bm * period;
                gamma[m] = gm;
            }
            direct = ss.d;
        }
    };

    /** phi_k(w) = sum_j w^j/(j+k)!, k = 0..4 (phi_0 = exp). Small |w|: a
     *  length-adapted Taylor series of phi_4 and the exact downward
     *  recurrence; otherwise the upward recurrence from exp. */
    inline void tubePreampPhi(double w, double* p) noexcept
    {
        const double aw = std::abs(w);
        if (aw < 0.5)
        {
            static constexpr double inv[16] = {1.0, 1.0, 1.0 / 2, 1.0 / 6, 1.0 / 24, 1.0 / 120,
                1.0 / 720, 1.0 / 5040, 1.0 / 40320, 1.0 / 362880, 1.0 / 3628800,
                1.0 / 39916800, 1.0 / 479001600, 1.0 / 6227020800.0,
                1.0 / 87178291200.0, 1.0 / 1307674368000.0};
            const int n = aw < 1e-3 ? 3 : (aw < 0.02 ? 5 : (aw < 0.1 ? 7 : 10));
            double v = inv[4 + n];
            for (int j = n - 1; j >= 0; --j) v = v * w + inv[4 + j];
            p[4] = v;
            p[3] = w * v + inv[3];
            p[2] = w * p[3] + inv[2];
            p[1] = w * p[2] + 1.0;
            p[0] = w * p[1] + 1.0;
        }
        else
        {
            p[0] = std::exp(w);
            p[1] = (p[0] - 1.0) / w;
            p[2] = (p[1] - 1.0) / w;
            p[3] = (p[2] - 0.5) / w;
            p[4] = (p[3] - 1.0 / 6.0) / w;
        }
    }
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
            design_ = std::make_unique<detail::TubePreampCoreDesign>(osFactor_);
        }
        else
        {
            oversampler_.reset();
            design_.reset();
        }

        // The continuous core freezes the cathode per interval (B = 0); the
        // 1x point circuit keeps the trapezoidal cathode coupling.
        loadTable_ = std::make_unique<LoadTable>(
            design_ ? std::numeric_limits<double>::infinity() : fs2_);
        channels_.clear();
        channels_.resize(static_cast<size_t>(numChannels_));
        for (auto& ch : channels_)
            ch = std::make_unique<ChannelState>(fs2_, loadTable_.get(), design_.get());

        latency_ = (oversampler_ ? oversampler_->getLatency() : 0) + (design_ ? design_->latency : 0);
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
     *  the default. Higher factors lower the residual error at higher CPU.
     *  Invalid or non-power-of-two values are ignored. getLatency()
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

    /** @brief Latency in prepared-rate samples added by the oversampler and the
     *  continuous core (0 at 1x = off); reflects the current factor, so hosts
     *  can compensate (PDC). */
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
        // poison the recursive circuit, tone-circuit, supply sag, output
        // DC-blocker and flatten-EQ state PERMANENTLY (only reset() clears
        // it, not clean input). Replace bad samples with silence before
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
                    state.compensateBlock(compensationScratch_.data(), osN);
                    for (int i = 0; i < osN; ++i)
                    {
                        const double g = outputRamping ? gainRamps_[1][static_cast<size_t>(i)] : outGain;
                        d[i] = static_cast<T>(g * compensationScratch_[static_cast<size_t>(i)]);
                    }
                }
            }
            if (osOn) oversampler_->downsample(buffer);
            supplyNow_.store(static_cast<T>(kBplus - sagR_ * channels_[0]->supplyCurrent()),
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
    using Clamp = detail::TubePreampGridClamp;

    static void koren(double vpk, double vgk, double& ip,
                      double& dIpdVpk, double& dIpdVgk) noexcept
    {
        LoadTable::koren(vpk, vgk, ip, dIpdVpk, dIpdVgk);
    }

    /** @brief Static operating point: Vk = Ip*Rk (capacitor fully charged). */
    static double settleCurrent(double bplus) noexcept
    {
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
        return i;
    }

    /** @brief Point-sampled common-cathode stage (1x path) with trapezoidal
     *  cathode bypass. */
    struct TriodeStage
    {
        const LoadTable* table = nullptr;
        double fs2 = 96000.0;
        double ip = 8e-4;          ///< Plate current state / NR seed.
        double vk = 1.2;           ///< Cathode voltage (bypass cap state).
        double fPrev = 0.0;        ///< Previous net capacitor current (A).
        double vpDC = 200.0;       ///< Plate voltage at the operating point.

        void settleDC(double bplus) noexcept
        {
            ip = settleCurrent(bplus);
            vk = ip * kRk;
            fPrev = 0.0;
            vpDC = bplus - ip * kRL;
        }

        /** @brief Processes one grid-volt sample, returns AC plate voltage. */
        [[nodiscard]] double processSample(double vg, double bplusEff) noexcept
        {
            vg = Clamp::value(vg);
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
            if (table->covers(bplusEff - kA, vg - kA))
                i = table->eval(bplusEff - kA, vg - kA);
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

    /**
     * @brief Continuous-time two-stage core (factors >= 2). One call per
     *  internal sample; returns the kernel-projected plate voltage of the
     *  output stage, delayed by taps/2 + 2 internal samples.
     */
    struct ContinuousCore
    {
        static constexpr int kLevels = 5;       ///< cutoff, A-3, A-1, +1 V, +6 V
        static constexpr int kMaxPieces = 48;
        static constexpr int kOrder = detail::TubePreampCoreDesign::kKernelOrder;
        static constexpr double kKneeRange = 1.0;   ///< Volts of x needed to split at knees.
        static constexpr double kSmallZ = 0.3, kStiffZ = 40.0;
        static constexpr double kLongPiece = 0.5;    ///< Samples; longer stage-2 pieces use 4 nodes.
        static constexpr double kShortPiece = 0.03;  ///< Samples; shorter stage-2 pieces use 2 nodes.

        const LoadTable* table = nullptr;
        const detail::TubePreampCoreDesign* design = nullptr;
        double period = 1.0 / 96000.0, cathodeDecay = 0.0, sagDecay = 0.0;
        detail::TubePreampToneModes modes;
        std::array<double, 16> ring {};
        int ringPos = 0;
        double q[3] {};                         ///< Modal tone-circuit state.
        double vk1 = 0.0, vk2 = 0.0, ipLP = 0.0, i1Prev = 0.0, i2Prev = 0.0;
        double vp1 = 0.0, vp2 = 0.0;
        double out[kOrder] {};                  ///< Pending outputs z[n-2] .. z[n+3].
        double cutSupply[2] = {-1.0, -1.0}, cutVgk[2] {};
        // Interval scratch
        typename LoadTable::Slice slice1, slice2;
        double supply = 0.0, a1 = 0.0, a2 = 0.0, s1 = 0.0, s2 = 0.0;
        double lev1[kLevels] {}, lev2[kLevels] {};
        double integral1 = 0.0, integral2 = 0.0, moments[kOrder] {};
        double hi1 = 0.0, hi2 = 0.0;
        bool haveHi1 = false, haveHi2 = false;
        int stages = 2;
        // Gauss-Legendre rules on [0, 1]
        static constexpr double kG4x[4] = {0.069431844202973713, 0.33000947820757187,
                                           0.66999052179242813, 0.93056815579702629};
        static constexpr double kG4w[4] = {0.17392742256872692, 0.32607257743127308,
                                           0.32607257743127308, 0.17392742256872692};
        static constexpr double kG3x[3] = {0.11270166537925831, 0.5, 0.88729833462074169};
        static constexpr double kG3w[3] = {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
        double lagrange4[4][4] {};              ///< Monomial coefficients of the node basis.
        double lagrange3[3][3] {};

        void init(double sampleRate, const LoadTable* tableIn,
                  const detail::TubePreampCoreDesign* designIn) noexcept
        {
            table = tableIn;
            design = designIn;
            period = 1.0 / sampleRate;
            cathodeDecay = std::exp(-period / (kRk * kCk));
            sagDecay = std::exp(-period / 0.07);
            for (int k = 0; k < 4; ++k)
            {
                double poly[4] = {1.0, 0.0, 0.0, 0.0};
                double den = 1.0;
                int deg = 0;
                for (int o = 0; o < 4; ++o)
                {
                    if (o == k) continue;
                    double next[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int e = 0; e <= deg; ++e)
                    {
                        next[e + 1] += poly[e];
                        next[e] -= kG4x[o] * poly[e];
                    }
                    ++deg;
                    for (int e = 0; e < 4; ++e) poly[e] = next[e];
                    den *= kG4x[k] - kG4x[o];
                }
                for (int e = 0; e < 4; ++e) lagrange4[k][e] = poly[e] / den;
            }
            for (int k = 0; k < 3; ++k)
            {
                const int o1 = (k + 1) % 3, o2 = (k + 2) % 3;
                const double den = (kG3x[k] - kG3x[o1]) * (kG3x[k] - kG3x[o2]);
                lagrange3[k][0] = kG3x[o1] * kG3x[o2] / den;
                lagrange3[k][1] = -(kG3x[o1] + kG3x[o2]) / den;
                lagrange3[k][2] = 1.0 / den;
            }
        }

        /** Re-designs the modal tone circuit, keeping capacitor voltages. */
        void setTone(const wdf::ToneStackFMV<double>::AnalogStateSpace& ss) noexcept
        {
            double physical[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
                for (int m = 0; m < 3; ++m)
                    physical[i] += modes.toPhysical[i][m] * q[m];
            modes.design(ss, 1.0 / period);
            for (int m = 0; m < 3; ++m)
            {
                q[m] = 0.0;
                for (int i = 0; i < 3; ++i) q[m] += modes.toModal[m][i] * physical[i];
            }
        }

        void reset(double sagR, int numStages) noexcept
        {
            double bp = kBplus, i1 = 0.0, i2 = 0.0, iTotal = 0.0;
            for (int it = 0; it < 40; ++it)
            {
                i1 = settleCurrent(bp);
                i2 = i1;
                iTotal = i1 + (numStages > 1 ? i2 : 0.0);
                const double next = kBplus - sagR * iTotal;
                if (std::abs(next - bp) < 1e-12) { bp = next; break; }
                bp = next;
            }
            vk1 = vk2 = i1 * kRk;
            vp1 = vp2 = bp - i1 * kRL;
            ipLP = iTotal;
            i1Prev = i2Prev = i1;
            q[0] = q[1] = q[2] = 0.0;
            ring.fill(0.0);
            ringPos = 0;
            for (double& v : out) v = 0.0;
            cutSupply[0] = cutSupply[1] = -1.0;
        }

        static double cutoffVgk(double s) noexcept
        {
            // Grid voltage relative to the cathode below which the plate
            // current is < 1e-10 A (plate effect < 10 uV) at vpk ~= s.
            constexpr double eps = 1e-10;
            const double e1 = std::pow(eps * LoadTable::kKg1 / 2.0, 1.0 / LoadTable::kEx);
            const double u = std::log(std::expm1(e1 * LoadTable::kKp / s));
            return std::sqrt(LoadTable::kKvb + s * s) * (u / LoadTable::kKp - 1.0 / LoadTable::kMu);
        }

        static double currentSlow(double s, double g) noexcept
        {
            // Frozen-cathode load line outside the table: Ip = Koren(s - RL Ip, g).
            double i = 0.5 * std::max(s, 0.0) / kRL, p = 0.0, dp = 0.0, dg = 0.0;
            for (int it = 0; it < 60; ++it)
            {
                koren(s - kRL * i, g, p, dp, dg);
                const double next = std::clamp(i - (i - p) / (1.0 + kRL * dp), 0.0,
                                               std::max(s, 0.0) / kRL);
                if (std::abs(next - i) < 1e-16) { i = next; break; }
                i = next;
            }
            return i;
        }

        template <int N>
        void currents(const typename LoadTable::Slice& sl, double cathode,
                      const double* x, double* result) const noexcept
        {
            double g[N];
            for (int j = 0; j < N; ++j) g[j] = Clamp::value(x[j]) - cathode;
            for (int j = 0; j < N; ++j)
                result[j] = (sl.inside && g[j] < 1.0) ? sl.eval(g[j]) : currentSlow(sl.s, g[j]);
        }

        double current(const typename LoadTable::Slice& sl, double cathode, double x) const noexcept
        {
            double r;
            currents<1>(sl, cathode, &x, &r);
            return r;
        }

        static int classify(const double* lev, double v) noexcept
        {
            int k = 0;
            while (k < kLevels && v > lev[k]) ++k;
            return k;
        }
        static int classifyEdges(const double* lev, double v) noexcept
        {
            return v <= lev[0] ? 0 : (v > lev[kLevels - 1] ? kLevels : 1);
        }

        void addMoments(double tau, double weight, double y) noexcept
        {
            const double d = tau - 0.5;
            double t = weight * y;
            for (int p = 0; p < kOrder; ++p) { moments[p] += t; t *= d; }
        }
        void addConstantMoments(double a, double b, double y) noexcept
        {
            const double pa = a - 0.5, pb = b - 0.5;
            double ea = pa, eb = pb;
            for (int p = 0; p < kOrder; ++p)
            {
                moments[p] += y * (eb - ea) / (p + 1);
                ea *= pa;
                eb *= pb;
            }
        }
        /// Moments of sum_e c[e] xi^e over the piece [a, a + h].
        void addPolynomialMoments(double a, double h, const double* c, int degree) noexcept
        {
            static constexpr double inv[16] = {1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6,
                1.0 / 7, 1.0 / 8, 1.0 / 9, 1.0 / 10, 1.0 / 11, 1.0 / 12, 1.0 / 13, 1.0 / 14,
                1.0 / 15, 1.0 / 16};
            static constexpr double binom[kOrder][kOrder] = {{1}, {1, 1}, {1, 2, 1},
                {1, 3, 3, 1}, {1, 4, 6, 4, 1}, {1, 5, 10, 10, 5, 1}};
            double sums[kOrder], hk = 1.0, dk[kOrder];
            dk[0] = 1.0;
            for (int k = 1; k < kOrder; ++k) dk[k] = dk[k - 1] * (a - 0.5);
            for (int k = 0; k < kOrder; ++k)
            {
                double acc = 0.0;
                for (int e = 0; e <= degree; ++e) acc += c[e] * inv[e + k];
                sums[k] = acc * hk;
                hk *= h;
            }
            for (int p = 0; p < kOrder; ++p)
            {
                double acc = 0.0;
                for (int k = 0; k <= p; ++k) acc += binom[p][k] * dk[p - k] * sums[k];
                moments[p] += h * acc;
            }
        }

        /// Interstage voltage on one piece: a degree-5 polynomial in the piece
        /// coordinate plus exact terms for modes of intermediate stiffness.
        struct Trajectory
        {
            double c[6] {};
            int exactCount = 0;
            double q0[3] {}, z[3] {}, bt[3] {}, g[3] {};
            double u[4] {};
            int uTerms = 1;

            double exactTerm(int k, double xi, double& derivative) const noexcept
            {
                double p[5];
                detail::tubePreampPhi(z[k] * xi, p);
                static constexpr double fact[4] = {1.0, 1.0, 2.0, 6.0};
                double forced = 0.0, xp = xi, ui = 0.0, xe = 1.0;
                for (int e = 0; e < uTerms; ++e)
                {
                    forced += u[e] * fact[e] * xp * p[e + 1];
                    xp *= xi;
                    ui += u[e] * xe;
                    xe *= xi;
                }
                const double qv = p[0] * q0[k] + bt[k] * forced;
                derivative = g[k] * (z[k] * qv + bt[k] * ui);
                return g[k] * qv;
            }
            double operator()(double xi) const noexcept
            {
                double v = ((((c[5] * xi + c[4]) * xi + c[3]) * xi + c[2]) * xi + c[1]) * xi + c[0];
                for (int k = 0; k < exactCount; ++k)
                {
                    double dq;
                    v += exactTerm(k, xi, dq);
                }
                return v;
            }
            double eval(double xi, double& derivative) const noexcept
            {
                double v = c[5], d = 0.0;
                for (int p = 4; p >= 0; --p)
                {
                    d = d * xi + v;
                    v = v * xi + c[p];
                }
                for (int k = 0; k < exactCount; ++k)
                {
                    double dq;
                    v += exactTerm(k, xi, dq);
                    d += dq;
                }
                derivative = d;
                return v;
            }
        };

        /// Propagates the modal state over a piece of length h (internal
        /// samples) whose stage-1 plate voltage is sum_e u[e] xi^e, and
        /// returns the tone-circuit output on the piece.
        void propagate(double h, const double* u, int terms, Trajectory& tr) noexcept
        {
            static constexpr double fact[4] = {1.0, 1.0, 2.0, 6.0};
            tr.exactCount = 0;
            tr.uTerms = terms;
            for (int e = 0; e < 4; ++e) tr.u[e] = e < terms ? u[e] : 0.0;
            for (double& x : tr.c) x = 0.0;
            const double u0 = u[0], u0p = terms > 1 ? u[1] : 0.0;
            double u1 = 0.0, u1p = 0.0;
            for (int e = 0; e < terms; ++e)
            {
                u1 += u[e];
                u1p += e * u[e];
            }
            // Non-stiff modal sum: exact values and first/second derivatives
            // at both ends (from dq/dxi = z q + bt u), then a quintic Hermite.
            double h0 = 0.0, h0p = 0.0, h0pp = 0.0, h1 = 0.0, h1p = 0.0, h1pp = 0.0;
            for (int m = 0; m < 3; ++m)
            {
                const double z = modes.lambda[m] * h, bt = modes.beta[m] * h, g = modes.gamma[m];
                const double az = std::abs(z);
                if (az >= kStiffZ)
                {
                    // Slow manifold of dq/dxi = z q + bt u:
                    // q = -(bt/z) sum_k u^(k)(xi) / z^k (boundary layer < e^-40).
                    const double iz = 1.0 / z;
                    double der[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int e = 0; e < terms; ++e) der[e] = u[e];
                    double izp = 1.0, endValue = 0.0;
                    for (int k = 0; k < terms; ++k)
                    {
                        double sumAtEnd = 0.0;
                        for (int e = 0; e < 4; ++e)
                        {
                            tr.c[e] += -g * bt * iz * izp * der[e];
                            sumAtEnd += der[e];
                        }
                        endValue += izp * sumAtEnd;
                        for (int e = 0; e < 3; ++e) der[e] = der[e + 1] * (e + 1);
                        der[3] = 0.0;
                        izp *= iz;
                    }
                    q[m] = -bt * iz * endValue;
                    continue;
                }
                double p[5];
                detail::tubePreampPhi(z, p);
                double forced = 0.0;
                for (int e = 0; e < terms; ++e) forced += u[e] * fact[e] * p[e + 1];
                const double qEnd = p[0] * q[m] + bt * forced;
                if (az > kSmallZ)
                {
                    const int k = tr.exactCount++;
                    tr.q0[k] = q[m];
                    tr.z[k] = z;
                    tr.bt[k] = bt;
                    tr.g[k] = g;
                    q[m] = qEnd;
                    continue;
                }
                const double d0 = z * q[m] + bt * u0, d1 = z * qEnd + bt * u1;
                h0 += g * q[m];
                h0p += g * d0;
                h0pp += g * (z * d0 + bt * u0p);
                h1 += g * qEnd;
                h1p += g * d1;
                h1pp += g * (z * d1 + bt * u1p);
                q[m] = qEnd;
            }
            const double quad = 0.5 * h0pp;
            const double r0 = h1 - (h0 + h0p + quad), r1 = h1p - (h0p + 2.0 * quad), r2 = h1pp - 2.0 * quad;
            tr.c[0] += h0;
            tr.c[1] += h0p;
            tr.c[2] += quad;
            tr.c[3] += 10.0 * r0 - 4.0 * r1 + 0.5 * r2;
            tr.c[4] += -15.0 * r0 + 7.0 * r1 - r2;
            tr.c[5] += 6.0 * r0 - 3.0 * r1 + 0.5 * r2;
            for (int e = 0; e < terms; ++e) tr.c[e] += modes.direct * u[e];
        }

        /// Stage 2 over the piece [a, a + h] (interval coordinates).
        void stage2(double a, double h, const Trajectory& tr) noexcept
        {
            double cuts[16];
            int count = 0;
            cuts[count++] = 0.0;
            const int samples = h > 0.5 ? 4 : 2;
            double sv[5];
            double lo = std::numeric_limits<double>::max(), hi = -lo;
            for (int k = 0; k <= samples; ++k)
            {
                sv[k] = kInterstage * tr(double(k) / samples);
                lo = std::min(lo, sv[k]);
                hi = std::max(hi, sv[k]);
            }
            const bool edges = hi - lo < kKneeRange;
            const auto cls = [&](double v) { return edges ? classifyEdges(lev2, v) : classify(lev2, v); };
            double pt = 0.0, pv = sv[0];
            int pc = cls(pv);
            const int first = pc;
            bool changed = false;
            for (int k = 1; k <= samples; ++k)
            {
                const double t = double(k) / samples, v = sv[k];
                const int cc = cls(v);
                if (cc != pc)
                {
                    changed = true;
                    const int low = std::min(cc, pc), high = std::max(cc, pc);
                    for (int r = 0; r < high - low; ++r)
                    {
                        int li = cc > pc ? low + r : high - 1 - r;
                        if (edges) li = li == 0 ? 0 : kLevels - 1;
                        const double level = lev2[li];
                        double tt = pt + (t - pt) * (pv - level) / (pv - v);
                        for (int it = 0; it < 2; ++it)
                        {
                            double dv;
                            const double f = kInterstage * tr.eval(tt, dv) - level;
                            dv *= kInterstage;
                            if (dv == 0.0) break;
                            const double next = tt - f / dv;
                            if (!(next > pt && next < t)) break;
                            tt = next;
                        }
                        if (tt > cuts[count - 1] + 1e-12 && count < 14) cuts[count++] = tt;
                    }
                }
                pt = t;
                pv = v;
                pc = cc;
            }
            cuts[count++] = 1.0;
            for (int k = 0; k + 1 < count; ++k)
            {
                const double xa = cuts[k], xb = cuts[k + 1], len = xb - xa;
                if (len <= 0.0) continue;
                const int cm = changed ? cls(kInterstage * tr(0.5 * (xa + xb))) : first;
                if (cm == 0 || cm == kLevels)
                {
                    double i2 = 0.0;
                    if (cm == kLevels)
                    {
                        if (!haveHi2) { hi2 = current(slice2, a2, 1e3); haveHi2 = true; }
                        i2 = hi2;
                    }
                    integral2 += h * len * i2;
                    addConstantMoments(a + h * xa, a + h * xb, supply - kRL * i2 - vp2);
                    continue;
                }
                if (h * len > kLongPiece)
                {
                    // Long smooth piece: a cubic interpolant keeps harmonics
                    // near 0.8 cycles/sample from leaking past the kernel.
                    double xs[4], is[4], y[4], poly[4];
                    for (int j = 0; j < 4; ++j) xs[j] = kInterstage * tr(xa + len * kG4x[j]);
                    currents<4>(slice2, a2, xs, is);
                    for (int j = 0; j < 4; ++j)
                    {
                        integral2 += h * len * kG4w[j] * is[j];
                        y[j] = supply - kRL * is[j] - vp2;
                    }
                    for (int e = 0; e < 4; ++e)
                        poly[e] = lagrange4[0][e] * y[0] + lagrange4[1][e] * y[1]
                                + lagrange4[2][e] * y[2] + lagrange4[3][e] * y[3];
                    addPolynomialMoments(a + h * xa, h * len, poly, 3);
                    continue;
                }
                if (h * len < kShortPiece)
                {
                    // Short piece: two nodes and the exact moments of their
                    // linear interpolant.
                    constexpr double g0 = 0.21132486540518713, g1 = 0.78867513459481287;
                    double xs[2] = {kInterstage * tr(xa + len * g0), kInterstage * tr(xa + len * g1)}, is[2];
                    currents<2>(slice2, a2, xs, is);
                    integral2 += h * len * 0.5 * (is[0] + is[1]);
                    const double y0 = supply - kRL * is[0] - vp2, y1 = supply - kRL * is[1] - vp2;
                    double poly[2];
                    poly[1] = (y1 - y0) / (g1 - g0);
                    poly[0] = y0 - poly[1] * g0;
                    addPolynomialMoments(a + h * xa, h * len, poly, 1);
                    continue;
                }
                double xs[3], is[3];
                for (int j = 0; j < 3; ++j) xs[j] = kInterstage * tr(xa + len * kG3x[j]);
                currents<3>(slice2, a2, xs, is);
                // Exact moments of the plate interpolant through the three
                // nodes: Gauss weights alone are not exact for the quintic
                // kernel weights times a curved plate trajectory.
                double y[3], poly[3];
                for (int j = 0; j < 3; ++j)
                {
                    integral2 += h * len * kG3w[j] * is[j];
                    y[j] = supply - kRL * is[j] - vp2;
                }
                for (int e = 0; e < 3; ++e)
                    poly[e] = lagrange3[0][e] * y[0] + lagrange3[1][e] * y[1] + lagrange3[2][e] * y[2];
                addPolynomialMoments(a + h * xa, h * len, poly, 2);
            }
        }

        void plate(double a, double h, const double* u, int terms) noexcept
        {
            Trajectory tr;
            propagate(h, u, terms, tr);
            if (stages > 1)
                stage2(a, h, tr);
            else if (tr.exactCount == 0)
                addPolynomialMoments(a, h, tr.c, 5);
            else
            {
                // Intermediate-stiffness mode present: interpolate the
                // tone-circuit output at the nodes, then take exact moments.
                const double y[3] = {tr(kG3x[0]), tr(kG3x[1]), tr(kG3x[2])};
                double poly[3];
                for (int e = 0; e < 3; ++e)
                    poly[e] = lagrange3[0][e] * y[0] + lagrange3[1][e] * y[1] + lagrange3[2][e] * y[2];
                addPolynomialMoments(a, h, poly, 2);
            }
        }

        double process(double x, int numStages, double sagR) noexcept
        {
            stages = numStages;
            const int taps = design->taps;
            ring[static_cast<size_t>(ringPos)] = x;
            ringPos = ringPos + 1 == taps ? 0 : ringPos + 1;
            interval(sagR);
            const double result = out[0];
            for (int k = 0; k + 1 < kOrder; ++k) out[k] = out[k + 1];
            out[kOrder - 1] = 0.0;
            return result;
        }

        void interval(double sagR) noexcept
        {
            // Stage-1 grid polynomial on the interval [n, n+1].
            const int taps = design->taps, deg = design->degree;
            double c[16] {};
            int idx = ringPos;
            for (int j = 0; j < taps; ++j)
            {
                const double v = ring[static_cast<size_t>(idx)];
                idx = idx + 1 == taps ? 0 : idx + 1;
                const double* row = design->farrow + j * (deg + 1);
                for (int p = 0; p <= deg; ++p) c[p] += row[p] * v;
            }
            const auto xAt = [&](double tau) {
                double v = c[deg];
                for (int p = deg - 1; p >= 0; --p) v = v * tau + c[p];
                return v;
            };
            // Slow states frozen at the predicted interval midpoint.
            const double iPrev = i1Prev + (stages > 1 ? i2Prev : 0.0);
            supply = kBplus - sagR * (ipLP + 0.5 * (1.0 - sagDecay) * (iPrev - ipLP));
            a1 = vk1 + 0.5 * (1.0 - cathodeDecay) * (i1Prev * kRk - vk1);
            a2 = vk2 + 0.5 * (1.0 - cathodeDecay) * (i2Prev * kRk - vk2);
            s1 = supply - a1;
            s2 = supply - a2;
            slice1.set(*table, s1);
            slice2.set(*table, s2);
            if (std::abs(s1 - cutSupply[0]) > 0.05) { cutSupply[0] = s1; cutVgk[0] = cutoffVgk(s1); }
            if (std::abs(s2 - cutSupply[1]) > 0.05) { cutSupply[1] = s2; cutVgk[1] = cutoffVgk(s2); }
            lev1[0] = a1 + cutVgk[0]; lev1[1] = a1 - 3.0; lev1[2] = a1 - 1.0; lev1[3] = 1.0; lev1[4] = 6.0;
            lev2[0] = a2 + cutVgk[1]; lev2[1] = a2 - 3.0; lev2[2] = a2 - 1.0; lev2[3] = 1.0; lev2[4] = 6.0;
            for (int j = 1; j < kLevels; ++j)
            {
                lev1[j] = std::max(lev1[j], lev1[j - 1] + 1e-6);
                lev2[j] = std::max(lev2[j], lev2[j - 1] + 1e-6);
            }
            // Split the interval where the grid crosses the stage-1 levels;
            // interior knees only when the interval spans at least 1 V.
            double qv[5];
            double lo = std::numeric_limits<double>::max(), hi = -lo;
            for (int k = 0; k <= 4; ++k)
            {
                qv[k] = xAt(0.25 * k);
                lo = std::min(lo, qv[k]);
                hi = std::max(hi, qv[k]);
            }
            const bool edges = hi - lo < kKneeRange;
            const auto cls = [&](double v) { return edges ? classifyEdges(lev1, v) : classify(lev1, v); };
            double cuts[kMaxPieces];
            int count = 0;
            cuts[count++] = 0.0;
            double pt = 0.0, pv = qv[0];
            int pc = cls(pv);
            const int first = pc;
            bool changed = false;
            for (int k = 1; k <= 4; ++k)
            {
                const double t = 0.25 * k, v = qv[k];
                const int cc = cls(v);
                if (cc != pc)
                {
                    changed = true;
                    const int low = std::min(cc, pc), high = std::max(cc, pc);
                    for (int r = 0; r < high - low; ++r)
                    {
                        int li = cc > pc ? low + r : high - 1 - r;
                        if (edges) li = li == 0 ? 0 : kLevels - 1;
                        const double level = lev1[li];
                        double tt = pt + (t - pt) * (pv - level) / (pv - v);
                        for (int it = 0; it < 2; ++it)
                        {
                            double f = c[deg], df = 0.0;
                            for (int p = deg - 1; p >= 0; --p) { df = df * tt + f; f = f * tt + c[p]; }
                            f -= level;
                            if (df == 0.0) break;
                            const double next = tt - f / df;
                            if (!(next > pt && next < t)) break;
                            tt = next;
                        }
                        if (tt > cuts[count - 1] + 1e-12 && count < kMaxPieces - 2) cuts[count++] = tt;
                    }
                }
                pt = t;
                pv = v;
                pc = cc;
            }
            cuts[count++] = 1.0;
            integral1 = integral2 = 0.0;
            for (double& m : moments) m = 0.0;
            haveHi1 = haveHi2 = false;
            for (int k = 0; k + 1 < count; ++k)
            {
                const double a = cuts[k], b = cuts[k + 1], h = b - a;
                if (h <= 0.0) continue;
                const int cm = changed ? cls(xAt(0.5 * (a + b))) : first;
                if (cm == 0 || cm == kLevels)
                {
                    // Stage 1 saturated: constant plate voltage on the piece.
                    double i1 = 0.0;
                    if (cm == kLevels)
                    {
                        if (!haveHi1) { hi1 = current(slice1, a1, 1e3); haveHi1 = true; }
                        i1 = hi1;
                    }
                    integral1 += h * i1;
                    const double u[1] = {supply - kRL * i1 - vp1};
                    plate(a, h, u, 1);
                    continue;
                }
                double xs[4], is[4];
                {
                    double t0 = a + h * kG4x[0], t1 = a + h * kG4x[1];
                    double t2 = a + h * kG4x[2], t3 = a + h * kG4x[3];
                    double v0 = c[deg], v1 = c[deg], v2 = c[deg], v3 = c[deg];
                    for (int p = deg - 1; p >= 0; --p)
                    {
                        v0 = v0 * t0 + c[p];
                        v1 = v1 * t1 + c[p];
                        v2 = v2 * t2 + c[p];
                        v3 = v3 * t3 + c[p];
                    }
                    xs[0] = v0; xs[1] = v1; xs[2] = v2; xs[3] = v3;
                }
                currents<4>(slice1, a1, xs, is);
                double y[4];
                for (int j = 0; j < 4; ++j)
                {
                    y[j] = supply - kRL * is[j] - vp1;
                    integral1 += h * kG4w[j] * is[j];
                }
                double u[4];
                for (int e = 0; e < 4; ++e)
                    u[e] = lagrange4[0][e] * y[0] + lagrange4[1][e] * y[1]
                         + lagrange4[2][e] * y[2] + lagrange4[3][e] * y[3];
                plate(a, h, u, 4);
            }
            // Exact exponential updates with the interval-average currents.
            vk1 = vk1 * cathodeDecay + (1.0 - cathodeDecay) * kRk * integral1;
            if (stages > 1)
            {
                vk2 = vk2 * cathodeDecay + (1.0 - cathodeDecay) * kRk * integral2;
                i2Prev = integral2;
            }
            ipLP = ipLP * sagDecay + (1.0 - sagDecay) * (integral1 + (stages > 1 ? integral2 : 0.0));
            i1Prev = integral1;
            for (int k = 0; k < kOrder; ++k)
            {
                const auto& piece = design->kernel[static_cast<size_t>(k)];
                double acc = 0.0;
                for (int p = 0; p < kOrder; ++p) acc += piece[static_cast<size_t>(p)] * moments[p];
                out[k] += acc;
            }
        }
    };

    /** @brief Full per-channel circuit: two stages + FMV tone stack + sag. */
    struct ChannelState
    {
        explicit ChannelState(double fs2In, const LoadTable* table,
                              const detail::TubePreampCoreDesign* designIn)
            : design(designIn), fmv(38e3, 1e6) // fixed source impedance
        {
            stage1.table = table;
            stage2.table = table;
            stage1.fs2 = fs2In;
            stage2.fs2 = fs2In;
            fs2 = fs2In;
            fmv.prepare(fs2In);
            if (design)
            {
                core.init(fs2In, table, design);
                core.setTone(fmv.analogStateSpace());
                compensation.prepare(static_cast<int>(design->compensation.size()), 1);
                compensation.setCoefficients(design->compensation);
            }
        }

        /** Settles the DC operating point CONSISTENTLY with the sagged
         *  supply: fixed point on B+ = kBplus - sagR*(Ip1+Ip2). Settling at
         *  the stiff kBplus while processing immediately applies the
         *  sag drop produced a ~19 V supply step at sag 0.3 - an audible
         *  activation thump, and it sat inside the old 10 ms calibration
         *  window, burying the whole wet path ~20 dB under unity. */
        void reset(double sagR, int numStages) noexcept
        {
            // Fixed point over the ACTIVE stage count: processing only
            // draws current from the stages in use, so seeding ipLP with both
            // stages' current at 1-stage settings left a ~70 ms sag transient.
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
            if (design)
            {
                core.reset(sagR, numStages);
                compensation.reset();
            }
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
            if (design) core.setTone(fmv.analogStateSpace());
        }

        /** Total plate current behind the sag meter. */
        [[nodiscard]] double supplyCurrent() const noexcept { return design ? core.ipLP : ipLP; }

        template <bool Compensate = true>
        [[nodiscard]] double processSample(double vgIn, int numStages, double sagR) noexcept
        {
            double v;
            if (design)
                v = core.process(vgIn, numStages, sagR);
            else
            {
                // Supply sag: B+ droops with smoothed total plate current.
                const double sagAlpha = 1.0 - std::exp(-1.0 / (0.07 * fs2));
                const double iTotal = stage1.ip + (numStages > 1 ? stage2.ip : 0.0);
                ipLP += sagAlpha * (iTotal - ipLP);
                const double bplusEff = kBplus - sagR * ipLP;
                // Stage 1 -> FMV tone stack -> (stage 2).
                v = stage1.processSample(vgIn, bplusEff);
                v = fmv.processSample(v);
                if (numStages > 1)
                    v = stage2.processSample(v * kInterstage, bplusEff);
            }

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
            // and the tone controls act RELATIVE to flat. This linear stage
            // adds no harmonics.
            auto& fl = flatten[numStages > 1 ? 1 : 0];
            out = fl[0].processSample(out, 0);
            out = fl[1].processSample(out, 0);
            out = fl[2].processSample(out, 0);
            if constexpr (!Compensate) return out;
            if (!design) return out;
            return compensation.processSample(out, 0);
        }

        void compensateBlock(double* data, int count) noexcept
        {
            // Core block processing hoists coefficient publication/atomics
            // out of the inner loop. The calibration uses the same FIR.
            compensation.processBlock(AudioBufferView<double>(&data, 1, count));
        }

        double fs2 = 96000.0;
        TriodeStage stage1, stage2;
        double ipLP = 1.6e-3;
        double outHpX = 0.0, outHpY = 0.0;
        std::array<std::array<Biquad<double, 1>, 3>, 2> flatten;   ///< Per stage count.
        const detail::TubePreampCoreDesign* design;
        ContinuousCore core;
        FIRFilter<double> compensation;

        wdf::ToneStackFMV<double> fmv;   ///< Bassman stack (R-type WDF / analog form).
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
            ChannelState cal(fs2_, loadTable_.get(), design_.get());
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
            ChannelState sweep(fs2_, loadTable_.get(), design_.get());
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
    std::unique_ptr<detail::TubePreampCoreDesign> design_;   ///< Continuous core (factor >= 2).
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
