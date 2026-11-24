// DSParkLab - Effect Registry
// Factory functions that create EffectSlot wrappers for all DSPark processors.

#pragma once

#include "../DSPark.h"
#include "EffectSlot.h"
#include "Presets.h"

#include <array>
#include <complex>
#include <memory>
#include <string>
#include <utility>

namespace dsplab {

// The display curves of the spectral processors sit at points evenly in log
// frequency from 20 Hz to 20 kHz (dspark::SpectralReadout); this reads them
// at the frequencies of the screen, between two points linearly.
template <typename ValueAt>
inline void readDisplayCurve(ValueAt valueAt, const float* f, float* out, int n)
{
    for (int i = 0; i < n; ++i)
        out[i] = static_cast<float>(dspark::SpectralReadout<float>::between(static_cast<double>(f[i]), valueAt));
}

// =============================================================================
// FILTERS
// =============================================================================

inline EffectSlot makeFilterEngine()
{
    auto p = std::make_shared<dspark::FilterEngine<float>>();
    EffectSlot s;
    s.name = "Filter"; s.category = "Filters";
    s.addChoice("Type", {"LowPass","HighPass","BandPass","Peak","LowShelf","HighShelf","Notch","AllPass","Tilt"}, 0);
    s.addSlider("Frequency", 20, 20000, 1000, "Hz", true);   // Visible mid default so the node isn't hidden at the edge
    s.addSlider("Resonance", 0.1f, 20, 0.707f, "Q", true);
    s.addSlider("Gain", -24, 24, 0, "dB");
    s.addSlider("Slope", 6, 48, 12, "dB/oct");
    s.addSlider("Nonlinearity", 0, 1, 0, "");
    s.addChoice("Structure", {"Direct Form","Ladder"}, 0);   // 6: ladder = passive under fast modulation
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        using Shape = dspark::FilterEngine<float>::Shape;
        switch(i) {
            case 0: {
                // Switch topology only. setShape() preserves the user's
                // current Frequency / Resonance / Gain slider values, so
                // changing the dropdown doesn't reset what they had dialled in.
                static constexpr Shape kShapes[] = {
                    Shape::LowPass, Shape::HighPass, Shape::BandPass,
                    Shape::Peak,    Shape::LowShelf, Shape::HighShelf,
                    Shape::Notch,   Shape::AllPass,  Shape::Tilt
                };
                int idx = std::clamp(static_cast<int>(v), 0,
                                     static_cast<int>(std::size(kShapes)) - 1);
                p->setShape(kShapes[idx]);
                break;
            }
            case 1: p->setFrequency(v); break;
            case 2: p->setResonance(v); break;
            case 3: p->setGain(v); break;
            case 4: {
                // Slope only applies to LP / HP - for other shapes setShape()
                // already pinned numStages_ = 1, so this is a no-op there.
                int slope = std::clamp(static_cast<int>(v), 6, 48);
                // Round to the nearest 6 dB/oct step so cascade size matches.
                slope = (slope / 6) * 6;
                if (slope < 6) slope = 6;
                // Re-apply current shape with the new slope. Doesn't disturb
                // freq/res/gain.
                p->setShape(p->getShape(), slope);
                break;
            }
            case 5: p->setNonlinearity(v); break;
            case 6: p->setStructure(static_cast<dspark::FilterEngine<float>::Structure>(static_cast<int>(v))); break;
        }
    };
    // Interactive response curve. Reconstructs the exact cascade from the public
    // BiquadCoeffs factories + FilterEngine::cascadeForSlope (single source of truth).
    s.magnitudeFn = [](const float* f, float* mdb, int n, double sr, const float* v) {
        using BC = dspark::BiquadCoeffs;
        using FE = dspark::FilterEngine<float>;
        const int type = static_cast<int>(v[0]);
        const float freq = v[1], Q = std::max(0.05f, v[2]), gain = v[3];
        int slope = std::clamp((static_cast<int>(v[4]) / 6) * 6, 6, 48);
        BC st[5]; int ns = 0;
        auto add = [&](const BC& c){ if (ns < 5) st[ns++] = c; };
        switch (type) {
            case 0: case 1: {  // LowPass / HighPass - Butterworth cascade for the slope
                auto info = FE::cascadeForSlope(slope, Q);  // user Q scales the final stage, like the engine
                const bool lp = (type == 0);
                if (info.hasFirstOrder) add(lp ? BC::makeFirstOrderLowPass(sr, freq) : BC::makeFirstOrderHighPass(sr, freq));
                for (int s = 0; s < info.numSecondOrder; ++s) add(lp ? BC::makeLowPass(sr, freq, info.qValues[s]) : BC::makeHighPass(sr, freq, info.qValues[s]));
                break;
            }
            case 2: add(BC::makeBandPass(sr, freq, Q)); break;
            case 3: add(BC::makePeak(sr, freq, Q, gain)); break;
            case 4: add(BC::makeLowShelf(sr, freq, gain)); break;
            case 5: add(BC::makeHighShelf(sr, freq, gain)); break;
            case 6: add(BC::makeNotch(sr, freq, Q)); break;
            case 7: add(BC::makeAllPass(sr, freq, Q)); break;
            case 8: add(BC::makeTilt(sr, freq, gain)); break;
        }
        for (int i = 0; i < n; ++i) {
            double mag = 1.0;
            for (int s = 0; s < ns; ++s) mag *= st[s].getMagnitude(static_cast<double>(f[i]), sr);
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    s.curveNodes = { {1, 3, 2} };  // X=freq, Y=gain (Peak/Shelf/Tilt), wheel=Q
    return s;
}

inline EffectSlot makeEqualizer()
{
    using EQ = dspark::Equalizer<float>;
    auto p = std::make_shared<EQ>();
    EffectSlot s;
    s.name = "Equalizer"; s.category = "Filters";
    // Per band: Freq, Gain, Q, Type, Slope. Type/Slope expose the full
    // framework BandType set (shelves, cuts, notch, bandpass, tilt) - they
    // were missing from the registry, leaving the EQ bells-only.
    static constexpr float kDefFreq[4] = { 100, 500, 2000, 8000 };
    for (int b = 0; b < 4; ++b)
    {
        char nm[24];
        std::snprintf(nm, sizeof(nm), "Band %d Freq", b + 1);
        s.addSlider(nm, 20, 20000, kDefFreq[b], "Hz", true);
        std::snprintf(nm, sizeof(nm), "Band %d Gain", b + 1);
        s.addSlider(nm, -24, 24, 0, "dB");
        std::snprintf(nm, sizeof(nm), "Band %d Q", b + 1);
        s.addSlider(nm, 0.1f, 10, 1, "");
        std::snprintf(nm, sizeof(nm), "Band %d Type", b + 1);
        s.addChoice(nm, {"Peak","Low Shelf","High Shelf","Low Pass","High Pass",
                         "Notch","Band Pass","Tilt","All Pass","Constant Tilt",
                         "Low Pass Wall","High Pass Wall"}, 0);
        std::snprintf(nm, sizeof(nm), "Band %d Slope", b + 1);
        // Cuts only. Whole steps of 6 from 6 to 48 with the cookbook engine;
        // any value from 0 to 96 with Analog Response on.
        s.addSlider(nm, 0, 96, 12, "dB/oct");
    }
    // Global controls (indices 20/21, appended so band indices stay stable):
    // Phase Mode switches the IIR cascade for the FFT overlap-save engine
    // (adds maxBlockSize of latency, audible as a small delay here) or for the
    // prototype-phase mode (the bands from their analog prototypes plus a short
    // correction of their phase, 32 samples late), and
    // Matched Bells selects the analog-matched de-cramped design for Peak bands
    // (audible on high-frequency bells; the curve follows).
    s.addChoice("Phase Mode", {"Minimum","Linear","Prototype"}, 0);
    s.addToggle("Matched Bells", true);
    // Analog Response (index 22): every band designed from its analog prototype
    // (no shrinking towards Nyquist, in both phase modes), with glided changes.
    // Band N Order (23..26): bells, shelves, notches, band-passes and all-passes
    // of order 1 to 16 (flat top, steeper sides); above 1 the band runs with the
    // analog response whatever the toggle says, as do the last four types.
    s.addToggle("Analog Response", false);
    for (int b = 0; b < 4; ++b) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "Band %d Order", b + 1);
        s.addSlider(nm, 1, 16, 1, "");
    }
    // Band N Channels (27..30): the channels the band acts on.
    for (int b = 0; b < 4; ++b) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "Band %d Channels", b + 1);
        s.addChoice(nm, {"Stereo","Left","Right","Mid","Side"}, 0);
    }
    // Structure (index 31): the minimum-phase bands as normalized ladders.
    s.addChoice("Structure", {"Direct Form","Ladder"}, 0);
    // Resonant Shelves (index 32): the Q of a shelf above 0.71 makes it rise
    // over its plateau next to the corner (and dip on the other side).
    s.addToggle("Resonant Shelves", false);
    // Loudness Compensation (index 33): a gain takes back the bands' change
    // of the K-weighted loudness of pink noise.
    s.addToggle("Loudness Compensation", false);
    // Type index -> framework enum. The UI groups cuts after the shelves;
    // the enum orders LowPass/HighPass there, so the mapping is direct.
    auto toType = [](float v) {
        return static_cast<EQ::BandType>(std::clamp(static_cast<int>(v), 0, 11));
    };
    auto toSlope = [](float v) { return std::clamp(v, 0.0f, 96.0f); };
    auto toOrder = [](float v) { return std::clamp(v, 1.0f, 16.0f); };   // whole or not
    // Equalizer docs require prepare() BEFORE setNumBands() so bands
    // know the sample rate when auto-spacing frequencies.
    s.prepareFn = [p](auto& sp) { p->prepare(sp); p->setNumBands(4); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, toType, toSlope, toOrder](int i, float v) {
        if (i == 22) { p->setAnalogResponse(v > 0.5f); return; }
        if (i == 31) { p->setStructure(static_cast<EQ::Structure>(std::clamp(static_cast<int>(v), 0, 1))); return; }
        if (i == 32) { p->setResonantShelves(v > 0.5f); return; }
        if (i == 33) { p->setLoudnessCompensation(v > 0.5f); return; }
        if (i >= 27 && i <= 30)
        {
            auto cfg = p->getBandConfig(i - 27);
            cfg.scope = static_cast<EQ::Scope>(std::clamp(static_cast<int>(v), 0, 4));
            p->setBand(i - 27, cfg);
            return;
        }
        if (i >= 23 && i <= 26)
        {
            auto cfg = p->getBandConfig(i - 23);
            cfg.order = toOrder(v);
            p->setBand(i - 23, cfg);
            return;
        }
        if (i == 20)
        {
            const int mode = static_cast<int>(v);
            p->setFilterMode(mode == 1 ? EQ::FilterMode::LinearPhase
                           : mode == 2 ? EQ::FilterMode::PrototypePhase
                                       : EQ::FilterMode::MinimumPhase);
            return;
        }
        if (i == 21) { p->setMatchedBells(v > 0.5f); return; }
        int band = i / 5;
        int param = i % 5;
        if (band >= 4) return;
        auto cfg = p->getBandConfig(band);
        switch(param) {
            case 0: cfg.frequency = v; break;
            case 1: cfg.gain = v; break;
            case 2: cfg.q = v; break;
            case 3: cfg.type = toType(v); break;
            case 4: cfg.slope = toSlope(v); break;
        }
        p->setBand(band, cfg);
    };
    // Interactive response curve drawn from the band's REAL sections
    // (buildBandSections), so every type, slope and order renders exactly what
    // the audio path applies in either engine. X=freq, Y=gain, wheel=Q.
    s.magnitudeFn = [p, toType, toSlope, toOrder](const float* f, float* mdb, int n, double sr, const float* v) {
        for (int i = 0; i < n; ++i) {
            double mag = 1.0;
            for (int b = 0; b < 4; ++b) {
                EQ::BandConfig cfg;
                cfg.frequency = v[b*5+0];
                cfg.gain      = v[b*5+1];
                cfg.q         = std::max(0.05f, v[b*5+2]);
                cfg.type      = toType(v[b*5+3]);
                cfg.slope     = toSlope(v[b*5+4]);
                cfg.order     = toOrder(v[23+b]);
                dspark::BiquadCoeffs stages[16];
                const int ns = p->buildBandSections(cfg, stages, 16);
                for (int k = 0; k < ns; ++k)
                    mag *= stages[k].getMagnitude(static_cast<double>(f[i]), sr);
            }
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    s.curveNodes = { {0,1,2}, {5,6,7}, {10,11,12}, {15,16,17} };
    s.liveLabel = "Response";
    s.group("Band 1", { 0, 1, 2, 3, 4, 23, 27 });
    s.group("Band 2", { 5, 6, 7, 8, 9, 24, 28 });
    s.group("Band 3", { 10, 11, 12, 13, 14, 25, 29 });
    s.group("Band 4", { 15, 16, 17, 18, 19, 26, 30 });
    s.group("Engine", { 20, 21, 22, 31, 32, 33 });
    return s;
}

inline EffectSlot makeLadderFilter()
{
    auto p = std::make_shared<dspark::LadderFilter<float>>();
    EffectSlot s;
    s.name = "Ladder Filter"; s.category = "Filters";
    // Character filter: audible default so moving the knob is obviously heard.
    s.addSlider("Cutoff", 20, 20000, 2000, "Hz", true);    // Default 2 kHz = clearly filtering
    s.addSlider("Resonance", 0, 1, 0, "");
    s.addSlider("Drive", 0, 20, 0, "dB");
    // Mode order MUST match dspark::LadderFilter<float>::Mode enum.
    // Enum: LP6=0, LP12=1, LP18=2, LP24=3, BP12=4, HP24=5
    s.addChoice("Mode", {"LP6","LP12","LP18","LP24","BP12","HP24"}, 3);  // Default LP24 (Moog-style)
    // A/B: Feedback (default) saturates only the resonance loop; Stages
    // saturates every stage (opt-in, solved per sample); Input saturates the
    // loop input and runs four sub-steps per sample (opt-in). Order MUST
    // match dspark::LadderFilter<float>::Saturation (Feedback=0, Stages=1,
    // Input=2).
    s.addChoice("Saturation", {"Feedback","Stages","Input"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setCutoff(v); break;
            case 1: p->setResonance(v); break;
            // Slider is labelled in dB; the setter takes a linear multiplier.
            case 2: p->setDrive(std::pow(10.0f, v / 20.0f)); break;
            case 3: p->setMode(static_cast<typename dspark::LadderFilter<float>::Mode>(static_cast<int>(v))); break;
            case 4: p->setSaturation(static_cast<typename dspark::LadderFilter<float>::Saturation>(static_cast<int>(v))); break;
        }
    };
    // The small-signal response of the ladder, from the same equations the
    // filter runs (four one-poles of the bilinear transform, the loop gain
    // k = 4 x resonance resolved without delay): L = 1 / (1 + s), the loop
    // input u = 1 / (1 + k L^4), and each mode's tap of the stages. Drive
    // only shapes the loop when it saturates, so it is not in the curve.
    s.magnitudeFn = [](const float* f, float* mdb, int n, double sr, const float* v) {
        using C = std::complex<double>;
        const double fc = std::clamp(static_cast<double>(v[0]), 20.0, 0.49 * sr);
        const double k = 4.0 * std::clamp(static_cast<double>(v[1]), 0.0, 1.0);
        const int mode = std::clamp(static_cast<int>(v[3]), 0, 5);
        const double wc = std::tan(3.14159265358979323846 * fc / sr);
        for (int i = 0; i < n; ++i)
        {
            const double fi = std::min(static_cast<double>(f[i]), 0.4999 * sr);
            const C sj(0.0, std::tan(3.14159265358979323846 * fi / sr) / wc);
            const C L = 1.0 / (1.0 + sj);
            const C u = 1.0 / (1.0 + k * L * L * L * L);
            C h;
            switch (mode)
            {
                case 0: h = L * u; break;
                case 1: h = L * L * u; break;
                case 2: h = L * L * L * u; break;
                case 3: h = L * L * L * L * u; break;
                case 4: h = (L - L * L * L) * u; break;
                default: { const C hp = 1.0 - L; h = hp * hp * hp * hp * u; break; }
            }
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, std::abs(h))));
        }
    };
    s.curveNodes = { {0, -1, -1} };   // X = cutoff (the node rides the curve)
    s.liveLabel = "Response (small signal)";
    return s;
}

inline EffectSlot makeStateVariableFilter()
{
    auto p = std::make_shared<dspark::StateVariableFilter<float>>();
    EffectSlot s;
    s.name = "SVF Filter"; s.category = "Filters";
    // Character filter: audible default so moving the knob is obviously heard.
    s.addSlider("Cutoff", 20, 20000, 2000, "Hz", true);    // Default 2 kHz = clearly filtering
    s.addSlider("Q", 0.1f, 20, 0.707f, "");
    s.addSlider("Gain", -24, 24, 0, "dB");
    s.addChoice("Mode", {"LowPass","HighPass","BandPass","Notch","AllPass","Bell","LowShelf","HighShelf"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setCutoff(v); break;
            case 1: p->setQ(v); break;
            case 2: p->setGain(v); break;
            case 3: p->setMode(static_cast<typename dspark::StateVariableFilter<float>::Mode>(static_cast<int>(v))); break;
        }
    };
    // The response of the state-variable filter, from the same equations it
    // runs: the bilinear transform pre-warped at the cutoff (the shelves by
    // the square root of the gain, as the filter does), damping R = 1 / (2Q)
    // (the bell's narrowed or widened by the gain), and each mode's mix of
    // the low-pass, band-pass and high-pass outputs.
    s.magnitudeFn = [](const float* f, float* mdb, int n, double sr, const float* v) {
        using C = std::complex<double>;
        const double pi = 3.14159265358979323846;
        const double fc = std::clamp(static_cast<double>(v[0]), 20.0, 0.499 * sr);
        const double R = 1.0 / (2.0 * std::max(0.1, static_cast<double>(v[1])));
        const double db = static_cast<double>(v[2]);
        const double A = std::pow(10.0, std::fabs(db) / 40.0);
        const double As = db >= 0.0 ? A : 1.0 / A;
        const int mode = std::clamp(static_cast<int>(v[3]), 0, 7);
        double g = std::tan(pi * fc / sr);
        if (mode == 6) g /= std::sqrt(As);
        if (mode == 7) g *= std::sqrt(As);
        const double Rb = db >= 0.0 ? R / A : R * A;
        const double r = mode == 5 ? Rb : R;
        for (int i = 0; i < n; ++i)
        {
            const double fi = std::min(static_cast<double>(f[i]), 0.4999 * sr);
            const C sj(0.0, std::tan(pi * fi / sr) / g);
            const C d = sj * sj + 2.0 * r * sj + 1.0;
            const C lp = 1.0 / d, bp = sj / d, hp = sj * sj / d;
            C h;
            switch (mode)
            {
                case 0: h = lp; break;
                case 1: h = hp; break;
                case 2: h = bp; break;
                case 3: h = lp + hp; break;
                case 4: h = lp + hp - 2.0 * R * bp; break;
                case 5: h = 1.0 + (db >= 0.0 ? A * A - 1.0 : 1.0 / (A * A) - 1.0) * 2.0 * Rb * bp; break;
                case 6: h = As * As * lp + As * 2.0 * R * bp + hp; break;
                default: h = lp + As * 2.0 * R * bp + As * As * hp; break;
            }
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, std::abs(h))));
        }
    };
    s.curveNodes = { {0, 2, 1} };   // X = cutoff, Y = gain (bell and shelves), wheel = Q
    s.liveLabel = "Response";
    return s;
}

inline EffectSlot makeDCBlocker()
{
    auto p = std::make_shared<dspark::DCBlocker<float>>();
    EffectSlot s;
    s.name = "DC Blocker"; s.category = "Filters";
    s.addSlider("Cutoff", 1, 100, 5, "Hz");
    s.addSlider("Order", 1, 10, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setCutoff(static_cast<float>(v)); break;
            case 1: p->setOrder(static_cast<int>(v)); break;
        }
    };
    return s;
}

// =============================================================================
// DYNAMICS
// =============================================================================

inline EffectSlot makeCompressor()
{
    auto p = std::make_shared<dspark::Compressor<float>>();
    EffectSlot s;
    s.name = "Compressor"; s.category = "Dynamics";
    s.addSlider("Threshold", -60, 0, 0, "dB");   // Default 0 dB = only compresses at full-scale (neutral)
    s.addSlider("Ratio", 1, 20, 4, ":1");
    s.addSlider("Attack", 0.1f, 100, 10, "ms");
    s.addSlider("Release", 10, 1000, 100, "ms");
    s.addSlider("Knee", 0, 30, 6, "dB");
    s.addSlider("Makeup", -12, 30, 0, "dB");
    // Order MUST match dspark::Compressor::AutoMakeupMode (Off=0, Static=1, Adaptive=2, Analytic=3).
    s.addChoice("Auto Makeup", {"Off","Static","Adaptive","Analytic"}, 0);
    s.addSlider("Mix", 0, 1, 1, "");
    // Detector order MUST match dspark::Compressor::DetectorType enum.
    // Enum: Peak=0, Rms=1, TruePeak=2, SplitPolarity=3, Hilbert=4, Hybrid=5
    s.addChoice("Detector", {"Peak","RMS","TruePeak","SplitPolarity","Hilbert","Hybrid"}, 0);
    // Character order MUST match dspark::Compressor::Character enum.
    // Enum: Clean=0, Opto=1, FET=2, Varimu=3, Layered=4
    s.addChoice("Character", {"Clean","Opto","FET","Varimu","Layered"}, 0);
    // Harmonic signature of the character (FET: 2nd order of the element;
    // Varimu: the gain through a push-pull pair).
    s.addSlider("Color", 0, 1, 0, "");
    // Order MUST match dspark::Compressor::Topology (FeedForward=0, FeedBack=1).
    s.addChoice("Topology", {"FeedForward","FeedBack"}, 0);              // index 11
    // A/B: Branching (default) or Decoupled (opt-in) clean ballistics. Order
    // MUST match dspark::Compressor::CleanBallistics (Branching=0, Decoupled=1).
    s.addChoice("Clean Ballistics", {"Branching","Decoupled"}, 0);       // index 12
    // Threshold counted from the key's own recent peak (index 13), a latency
    // kept for the lookahead and the lookahead inside it (14, 15), the share
    // of the gain change that is applied (16) and a range with no corner (17, 18).
    s.addToggle("Auto Threshold", false);
    s.addSlider("Max Lookahead", 0, 10, 0, "ms");
    s.addSlider("Lookahead", 0, 10, 0, "ms");
    s.addSlider("Amount", 0, 1, 1, "");
    s.addSlider("Range", 0, 60, 60, "dB");
    s.addToggle("Soft Range", false);
    // The control signal worked out between the samples (index 19). Order
    // MUST match dspark::Compressor::Precision (Standard=0, High=1).
    s.addChoice("Precision", {"Standard","High"}, 0);
    // Timing that follows the programme (indices 20 to 23): release after a
    // sustained reduction, attack by the distance to the target, both by the
    // spectral balance of the key, and the emphasis of the attack.
    s.addSlider("Program Release", 0, 1, 0, "");
    s.addSlider("Adaptive Attack", 0, 1, 0, "");
    s.addSlider("Spectral Timing", 0, 1, 0, "");
    s.addSlider("Attack Emphasis", -1, 1, 0, "");
    // The equalizer of the key (indices 24 to 31): listen to the key, a low
    // cut with its slope (20 Hz: off), a high cut (20 kHz: off), a bell and a
    // tilt about 1 kHz.
    s.addToggle("Key Listen", false);
    s.addSlider("Key Low Cut", 20, 2000, 20, "Hz");
    s.addSlider("Key Low Cut Slope", 6, 48, 12, "dB/oct");
    s.addSlider("Key High Cut", 500, 20000, 20000, "Hz");
    s.addSlider("Key Bell Freq", 40, 16000, 1000, "Hz");
    s.addSlider("Key Bell Gain", -24, 24, 0, "dB");
    s.addSlider("Key Bell Q", 0.3f, 10, 1, "");
    s.addSlider("Key Tilt", -6, 6, 0, "dB/oct");
    // The gain smoothed after the ballistics (index 32).
    s.addSlider("Gain Smoothing", 0, 50, 0, "ms");
    s.addToggle("Hear Removed", false);   // 33: what the compression takes out
    s.addChoice("Link Mode", {"Loudest","Dominant"}, 0);   // 34: the pair read as its dominant component
    s.addChoice("Opto Cell", {"Two stages","Three stages"}, 0); // 35: the Opto character with a medium memory
    s.addChoice("Colour Stage", {"Off","Before","After"}, 0);   // 36: the analog colour stage
    s.addSlider("Stage Drive", 0, 24, 6, "dB");                 // 37
    s.addSlider("Stage Emphasis", 0, 12, 6, "dB");              // 38
    s.addSlider("Stage Asymmetry", 0, 1, 0.3f, "");             // 39
    s.addSlider("Stage Drift", 0, 1, 0, "");                    // 40
    s.addSlider("Stage Sag", 0, 1, 0, "");                      // 41
    s.addSlider("Stage Noise", -140, -40, -140, "dB");          // 42
    s.addSlider("Hold", 0, 500, 0, "ms");                       // 43: the reduction kept after a peak
    s.addChoice("Hold Mode", {"Deepest","Every peak"}, 0);      // 44: the hold as a sliding window
    s.addChoice("Lookahead Shape", {"Lead","Ramp"}, 0);         // 45: the reduction arriving with the peak along a ramp
    s.addToggle("Feedback Keyed", false);                       // 46: a feedback loop closed around the key (lookahead and external key in feedback)
    s.addChoice("Mode", {"Downward","Upward","Both","Leveler"}, 0);   // 47: the curve of the gain computer
    s.addSlider("Lift Below", 0, 80, 30, "dB");                 // 48: Both: the lower threshold, under the main one
    s.addSlider("Lift Ratio", 1, 20, 2, ":1");                  // 49
    s.addSlider("Lift Range", 0, 40, 6, "dB");                  // 50: the most it raises
    s.addChoice("Range Shape", {"Corner","Asymptotic","Local"}, 0);   // 51: how the gain change meets the range
    s.addSlider("Release Landing", 0, 2000, 0, "ms");           // 52: the last decibels of a release come back slowly
    s.addSlider("Network Depth", 0, 12, 0, "dB");               // 53: only a share of a fast move of the gain at once
    s.addSlider("Network Corner", 1, 100, 10, "Hz");            // 54
    s.addSlider("Tone Low", -3, 3, 0, "dB");                    // 55: the tone that follows the reduction, under 200 Hz
    s.addSlider("Tone High", -3, 3, 0, "dB");                   // 56: and over 4 kHz
    s.addSlider("Control Leak", 0, 1, 0, "");                   // 57: the moves of the gain as low thumps
    s.addSlider("Detector Asymmetry", -1, 1, 0, "");            // 58: the two halves of the wave read differently
    s.addChoice("Route", {"Stereo","Mid + Side","Mid","Side","Mid keys Side","Side keys Mid"}, 0);   // 59
    // Where the key's low cut and its bell sit (60, 61). Order MUST match
    // dspark::Compressor<float>::KeyPlacement (Stereo=0, Mid=1, Side=2).
    s.addChoice("Key Low Cut On", {"Both channels","Middle","Sides"}, 0);
    s.addChoice("Key Bell On", {"Both channels","Middle","Sides"}, 0);
    s.addSlider("Mid/Side Focus", -1, 1, 0, "");   // 62: in Mid + Side, the lean to the middle (-1) or the sides (1)
    s.addSlider("Detector Drift", 0, 1, 0, "");    // 63: the detector's asymmetry wanders slowly at random
    using CompKeyBand = dspark::Compressor<float>::KeyBand;
    using CompKeyShape = dspark::Compressor<float>::KeyShape;
    auto key = std::make_shared<std::array<CompKeyBand, 4>>();
    (*key)[0].frequency = 20.0f;
    (*key)[1].frequency = 20000.0f;
    (*key)[2].shape = CompKeyShape::Bell;
    (*key)[2].q = 1.0f;
    (*key)[3].shape = CompKeyShape::Tilt;
    (*key)[3].slope = 0.0f;
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    auto thrKnob = std::make_shared<std::atomic<float>>(0.0f);
    s.setParamFn = [p, key, thrKnob](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); thrKnob->store(v, std::memory_order_relaxed); break;
            case 1: p->setRatio(v); break;
            case 2: p->setAttack(v); break;
            case 3: p->setRelease(v); break;
            case 4: p->setKnee(v); break;
            case 5: p->setMakeupGain(v); break;
            case 6: p->setAutoMakeup(static_cast<dspark::Compressor<float>::AutoMakeupMode>(static_cast<int>(v))); break;
            case 7: p->setMix(v); break;
            case 8: p->setDetector(static_cast<dspark::Compressor<float>::DetectorType>(static_cast<int>(v))); break;
            case 9: p->setCharacter(static_cast<dspark::Compressor<float>::Character>(static_cast<int>(v))); break;
            case 10: p->setCharacterColor(v); break;
            case 11: p->setTopology(static_cast<dspark::Compressor<float>::Topology>(static_cast<int>(v))); break;
            case 12: p->setCleanBallistics(static_cast<dspark::Compressor<float>::CleanBallistics>(static_cast<int>(v))); break;
            case 13: p->setAutoThreshold(v > 0.5f); break;
            case 14: p->setMaxLookahead(v); break;
            case 15: p->setLookahead(v); break;
            case 16: p->setAmount(v); break;
            case 17: p->setRange(v); break;
            case 18: p->setSoftRange(v > 0.5f); break;
            case 19: p->setPrecision(static_cast<dspark::Compressor<float>::Precision>(static_cast<int>(v))); break;
            case 20: p->setProgramRelease(v); break;
            case 21: p->setAdaptiveAttack(v); break;
            case 22: p->setSpectralTiming(v); break;
            case 23: p->setAttackEmphasis(v); break;
            case 24: p->setSidechainListen(v > 0.5f); break;
            case 25: case 26: case 27: case 28: case 29: case 30: case 31:
            {
                auto& k = *key;
                if (i == 25) k[0].frequency = v;
                if (i == 26) k[0].slope = v;
                if (i == 27) k[1].frequency = v;
                if (i == 28) k[2].frequency = v;
                if (i == 29) k[2].gainDb = v;
                if (i == 30) k[2].q = v;
                if (i == 31) k[3].slope = v;
                k[0].shape = k[0].frequency > 20.5f ? CompKeyShape::LowCut : CompKeyShape::Off;
                k[1].shape = k[1].frequency < 19999.0f ? CompKeyShape::HighCut : CompKeyShape::Off;
                for (int b = 0; b < 4; ++b) p->setSidechainBand(b, k[static_cast<size_t>(b)]);
                break;
            }
            case 32: p->setGainSmoothing(v); break;
            case 33: p->setRemovalListen(v > 0.5f); break;
            case 34: p->setLinkMode(static_cast<dspark::Compressor<float>::LinkMode>(static_cast<int>(v))); break;
            case 35: p->setOptoCell(static_cast<dspark::Compressor<float>::OptoCell>(static_cast<int>(v))); break;
            case 36: p->setCharacterStage(static_cast<dspark::Compressor<float>::StagePosition>(static_cast<int>(v))); break;
            case 37: case 38: case 39: case 40: case 41: case 42:
            {
                auto st = p->getCharacterStageSettings();
                if (i == 37) st.driveDb = v;
                if (i == 38) st.emphasisDb = v;
                if (i == 39) st.asymmetry = v;
                if (i == 40) st.drift = v;
                if (i == 41) st.sag = v;
                if (i == 42) st.noiseDb = v;
                p->setCharacterStageSettings(st);
                break;
            }
            case 43: p->setHoldTime(v); break;
            case 44: p->setHoldMode(static_cast<dspark::Compressor<float>::HoldMode>(static_cast<int>(v))); break;
            case 45: p->setLookaheadShape(static_cast<dspark::Compressor<float>::LookaheadShape>(static_cast<int>(v))); break;
            case 46: p->setFeedbackKeyed(v > 0.5f); break;
            case 47: p->setMode(static_cast<dspark::Compressor<float>::Mode>(static_cast<int>(v))); break;
            case 48: p->setLift(v, p->getLiftRatio(), p->getLiftRange()); break;
            case 49: p->setLift(p->getLiftBelow(), v, p->getLiftRange()); break;
            case 50: p->setLift(p->getLiftBelow(), p->getLiftRatio(), v); break;
            case 51: p->setRangeShape(static_cast<dspark::Compressor<float>::RangeShape>(static_cast<int>(v))); break;
            case 52: p->setReleaseLanding(v); break;
            case 53: p->setControlNetwork(v, p->getControlNetworkCorner()); break;
            case 54: p->setControlNetwork(p->getControlNetworkDepth(), v); break;
            case 55: p->setReductionTone(v, p->getReductionToneHigh()); break;
            case 56: p->setReductionTone(p->getReductionToneLow(), v); break;
            case 57: p->setControlLeak(v); break;
            case 58: p->setDetectorAsymmetry(v); break;
            case 59: p->setRoute(static_cast<dspark::Compressor<float>::Route>(static_cast<int>(v))); break;
            case 62: p->setMidSideFocus(v); break;
            case 63: p->setDetectorDrift(v); break;
            case 60: case 61:
            {
                auto& k = *key;
                k[i == 60 ? 0 : 2].placement = static_cast<dspark::Compressor<float>::KeyPlacement>(std::clamp(static_cast<int>(v), 0, 2));
                for (int b = 0; b < 4; ++b) p->setSidechainBand(b, k[static_cast<size_t>(b)]);
                break;
            }
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
    // The threshold in use: with Auto Threshold it is the programme's recent
    // peak plus the Threshold knob, so the knob reads as "dB under the peak".
    s.statusFn = [p, thrKnob]() {
        char b[96];
        const double now = static_cast<double>(p->getThresholdInUseDb());
        if (p->isAutoThreshold())
            std::snprintf(b, sizeof(b), "Threshold now %.1f dBFS (%.1f dB under the recent peak)",
                          now, 0.0 - static_cast<double>(thrKnob->load(std::memory_order_relaxed)));
        else
            std::snprintf(b, sizeof(b), "Threshold %.1f dBFS", now);
        return std::string(b);
    };
    s.group("Main", { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18 });
    s.group("Character", { 8, 9, 10, 11, 12, 19, 34, 35 });
    s.group("Timing", { 13, 14, 15, 20, 21, 22, 23, 32 });
    s.group("Key", { 24, 25, 26, 27, 28, 29, 30, 31, 33, 60, 61 });
    s.group("Colour Stage", { 36, 37, 38, 39, 40, 41, 42 });
    s.group("Curve", { 47, 48, 49, 50, 51, 59, 62 });
    s.group("Control Path", { 52, 53, 54, 55, 56, 57, 58, 63 });
    return s;
}

inline EffectSlot makeLimiter()
{
    auto p = std::make_shared<dspark::Limiter<float>>();
    auto inputGainDb = std::make_shared<float>(0.0f);
    EffectSlot s;
    s.name = "Limiter"; s.category = "Dynamics";
    s.addSlider("Input Gain", 0, 30, 0, "dB");
    s.addSlider("Ceiling", -30, 0, -1, "dB");
    s.addSlider("Release", 10, 1000, 100, "ms");
    s.addSlider("Lookahead", 0, 10, 5, "ms");
    s.addToggle("Safety Clip", true);
    s.addToggle("Adaptive Release", true);
    s.addSlider("Hold", 0, 50, 10, "ms");
    s.addToggle("True Peak", false);
    s.addToggle("Hold Every Peak", false);
    s.addToggle("True Peak High Precision", false);
    // A latency kept for the lookahead to move inside (0: the lookahead itself).
    s.addSlider("Max Lookahead", 0, 10, 0, "ms");
    // The gain computer at eight points and steps per sample (sample-peak mode).
    s.addToggle("Gain Precision High", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p, inputGainDb](auto b) {
        // Input Gain MUST be applied BEFORE the limiter - otherwise
        // amplifying post-limit would exceed the ceiling and the limiter
        // would never engage for sub-threshold signals.
        float gDb = *inputGainDb;
        if (gDb > 0.01f) {
            float g = dspark::decibelsToGain(gDb);
            for (int ch = 0; ch < b.getNumChannels(); ++ch)
                for (int i = 0; i < b.getNumSamples(); ++i)
                    b.getChannel(ch)[i] *= g;
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, inputGainDb](int i, float v) {
        switch(i) {
            case 0: *inputGainDb = v; break;
            case 1: p->setCeiling(v); break;
            case 2: p->setRelease(v); break;
            case 3: p->setLookahead(v); break;
            case 4: p->setSafetyClip(v > 0.5f); break;
            case 5: p->setAdaptiveRelease(v > 0.5f); break;
            case 6: p->setHold(v); break;
            case 7: p->setTruePeak(v > 0.5f); break;
            case 8: p->setHoldMode(v > 0.5f ? dspark::Limiter<float>::HoldMode::EveryPeak
                                            : dspark::Limiter<float>::HoldMode::Deepest); break;
            case 9: p->setTruePeakPrecision(v > 0.5f ? dspark::Limiter<float>::TruePeakPrecision::High
                                                     : dspark::Limiter<float>::TruePeakPrecision::Standard); break;
            case 10: p->setMaxLookahead(v); break;
            case 11: p->setPrecision(v > 0.5f ? dspark::Limiter<float>::Precision::High
                                              : dspark::Limiter<float>::Precision::Standard); break;
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
    return s;
}

inline EffectSlot makeNoiseGate()
{
    auto p = std::make_shared<dspark::NoiseGate<float>>();
    EffectSlot s;
    s.name = "Noise Gate"; s.category = "Dynamics";
    s.addSlider("Threshold", -80, 0, -80, "dB");   // Default -80 dB = gate fully open (neutral)
    s.addSlider("Attack", 0.1f, 50, 1, "ms");
    s.addSlider("Hold", 0, 500, 50, "ms");
    s.addSlider("Release", 5, 500, 50, "ms");
    s.addSlider("Range", -80, 0, -80, "dB");
    s.addChoice("Gate Mode", {"Amplitude","Frequency"}, 0);
    s.addToggle("Adaptive Hold", false);
    // Threshold counted from the key's own recent level.
    s.addToggle("Auto Threshold", false);
    // The key (indices 8 to 10): listen to it, a low cut (20 Hz: off) and a
    // high cut (20 kHz: off), both at 24 dB per octave.
    s.addToggle("Key Listen", false);
    s.addSlider("Key Low Cut", 20, 2000, 20, "Hz");
    s.addSlider("Key High Cut", 500, 20000, 20000, "Hz");
    // Where the key's low cut sits (11). Order MUST match
    // dspark::NoiseGate<float>::KeyPlacement (Stereo=0, Mid=1, Side=2).
    s.addChoice("Key Low Cut On", {"Both channels","Middle","Sides"}, 0);
    auto lowCut = std::make_shared<std::pair<float, int>>(20.0f, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, lowCut](int i, float v) {
        using Gate = dspark::NoiseGate<float>;
        switch(i) {
            case 8: p->setSidechainListen(v > 0.5f); break;
            case 9: case 11:
            {
                if (i == 9) lowCut->first = v;
                else        lowCut->second = static_cast<int>(v);
                Gate::KeyBand band;
                band.frequency = lowCut->first;
                band.slope = 24.0f;
                band.shape = lowCut->first > 20.5f ? Gate::KeyShape::LowCut : Gate::KeyShape::Off;
                band.placement = static_cast<Gate::KeyPlacement>(lowCut->second);
                p->setSidechainBand(0, band);
                break;
            }
            case 10:
            {
                Gate::KeyBand band;
                band.frequency = v;
                band.slope = 24.0f;
                band.shape = v < 19999.0f ? Gate::KeyShape::HighCut : Gate::KeyShape::Off;
                p->setSidechainBand(1, band);
                break;
            }
            case 0: p->setThreshold(v); break;
            case 1: p->setAttack(v); break;
            case 2: p->setHold(v); break;
            case 3: p->setRelease(v); break;
            case 4: p->setRange(v); break;
            case 5: p->setGateMode(static_cast<dspark::NoiseGate<float>::GateMode>(static_cast<int>(v))); break;
            case 6: p->setAdaptiveHold(v > 0.5f); break;
            case 7: p->setAutoThreshold(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeExpander()
{
    auto p = std::make_shared<dspark::Expander<float>>();
    EffectSlot s;
    s.name = "Expander"; s.category = "Dynamics";
    s.addSlider("Threshold", -60, 0, -60, "dB");   // Default -60 dB = inactive (expander threshold = bottom of range)
    s.addSlider("Ratio", 1, 20, 1, ":1");          // Default 1:1 = no expansion
    s.addSlider("Attack", 0.1f, 50, 5, "ms");
    s.addSlider("Hold", 0, 500, 50, "ms");
    s.addSlider("Release", 5, 500, 100, "ms");
    s.addSlider("Range", -80, 0, -60, "dB");
    // Threshold counted from the key's own recent level.
    s.addToggle("Auto Threshold", false);
    // The key (indices 7 to 9): listen to it, a low cut (20 Hz: off) and a
    // high cut (20 kHz: off), both at 24 dB per octave.
    s.addToggle("Key Listen", false);
    s.addSlider("Key Low Cut", 20, 2000, 20, "Hz");
    s.addSlider("Key High Cut", 500, 20000, 20000, "Hz");
    // Where the key's low cut sits (10). Order MUST match
    // dspark::Expander<float>::KeyPlacement (Stereo=0, Mid=1, Side=2).
    s.addChoice("Key Low Cut On", {"Both channels","Middle","Sides"}, 0);
    auto lowCut = std::make_shared<std::pair<float, int>>(20.0f, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, lowCut](int i, float v) {
        using Exp = dspark::Expander<float>;
        switch(i) {
            case 7: p->setSidechainListen(v > 0.5f); break;
            case 8: case 10:
            {
                if (i == 8) lowCut->first = v;
                else        lowCut->second = static_cast<int>(v);
                Exp::KeyBand band;
                band.frequency = lowCut->first;
                band.slope = 24.0f;
                band.shape = lowCut->first > 20.5f ? Exp::KeyShape::LowCut : Exp::KeyShape::Off;
                band.placement = static_cast<Exp::KeyPlacement>(lowCut->second);
                p->setSidechainBand(0, band);
                break;
            }
            case 9:
            {
                Exp::KeyBand band;
                band.frequency = v;
                band.slope = 24.0f;
                band.shape = v < 19999.0f ? Exp::KeyShape::HighCut : Exp::KeyShape::Off;
                p->setSidechainBand(1, band);
                break;
            }
            case 0: p->setThreshold(v); break;
            case 1: p->setRatio(v); break;
            case 2: p->setAttack(v); break;
            case 3: p->setHold(v); break;
            case 4: p->setRelease(v); break;
            case 5: p->setRange(v); break;
            case 6: p->setAutoThreshold(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeTransientDesigner()
{
    auto p = std::make_shared<dspark::TransientDesigner<float>>();
    EffectSlot s;
    s.name = "Transient Designer"; s.category = "Dynamics";
    s.addSlider("Attack", -100, 100, 0, "%");
    s.addSlider("Sustain", -100, 100, 0, "%");
    s.addSlider("Character", -1, 1, 0, "");
    s.addToggle("Output-Dep Recovery", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setAttack(v); break;
            case 1: p->setSustain(v); break;
            case 2: p->setCharacter(v); break;
            case 3: p->setOutputDepRecovery(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeDeEsser()
{
    auto p = std::make_shared<dspark::DeEsser<float>>();
    EffectSlot s;
    s.name = "De-Esser"; s.category = "Dynamics";
    s.addSlider("Frequency", 2000, 12000, 6000, "Hz", true);
    s.addSlider("Bandwidth", 0.5f, 4, 1, "oct");
    s.addSlider("Threshold", -40, 0, 0, "dB");     // Default 0 dB = never triggers (neutral)
    s.addSlider("Reduction", 0, 24, 0, "dB");      // Default 0 dB = no reduction (neutral)
    s.addChoice("Detection", {"Bandpass","Derivative"}, 0);
    // Threshold counted from the level of the whole input.
    s.addToggle("Auto Threshold", false);
    s.addToggle("Hear Removed", false);   // 6: what the de-esser takes out
    s.addChoice("Structure", {"Direct Form","Ladder"}, 0);   // 7: the cut bell as a normalized ladder
    s.addToggle("Mixed Bell", false);                        // 8: one fixed filter mixed by the reduction
    s.addSlider("Attack", 0.1f, 20, 0.5f, "ms");             // 9
    s.addSlider("Release", 1, 500, 20, "ms");                // 10: short releases show the mixed bell
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setFrequency(v); break;
            case 1: p->setBandwidth(v); break;
            case 2: p->setThreshold(v); break;
            case 3: p->setReduction(v); break;
            case 4: p->setDetectionMode(static_cast<dspark::DeEsser<float>::DetectionMode>(static_cast<int>(v))); break;
            case 5: p->setAutoThreshold(v > 0.5f); break;
            case 6: p->setRemovalListen(v > 0.5f); break;
            case 7: p->setStructure(static_cast<dspark::DeEsser<float>::Structure>(static_cast<int>(v))); break;
            case 8: p->setMixedBell(v > 0.5f); break;
            case 9: p->setAttack(v); break;
            case 10: p->setRelease(v); break;
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
    return s;
}

// =============================================================================
// DISTORTION
// =============================================================================

inline EffectSlot makeSaturation()
{
    auto p = std::make_shared<dspark::Saturation<float>>();
    // Oversampling is applied lazily on the audio thread (see processFn): changing
    // it reallocates the polyphase filters, which is not safe to do concurrently
    // with process() from the UI thread. osWanted is set by the UI, osApplied is
    // reconciled inside process() where nothing else touches the processor.
    auto osWanted  = std::make_shared<std::atomic<int>>(2);   // the library default
    auto osApplied = std::make_shared<std::atomic<int>>(2);
    EffectSlot s;
    s.name = "Saturation"; s.category = "Distortion";
    // Algorithm indices MUST match dspark::Saturation<float>::Algorithm enum order.
    // Enum: Tube=0, Tape=1, Transformer=2, SoftClip=3, HardClip=4,
    //       Exciter=5, Wavefolder=6, Bitcrusher=7, Downsample=8, MultiStage=9, Curve=10
    s.addChoice("Algorithm",
                {"Tube","Tape","Transformer","SoftClip","HardClip",
                 "Exciter","Wavefolder","Bitcrusher","Downsample","MultiStage","Curve"},
                3);  // Default: SoftClip (cleanest, matches framework default)
    s.addSlider("Drive", 0, 40, 0, "dB");   // Default 0 dB = neutral (no saturation)
    s.addSlider("Mix", 0, 1, 1, "");
    s.addSlider("Output", -24, 12, 0, "dB");
    s.addToggle("Adaptive Blend", false);
    s.addSlider("Slew Sensitivity", 0, 1, 0, "");
    s.addChoice("Oversampling", {"Off","2x","4x","8x","16x"}, 1);  // index 6
    // A/B: antiderivative antialiasing. Order MUST match
    // dspark::Saturation<float>::Antialiasing (Off=0, FirstOrder=1, SecondOrder=2).
    s.addChoice("Antialiasing", {"Off","1st order","2nd order"}, 0); // index 7
    // The Curve algorithm's three controls (indices 8 to 10), then the stages
    // around any curve (11 to 17) and the filters of the oversampler (18).
    s.addSlider("Curve Hardness", 0, 0.95f, 0, "");
    s.addSlider("Curve Curvature", 0.5f, 4, 1, "");
    s.addSlider("Curve Asymmetry", -1, 1, 0, "");
    s.addSlider("Asymmetry", -1, 1, 0, "");
    s.addSlider("Program Noise", -120, -20, -120, "dB");
    s.addSlider("Level Tracking", -1, 1, 0, "");
    s.addSlider("Tracking Corner", 0, 2000, 0, "Hz");
    s.addSlider("Comb Loop", -0.95f, 0.95f, 0, "");
    s.addSlider("Comb Loop Freq", 20, 2000, 200, "Hz");
    s.addToggle("Drive Compensation", false);
    s.addChoice("Oversampling Mode", {"Linear phase","Minimum phase"}, 0);
    s.addToggle("Comb Loop Closing", false);   // 19: the loop closes when the programme stops
    // 20: the band the oversampler's first stage keeps. Order MUST match
    // dspark::Oversampling<float>::Band (Preset=0, Audible=1).
    s.addChoice("Oversampling Band", {"Preset","Audible (to 20 kHz)"}, 0);
    // 21: the base curve under the Curve controls. Order MUST match
    // dspark::Saturation<float>::CurveBase.
    s.addChoice("Curve Base", {"Tanh","Algebraic","Arctangent","Erf","Cubic","Valve","Rectifier"}, 0);
    s.addSlider("Drive Pan", -1, 1, 0, "");                 // 22: the drive leans to one channel
    // 23 to 30: the tone after the curve, gains then frequencies of its four sections.
    s.addSlider("Low Shelf", -18, 18, 0, "dB");
    s.addSlider("Low Mid", -18, 18, 0, "dB");
    s.addSlider("High Mid", -18, 18, 0, "dB");
    s.addSlider("High Shelf", -18, 18, 0, "dB");
    s.addSlider("Low Shelf Freq", 20, 500, 100, "Hz", true);
    s.addSlider("Low Mid Freq", 100, 2000, 500, "Hz", true);
    s.addSlider("High Mid Freq", 500, 8000, 2500, "Hz", true);
    s.addSlider("High Shelf Freq", 2000, 20000, 8000, "Hz", true);
    auto tone = std::make_shared<std::array<std::atomic<float>, 8>>();
    {
        static constexpr float kToneDefaults[8] = { 0, 0, 0, 0, 100, 500, 2500, 8000 };
        for (size_t t = 0; t < 8; ++t) (*tone)[t] = kToneDefaults[t];
    }
    auto shape = std::make_shared<std::array<std::atomic<float>, 4>>();
    (*shape)[0] = 0.0f; (*shape)[1] = 1.0f; (*shape)[2] = 0.0f; (*shape)[3] = 200.0f;
    auto modeWanted  = std::make_shared<std::atomic<int>>(0);
    auto modeApplied = std::make_shared<std::atomic<int>>(0);
    auto bandWanted  = std::make_shared<std::atomic<int>>(0);
    auto bandApplied = std::make_shared<std::atomic<int>>(0);
    s.prepareFn = [p, osApplied, modeApplied, bandApplied](auto& sp) {
        p->setOversampling(osApplied->load(std::memory_order_relaxed));
        p->setOversamplingMode(static_cast<dspark::Saturation<float>::OversamplingMode>(modeApplied->load(std::memory_order_relaxed)));
        p->setOversamplingBand(static_cast<dspark::Saturation<float>::OversamplingBand>(bandApplied->load(std::memory_order_relaxed)));
        p->prepare(sp);
    };
    s.processFn = [p, osWanted, osApplied, modeWanted, modeApplied, bandWanted, bandApplied](auto b) {
        const int band = bandWanted->load(std::memory_order_relaxed);
        if (band != bandApplied->load(std::memory_order_relaxed)) {
            p->setOversamplingBand(static_cast<dspark::Saturation<float>::OversamplingBand>(band));   // refits the first stage
            bandApplied->store(band, std::memory_order_relaxed);
        }
        const int want = osWanted->load(std::memory_order_relaxed);
        if (want != osApplied->load(std::memory_order_relaxed)) {
            p->setOversampling(want);   // reallocates - safe here (audio thread, no concurrent process)
            osApplied->store(want, std::memory_order_relaxed);
        }
        const int mode = modeWanted->load(std::memory_order_relaxed);
        if (mode != modeApplied->load(std::memory_order_relaxed)) {
            p->setOversamplingMode(static_cast<dspark::Saturation<float>::OversamplingMode>(mode));   // reallocates too
            modeApplied->store(mode, std::memory_order_relaxed);
        }
        p->process(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, osWanted, shape, modeWanted, bandWanted, tone](int i, float v) {
        auto& k = *shape;
        if (i >= 23 && i <= 30) {
            auto& t = *tone;
            t[static_cast<size_t>(i - 23)] = v;
            const int section = (i - 23) % 4;
            p->setTone(section, t[static_cast<size_t>(section + 4)].load(), t[static_cast<size_t>(section)].load());
            return;
        }
        switch(i) {
            case 21: p->setCurveBase(static_cast<dspark::Saturation<float>::CurveBase>(std::clamp(static_cast<int>(v), 0, 6))); break;
            case 22: p->setDrivePan(v); break;
            case 20: bandWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
            case 8: case 9: case 10:
                k[static_cast<size_t>(i - 8)] = v;
                p->setCurveShape(k[0].load(), k[1].load(), k[2].load());
                break;
            case 11: p->setAsymmetry(v); break;
            case 12: p->setProgramNoise(v); break;
            case 13: p->setLevelTracking(v); break;
            case 14: p->setTrackingCorner(v); break;
            case 15: p->setCombLoop(v, k[3].load()); break;
            case 16: k[3] = v; p->setCombLoop(p->getCombLoop(), v); break;
            case 17: p->setDriveCompensation(v > 0.5f); break;
            case 18: modeWanted->store(static_cast<int>(v), std::memory_order_relaxed); break;
            case 19: p->setCombLoopClosing(v > 0.5f); break;
            case 0: p->setAlgorithm(static_cast<typename dspark::Saturation<float>::Algorithm>(static_cast<int>(v))); break;
            case 1: p->setDrive(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setOutputGain(v); break;
            case 4: p->setAdaptiveBlend(v > 0.5f); break;
            case 5: p->setSlewSensitivity(v); break;
            case 6: { static constexpr int f[] = {1,2,4,8,16}; osWanted->store(f[std::clamp(static_cast<int>(v),0,4)], std::memory_order_relaxed); break; }
            case 7: p->setAntialiasing(static_cast<dspark::Saturation<float>::Antialiasing>(static_cast<int>(v))); break;
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
    s.group("Main", { 0, 1, 2, 3, 4, 5, 17, 22 });
    s.group("Curve", { 8, 9, 10, 21, 11, 12 });
    s.group("Tone", { 23, 24, 25, 26, 27, 28, 29, 30 });
    s.group("Tracking & Loop", { 13, 14, 15, 16 });
    s.group("Oversampling", { 6, 7, 18, 20 });
    return s;
}

inline EffectSlot makeMultibandSaturation()
{
    using MbSat = dspark::MultibandSaturation<float>;
    auto p = std::make_shared<MbSat>();
    auto splits = std::make_shared<std::array<std::atomic<float>, 3>>();
    (*splits)[0] = 200.0f; (*splits)[1] = 1500.0f; (*splits)[2] = 6000.0f;
    auto bands = std::make_shared<std::atomic<int>>(3);
    // The filters of the oversampler reallocate: applied on the audio thread.
    auto modeWanted  = std::make_shared<std::atomic<int>>(0);
    auto modeApplied = std::make_shared<std::atomic<int>>(0);
    p->setNumBands(3);
    for (int i = 0; i < 2; ++i) p->setCrossoverFrequency(i, (*splits)[static_cast<size_t>(i)].load());
    EffectSlot s;
    s.name = "Multiband Saturation"; s.category = "Distortion";
    s.addSlider("Bands", 2, 4, 3, "");                               // index 0
    s.addSlider("Split 1", 40, 2000, 200, "Hz");
    s.addSlider("Split 2", 200, 8000, 1500, "Hz");
    s.addSlider("Split 3", 1000, 16000, 6000, "Hz");
    // Order MUST match dspark::Saturation<float>::Algorithm (applied to every band).
    s.addChoice("Algorithm",
                {"Tube","Tape","Transformer","SoftClip","HardClip",
                 "Exciter","Wavefolder","Bitcrusher","Downsample","MultiStage","Curve"}, 3);  // index 4
    s.addChoice("Oversampling Mode", {"Linear phase","Minimum phase"}, 0);                   // index 5
    // Per band (indices 6 + 5 * band): drive, level, pan, mute, solo.
    for (int b = 1; b <= 4; ++b)
    {
        const std::string n = "B" + std::to_string(b) + " ";
        s.addSlider((n + "Drive").c_str(), 0, 40, 0, "dB");
        s.addSlider((n + "Level").c_str(), -24, 12, 0, "dB");
        s.addSlider((n + "Pan").c_str(), -1, 1, 0, "");
        s.addToggle((n + "Mute").c_str(), false);
        s.addToggle((n + "Solo").c_str(), false);
    }
    s.addChoice("Crossover Structure", {"Direct form","Ladder"}, 0);                        // index 26
    s.addChoice("Crossover Slope", {"6 dB/oct","12 dB/oct","24 dB/oct","36 dB/oct","48 dB/oct"}, 2); // index 27
    s.addSlider("Mix", 0, 1, 1, "");                                                       // index 28: against the programme as the crossover leaves it
    s.addToggle("Output Clip", false);                                                      // index 29: an oversampled clipper after the sum
    s.addSlider("Output Ceiling", -24, 0, 0, "dB");                                         // index 30
    s.addChoice("Oversampling Band", {"Preset","Audible (to 20 kHz)"}, 0);                  // index 31: every band's first stage
    // Order MUST match CrossoverFilter::Topology (Cascade=0, Balanced=1).
    s.addChoice("Crossover Topology", {"Cascade","Balanced"}, 0);                           // index 32
    auto bandWanted  = std::make_shared<std::atomic<int>>(0);
    auto bandApplied = std::make_shared<std::atomic<int>>(0);
    s.prepareFn = [p, modeApplied, bandApplied](auto& sp) {
        p->setOversamplingMode(static_cast<dspark::Saturation<float>::OversamplingMode>(modeApplied->load(std::memory_order_relaxed)));
        p->setOversamplingBand(static_cast<dspark::Saturation<float>::OversamplingBand>(bandApplied->load(std::memory_order_relaxed)));
        p->prepare(sp);
    };
    s.processFn = [p, modeWanted, modeApplied, bandWanted, bandApplied](auto b) {
        const int band = bandWanted->load(std::memory_order_relaxed);
        if (band != bandApplied->load(std::memory_order_relaxed)) {
            p->setOversamplingBand(static_cast<dspark::Saturation<float>::OversamplingBand>(band));
            bandApplied->store(band, std::memory_order_relaxed);
        }
        const int mode = modeWanted->load(std::memory_order_relaxed);
        if (mode != modeApplied->load(std::memory_order_relaxed)) {
            p->setOversamplingMode(static_cast<dspark::Saturation<float>::OversamplingMode>(mode));
            modeApplied->store(mode, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn = [p]() { p->reset(); };
    s.setParamFn = [p, splits, bands, modeWanted, bandWanted](int i, float v) {
        auto& k = *splits;
        if (i == 31) { bandWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); return; }
        const auto placeSplits = [&]() {
            for (int x = 0; x < bands->load() - 1; ++x) p->setCrossoverFrequency(x, k[static_cast<size_t>(x)].load());
        };
        // The splits that stay keep their frequency and memory.
        if (i == 0) { bands->store(static_cast<int>(v + 0.5f)); p->setNumBandsKeepingSplits(bands->load()); placeSplits(); return; }
        if (i == 32) { p->setCrossoverTopology(static_cast<MbSat::Crossover::Topology>(std::clamp(static_cast<int>(v), 0, 1))); return; }
        if (i >= 1 && i <= 3) { k[static_cast<size_t>(i - 1)] = v; placeSplits(); return; }
        if (i == 4) { for (int b = 0; b < 4; ++b) p->setBandAlgorithm(b, static_cast<dspark::Saturation<float>::Algorithm>(static_cast<int>(v))); return; }
        if (i == 5) { modeWanted->store(static_cast<int>(v), std::memory_order_relaxed); return; }
        if (i == 26) { p->setCrossoverStructure(static_cast<MbSat::Crossover::Structure>(static_cast<int>(v))); return; }
        if (i == 27) { static constexpr int slopes[] = {6,12,24,36,48}; p->setOrder(slopes[std::clamp(static_cast<int>(v), 0, 4)]); return; }
        if (i == 28) { p->setMix(v); return; }
        if (i == 29) { p->setOutputClip(v > 0.5f); return; }
        if (i == 30) { p->setOutputCeiling(v); return; }
        const int band = (i - 6) / 5, field = (i - 6) % 5;
        switch (field) {
            case 0: p->setBandDrive(band, v); break;
            case 1: p->setBandLevel(band, v); break;
            case 2: p->setBandPan(band, v); break;
            case 3: p->setBandMute(band, v > 0.5f); break;
            case 4: p->setBandSolo(band, v > 0.5f); break;
        }
    };
    s.group("Main", { 0, 1, 2, 3, 4, 5, 26, 32 });
    s.group("Band 1", { 6, 7, 8, 9, 10 });
    s.group("Band 2", { 11, 12, 13, 14, 15 });
    s.group("Band 3", { 16, 17, 18, 19, 20 });
    s.group("Band 4", { 21, 22, 23, 24, 25 });
    return s;
}

inline EffectSlot makeLoFi()
{
    auto p = std::make_shared<dspark::LoFi<float>>();
    EffectSlot s;
    s.name = "Lo-Fi"; s.category = "Distortion";
    s.addSlider("Smear", 0, 1, 0, "");
    s.addSlider("Pitch", -24, 12, 0, "st");
    s.addSlider("Fold", 0, 36, 0, "dB");
    s.addSlider("Rate", 500, 48000, 48000, "Hz", true);
    s.addToggle("Rate Filter", false);
    s.addSlider("Bits", 1, 24, 24, "");
    s.addToggle("Dither", false);
    s.addSlider("Mix", 0, 1, 1, "");
    s.addSlider("Output", -24, 12, 0, "dB");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setSmear(v); break;
            case 1: p->setPitch(v); break;
            case 2: p->setFold(v); break;
            case 3: p->setRate(v >= 47999.0f ? 1e9f : v); break;
            case 4: p->setRateFilter(v > 0.5f); break;
            case 5: p->setBits(v); break;
            case 6: p->setDither(v > 0.5f); break;
            case 7: p->setMix(v); break;
            case 8: p->setOutput(v); break;
        }
    };
    return s;
}

inline EffectSlot makeClipper()
{
    auto p = std::make_shared<dspark::Clipper<float>>();
    // Clipper::setOversampling only stores the factor; the filters are (re)built in
    // prepare(). We therefore re-prepare on the audio thread when the factor changes
    // (safe: no concurrent process()), so oversampling is auditionable live.
    auto spec      = std::make_shared<dspark::AudioSpec>();
    auto osWanted  = std::make_shared<std::atomic<int>>(1);
    auto osApplied = std::make_shared<std::atomic<int>>(1);
    EffectSlot s;
    s.name = "Clipper"; s.category = "Distortion";
    s.addChoice("Mode", {"Hard","Soft","Analog","GoldenRatio"}, 0);
    s.addSlider("Ceiling", -60, 0, 0, "dB");
    s.addSlider("Input Gain", 0, 48, 0, "dB");
    s.addSlider("Stages", 1, 4, 1, "");
    s.addSlider("Mix", 0, 1, 1, "");
    s.addSlider("Slew Limit", 0, 1, 0, "");
    s.addChoice("Oversampling", {"Off","2x","4x","8x","16x"}, 0);
    // A/B: second-order antiderivative antialiasing (opt-in; adds a few
    // samples of latency, matched on the dry path).
    s.addToggle("Antialiasing (2nd order)", false);                  // index 7
    // Linear-phase (default) or minimum-phase oversampling filters; the
    // dry copy of the mix follows the minimum-phase ones (index 8).
    s.addChoice("Oversampling Filters", {"Linear Phase","Minimum Phase"}, 0);
    // The band the oversampler's first stage keeps (index 9). Order MUST
    // match dspark::Oversampling<float>::Band (Preset=0, Audible=1).
    s.addChoice("Oversampling Band", {"Preset","Audible (to 20 kHz)"}, 0);
    auto filtWanted  = std::make_shared<std::atomic<int>>(0);
    auto filtApplied = std::make_shared<std::atomic<int>>(0);
    auto bandWanted  = std::make_shared<std::atomic<int>>(0);
    auto bandApplied = std::make_shared<std::atomic<int>>(0);
    s.prepareFn = [p, spec, osWanted, osApplied, filtWanted, filtApplied, bandWanted, bandApplied](auto& sp) {
        *spec = sp;
        p->setOversampling(osWanted->load(std::memory_order_relaxed));
        p->setOversamplingMode(static_cast<dspark::Clipper<float>::OversamplingMode>(filtWanted->load(std::memory_order_relaxed)));
        p->setOversamplingBand(static_cast<dspark::Clipper<float>::OversamplingBand>(bandWanted->load(std::memory_order_relaxed)));
        p->prepare(sp);
        osApplied->store(osWanted->load(std::memory_order_relaxed), std::memory_order_relaxed);
        filtApplied->store(filtWanted->load(std::memory_order_relaxed), std::memory_order_relaxed);
        bandApplied->store(bandWanted->load(std::memory_order_relaxed), std::memory_order_relaxed);
    };
    s.processFn = [p, spec, osWanted, osApplied, filtWanted, filtApplied, bandWanted, bandApplied](auto b) {
        const int want = osWanted->load(std::memory_order_relaxed);
        const int filt = filtWanted->load(std::memory_order_relaxed);
        const int band = bandWanted->load(std::memory_order_relaxed);
        if ((want != osApplied->load(std::memory_order_relaxed) || filt != filtApplied->load(std::memory_order_relaxed)
             || band != bandApplied->load(std::memory_order_relaxed))
            && spec->sampleRate > 0) {
            p->setOversampling(want);
            p->setOversamplingMode(static_cast<dspark::Clipper<float>::OversamplingMode>(filt));
            p->setOversamplingBand(static_cast<dspark::Clipper<float>::OversamplingBand>(band));
            p->prepare(*spec);          // rebuilds filters - safe here (audio thread)
            osApplied->store(want, std::memory_order_relaxed);
            filtApplied->store(filt, std::memory_order_relaxed);
            bandApplied->store(band, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, osWanted, filtWanted, bandWanted](int i, float v) {
        switch(i) {
            case 9: bandWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
            case 0: p->setMode(static_cast<dspark::Clipper<float>::Mode>(static_cast<int>(v))); break;
            case 1: p->setCeiling(v); break;
            case 2: p->setInputGain(v); break;
            case 3: p->setStages(static_cast<int>(v)); break;
            case 4: p->setMix(v); break;
            case 5: p->setSlewLimit(v); break;
            case 6: { static constexpr int f[] = {1,2,4,8,16}; osWanted->store(f[std::clamp(static_cast<int>(v),0,4)], std::memory_order_relaxed); break; }
            case 7: p->setAntialiasing(v > 0.5f); break;
            case 8: filtWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
    return s;
}

// =============================================================================
// MODULATION
// =============================================================================

inline EffectSlot makeChorus()
{
    auto p = std::make_shared<dspark::Chorus<float>>();
    EffectSlot s;
    s.name = "Chorus"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 10, 1.5f, "Hz");
    s.addSlider("Depth", 0, 20, 5, "ms");          // depth in milliseconds
    s.addSlider("Mix", 0, 1, 0, "");               // Default 0 = dry only (neutral)
    s.addSlider("Voices", 1, 6, 2, "");
    s.addSlider("Feedback", -0.95f, 0.95f, 0, "");
    s.addToggle("Auto Depth", false);
    s.addSlider("Stereo Spread", 0, 1, 0.5f, "");
    // The bucket-brigade model (indices 7 to 10). Orders MUST match
    // dspark::Chorus<float>::Model and AnalogMode.
    s.addSlider("Centre Delay", 1, 30, 7, "ms");                       // 7
    s.addChoice("Model", {"Digital","Analog"}, 0);                      // 8
    s.addChoice("Analog Mode", {"Slow","Fast","Vibrato"}, 0);           // 9
    s.addSlider("Analog Hiss", -120, -50, -84, "dBFS");                 // 10
    s.addChoice("Interpolation", {"Cubic","Sinc","Linear"}, 0);         // 11: the reads of the digital lines
    s.addSlider("Analog Tap Phase", 0, 180, 180, "deg");                // 12: the analog taps' offset (a mode loads it)
    s.addSlider("Analog Tone", -1, 1, 0, "");                           // 13: the corner after the lines
    s.addToggle("Circuit Filters", false);                              // 14: the filter pair of a chorus circuit
    // 15: order MUST match dspark::DryWetMixer<float>::MixRule.
    s.addChoice("Mix Rule", {"Linear","Equal Power","Full Dry To Half"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        using CH = dspark::Chorus<float>;
        switch(i) {
            case 12: p->setAnalogTapPhase(v); break;
            case 13: p->setAnalogTone(v); break;
            case 14: p->setAnalogCircuitFilters(v > 0.5f); break;
            case 15: p->setMixRule(static_cast<dspark::DryWetMixer<float>::MixRule>(std::clamp(static_cast<int>(v), 0, 2))); break;
            case 0: p->setRate(v); break;
            case 1: p->setDepthMs(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setVoices(static_cast<int>(v)); break;
            case 4: p->setFeedback(v); break;
            case 5: p->setAutoDepth(v > 0.5f); break;
            case 6: p->setStereoSpread(v); break;
            case 7: p->setCenterDelay(v); break;
            case 8: p->setModel(static_cast<CH::Model>(static_cast<int>(v))); break;
            case 9: p->setAnalogMode(static_cast<CH::AnalogMode>(static_cast<int>(v))); break;
            case 10: p->setAnalogNoise(v); break;
            case 11: p->setInterpolation(static_cast<CH::Interpolation>(static_cast<int>(v))); break;
        }
    };
    // A mode loads rate, depth, centre and the taps' offset: show them.
    s.readbackFn = [p](std::vector<float>& v) {
        v[0] = static_cast<float>(p->getRate());
        v[1] = static_cast<float>(p->getDepthMs());
        v[7] = static_cast<float>(p->getCenterDelay());
        v[12] = static_cast<float>(p->getAnalogTapPhase());
    };
    s.group("Main", { 0, 1, 2, 3, 4, 5, 6, 7, 11, 15 });
    s.group("Analog", { 8, 9, 10, 12, 13, 14 });
    return s;
}

// A state-variable filter whose corner a step sequence (Lfo's Steps shape) or
// an envelope triggered by the audio (TriggeredEnvelope) moves. The two
// sources are owner-managed: the interface writes its wishes to atomics and
// the audio thread hands them over between blocks.
inline EffectSlot makeModFilter()
{
    struct Engine
    {
        dspark::StateVariableFilter<float> filter;
        dspark::Lfo<float> lfo;
        dspark::TriggeredEnvelope<float> env;
        std::array<std::atomic<float>, 16> wish {};
        std::atomic<int> version { 1 };
        int applied = 0;
        double rate = 48000.0;
    };
    auto e = std::make_shared<Engine>();
    EffectSlot s;
    s.name = "Mod Filter"; s.category = "Modulation";
    s.addChoice("Source", {"Step Sequence","Triggered Envelope"}, 0);   // 0
    s.addChoice("Mode", {"LowPass","BandPass","HighPass"}, 0);           // 1
    s.addSlider("Cutoff", 60, 8000, 600, "Hz", true);                    // 2: the corner at rest
    s.addSlider("Depth", 0, 5, 2.5f, "oct");                             // 3: how far the source moves it
    s.addSlider("Q", 0.5f, 12, 3, "");                                   // 4
    s.addSlider("Step Rate", 0.5f, 32, 8, "steps/s");                    // 5
    s.addSlider("Steps", 2, 16, 8, "");                                  // 6
    s.addSlider("Glide", 0, 1, 0.15f, "");                               // 7: the share of each step it glides over
    s.addToggle("Random Steps", false);                                  // 8: each cycle draws its values
    s.addSlider("Swing", -0.9f, 0.9f, 0, "");                            // 9
    s.addSlider("Threshold", -60, 0, -30, "dB");                         // 10: the envelope's trigger
    s.addSlider("Attack", 0.5f, 300, 5, "ms", true);                     // 11
    s.addSlider("Decay", 10, 3000, 300, "ms", true);                     // 12
    s.addSlider("Sustain", 0, 1, 0, "");                                 // 13
    s.addSlider("Release", 10, 3000, 200, "ms", true);                   // 14
    s.addSlider("Curve", -1, 1, 0, "");                                  // 15: the envelope's bend
    {
        static constexpr float kDefaults[16] = { 0, 0, 600, 2.5f, 3, 8, 8, 0.15f, 0, 0, -30, 5, 300, 0, 200, 0 };
        for (size_t i = 0; i < 16; ++i) e->wish[i] = kDefaults[i];
    }
    // The sequence: eight values that walk around the corner at rest.
    static constexpr double kPattern[16] = { 1.0, -0.6, 0.3, -1.0, 0.7, -0.3, 0.5, -0.8,
                                             0.9, -0.5, 0.2, -0.9, 0.6, -0.2, 0.4, -0.7 };
    const auto apply = [e]() {
        const auto w = [&](int i) { return e->wish[static_cast<size_t>(i)].load(std::memory_order_relaxed); };
        using SVF = dspark::StateVariableFilter<float>;
        static constexpr SVF::Mode kModes[3] = { SVF::Mode::LowPass, SVF::Mode::BandPass, SVF::Mode::HighPass };
        e->filter.setMode(kModes[std::clamp(static_cast<int>(w(1)), 0, 2)]);
        e->filter.setQ(w(4));
        const int steps = std::clamp(static_cast<int>(w(6) + 0.5f), 2, 16);
        e->lfo.setShape(dspark::Lfo<float>::Shape::Steps);
        e->lfo.setNumSteps(steps);
        e->lfo.setRateHz(static_cast<double>(w(5)) / steps);
        for (int k = 0; k < steps; ++k)
        {
            e->lfo.setStep(k, kPattern[k], static_cast<double>(w(7)));
            e->lfo.setStepRandom(k, w(8) > 0.5f);
        }
        e->lfo.setStepSwing(static_cast<double>(w(9)));
        e->env.setTrigger(dspark::TriggeredEnvelope<float>::Trigger::Audio);
        e->env.setThresholdDb(w(10));
        e->env.setTimes(0.0f, w(11), 0.0f, w(12), w(14));
        e->env.setSustain(w(13));
        e->env.setCurves(w(15), w(15), w(15));
    };
    s.prepareFn = [e, apply](auto& sp) {
        e->rate = sp.sampleRate;
        e->filter.prepare(sp);
        e->lfo.prepare(sp.sampleRate);
        e->lfo.setSeed(7);
        e->env.prepare(sp.sampleRate);
        apply();
        e->applied = e->version.load(std::memory_order_acquire);
    };
    s.processFn = [e, apply](auto b) {
        const int v = e->version.load(std::memory_order_acquire);
        if (v != e->applied) { apply(); e->applied = v; }
        const bool steps = e->wish[0].load(std::memory_order_relaxed) < 0.5f;
        const double rest = static_cast<double>(e->wish[2].load(std::memory_order_relaxed));
        const double depth = static_cast<double>(e->wish[3].load(std::memory_order_relaxed));
        const int numCh = std::min(b.getNumChannels(), 2), n = b.getNumSamples();
        for (int i = 0; i < n; ++i)
        {
            float key = 0.0f;
            for (int ch = 0; ch < numCh; ++ch) key += b.getChannel(ch)[i];
            key /= static_cast<float>(std::max(numCh, 1));
            // The sequence moves the corner both ways around its rest, the
            // envelope only up from it.
            const double m = steps ? static_cast<double>(e->lfo.advance()) : static_cast<double>(e->env.process(key));
            const double hz = std::clamp(rest * std::exp2(depth * m), 20.0, 0.45 * e->rate);
            e->filter.setCutoff(static_cast<float>(hz));
            for (int ch = 0; ch < numCh; ++ch) b.getChannel(ch)[i] = e->filter.processSample(b.getChannel(ch)[i], ch);
        }
    };
    s.resetFn = [e]() { e->filter.reset(); e->lfo.reset(); e->env.reset(); };
    s.setParamFn = [e](int i, float v) {
        if (i < 0 || i >= 16) return;
        e->wish[static_cast<size_t>(i)].store(v, std::memory_order_relaxed);
        e->version.fetch_add(1, std::memory_order_release);
    };
    s.group("Main", { 0, 1, 2, 3, 4 });
    s.group("Sequence", { 5, 6, 7, 8, 9 });
    s.group("Envelope", { 10, 11, 12, 13, 14, 15 });
    return s;
}

inline EffectSlot makePhaser()
{
    auto p = std::make_shared<dspark::Phaser<float>>();
    EffectSlot s;
    s.name = "Phaser"; s.category = "Modulation";
    s.addSlider("Rate", 0.01f, 10, 0.5f, "Hz");
    s.addSlider("Depth", 0, 1, 0.7f, "");
    s.addSlider("Mix", 0, 1, 0, "");               // Default 0 = dry only
    s.addSlider("Stages", 2, 12, 4, "");
    s.addSlider("Feedback", -0.95f, 0.95f, 0.3f, "");
    s.addSlider("Stereo Spread", 0, 1, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setStages(static_cast<int>(v)); break;
            case 4: p->setFeedback(v); break;
            case 5: p->setStereoSpread(v); break;
        }
    };
    return s;
}

inline EffectSlot makeTremolo()
{
    using TR = dspark::Tremolo<float>;
    auto p = std::make_shared<TR>();
    EffectSlot s;
    s.name = "Tremolo"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 20, 4, "Hz");
    s.addSlider("Depth", 0, 1, 0, "");             // Default 0 = no modulation (neutral)
    // Order MUST match Tremolo::Shape (Sine=0, Triangle=1, Square=2).
    s.addChoice("Shape", {"Sine","Triangle","Square"}, 0);
    s.addToggle("Stereo (auto-pan)", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setShape(static_cast<TR::Shape>(static_cast<int>(v))); break;
            case 3: p->setStereo(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeVibrato()
{
    auto p = std::make_shared<dspark::Vibrato<float>>();
    EffectSlot s;
    s.name = "Vibrato"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 15, 5, "Hz");
    s.addSlider("Depth", 0, 2, 0, "st");           // Default 0 = no pitch modulation (neutral)
    s.addSlider("Mod Rate", 0, 5, 0, "Hz");
    s.addSlider("Mod Depth", 0, 1, 0, "");
    // The bucket-brigade model (indices 4 and 5). Order MUST match
    // dspark::Vibrato<float>::Model.
    s.addChoice("Model", {"Digital","Analog"}, 0);                // 4
    s.addSlider("Analog Hiss", -120, -50, -84, "dBFS");           // 5
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setModRate(v); break;
            case 3: p->setModDepth(v); break;
            case 4: p->setModel(static_cast<dspark::Vibrato<float>::Model>(static_cast<int>(v))); break;
            case 5: p->setAnalogNoise(v); break;
        }
    };
    return s;
}

inline EffectSlot makeRingModulator()
{
    auto p = std::make_shared<dspark::RingModulator<float>>();
    EffectSlot s;
    s.name = "Ring Modulator"; s.category = "Modulation";
    s.addSlider("Frequency", 20, 5000, 440, "Hz", true);
    s.addSlider("Mix", 0, 1, 0, "");               // Default 0 = dry only
    s.addChoice("Mode", {"Classic","GeometricMean"}, 0);
    s.addSlider("Soar", 0, 0.2f, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setFrequency(v); break;
            case 1: p->setMix(v); break;
            case 2: p->setMode(static_cast<dspark::RingModulator<float>::Mode>(static_cast<int>(v))); break;
            case 3: p->setSoar(v); break;
        }
    };
    return s;
}

inline EffectSlot makeFrequencyShifter()
{
    auto p = std::make_shared<dspark::FrequencyShifter<float>>();
    EffectSlot s;
    s.name = "Freq Shifter"; s.category = "Modulation";
    s.addSlider("Shift", -500, 500, 0, "Hz");
    s.addSlider("Mix", 0, 1, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setShift(v); break;
            case 1: p->setMix(v); break;
        }
    };
    return s;
}

// =============================================================================
// SPATIAL
// =============================================================================

inline EffectSlot makeDelay()
{
    using DL = dspark::Delay<float>;
    auto p = std::make_shared<DL>();
    EffectSlot s;
    s.name = "Delay"; s.category = "Spatial";
    s.addSlider("Time", 1, 2000, 250, "ms", true);                // 0
    s.addSlider("Feedback", 0, 0.95f, 0.4f, "");                  // 1
    s.addSlider("LP Filter", 200, 20000, 8000, "Hz", true);       // 2 (in the feedback path)
    s.addSlider("HP Filter", 20, 2000, 80, "Hz", true);           // 3 (in the feedback path)
    s.addSlider("Mix", 0, 1, 0, "");                              // 4 - default 0 = dry only
    // Order MUST match Delay::FeedbackMode (Clean=0, Analog=1).
    s.addChoice("Feedback Mode", {"Clean","Analog"}, 1);          // 5
    // Order MUST match Delay::Character (Clean=0, Tape=1, Bucket=2).
    s.addChoice("Character", {"Clean","Tape","Bucket"}, 0);       // 6
    s.addSlider("Wow & Flutter", 0, 1, 0.5f, "");                 // 7 (tape)
    s.addSlider("Character Hiss", -120, -50, -84, "dBFS");        // 8 (tape and bucket)
    // Insert use: the processor blends dry/wet itself (ramped mix) and
    // smooths delay-time changes; 4 s of capacity.
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setDelayMs(v); break;
            case 1: p->setFeedback(v); break;
            case 2: p->setFeedbackLpHz(v); break;
            case 3: p->setFeedbackHpHz(v); break;
            case 4: p->setMix(v); break;
            case 5: p->setFeedbackMode(static_cast<DL::FeedbackMode>(static_cast<int>(v))); break;
            case 6: p->setCharacter(static_cast<DL::Character>(static_cast<int>(v))); break;
            case 7: p->setWowFlutter(v); break;
            case 8: p->setCharacterNoise(v); break;
        }
    };
    return s;
}

inline EffectSlot makeAlgorithmicReverb()
{
    using ARV = dspark::AlgorithmicReverb<float>;
    auto p = std::make_shared<ARV>();
    EffectSlot s;
    s.name = "Reverb"; s.category = "Spatial";
    // Defaults are the Room preset's own values, so the slot starts as the
    // real preset (the Lab applies every default after Type). Picking another
    // Type loads that preset in the processor; the read-back below then moves
    // the sliders to it. Ranges are the processor's own clamps.
    s.addChoice("Type", {"Room","Hall","Chamber","Plate","Spring","Cathedral"}, 0); // 0
    s.addChoice("Quality", {"Full","Eco"}, 0);                   // 1 (Eco = 16-line engine, ~55% CPU)
    s.addSlider("Mix", 0, 1, 0, "");                             // 2 - default 0 = dry only
    s.addSlider("Decay", 0.1f, 30, 0.5f, "s", true);             // 3 (mid-band T60)
    s.addSlider("Size", 0.01f, 1, 0.22f, "");                    // 4 (glides: automatable)
    s.addSlider("Pre-Delay", 0, 500, 0, "ms");                   // 5
    s.addSlider("Damping", 0, 1, 0.6667f, "");                   // 6 (drives HF Decay)
    s.addSlider("HF Decay", 0.05f, 1, 0.40f, "x");               // 7 (HF T60 / mid T60)
    s.addSlider("HF Crossover", 1000, 16000, 5000, "Hz", true);  // 8
    s.addSlider("Bass Decay", 0.3f, 3, 1.1f, "x");               // 9 (bass T60 / mid T60)
    s.addSlider("Bass Crossover", 50, 500, 250, "Hz", true);     // 10
    s.addSlider("Diffusion", 0, 1, 0.72f, "");                   // 11 (Spring: chirp strength)
    s.addSlider("Modulation", 0, 1, 0.14f, "");                  // 12
    s.addSlider("Mod Rate", 0.1f, 5, 1.2f, "Hz", true);          // 13
    s.addSlider("ER-Late Gap", 0, 200, 0, "ms");                 // 14
    s.addSlider("Early Level", -60, 6, 0, "dB");                 // 15
    s.addSlider("Late Level", -60, 6, -1.94f, "dB");             // 16
    s.addSlider("Width", 0, 2, 1, "");                           // 17
    s.addSlider("Low Cut (min = off)", 10, 500, 10, "Hz", true);       // 18
    s.addSlider("High Cut (max = off)", 2000, 20000, 20000, "Hz", true); // 19
    // Around the tank (indices 20 to 43). Every choice's order MUST match its
    // enum in dspark::AlgorithmicReverb (or KeyEqualizer for the wet EQ).
    s.addToggle("Hold Tail", false);                                   // 20
    s.addSlider("Dip Under Dry", 0, 24, 0, "dB");                      // 21
    s.addSlider("Gate (min = off)", -80, 0, -80, "dB");                // 22
    s.addSlider("Gate Hold", 0, 1000, 100, "ms");                      // 23
    s.addSlider("Gate Release", 5, 1000, 150, "ms", true);             // 24
    s.addSlider("Source Depth", 0, 1, 0, "");                          // 25
    s.addSlider("Tank Leveling", 0, 1, 0, "");                         // 26
    s.addToggle("Tank Saturation", false);                             // 27
    s.addSlider("Tempo", 40, 240, 120, "BPM");                         // 28
    s.addChoice("Pre-Delay Note", {"Off","1/1","1/2","1/4","1/8","1/16","1/32"}, 0); // 29
    s.addChoice("Note Feel", {"Straight","Dotted","Triplet"}, 0);      // 30
    s.addSlider("Input Width", 0, 1, 1, "");                           // 31
    s.addChoice("Width Law", {"Side","Power Kept"}, 0);                // 32
    s.addChoice("Modulation Style", {"Smooth","Segments"}, 0);         // 33
    s.addChoice("Layout", {"Prime","Geometric"}, 0);                   // 34
    s.addSlider("Shimmer", 0, 1, 0, "");                               // 35
    s.addChoice("Decay Band", {"Off","Bell","Low Shelf","High Shelf"}, 0); // 36
    s.addSlider("Decay Band Freq", 20, 20000, 2000, "Hz", true);       // 37
    s.addSlider("Decay Band Q", 0.1f, 10, 1, "", true);                // 38
    s.addSlider("Decay Band Time", 0.1f, 4, 1, "x", true);             // 39
    s.addChoice("Wet EQ", {"Off","Low Cut","High Cut","Bell","Low Shelf","High Shelf","Tilt"}, 0); // 40
    s.addSlider("Wet EQ Freq", 20, 20000, 1000, "Hz", true);           // 41
    s.addSlider("Wet EQ Gain", -24, 24, 0, "dB");                      // 42
    s.addSlider("Wet EQ Q", 0.1f, 10, 0.7071f, "", true);              // 43
    s.addSlider("Band Limit (max = off)", 2000, 20000, 20000, "Hz", true); // 44: an elliptic wall on the wet signal
    s.addSlider("Double Slope", 0, 1, 0, "");                          // 45: a second, steeper slope beside the tank
    s.addToggle("Plate Tank", false);                                  // 46: Plate only: its own figure-eight tank
    s.addToggle("Wet EQ Loudness", false);                             // 47: the wet equalizer keeps the loudness
    s.addSlider("Air Absorption", 0, 2, 0, "");                         // 48: the air takes the top of the tail sooner (1 = standard air)
    auto decayBand = std::make_shared<ARV::DecayBand>();
    auto wetBand = std::make_shared<dspark::KeyEqualizer<float, 2>::Band>();
    auto noteFeel = std::make_shared<std::array<int, 2>>(std::array<int, 2> { 0, 0 });
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, decayBand, wetBand, noteFeel](int i, float v) {
        using KE = dspark::KeyEqualizer<float, 2>;
        switch(i) {
            case 20: p->setTailHold(v > 0.5f); return;
            case 21: p->setDryDip(v); return;
            case 22: p->setGate(v <= -79.5f ? -200.0f : v); return;
            case 23: p->setGateHold(v); return;
            case 24: p->setGateRelease(v); return;
            case 25: p->setSourceDepth(v); return;
            case 26: p->setTankLeveling(v); return;
            case 27: p->setTankSaturation(v > 0.5f); return;
            case 28: p->setTempo(v); return;
            case 29: (*noteFeel)[0] = static_cast<int>(v);
                     p->setPreDelayNote(static_cast<ARV::NoteValue>((*noteFeel)[0]), static_cast<ARV::NoteFeel>((*noteFeel)[1])); return;
            case 30: (*noteFeel)[1] = static_cast<int>(v);
                     p->setPreDelayNote(static_cast<ARV::NoteValue>((*noteFeel)[0]), static_cast<ARV::NoteFeel>((*noteFeel)[1])); return;
            case 31: p->setInputWidth(v); return;
            case 32: p->setWidthLaw(static_cast<ARV::WidthLaw>(static_cast<int>(v))); return;
            case 33: p->setModulationStyle(static_cast<ARV::ModulationStyle>(static_cast<int>(v))); return;
            case 34: p->setSpaceLayout(static_cast<ARV::SpaceLayout>(static_cast<int>(v))); return;
            case 35: p->setShimmer(v); return;
            case 36: decayBand->shape = static_cast<ARV::DecayShape>(static_cast<int>(v)); p->setDecayBand(0, *decayBand); return;
            case 37: decayBand->frequency = v; p->setDecayBand(0, *decayBand); return;
            case 38: decayBand->q = v; p->setDecayBand(0, *decayBand); return;
            case 39: decayBand->multiplier = v; p->setDecayBand(0, *decayBand); return;
            case 40: wetBand->shape = static_cast<KE::Shape>(static_cast<int>(v)); p->setPostEqBand(0, *wetBand); return;
            case 41: wetBand->frequency = v; p->setPostEqBand(0, *wetBand); return;
            case 42: wetBand->gainDb = v; p->setPostEqBand(0, *wetBand); return;
            case 43: wetBand->q = v; p->setPostEqBand(0, *wetBand); return;
            case 44: p->setBandLimit(v >= 19900.0f ? 0.0f : v); return;
            case 45: p->setDoubleSlope(v); return;
            case 46: p->setPlateTank(v > 0.5f); return;
            case 47: p->setPostEqLoudness(v > 0.5f); return;
            case 48: p->setAirAbsorption(v); return;
            default: break;
        }
        switch(i) {
            case 0:  p->setType(static_cast<ARV::Type>(static_cast<int>(v))); break;
            case 1:  p->setQuality(v > 0.5f ? ARV::Quality::Eco : ARV::Quality::Full); break;
            case 2:  p->setMix(v); break;
            case 3:  p->setDecay(v); break;
            case 4:  p->setSize(v); break;
            case 5:  p->setPreDelay(v); break;
            case 6:  p->setDamping(v); break;
            case 7:  p->setHighDecayMultiplier(v); break;
            case 8:  p->setHighCrossover(v); break;
            case 9:  p->setBassDecayMultiplier(v); break;
            case 10: p->setBassCrossover(v); break;
            case 11: p->setDiffusion(v); break;
            case 12: p->setModulation(v); break;
            case 13: p->setModRate(v); break;
            case 14: p->setErToLateDelay(v); break;
            case 15: p->setEarlyLevel(v); break;
            case 16: p->setLateLevel(v); break;
            case 17: p->setWidth(v); break;
            case 18: p->setToneLowCut(v <= 10.5f ? 0.0f : v); break;       // 0 = off
            case 19: p->setToneHighCut(v >= 19900.0f ? 0.0f : v); break;   // 0 = off
        }
    };
    // Mirror the processor: a Type change loads a whole preset (applied on
    // the next audio block), and Damping / HF Decay drive each other.
    s.readbackFn = [p](std::vector<float>& v) {
        v[0]  = static_cast<float>(static_cast<int>(p->getType()));
        v[1]  = p->getQuality() == ARV::Quality::Eco ? 1.0f : 0.0f;
        v[2]  = p->getMix();
        v[3]  = p->getDecay();
        v[4]  = p->getSize();
        v[5]  = p->getPreDelay();
        v[6]  = p->getDamping();
        v[7]  = p->getHighDecayMultiplier();
        v[8]  = p->getHighCrossover();
        v[9]  = p->getBassDecayMultiplier();
        v[10] = p->getBassCrossover();
        v[11] = p->getDiffusion();
        v[12] = p->getModulation();
        v[13] = p->getModRate();
        v[14] = p->getErToLateDelay();
        v[15] = std::max(p->getEarlyLevel(), -60.0f);   // a muted field shows as the floor
        v[16] = std::max(p->getLateLevel(), -60.0f);
        v[17] = p->getWidth();
    };
    s.group("Main", { 0, 1, 2, 3, 4, 5, 17, 46 });
    s.group("Decay & Tone", { 6, 7, 8, 9, 10, 48, 18, 19, 44 });
    s.group("Space", { 11, 12, 13, 14, 15, 16, 31, 32, 33, 34 });
    s.group("Tail", { 20, 21, 22, 23, 24, 25, 26, 27, 35, 45 });
    s.group("Tempo", { 28, 29, 30 });
    s.group("Bands", { 36, 37, 38, 39, 40, 41, 42, 43, 47 });
    return s;
}

inline EffectSlot makePanner()
{
    auto p = std::make_shared<dspark::Panner<float>>();
    EffectSlot s;
    s.name = "Panner"; s.category = "Spatial";
    s.addSlider("Pan", -1, 1, 0, "");                     // 0
    s.addChoice("Algorithm", {"EqualPower","Binaural","MidPan","SidePan","Haas","Spectral"}, 0); // 1
    s.addSlider("Binaural ITD", 0.1f, 5, 0.66f, "ms");    // 2
    s.addSlider("Haas Delay", 0.1f, 40, 30, "ms");        // 3
    s.addSlider("Spectral Freq", 200, 16000, 4000, "Hz", true); // 4 (Spectral algorithm)
    s.addSlider("Spectral Gain", 0, 18, 6, "dB");         // 5 (Spectral algorithm)
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setPan(v); break;
            case 1: p->setAlgorithm(static_cast<typename dspark::Panner<float>::Algorithm>(static_cast<int>(v))); break;
            case 2: p->setBinauralMaxITD(v); break;
            case 3: p->setHaasMaxDelay(v); break;
            case 4: p->setSpectralFrequency(v); break;
            case 5: p->setSpectralMaxGain(v); break;
        }
    };
    return s;
}

inline EffectSlot makeStereoWidth()
{
    auto p = std::make_shared<dspark::StereoWidth<float>>();
    EffectSlot s;
    s.name = "Stereo Width"; s.category = "Spatial";
    s.addSlider("Width", 0, 2, 1, "");
    s.addToggle("Bass Mono", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setWidth(v); break;
            case 1: p->setBassMono(v > 0.5f); break;
        }
    };
    return s;
}

// =============================================================================
// UTILITY
// =============================================================================

inline EffectSlot makeGain()
{
    auto p = std::make_shared<dspark::Gain<float>>();
    EffectSlot s;
    s.name = "Gain"; s.category = "Utility";
    s.addSlider("Gain", -60, 24, 0, "dB");
    s.addToggle("Mute", false);
    s.addToggle("Invert", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setGainDb(v); break;
            case 1: p->setMuted(v > 0.5f); break;
            case 2: p->setInverted(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeNoiseGenerator()
{
    auto p = std::make_shared<dspark::NoiseGenerator<float>>();
    EffectSlot s;
    s.name = "Noise Generator"; s.category = "Utility";
    s.addChoice("Type", {"White","Pink","Brown"}, 0);
    s.addSlider("Level", -60, 0, -60, "dB");   // Default -60 dB = silent (neutral at activation)
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setType(static_cast<typename dspark::NoiseGenerator<float>::Type>(static_cast<int>(v))); break;
            case 1: p->setLevel(v); break;
        }
    };
    return s;
}

// =============================================================================
// MASTER REGISTRY
// =============================================================================

inline EffectSlot makeDynamicEQ()
{
    using DEQ = dspark::DynamicEQ<float>;
    auto p = std::make_shared<DEQ>();

    // Per band, 17 controls: what the band is (frequency, Q, type, threshold,
    // knee, rest gain, channels, timing, relative threshold), then what it
    // does ABOVE the threshold and what it does BELOW it, each side with its
    // ratio (1: the side does nothing), its amount (+dB raises the band, -dB
    // cuts it, the size is the most it moves), its attack and its release.
    constexpr int kFields = 17;
    constexpr int kAboveRatio = 9, kAboveAmount = 10, kBelowRatio = 13, kBelowAmount = 14;
    auto ui = std::make_shared<std::array<std::array<float, kFields>, 2>>();
    (*ui)[0] = {  200.0f, 1.0f, 0.0f, -20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                  3.0f, -6.0f, 10.0f, 80.0f,   1.0f, 6.0f, 10.0f, 120.0f };
    (*ui)[1] = { 3000.0f, 1.0f, 0.0f, -20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                  3.0f, -6.0f, 10.0f, 80.0f,   1.0f, 6.0f, 10.0f, 120.0f };

    // The zone a band listens to (on, low edge, high edge), kept apart from
    // the fields of the band so their indices stay where the curves expect them.
    auto zones = std::make_shared<std::array<std::array<float, 3>, 2>>();
    (*zones)[0] = { 0.0f, 50.0f, 120.0f };
    (*zones)[1] = { 0.0f, 5000.0f, 10000.0f };

    auto buildCfg = [](const std::array<float, kFields>& v, const std::array<float, 3>& zone) {
        DEQ::BandConfig c{};
        c.listenLow  = zone[0] > 0.5f ? zone[1] : 0.0f;
        c.listenHigh = zone[0] > 0.5f ? zone[2] : 0.0f;
        c.frequency = v[0];
        c.q         = std::max(0.1f, v[1]);
        c.shape     = static_cast<DEQ::BandShape>(std::clamp(static_cast<int>(v[2]), 0, 2));
        c.threshold = v[3];
        c.knee      = v[4];                   // below 0: the automatic knee
        c.gain      = v[5];
        c.scope     = static_cast<DEQ::Scope>(std::clamp(static_cast<int>(v[6]), 0, 4));
        c.timing    = static_cast<DEQ::Timing>(std::clamp(static_cast<int>(v[7]), 0, 1));
        c.relativeThreshold = v[8] > 0.5f;
        c.enabled   = true;
        c.aboveRatio = std::max(1.0f, v[kAboveRatio]);
        c.aboveRangeDb = std::fabs(v[kAboveAmount]);
        c.aboveBoost = v[kAboveAmount] > 0.0f;
        c.aboveAttackMs = v[11];
        c.aboveReleaseMs = v[12];
        c.belowRatio = std::max(1.0f, v[kBelowRatio]);
        c.belowRangeDb = std::fabs(v[kBelowAmount]);
        c.belowBoost = v[kBelowAmount] > 0.0f;
        c.belowAttackMs = v[15];
        c.belowReleaseMs = v[16];
        return c;
    };

    EffectSlot s;
    s.name = "Dynamic EQ"; s.category = "Filters";
    for (int b = 0; b < 2; ++b)
    {
        const std::string pfx = "B" + std::to_string(b + 1) + " ";
        const auto& d = (*ui)[static_cast<std::size_t>(b)];
        const auto nm = [&](const char* n) { return pfx + n; };
        s.addSlider(nm("Freq").c_str(),      20, 20000, d[0], "Hz", true);
        s.addSlider(nm("Q").c_str(),       0.2f,    10, d[1], "");
        s.addChoice(nm("Type").c_str(), {"Bell","Low Shelf","High Shelf"}, 0);
        s.addSlider(nm("Threshold").c_str(), -60,    0, d[3], "dB");
        s.addSlider(nm("Knee").c_str(),       -1,   24, d[4], "dB");     // -1: automatic
        s.addSlider(nm("Rest Gain").c_str(), -24,   24, d[5], "dB");
        s.addChoice(nm("Channels").c_str(), {"Stereo","Left","Right","Mid","Side"}, 0);
        s.addChoice(nm("Timing").c_str(), {"Manual","Program"}, 0);
        s.addToggle(nm("Relative Thr").c_str(), false);
        s.addSlider(nm("Above Ratio").c_str(),   1, 12, d[9], ":1");     // 1: no action above
        s.addSlider(nm("Above Amount").c_str(), -24, 24, d[10], "dB");   // node Y: +raise / -cut
        s.addSlider(nm("Above Attack").c_str(), 0.1f, 100, d[11], "ms");
        s.addSlider(nm("Above Release").c_str(),   5, 500, d[12], "ms");
        s.addSlider(nm("Below Ratio").c_str(),   1, 12, d[13], ":1");    // 1: no action below
        s.addSlider(nm("Below Amount").c_str(), -24, 24, d[14], "dB");
        s.addSlider(nm("Below Attack").c_str(), 0.1f, 100, d[15], "ms");
        s.addSlider(nm("Below Release").c_str(),   5, 500, d[16], "ms");
    }
    constexpr int kGlobal = 2 * kFields;      // 34: the controls of the whole equalizer
    s.addToggle("Mixed Bands", false);        // 34
    s.addToggle("Linear Phase", false);       // 35
    s.addToggle("Hear Removed", false);       // 36: what the bands take out (or add)
    s.addChoice("Link Mode", {"Loudest","Dominant"}, 0);   // 37: stereo bands read the dominant component
    s.addChoice("Structure", {"Direct Form","Ladder"}, 0); // 38: the band filter as a normalized ladder
    // 39 to 44: a zone of its own for each band's detector; 45: the key a band hears.
    s.addToggle("B1 Own Zone", false);
    s.addSlider("B1 Zone Low",  20, 20000, (*zones)[0][1], "Hz", true);
    s.addSlider("B1 Zone High", 20, 20000, (*zones)[0][2], "Hz", true);
    s.addToggle("B2 Own Zone", false);
    s.addSlider("B2 Zone Low",  20, 20000, (*zones)[1][1], "Hz", true);
    s.addSlider("B2 Zone High", 20, 20000, (*zones)[1][2], "Hz", true);
    s.addChoice("Hear Key", {"Off","Band 1","Band 2"}, 0);
    // Linear phase is a setup call: the interface records the wish and the
    // audio thread prepares the processor again between blocks.
    auto linearWanted  = std::make_shared<std::atomic<bool>>(false);
    auto linearApplied = std::make_shared<std::atomic<bool>>(false);
    auto lastSpec = std::make_shared<dspark::AudioSpec>();
    s.prepareFn = [p, ui, zones, buildCfg, linearWanted, linearApplied, lastSpec](auto& sp) {
        const bool want = linearWanted->load(std::memory_order_relaxed);
        p->setLinearPhase(want);
        linearApplied->store(want, std::memory_order_relaxed);
        *lastSpec = sp;
        p->prepare(sp);
        p->setNumBands(2);
        p->setBand(0, buildCfg((*ui)[0], (*zones)[0]));
        p->setBand(1, buildCfg((*ui)[1], (*zones)[1]));
    };
    s.processFn = [p, linearWanted, linearApplied, lastSpec](auto b) {
        const bool want = linearWanted->load(std::memory_order_relaxed);
        if (want != linearApplied->load(std::memory_order_relaxed)) {
            p->setLinearPhase(want);            // setup call - safe here (audio thread)
            p->prepare(*lastSpec);
            linearApplied->store(want, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, ui, zones, buildCfg, linearWanted](int i, float v) {
        if (i >= kGlobal + 5 && i < kGlobal + 11) {
            const int band = (i - kGlobal - 5) / 3, field = (i - kGlobal - 5) % 3;
            (*zones)[static_cast<std::size_t>(band)][static_cast<std::size_t>(field)] = v;
            p->setBand(band, buildCfg((*ui)[static_cast<std::size_t>(band)], (*zones)[static_cast<std::size_t>(band)]));
            return;
        }
        if (i == kGlobal + 11) { p->setKeyListen(static_cast<int>(v) - 1); return; }
        switch (i - kGlobal) {
            case 0: p->setMixedBands(v > 0.5f); return;
            case 1: linearWanted->store(v > 0.5f, std::memory_order_relaxed); return;
            case 2: p->setRemovalListen(v > 0.5f); return;
            case 3: p->setLinkMode(static_cast<DEQ::LinkMode>(static_cast<int>(v))); return;
            case 4: p->setStructure(static_cast<DEQ::Structure>(static_cast<int>(v))); return;
            default: break;
        }
        if (i < 0 || i >= kGlobal) return;
        const int band = i / kFields, field = i % kFields;
        (*ui)[static_cast<std::size_t>(band)][static_cast<std::size_t>(field)] = v;
        p->setBand(band, buildCfg((*ui)[static_cast<std::size_t>(band)], (*zones)[static_cast<std::size_t>(band)]));
    };
    // The engine's own curves (getBandMagnitude): what is drawn is what plays.
    // Live: each band at the gain it applies now. Behind it: the bands at the
    // full amount of each side (above in grey, below in blue), where the
    // band can go; the node is the above amount (drag it up to raise the
    // band when the level passes the threshold, down to cut it).
    auto curve = [p](const float* f, float* mdb, int n, float g0, float g1) {
        std::vector<float> first(static_cast<std::size_t>(std::max(n, 0))), second(first.size());
        p->getBandMagnitude(0, g0, f, first.data(), n);
        p->getBandMagnitude(1, g1, f, second.data(), n);
        for (int i = 0; i < n; ++i)
        {
            const double mag = static_cast<double>(first[static_cast<std::size_t>(i)])
                             * static_cast<double>(second[static_cast<std::size_t>(i)]);
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    // A side with ratio 1 does nothing: its curve stays at the rest gain.
    const auto sideAt = [](const float* v, int ratio, int amount) {
        return v[ratio] > 1.0f ? v[amount] : 0.0f;
    };
    s.magnitudeFn = [p, curve](const float* f, float* mdb, int n, double, const float*) {
        curve(f, mdb, n, p->getBandGainDb(0), p->getBandGainDb(1));
    };
    s.targetMagnitudeFn = [curve, sideAt](const float* f, float* mdb, int n, double, const float* v) {
        curve(f, mdb, n, sideAt(v, kAboveRatio, kAboveAmount), sideAt(v + kFields, kAboveRatio, kAboveAmount));
    };
    s.secondaryMagnitudeFn = [curve, sideAt](const float* f, float* mdb, int n, double, const float* v) {
        curve(f, mdb, n, sideAt(v, kBelowRatio, kBelowAmount), sideAt(v + kFields, kBelowRatio, kBelowAmount));
    };
    s.curveNodes = { {0, kAboveAmount, 1}, {kFields, kFields + kAboveAmount, kFields + 1} };
    s.gainReductionDbFn = [p]() {
        return std::min({ 0.0f, static_cast<float>(p->getBandGainDb(0)), static_cast<float>(p->getBandGainDb(1)) });
    };
    s.liveLabel = "Gain now";
    s.targetLabel = "Full above";
    s.secondaryLabel = "Full below";
    s.group("Band 1", { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    s.group("Band 2", { 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 });
    s.group("Engine", { 34, 35, 36, 37, 38 });
    s.group("Listening", { 39, 40, 41, 42, 43, 44, 45 });
    return s;
}

inline EffectSlot makeMultibandCompressor()
{
    auto p = std::make_shared<dspark::MultibandCompressor<float>>();
    auto xover = std::make_shared<std::array<float,2>>(std::array<float,2>{250.0f, 2500.0f});
    EffectSlot s;
    s.name = "Multiband Compressor"; s.category = "Dynamics";
    // 3 bands split by two crossovers; per-band threshold/ratio + shared time constants.
    s.addSlider("XOver Low", 40, 1000, 250, "Hz", true);   // 0
    s.addSlider("XOver High", 1000, 16000, 2500, "Hz", true); // 1
    s.addSlider("Low Thresh", -60, 0, -20, "dB");          // 2
    s.addSlider("Low Ratio", 1, 20, 3, ":1");              // 3
    s.addSlider("Mid Thresh", -60, 0, -20, "dB");          // 4
    s.addSlider("Mid Ratio", 1, 20, 3, ":1");              // 5
    s.addSlider("High Thresh", -60, 0, -20, "dB");         // 6
    s.addSlider("High Ratio", 1, 20, 3, ":1");             // 7
    s.addSlider("Attack", 0.1f, 100, 10, "ms");            // 8
    s.addSlider("Release", 10, 1000, 120, "ms");           // 9
    // Order MUST match CrossoverFilter::FilterMode (MinimumPhase=0, LinearPhase=1).
    s.addChoice("Crossover", {"Minimum phase","Linear phase"}, 0); // 10
    // Order MUST match CrossoverFilter::Structure (DirectForm=0, Ladder=1).
    s.addChoice("Crossover Structure", {"Direct form","Ladder"}, 0); // 11
    s.addChoice("Crossover Slope", {"6 dB/oct","12 dB/oct","24 dB/oct","36 dB/oct","48 dB/oct"}, 2); // 12
    s.addSlider("Mix", 0, 1, 1, "");   // 13: every band against itself as the crossover leaves it
    // Order MUST match CrossoverFilter::Topology (Cascade=0, Balanced=1).
    s.addChoice("Crossover Topology", {"Cascade","Balanced"}, 0); // 14
    s.prepareFn = [p](auto& sp) {
        p->prepare(sp);
        p->setNumBands(3);
        p->setCrossoverFrequency(0, 250.0f);
        p->setCrossoverFrequency(1, 2500.0f);
    };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, xover](int i, float v) {
        switch (i) {
            case 0: p->setCrossoverFrequency(0, v); (*xover)[0] = v; break;
            case 1: p->setCrossoverFrequency(1, v); (*xover)[1] = v; break;
            case 2: p->setBandThreshold(0, v); break;
            case 3: p->setBandRatio(0, v); break;
            case 4: p->setBandThreshold(1, v); break;
            case 5: p->setBandRatio(1, v); break;
            case 6: p->setBandThreshold(2, v); break;
            case 7: p->setBandRatio(2, v); break;
            case 8: for (int b = 0; b < 3; ++b) p->setBandAttack(b, v); break;
            case 9: for (int b = 0; b < 3; ++b) p->setBandRelease(b, v); break;
            case 10: p->setCrossoverMode(static_cast<typename dspark::CrossoverFilter<float, 12>::FilterMode>(static_cast<int>(v))); break;
            case 11: p->setCrossoverStructure(static_cast<typename dspark::CrossoverFilter<float, 12>::Structure>(static_cast<int>(v))); break;
            case 12: { static constexpr int slopes[] = {6,12,24,36,48}; p->setOrder(slopes[std::clamp(static_cast<int>(v), 0, 4)]); break; }
            case 13: p->setMix(v); break;
            case 14: p->setCrossoverTopology(static_cast<typename dspark::CrossoverFilter<float, 12>::Topology>(std::clamp(static_cast<int>(v), 0, 1))); break;
        }
    };
    s.gainReductionDbFn = [p]() {
        float gr = 0.0f;
        for (int b = 0; b < 3; ++b) gr = std::max(gr, static_cast<float>(p->getBandGainReductionDb(b)));
        return -gr;   // the meter reads a gain change: negative is a reduction
    };
    // Band visualiser: 2 crossover frequencies + live per-band gain reduction.
    s.multibandFn = [p, xover](float* xHz, float* grDb, int maxBands) -> int {
        if (maxBands >= 3) {
            xHz[0] = (*xover)[0]; xHz[1] = (*xover)[1];
            for (int b = 0; b < 3; ++b) grDb[b] = static_cast<float>(p->getBandGainReductionDb(b));
        }
        return 3;
    };
    return s;
}

inline EffectSlot makeConvolutionReverb()
{
    auto p = std::make_shared<dspark::Reverb<float>>();
    auto sr         = std::make_shared<double>(48000.0);
    auto decayS     = std::make_shared<float>(1.5f);
    auto fileLoaded = std::make_shared<bool>(false);  // true once a real IR WAV is loaded

    // Builds a synthetic IR (exponentially-decaying white noise) so the slot is
    // auditionable with no file. Use "Load IR (WAV)" for a real space/cab/plate.
    // Deterministic LCG + energy normalisation so the wet level tracks the dry.
    auto regenIR = [p, sr, decayS, fileLoaded]() {
        const double fs = (*sr > 0.0) ? *sr : 48000.0;
        const int len = std::max(64, static_cast<int>(fs * static_cast<double>(*decayS)));
        std::vector<float> ir(static_cast<std::size_t>(len));
        uint32_t rng = 0x1234567u;
        const double tau = (fs * static_cast<double>(*decayS)) / 6.9077; // ~ -60 dB over decay
        double energy = 0.0;
        for (int n = 0; n < len; ++n) {
            rng = rng * 1664525u + 1013904223u;
            const float white = static_cast<float>(static_cast<int32_t>(rng)) * (1.0f / 2147483648.0f);
            const float env = static_cast<float>(std::exp(-static_cast<double>(n) / tau));
            const float v = white * env;
            ir[static_cast<std::size_t>(n)] = v;
            energy += static_cast<double>(v) * v;
        }
        const float norm = (energy > 1e-12) ? static_cast<float>(1.0 / std::sqrt(energy)) : 1.0f;
        for (auto& v : ir) v *= norm;
        p->loadIR(ir.data(), len, fs);
        *fileLoaded = false;
    };

    EffectSlot s;
    s.name = "Convolution Reverb"; s.category = "Spatial";
    s.addSlider("Decay", 0.2f, 6.0f, 1.5f, "s");   // 0 - (re)builds the synthetic IR
    s.addSlider("Pre-Delay", 0, 100, 10, "ms");    // 1
    s.addSlider("Mix", 0, 1, 0, "");               // 2 - default dry
    s.addSlider("Decay Scale", 0.25f, 2.0f, 1.0f, "x"); // 3 (reshapes loaded/synthetic IR)
    s.addSlider("Stretch", 0.5f, 2.0f, 1.0f, "x");      // 4 (tape-speed style)
    s.addSlider("Dip Under Dry", 0, 24, 0, "dB");       // 5
    s.addSlider("Gate (min = off)", -80, 0, -80, "dB"); // 6
    s.addSlider("Gate Hold", 0, 1000, 100, "ms");       // 7
    s.addSlider("Gate Release", 5, 1000, 150, "ms", true); // 8
    s.prepareFn = [p, sr, fileLoaded, regenIR](auto& sp) {
        *sr = sp.sampleRate; p->prepare(sp);
        if (!*fileLoaded) regenIR();   // keep a user-loaded IR across re-prepares
    };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, decayS, regenIR](int i, float v) {
        switch (i) {
            case 0: *decayS = v; regenIR(); break;   // switch to synthetic IR (atomic publish - safe live)
            case 1: p->setPreDelay(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setDecayScale(v); break;      // IR rebuild, atomic publish (safe live)
            case 4: p->setStretch(v); break;         // IR rebuild, atomic publish (safe live)
            case 5: p->setDryDip(v); break;
            case 6: p->setGate(v <= -79.5f ? -200.0f : v); break;
            case 7: p->setGateHold(v); break;
            case 8: p->setGateRelease(v); break;
        }
    };
    // Load a real impulse response from a WAV (summed to mono, capped at 10 s).
    s.loadIRFn = [p, fileLoaded](const char* path) {
        dspark::WavFile wav;
        if (!wav.openRead(path)) return;
        auto info = wav.getInfo();
        const int ch = std::clamp(static_cast<int>(info.numChannels), 1, 16);
        int len = static_cast<int>(std::min<int64_t>(info.numSamples, static_cast<int64_t>(info.sampleRate * 10.0)));
        if (len <= 0) return;
        dspark::AudioBuffer<float> tmp; tmp.resize(ch, len);
        if (!wav.readSamples(tmp.toView(), 0, len)) return;
        std::vector<float> mono(static_cast<std::size_t>(len));
        for (int i = 0; i < len; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < ch; ++c) acc += tmp.getChannel(c)[i];
            mono[static_cast<std::size_t>(i)] = acc / static_cast<float>(ch);
        }
        p->loadIR(mono.data(), len, info.sampleRate);
        *fileLoaded = true;
    };
    return s;
}

// =============================================================================
// ANALOG (physical models)
// =============================================================================

inline EffectSlot makeTapeMachine()
{
    auto p = std::make_shared<dspark::TapeMachine<float>>();
    EffectSlot s;
    s.name = "Tape Machine"; s.category = "Analog";
    s.addSlider("Drive", -12, 24, 0, "dB");
    s.addSlider("Bias", 0, 1, 0.5f, "");
    s.addChoice("Speed", {"7.5 ips","15 ips","30 ips","3.75 ips"}, 1);
    s.addChoice("Standard", {"NAB","CCIR"}, 0);
    s.addSlider("Loss", 0, 1, 0.5f, "");
    s.addSlider("Head Bump", 0, 1, 0.5f, "");
    s.addSlider("Wow/Flutter", 0, 1, 0.15f, "");
    s.addSlider("Mix", 0, 1, 1, "");
    // The machine around the tape (indices 8 to 24). Order of every choice
    // MUST match its enum in dspark::TapeMachine.
    s.addChoice("Transport", {"Modulated","Speed"}, 0);                    // 8
    s.addSlider("Hiss", -120, -40, -120, "dBFS");                          // 9
    s.addSlider("Hum", -120, -40, -120, "dBFS");                           // 10
    s.addChoice("Mains", {"50 Hz","60 Hz"}, 0);                            // 11
    s.addSlider("Channel Bleed", -120, -30, -120, "dB");                   // 12
    s.addSlider("Even Harmonics", 0, 1, 0, "");                            // 13
    s.addSlider("Output Iron", 0, 1, 0, "");                               // 14
    s.addChoice("Head", {"Bump","Contour"}, 0);                            // 15
    s.addChoice("Electronics", {"Ideal","Circuit"}, 0);                    // 16
    s.addSlider("Align HF", -6, 6, 0, "dB");                               // 17
    s.addSlider("Align LF", -6, 6, 0, "dB");                               // 18
    s.addChoice("Formulation", {"Classic","High Output","Vintage"}, 0);    // 19
    s.addChoice("Track", {"Half","Full","Quarter"}, 0);                    // 20
    s.addSlider("Calibration", 0, 9, 0, "dB");                             // 21
    s.addChoice("Monitor", {"Play Head","Record Head","Source"}, 0);                   // 22
    s.addToggle("Auto Oversampling", false);                               // 23
    s.addChoice("Oversampling Filters", {"Linear Phase","Minimum Phase"}, 0); // 24
    s.addSlider("Record Emphasis", -6, 6, 0, "dB");                         // 25
    s.addChoice("Bias Shape", {"Two Level","Four Level"}, 1);               // 26 - the library's default
    s.addSlider("Alignment Tolerance", 0, 1, 0, "");                        // 27: each channel strays its own way
    auto shapeWanted = std::make_shared<std::atomic<int>>(0);
    // setAutomaticOversampling() and setBiasShape() are setup calls (they
    // re-prepare): the UI records the wish and the audio thread applies it
    // between blocks.
    auto autoWanted  = std::make_shared<std::atomic<int>>(0);
    auto autoApplied = std::make_shared<std::atomic<int>>(0);
    auto hum = std::make_shared<std::atomic<float>>(-120.0f);
    auto mains = std::make_shared<std::atomic<int>>(0);
    auto align = std::make_shared<std::array<std::atomic<float>, 2>>();
    auto filtWanted  = std::make_shared<std::atomic<int>>(0);
    auto filtApplied = std::make_shared<std::atomic<int>>(0);
    s.prepareFn = [p, autoWanted, autoApplied, filtWanted, filtApplied, shapeWanted](auto& sp) {
        p->setBiasShape(static_cast<dspark::TapeMachine<float>::BiasShape>(shapeWanted->load(std::memory_order_relaxed)));
        const int want = autoWanted->load(std::memory_order_relaxed);
        p->setAutomaticOversampling(want != 0);
        autoApplied->store(want, std::memory_order_relaxed);
        const int filt = filtWanted->load(std::memory_order_relaxed);
        p->setOversamplingMode(static_cast<dspark::TapeMachine<float>::OversamplingMode>(filt));
        filtApplied->store(filt, std::memory_order_relaxed);
        p->prepare(sp);
    };
    s.processFn = [p, autoWanted, autoApplied, filtWanted, filtApplied, shapeWanted](auto b) {
        p->setBiasShape(static_cast<dspark::TapeMachine<float>::BiasShape>(shapeWanted->load(std::memory_order_relaxed))); // returns at once when it is the one in use
        const int want = autoWanted->load(std::memory_order_relaxed);
        if (want != autoApplied->load(std::memory_order_relaxed)) {
            p->setAutomaticOversampling(want != 0);   // setup call - safe here (audio thread)
            if (want == 0) p->setOversampling(4);
            autoApplied->store(want, std::memory_order_relaxed);
        }
        const int filt = filtWanted->load(std::memory_order_relaxed);
        if (filt != filtApplied->load(std::memory_order_relaxed)) {
            p->setOversamplingMode(static_cast<dspark::TapeMachine<float>::OversamplingMode>(filt));   // re-prepares - safe here
            filtApplied->store(filt, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, autoWanted, filtWanted, hum, mains, align, shapeWanted](int i, float v) {
        using TM = dspark::TapeMachine<float>;
        switch (i) {
            case 0: p->setDrive(v); break;
            case 1: p->setBias(v); break;
            case 2: p->setSpeed(static_cast<TM::Speed>(static_cast<int>(v))); break;
            case 3: p->setStandard(static_cast<TM::Standard>(static_cast<int>(v))); break;
            case 4: p->setLossEffects(v); break;
            case 5: p->setHeadBump(v); break;
            case 6: p->setWowFlutter(v); break;
            case 7: p->setMix(v); break;
            case 8: p->setTransport(static_cast<TM::Transport>(static_cast<int>(v))); break;
            case 9: p->setNoise(v); break;
            case 10: hum->store(v); p->setHum(v, static_cast<TM::Mains>(mains->load())); break;
            case 11: mains->store(static_cast<int>(v)); p->setHum(hum->load(), static_cast<TM::Mains>(static_cast<int>(v))); break;
            case 12: p->setChannelBleed(v); break;
            case 13: p->setEvenHarmonics(v); break;
            case 14: p->setOutputIron(v); break;
            case 15: p->setHeadModel(static_cast<TM::HeadModel>(static_cast<int>(v))); break;
            case 16: p->setElectronics(static_cast<TM::Electronics>(static_cast<int>(v))); break;
            case 17: (*align)[0].store(v); p->setAlignment(v, (*align)[1].load()); break;
            case 18: (*align)[1].store(v); p->setAlignment((*align)[0].load(), v); break;
            case 19: p->setFormulation(static_cast<TM::Formulation>(static_cast<int>(v))); break;
            case 20: p->setTrackFormat(static_cast<TM::TrackFormat>(static_cast<int>(v))); break;
            case 21: p->setCalibrationLevel(v); break;
            case 22: p->setMonitor(static_cast<TM::Monitor>(static_cast<int>(v))); break;
            case 23: autoWanted->store(v > 0.5f ? 1 : 0, std::memory_order_relaxed); break;
            case 24: filtWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
            case 25: p->setRecordEmphasis(v); break;
            case 26: shapeWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
            case 27: p->setAlignmentTolerance(v); break;
        }
    };
    // The overload lamp before the tape, read at the internal rate.
    s.statusFn = [p]() {
        char b[96];
        std::snprintf(b, sizeof(b), "Record peak L %.1f dB, R %.1f dB",
                      static_cast<double>(p->getRecordPeakDb(0)), static_cast<double>(p->getRecordPeakDb(1)));
        return std::string(b);
    };
    s.group("Main", { 0, 1, 26, 2, 3, 4, 5, 6, 7 });
    s.group("Machine", { 8, 15, 16, 19, 20, 21, 22 });
    s.group("Noise & Colour", { 9, 10, 11, 12, 13, 14 });
    s.group("Alignment", { 17, 18, 25, 27, 23, 24 });
    return s;
}

inline EffectSlot makeTubePreamp()
{
    using TP = dspark::TubePreamp<float>;
    auto p = std::make_shared<TP>();
    // setGridModel() is a setup call (it rebuilds the circuits and re-runs the
    // calibration), so the UI only records the wish and the audio thread
    // applies it between blocks, where nothing else touches the processor.
    auto gridWanted  = std::make_shared<std::atomic<int>>(0);
    auto gridApplied = std::make_shared<std::atomic<int>>(0);
    EffectSlot s;
    s.name = "Tube Preamp"; s.category = "Analog";
    s.addSlider("Drive", -12, 36, 0, "dB");
    s.addChoice("Stages", {"1 (clean)","2 (crunch)"}, 0);
    s.addSlider("Treble", 0, 1, 0.5f, "");
    s.addSlider("Middle", 0, 1, 0.5f, "");
    s.addSlider("Bass", 0, 1, 0.5f, "");
    s.addSlider("Sag", 0, 1, 0.3f, "");
    s.addSlider("Output", -24, 12, 0, "dB");
    s.addSlider("Mix", 0, 1, 1, "");
    // A/B: Clamp (default) or Conduction (opt-in) grid model. Order MUST
    // match dspark::TubePreamp<float>::GridModel (Clamp=0, Conduction=1).
    s.addChoice("Grid Model", {"Clamp","Conduction"}, 0);            // index 8
    // The cabinet stage (indices 9 to 12).
    s.addToggle("Cabinet", false);
    s.addSlider("Cab Size", 0, 1, 0.5f, "");
    s.addSlider("Cab Brightness", 0, 1, 0.5f, "");
    s.addSlider("Cab Mic Distance", 0, 1, 0.3f, "");
    s.addChoice("Oversampling Filters", {"Linear Phase","Minimum Phase"}, 0);   // index 13
    auto filtWanted  = std::make_shared<std::atomic<int>>(0);
    auto filtApplied = std::make_shared<std::atomic<int>>(0);
    s.prepareFn = [p, gridWanted, gridApplied, filtWanted, filtApplied](auto& sp) {
        const int want = gridWanted->load(std::memory_order_relaxed);
        p->setGridModel(static_cast<TP::GridModel>(want));
        gridApplied->store(want, std::memory_order_relaxed);
        const int filt = filtWanted->load(std::memory_order_relaxed);
        p->setOversamplingMode(static_cast<TP::OversamplingMode>(filt));
        filtApplied->store(filt, std::memory_order_relaxed);
        p->prepare(sp);
    };
    s.processFn = [p, gridWanted, gridApplied, filtWanted, filtApplied](auto b) {
        const int want = gridWanted->load(std::memory_order_relaxed);
        if (want != gridApplied->load(std::memory_order_relaxed)) {
            p->setGridModel(static_cast<TP::GridModel>(want));   // setup call - safe here (audio thread)
            gridApplied->store(want, std::memory_order_relaxed);
        }
        const int filt = filtWanted->load(std::memory_order_relaxed);
        if (filt != filtApplied->load(std::memory_order_relaxed)) {
            p->setOversamplingMode(static_cast<TP::OversamplingMode>(filt));   // re-prepares - safe here (audio thread)
            filtApplied->store(filt, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, gridWanted, filtWanted](int i, float v) {
        switch (i) {
            case 0: p->setDrive(v); break;
            case 1: p->setStages(static_cast<int>(v) + 1); break;
            case 2: p->setTreble(v); break;
            case 3: p->setMiddle(v); break;
            case 4: p->setBass(v); break;
            case 5: p->setSag(v); break;
            case 6: p->setOutput(v); break;
            case 7: p->setMix(v); break;
            case 8: gridWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
            case 9: p->setCabinet(v > 0.5f); break;
            case 10: p->getCabinet().setSize(v); break;
            case 11: p->getCabinet().setBrightness(v); break;
            case 12: p->getCabinet().setDistance(v); break;
            case 13: filtWanted->store(std::clamp(static_cast<int>(v), 0, 1), std::memory_order_relaxed); break;
        }
    };
    s.group("Main", { 0, 1, 2, 3, 4, 5, 6, 7, 8, 13 });
    s.group("Cabinet", { 9, 10, 11, 12 });
    return s;
}

inline EffectSlot makeTransformerModel()
{
    auto p = std::make_shared<dspark::TransformerModel<float>>();
    // setOversampling() re-prepares, which is not safe concurrently with
    // process() from the UI thread: the UI sets osWanted and process()
    // reconciles it, as the Saturation slot does.
    auto osWanted  = std::make_shared<std::atomic<int>>(2);   // the library default
    auto osApplied = std::make_shared<std::atomic<int>>(2);
    EffectSlot s;
    s.name = "Transformer"; s.category = "Analog";
    s.addSlider("Drive", -12, 24, 0, "dB");
    s.addSlider("Core Size", 0, 1, 0.5f, "");
    s.addSlider("Resonance", 0, 1, 0.3f, "");
    s.addSlider("Mix", 0, 1, 1, "");
    s.addChoice("Oversampling", {"Off","2x","4x","8x","16x"}, 1);  // index 4
    s.prepareFn = [p, osApplied](auto& sp) {
        p->setOversampling(osApplied->load(std::memory_order_relaxed));
        p->prepare(sp);
    };
    s.processFn = [p, osWanted, osApplied](auto b) {
        const int want = osWanted->load(std::memory_order_relaxed);
        if (want != osApplied->load(std::memory_order_relaxed)) {
            p->setOversampling(want);   // re-prepares - safe here (audio thread, no concurrent process)
            osApplied->store(want, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, osWanted](int i, float v) {
        switch (i) {
            case 0: p->setDrive(v); break;
            case 1: p->setCoreSize(v); break;
            case 2: p->setResonance(v); break;
            case 3: p->setMix(v); break;
            case 4: { static constexpr int f[] = {1,2,4,8,16}; osWanted->store(f[std::clamp(static_cast<int>(v),0,4)], std::memory_order_relaxed); break; }
        }
    };
    return s;
}

// =============================================================================
// PITCH & TEXTURE
// =============================================================================

inline EffectSlot makePitchShifter()
{
    using PS = dspark::PitchShifter<float>;
    auto p = std::make_shared<PS>();
    EffectSlot s;
    s.name = "Pitch Shifter"; s.category = "Pitch";
    s.addSlider("Semitones", -12, 12, 0, "st");
    s.addSlider("Mix", 0, 1, 1, "");
    s.addToggle("Transient Preserve", true);
    s.addToggle("Formant Preserve", false);
    // Order MUST match PitchShifter::Quality (Standard=0, High=1, Studio=2):
    // High reads the 1.8 engine through a 32-tap windowed sinc (transparent
    // top octave, same latency); Studio is the default engine. Crossing
    // between Studio and the other two restarts the stream.
    s.addChoice("Quality", {"Standard","High","Studio"}, 2);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setSemitones(v); break;
            case 1: p->setMix(v); break;
            case 2: p->setTransientPreserve(v > 0.5f); break;
            case 3: p->setFormantPreserve(v > 0.5f); break;
            case 4: p->setQuality(static_cast<PS::Quality>(static_cast<int>(v))); break;
        }
    };
    return s;
}

inline EffectSlot makePitchCorrector()
{
    using PC = dspark::PitchCorrector<float>;
    auto p = std::make_shared<PC>();
    // Root and scale travel to setScale() together, so both are kept here.
    auto root  = std::make_shared<std::atomic<int>>(0);
    auto scale = std::make_shared<std::atomic<int>>(0);
    EffectSlot s;
    s.name = "Pitch Corrector"; s.category = "Pitch";
    s.addChoice("Root", {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"}, 0); // 0
    s.addChoice("Scale", {"Chromatic","Major","Natural minor"}, 0);                 // 1
    s.addSlider("Retune Speed", 0, 400, 0, "ms");                                    // 2 (0 = hard snap)
    s.addToggle("Formant Preserve", false);                                          // 3
    // A/B: the low-latency engine (opt-in; 1024 samples instead of 4096 at
    // 48 kHz, no formant preservation). A switch restarts the stream.
    s.addToggle("Low Latency", false);                                               // 4
    const auto applyScale = [p, root, scale]() {
        static constexpr std::uint16_t masks[] = { 0x0FFF, 0x0AB5, 0x05AD };   // bit k = k semitones above the root
        p->setScale(masks[std::clamp(scale->load(std::memory_order_relaxed), 0, 2)],
                    root->load(std::memory_order_relaxed));
    };
    s.prepareFn = [p, applyScale](auto& sp) { p->prepare(sp); applyScale(); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, root, scale, applyScale](int i, float v) {
        switch (i) {
            case 0: root->store(static_cast<int>(v), std::memory_order_relaxed); applyScale(); break;
            case 1: scale->store(static_cast<int>(v), std::memory_order_relaxed); applyScale(); break;
            case 2: p->setRetuneSpeedMs(v); break;
            case 3: p->setFormantPreserve(v > 0.5f); break;
            case 4: p->setLowLatency(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeGranularProcessor()
{
    auto p = std::make_shared<dspark::GranularProcessor<float>>();
    EffectSlot s;
    s.name = "Granular"; s.category = "Pitch";
    s.addSlider("Grain Size", 10, 500, 80, "ms", true);
    s.addSlider("Density", 1, 200, 25, "/s", true);
    s.addSlider("Jitter", 0, 1, 0.3f, "");
    s.addSlider("Pitch", -24, 24, 0, "st");
    s.addSlider("Pitch Jitter", 0, 12, 0, "st");
    s.addSlider("Spread", 0, 1, 0.5f, "");
    s.addToggle("Freeze", false);
    s.addSlider("Mix", 0, 1, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setGrainSize(v); break;
            case 1: p->setDensity(v); break;
            case 2: p->setJitter(v); break;
            case 3: p->setPitch(v); break;
            case 4: p->setPitchJitter(v); break;
            case 5: p->setSpread(v); break;
            case 6: p->setFreeze(v > 0.5f); break;
            case 7: p->setMix(v); break;
        }
    };
    return s;
}

inline EffectSlot makeSpectralDenoiser()
{
    auto p = std::make_shared<dspark::SpectralDenoiser<float>>();
    EffectSlot s;
    s.name = "Denoiser"; s.category = "Utility";
    s.addToggle("Learn Noise", false);
    s.addSlider("Reduction", 0, 40, 18, "dB");
    s.addSlider("Threshold", 1, 8, 2, "x");
    // The options (indices 3 to 10).
    s.addSlider("Detail", 1, 96, 96, "/oct", true);                     // 3 - parts of an octave; at the top each bin
    s.addSlider("Gain Release", 0, 500, 0, "ms");                       // 4
    s.addToggle("Hear Removed", false);                                 // 5
    s.addToggle("Filter Mode (5 ms at attacks)", false);                // 6
    s.addSlider("Link (filter mode)", 0, 1, 0, "");                     // 7
    // Order MUST match dspark::KeyEqualizer::Shape.
    s.addChoice("Sensitivity Band", {"Off","Low Cut","High Cut","Bell","Low Shelf","High Shelf","Tilt"}, 0); // 8
    s.addSlider("Sensitivity Freq", 20, 20000, 3000, "Hz", true);       // 9
    s.addSlider("Sensitivity Gain", -12, 12, 6, "dB");                  // 10
    auto sens = std::make_shared<dspark::KeyEqualizer<float, 2>::Band>();
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, sens](int i, float v) {
        using KE = dspark::KeyEqualizer<float, 2>;
        switch (i) {
            case 0: p->setLearning(v > 0.5f); break;
            case 1: p->setReduction(v); break;
            case 2: p->setThreshold(v); break;
            case 3: p->setDetail(v >= 94.0f ? 0.0f : 1.0f / v); break;
            case 4: p->setGainRelease(v); break;
            case 5: p->setRemovalListen(v > 0.5f); break;
            case 6: p->setFilterMode(v > 0.5f); break;
            case 7: p->setLink(v); break;
            case 8: sens->shape = static_cast<KE::Shape>(static_cast<int>(v)); p->setSensitivityBand(0, *sens); break;
            case 9: sens->frequency = v; p->setSensitivityBand(0, *sens); break;
            case 10: sens->gainDb = v; p->setSensitivityBand(0, *sens); break;
        }
    };
    // Live: the gain applied to the first channel.
    s.magnitudeFn = [p](const float* f, float* mdb, int n, double, const float*) {
        readDisplayCurve([&](int k) { return p->getDisplayGain(k); }, f, mdb, n);
    };
    s.liveLabel = "Gain now";
    s.displayRangeDb = 36.0f;
    s.leadParam = 3;
    return s;
}

// =============================================================================
// SPECTRAL
// =============================================================================

inline EffectSlot makeTonalBalance()
{
    auto p = std::make_shared<dspark::TonalBalance<float>>();
    EffectSlot s;
    s.name = "Tonal Balance"; s.category = "Spectral";
    s.addSlider("Balance", 0, 1, 0.5f, "");                    // 0 - the effect's defaults: Tilt acts through Balance
    s.addSlider("Fill Dips", 0, 1, 0.3f, "");                         // 1
    s.addSlider("Trim Peaks", 0, 1, 0.5f, "");                          // 2
    s.addSlider("Tilt", -6, 6, 0, "dB/oct");                   // 3
    s.addSlider("Range", 0, 24, 6, "dB");                      // 4
    s.addSlider("Knee", 0.1f, 12, 3, "dB");                    // 5
    s.addSlider("Low Freq", 20, 2000, 40, "Hz", true);         // 6
    s.addSlider("High Freq", 1000, 20000, 16000, "Hz", true);  // 7
    s.addToggle("Level Compensation", false);                  // 8
    s.addSlider("Reference Level", 60, 130, 100, "dB SPL");    // 9
    s.addSlider("Range Knee", 0, 6, 0, "dB");                  // 10
    s.addSlider("Floor", 10, 120, 30, "dB");                   // 11
    s.addChoice("Reference", {"Flat Loudness","Even Power"}, 0); // 12 - order MUST match TonalBalance::Reference
    s.addSlider("Listening Level", -30, 30, 0, "dB");          // 13 - compensates a listening level this much lower
    s.addChoice("Response", {"By Band","Common"}, 0);          // 14 - order MUST match TonalBalance::Response
    s.addSlider("Response Time", 10, 2000, 60, "ms", true);    // 15
    s.addToggle("Local Slope", false);                         // 16
    s.addChoice("Quality", {"Realtime","Render"}, 0); // 17 - order MUST match TonalBalance::Quality
    s.addToggle("Time Focus", false);                          // 18
    s.addSlider("Contrast Knee", 0.1f, 12, 3, "dB");           // 19 - the knee of dips and peaks
    s.addSlider("Detail", 1, 24, 4, "/oct", true);                       // 20 - parts of an octave
    s.addSlider("Balance Asymmetry", -1, 1, 0, "");            // 21 - lift what is lacking (+) or lower what is in excess (-) more
    s.addSlider("Balance Bias", -6, 6, 0, "dB");               // 22 - the point between lacking and in excess
    auto range = std::make_shared<std::array<float, 2>>(std::array<float, 2> { 40.0f, 16000.0f });
    auto spec = std::make_shared<dspark::AudioSpec>();
    s.prepareFn = [p, spec](auto& sp) { *spec = sp; p->prepare(sp); };
    // Live: the correction the filter plays, point by point. Behind it: the
    // reference shape the trend is pulled to.
    s.magnitudeFn = [p](const float* f, float* mdb, int n, double, const float*) {
        readDisplayCurve([&](int k) { return p->getDisplayCorrection(k); }, f, mdb, n);
    };
    s.targetMagnitudeFn = [](const float* f, float* mdb, int n, double, const float* v) {
        for (int i = 0; i < n; ++i) mdb[i] = v[3] * std::log2(std::max(1.0f, f[i]) / 1000.0f);
    };
    s.liveLabel = "Correction now";
    s.targetLabel = "Reference tilt";
    s.displayRangeDb = 12.0f;
    // Quality takes a prepare(): made here, where nothing processes beside
    // it, when what is asked is not what runs.
    s.processFn = [p, spec](auto b) {
        if (spec->sampleRate > 0 && p->getQuality() != p->getQualityInUse()) p->prepare(*spec);
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, range](int i, float v) {
        switch (i) {
            case 0: p->setBalance(v); break;
            case 1: p->setFillDips(v); break;
            case 2: p->setTrimPeaks(v); break;
            case 3: p->setTilt(v); break;
            case 4: p->setRange(v); break;
            case 5: p->setKnee(v); break;
            case 6: (*range)[0] = v; p->setFrequencyRange((*range)[0], (*range)[1]); break;
            case 7: (*range)[1] = v; p->setFrequencyRange((*range)[0], (*range)[1]); break;
            case 8: p->setLevelCompensation(v > 0.5f); break;
            case 9: p->setReferenceLevel(v); break;
            case 10: p->setRangeKnee(v); break;
            case 11: p->setFloor(v); break;
            case 12: p->setReference(static_cast<dspark::TonalBalance<float>::Reference>(static_cast<int>(v))); break;
            case 13: p->setListeningCompensation(v); break;
            case 14: p->setResponse(static_cast<dspark::TonalBalance<float>::Response>(static_cast<int>(v))); break;
            case 15: p->setResponseTime(v); break;
            case 16: p->setLocalSlope(v > 0.5f); break;
            case 17: p->setQuality(static_cast<dspark::TonalBalance<float>::Quality>(static_cast<int>(v))); break;
            case 18: p->setTimeFocus(v > 0.5f); break;
            case 19: p->setContrastKnee(v); break;
            case 20: p->setDetail(1.0f / v); break;
            case 21: p->setBalanceAsymmetry(v); break;
            case 22: p->setBalanceBias(v); break;
        }
    };
    s.leadParam = 20;
    s.group("Main", { 20, 0, 1, 2, 3, 4, 6, 7 });
    s.group("Shape", { 5, 21, 22, 19, 10, 11, 12, 8, 9, 13 });
    s.group("Response", { 14, 15, 16, 17, 18 });
    return s;
}

inline EffectSlot makeResonanceSuppressor()
{
    using RS = dspark::ResonanceSuppressor<float>;
    auto p = std::make_shared<RS>();
    EffectSlot s;
    s.name = "Resonance Suppressor"; s.category = "Spectral";
    s.addSlider("Threshold", 0, 30, 5, "dB");                   // 0 - the defaults are the library's
    s.addSlider("Strength", 0, 1, 1, "");                       // 1
    s.addSlider("Range", 0, 48, 9, "dB");                       // 2
    s.addSlider("Width", 0.05f, 4, 0.333f, "oct", true);        // 3
    s.addSlider("Attack", 0.5f, 500, 5, "ms", true);            // 4
    s.addSlider("Release", 5, 2000, 80, "ms", true);            // 5
    s.addChoice("Mode", {"Smooth","Deep"}, 0);                   // 6 - order MUST match RS::Mode
    s.addChoice("Stereo", {"Per Bin","Mid/Side"}, 0);           // 7 - order MUST match RS::Link
    s.addSlider("Link", 0, 1, 1, "");                           // 8
    s.addSlider("Side Weight", 0, 2, 1, "");                    // 9
    s.addToggle("Hear Removed", false);                         // 10
    s.addToggle("Hear Focus", false);                           // 11
    s.addChoice("Quality", {"Realtime","Offline"}, 0); // 12
    s.addChoice("Sensitivity Band", {"Off","Low Cut","High Cut","Bell","Low Shelf","High Shelf","Tilt"}, 0); // 13
    s.addSlider("Sensitivity Freq", 20, 20000, 3000, "Hz", true); // 14
    s.addSlider("Sensitivity Gain", -12, 12, 6, "dB");          // 15
    s.addSlider("Mix", 0, 1, 1, "");                            // 16
    s.addSlider("Knee", 0, 12, 0, "dB");                        // 17
    s.addSlider("Persistence", 0, 500, 50, "ms");               // 18
    s.addSlider("Quiet Level", -120, 0, -120, "dBFS");          // 19 - at -120: off
    s.addToggle("Ear Weighting", false);                        // 20
    s.addSlider("Detail", 1, 24, 12, "/oct", true);             // 21 - parts of an octave
    s.addSlider("Width Taper", 0, 2, 0, "");                    // 22 - above 2 kHz
    s.addSlider("Hear Band Low", 0, 20000, 0, "Hz");            // 23
    s.addSlider("Hear Band High", 0, 20000, 0, "Hz");           // 24 - 0: all of it
    s.addSlider("Sub Guard", 0, 1, 1, "");                      // 25 - how far a lone sub is spared, either mode
    s.addToggle("Short Kernel", false);                         // 26 - a cut at the top rings for less time
    auto heardBand = std::make_shared<std::array<float, 2>>(std::array<float, 2>{ 0.0f, 0.0f });
    auto sens = std::make_shared<dspark::KeyEqualizer<float, 2>::Band>();
    auto spec = std::make_shared<dspark::AudioSpec>();
    s.prepareFn = [p, spec](auto& sp) { *spec = sp; p->prepare(sp); };
    // Quality takes a prepare(): made here, where nothing processes beside it.
    s.processFn = [p, spec](auto b) {
        if (spec->sampleRate > 0 && p->getQuality() != p->getQualityInUse()) p->prepare(*spec);
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, sens, heardBand](int i, float v) {
        using KE = dspark::KeyEqualizer<float, 2>;
        switch (i) {
            case 16: p->setMix(v); break;
            case 17: p->setKnee(v); break;
            case 18: p->setPersistence(v); break;
            case 19: p->setQuietLevel(v <= -119.5f ? -200.0f : v); break;
            case 20: p->setAuditoryWeighting(v > 0.5f); break;
            case 21: p->setDetail(1.0f / v); break;
            case 22: p->setWidthTaper(2000.0f, v); break;
            case 25: p->setFoundationGuard(v); break;
            case 26: p->setShortKernel(v > 0.5f); break;
            case 23: (*heardBand)[0] = v; p->setRemovalBand((*heardBand)[0], (*heardBand)[1]); break;
            case 24: (*heardBand)[1] = v; p->setRemovalBand((*heardBand)[0], (*heardBand)[1]); break;
            case 0: p->setThreshold(v); break;
            case 1: p->setStrength(v); break;
            case 2: p->setRange(v); break;
            case 3: p->setWidth(v); break;
            case 4: p->setAttack(v); break;
            case 5: p->setRelease(v); break;
            case 6: p->setMode(static_cast<RS::Mode>(static_cast<int>(v))); break;
            case 7: p->setLinkMode(static_cast<RS::Link>(static_cast<int>(v))); break;
            case 8: p->setLink(v); break;
            case 9: p->setSideWeight(v); break;
            case 10: p->setRemovalListen(v > 0.5f); break;
            case 11: p->setFocusListen(v > 0.5f); break;
            case 12: p->setQuality(static_cast<RS::Quality>(static_cast<int>(v))); break;
            case 13: sens->shape = static_cast<KE::Shape>(static_cast<int>(v)); p->setSensitivityBand(0, *sens); break;
            case 14: sens->frequency = v; p->setSensitivityBand(0, *sens); break;
            case 15: sens->gainDb = v; p->setSensitivityBand(0, *sens); break;
        }
    };
    // Live: the reduction of the last frame on each channel (the mid and the
    // side with Mid/Side); filled: the resonances found over the threshold;
    // behind: the sensitivity curve, whose band is the node.
    s.magnitudeFn = [p](const float* f, float* mdb, int n, double, const float*) {
        readDisplayCurve([&](int k) { return p->getDisplayReduction(0, k); }, f, mdb, n);
    };
    s.secondaryMagnitudeFn = [p](const float* f, float* mdb, int n, double, const float*) {
        readDisplayCurve([&](int k) { return p->getDisplayReduction(1, k); }, f, mdb, n);
    };
    // Filled: only what will be reduced, by the strength and within the range.
    s.activityFn = [p](const float* f, float* mdb, int n, double, const float*) {
        const float strength = p->getStrength(), range = p->getRange();
        const float depth = p->getMode() == RS::Mode::Deep ? 1.5f : 1.0f; // Deep takes half as deep again
        readDisplayCurve([&](int k) { return strength * p->getDisplayExcess(0, k); }, f, mdb, n);
        for (int i = 0; i < n; ++i) mdb[i] = depth * std::clamp(mdb[i], 0.0f, range);
    };
    auto view = std::make_shared<dspark::KeyEqualizer<float, 2>>();
    s.targetMagnitudeFn = [view](const float* f, float* mdb, int n, double sr, const float* v) {
        using KE = dspark::KeyEqualizer<float, 2>;
        view->prepare(sr);
        KE::Band b;
        b.shape = static_cast<KE::Shape>(std::clamp(static_cast<int>(v[13]), 0, 6));
        b.frequency = v[14];
        b.gainDb = v[15];
        view->setBand(0, b);
        const auto response = view->getResponse();
        for (int i = 0; i < n; ++i) mdb[i] = static_cast<float>(response.magnitudeDb(static_cast<double>(f[i])));
    };
    s.curveNodes = { {14, 15, -1, 13} };   // shown while the sensitivity band is on
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getPeakReduction()); };
    s.liveLabel = "Reduction";
    s.secondaryLabel = "Reduction, 2nd channel";
    s.activityLabel = "Resonance found";
    s.targetLabel = "Sensitivity";
    s.leadParam = 21;
    s.group("Main", { 21, 0, 1, 2, 3, 4, 5, 6, 16 });
    s.group("Stereo & Listen", { 7, 8, 9, 10, 11, 23, 24, 12 });
    s.group("Sensitivity", { 13, 14, 15, 20 });
    s.group("Detection", { 25, 17, 18, 19, 22, 26 });
    return s;
}

inline EffectSlot makeSpectralDynamicEQ()
{
    using SD = dspark::SpectralDynamicEQ<float>;
    auto p = std::make_shared<SD>();
    EffectSlot s;
    s.name = "Spectral Dynamic EQ"; s.category = "Spectral";
    // Two bands of eight controls: order MUST match the setParamFn below and
    // the shape choice dspark::SpectralDynamicEQ::Shape.
    for (int b = 0; b < 2; ++b)
    {
        const std::string n = "B" + std::to_string(b + 1) + " ";
        s.addChoice((n + "Shape").c_str(), {"Off","Bell","Low Shelf","High Shelf"}, 0);
        s.addSlider((n + "Freq").c_str(), 20, 20000, b == 0 ? 2500.0f : 7000.0f, "Hz", true);
        s.addSlider((n + "Q").c_str(), 0.1f, 20, 1, "", true);
        s.addSlider((n + "Threshold").c_str(), -60, 0, -12, "dB");
        s.addSlider((n + "Ratio").c_str(), 1, 20, 3, ":1");
        s.addSlider((n + "Range").c_str(), -24, 24, -12, "dB");
        s.addSlider((n + "Attack").c_str(), 0.5f, 500, 10, "ms", true);
        s.addSlider((n + "Release").c_str(), 5, 2000, 150, "ms", true);
    }
    s.addSlider("Detail", 1, 24, 12, "/oct", true); // 16 - parts of an octave
    s.addToggle("Time Focus", false);                // 17 - the top read where it is played, brief moves as brief
    auto bands = std::make_shared<std::array<SD::Band, 2>>();
    (*bands)[0].frequency = 2500.0f;
    (*bands)[1].frequency = 7000.0f;
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, bands](int i, float v) {
        if (i == 16) { p->setDetail(1.0f / v); return; }
        if (i == 17) { p->setTimeFocus(v > 0.5f); return; }
        if (i < 0 || i >= 16) return;
        auto& b = (*bands)[static_cast<std::size_t>(i / 8)];
        switch (i % 8) {
            case 0: b.shape = static_cast<SD::Shape>(static_cast<int>(v)); break;
            case 1: b.frequency = v; break;
            case 2: b.q = v; break;
            case 3: b.threshold = v; break;
            case 4: b.ratio = v; break;
            case 5: b.range = v; break;
            case 6: b.attack = v; break;
            case 7: b.release = v; break;
        }
        p->setBand(i / 8, b);
    };
    // Live: the gain of the last frame, point by point; behind: each band's
    // shape at its full range, where the nodes sit.
    s.magnitudeFn = [p](const float* f, float* mdb, int n, double, const float*) {
        readDisplayCurve([&](int k) { return p->getDisplayGain(k); }, f, mdb, n);
    };
    s.targetMagnitudeFn = [](const float* f, float* mdb, int n, double sr, const float* v) {
        using BC = dspark::BiquadCoeffs;
        BC c[2]; bool on[2] = { false, false };
        for (int b = 0; b < 2; ++b)
        {
            const float* x = v + 8 * b;
            const int shape = static_cast<int>(x[0]);
            const double fr = std::clamp(static_cast<double>(x[1]), 20.0, 0.45 * sr), q = std::max(0.1, static_cast<double>(x[2]));
            const double range = static_cast<double>(x[5]);
            on[b] = shape != 0;
            if (shape == 1) c[b] = BC::makePeak(sr, fr, q, range);
            else if (shape == 2) c[b] = BC::makeLowShelf(sr, fr, range);
            else if (shape == 3) c[b] = BC::makeHighShelf(sr, fr, range);
        }
        for (int i = 0; i < n; ++i)
        {
            double mag = 1.0;
            for (int b = 0; b < 2; ++b) if (on[b]) mag *= c[b].getMagnitude(static_cast<double>(f[i]), sr);
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    s.curveNodes = { {1, 5, 2, 0}, {9, 13, 10, 8} };   // shown while the band is on
    s.gainReductionDbFn = [p]() { return std::min(0.0f, static_cast<float>(p->getPeakGain())); };
    s.liveLabel = "Gain now";
    s.targetLabel = "Full range";
    s.leadParam = 16;
    s.group("Main", { 16, 17, 0, 1, 2, 3, 4, 5, 6, 7 });
    s.group("Band 2", { 8, 9, 10, 11, 12, 13, 14, 15 });
    return s;
}

inline std::vector<EffectSlot> createAllEffects()
{
    std::vector<EffectSlot> all {
        // Filters
        makeFilterEngine(),
        makeEqualizer(),
        makeLadderFilter(),
        makeStateVariableFilter(),
        makeDCBlocker(),
        makeDynamicEQ(),
        // Dynamics
        makeCompressor(),
        makeLimiter(),
        makeNoiseGate(),
        makeExpander(),
        makeTransientDesigner(),
        makeDeEsser(),
        makeMultibandCompressor(),
        // Distortion
        makeSaturation(),
        makeMultibandSaturation(),
        makeLoFi(),
        makeClipper(),
        // Analog (physical models)
        makeTapeMachine(),
        makeTubePreamp(),
        makeTransformerModel(),
        // Pitch and texture
        makePitchShifter(),
        makePitchCorrector(),
        makeGranularProcessor(),
        // Modulation
        makeChorus(),
        makePhaser(),
        makeTremolo(),
        makeVibrato(),
        makeRingModulator(),
        makeFrequencyShifter(),
        makeModFilter(),
        // Spatial
        makeDelay(),
        makeAlgorithmicReverb(),
        makeConvolutionReverb(),
        makePanner(),
        makeStereoWidth(),
        // Utility
        makeGain(),
        makeNoiseGenerator(),
        makeSpectralDenoiser(),
        // Spectral
        makeTonalBalance(),
        makeResonanceSuppressor(),
        makeSpectralDynamicEQ(),
    };
    // Settings to start listening from, by effect (Presets.h).
    addLabPresets(all);
    return all;
}

} // namespace dsplab
