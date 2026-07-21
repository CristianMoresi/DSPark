// DSParkLab — Effect Registry
// Factory functions that create EffectSlot wrappers for all DSPark processors.

#pragma once

#include "../DSPark.h"
#include "EffectSlot.h"

#include <memory>

namespace dsplab {

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
                // Slope only applies to LP / HP — for other shapes setShape()
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
        }
    };
    // Interactive response curve. Reconstructs the exact cascade from the public
    // BiquadCoeffs factories + FilterEngine::cascadeForSlope (single source of truth).
    s.magnitudeFn = [](const float* f, float* mdb, int n, double sr, const float* v) {
        using BC = dspark::BiquadCoeffs<float>;
        using FE = dspark::FilterEngine<float>;
        const int type = static_cast<int>(v[0]);
        const float freq = v[1], Q = std::max(0.05f, v[2]), gain = v[3];
        int slope = std::clamp((static_cast<int>(v[4]) / 6) * 6, 6, 48);
        BC st[5]; int ns = 0;
        auto add = [&](BC c){ if (ns < 5) st[ns++] = c; };
        switch (type) {
            case 0: case 1: {  // LowPass / HighPass — Butterworth cascade for the slope
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
    // framework BandType set (shelves, cuts, notch, bandpass, tilt) — they
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
        s.addChoice(nm, {"Peak","Low Shelf","High Shelf","Low Cut","High Cut",
                         "Notch","Band Pass","Tilt"}, 0);
        std::snprintf(nm, sizeof(nm), "Band %d Slope", b + 1);
        s.addSlider(nm, 6, 48, 12, "dB/oct");   // cuts (Low/High Cut) only
    }
    // Global controls (indices 20/21, appended so band indices stay stable):
    // Phase Mode switches the IIR cascade for the FFT overlap-save engine
    // (adds maxBlockSize of latency, audible as a small delay here), and
    // Matched Bells selects the Orfanidis de-cramped design for Peak bands
    // (audible on high-frequency bells; the curve follows).
    s.addChoice("Phase Mode", {"Minimum","Linear"}, 0);
    s.addToggle("Matched Bells", false);
    // Type index -> framework enum. The UI groups cuts after the shelves;
    // the enum orders LowPass/HighPass there, so the mapping is direct.
    auto toType = [](float v) {
        return static_cast<EQ::BandType>(std::clamp(static_cast<int>(v), 0, 7));
    };
    auto toSlope = [](float v) {
        return std::clamp((static_cast<int>(v) / 6) * 6, 6, 48);
    };
    // Equalizer docs require prepare() BEFORE setNumBands() so bands
    // know the sample rate when auto-spacing frequencies.
    s.prepareFn = [p](auto& sp) { p->prepare(sp); p->setNumBands(4); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, toType, toSlope](int i, float v) {
        if (i == 20)
        {
            p->setFilterMode(static_cast<int>(v) == 1 ? EQ::FilterMode::LinearPhase
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
    // Interactive response curve drawn from the band's REAL biquad cascade
    // (buildBandStages), so shelves, cuts at any slope, notch and tilt all
    // render exactly what the audio path applies. X=freq, Y=gain, wheel=Q.
    s.magnitudeFn = [p, toType, toSlope](const float* f, float* mdb, int n, double sr, const float* v) {
        for (int i = 0; i < n; ++i) {
            double mag = 1.0;
            for (int b = 0; b < 4; ++b) {
                EQ::BandConfig cfg;
                cfg.frequency = v[b*5+0];
                cfg.gain      = v[b*5+1];
                cfg.q         = std::max(0.05f, v[b*5+2]);
                cfg.type      = toType(v[b*5+3]);
                cfg.slope     = toSlope(v[b*5+4]);
                dspark::BiquadCoeffs<float> stages[5];
                const int ns = p->buildBandStages(cfg, stages);
                for (int k = 0; k < ns; ++k)
                    mag *= stages[k].getMagnitude(static_cast<double>(f[i]), sr);
            }
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    s.curveNodes = { {0,1,2}, {5,6,7}, {10,11,12}, {15,16,17} };
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
        }
    };
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
    // Order MUST match dspark::Compressor::AutoMakeupMode (Off=0, Static=1, Adaptive=2).
    s.addChoice("Auto Makeup", {"Off","Static","Adaptive"}, 0);
    s.addSlider("Mix", 0, 1, 1, "");
    // Detector order MUST match dspark::Compressor::DetectorType enum.
    // Enum: Peak=0, Rms=1, TruePeak=2, SplitPolarity=3, Hilbert=4
    s.addChoice("Detector", {"Peak","RMS","TruePeak","SplitPolarity","Hilbert"}, 0);
    // Character order MUST match dspark::Compressor::Character enum.
    // Enum: Clean=0, Opto=1, FET=2, Varimu=3
    s.addChoice("Character", {"Clean","Opto","FET","Varimu"}, 0);
    // Harmonic signature of the character (FET: 1176-calibrated 2nd order).
    s.addSlider("Color", 0, 1, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); break;
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
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p, inputGainDb](auto b) {
        // Input Gain MUST be applied BEFORE the limiter — otherwise
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); break;
            case 1: p->setAttack(v); break;
            case 2: p->setHold(v); break;
            case 3: p->setRelease(v); break;
            case 4: p->setRange(v); break;
            case 5: p->setGateMode(static_cast<dspark::NoiseGate<float>::GateMode>(static_cast<int>(v))); break;
            case 6: p->setAdaptiveHold(v > 0.5f); break;
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); break;
            case 1: p->setRatio(v); break;
            case 2: p->setAttack(v); break;
            case 3: p->setHold(v); break;
            case 4: p->setRelease(v); break;
            case 5: p->setRange(v); break;
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
    auto osWanted  = std::make_shared<std::atomic<int>>(1);
    auto osApplied = std::make_shared<std::atomic<int>>(1);
    EffectSlot s;
    s.name = "Saturation"; s.category = "Distortion";
    // Algorithm indices MUST match dspark::Saturation<float>::Algorithm enum order.
    // Enum: Tube=0, Tape=1, Transformer=2, SoftClip=3, HardClip=4,
    //       Exciter=5, Wavefolder=6, Bitcrusher=7, Downsample=8, MultiStage=9
    s.addChoice("Algorithm",
                {"Tube","Tape","Transformer","SoftClip","HardClip",
                 "Exciter","Wavefolder","Bitcrusher","Downsample","MultiStage"},
                3);  // Default: SoftClip (cleanest, matches framework default)
    s.addSlider("Drive", 0, 40, 0, "dB");   // Default 0 dB = neutral (no saturation)
    s.addSlider("Mix", 0, 1, 1, "");
    s.addSlider("Output", -24, 12, 0, "dB");
    s.addToggle("Adaptive Blend", false);
    s.addSlider("Slew Sensitivity", 0, 1, 0, "");
    s.addChoice("Oversampling", {"Off","2x","4x","8x","16x"}, 0);  // index 6
    s.prepareFn = [p, osApplied](auto& sp) {
        p->setOversampling(osApplied->load(std::memory_order_relaxed));
        p->prepare(sp);
    };
    s.processFn = [p, osWanted, osApplied](auto b) {
        const int want = osWanted->load(std::memory_order_relaxed);
        if (want != osApplied->load(std::memory_order_relaxed)) {
            p->setOversampling(want);   // reallocates — safe here (audio thread, no concurrent process)
            osApplied->store(want, std::memory_order_relaxed);
        }
        p->process(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, osWanted](int i, float v) {
        switch(i) {
            case 0: p->setAlgorithm(static_cast<typename dspark::Saturation<float>::Algorithm>(static_cast<int>(v))); break;
            case 1: p->setDrive(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setOutputGain(v); break;
            case 4: p->setAdaptiveBlend(v > 0.5f); break;
            case 5: p->setSlewSensitivity(v); break;
            case 6: { static constexpr int f[] = {1,2,4,8,16}; osWanted->store(f[std::clamp(static_cast<int>(v),0,4)], std::memory_order_relaxed); break; }
        }
    };
    s.gainReductionDbFn = [p]() { return static_cast<float>(p->getGainReductionDb()); };
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
    s.prepareFn = [p, spec, osWanted, osApplied](auto& sp) {
        *spec = sp;
        p->setOversampling(osWanted->load(std::memory_order_relaxed));
        p->prepare(sp);
        osApplied->store(osWanted->load(std::memory_order_relaxed), std::memory_order_relaxed);
    };
    s.processFn = [p, spec, osWanted, osApplied](auto b) {
        const int want = osWanted->load(std::memory_order_relaxed);
        if (want != osApplied->load(std::memory_order_relaxed) && spec->sampleRate > 0) {
            p->setOversampling(want);
            p->prepare(*spec);          // rebuilds filters — safe here (audio thread)
            osApplied->store(want, std::memory_order_relaxed);
        }
        p->processBlock(b);
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, osWanted](int i, float v) {
        switch(i) {
            case 0: p->setMode(static_cast<dspark::Clipper<float>::Mode>(static_cast<int>(v))); break;
            case 1: p->setCeiling(v); break;
            case 2: p->setInputGain(v); break;
            case 3: p->setStages(static_cast<int>(v)); break;
            case 4: p->setMix(v); break;
            case 5: p->setSlewLimit(v); break;
            case 6: { static constexpr int f[] = {1,2,4,8,16}; osWanted->store(f[std::clamp(static_cast<int>(v),0,4)], std::memory_order_relaxed); break; }
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepthMs(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setVoices(static_cast<int>(v)); break;
            case 4: p->setFeedback(v); break;
            case 5: p->setAutoDepth(v > 0.5f); break;
        }
    };
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
        }
    };
    return s;
}

inline EffectSlot makeTremolo()
{
    auto p = std::make_shared<dspark::Tremolo<float>>();
    EffectSlot s;
    s.name = "Tremolo"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 20, 4, "Hz");
    s.addSlider("Depth", 0, 1, 0, "");             // Default 0 = no modulation (neutral)
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setModRate(v); break;
            case 3: p->setModDepth(v); break;
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
    struct Params { float ms=250; float fb=0.4f; float lp=8000; float hp=80; float mix=0.0f; };
    auto p = std::make_shared<dspark::Delay<float>>();
    auto par = std::make_shared<Params>();
    // Dry buffer for Mix (Delay::processBlock returns 100% wet).
    auto dry = std::make_shared<dspark::AudioBuffer<float>>();
    EffectSlot s;
    s.name = "Delay"; s.category = "Spatial";
    s.addSlider("Time", 1, 1000, 250, "ms");
    s.addSlider("Feedback", 0, 0.95f, 0.4f, "");
    s.addSlider("LP Filter", 200, 20000, 8000, "Hz", true);
    s.addSlider("HP Filter", 20, 2000, 80, "Hz", true);
    s.addSlider("Mix", 0, 1, 0, "");   // Default 0 = dry only (neutral at activation)
    s.prepareFn = [p, dry](auto& sp) {
        p->prepare(sp, 2.0);
        dry->resize(sp.numChannels, sp.maxBlockSize);
    };
    s.processFn = [p, par, dry](auto b) {
        const int nCh = b.getNumChannels();
        const int nS  = b.getNumSamples();
        const float mix = par->mix;

        // Snapshot dry
        for (int ch = 0; ch < nCh; ++ch)
            std::copy(b.getChannel(ch), b.getChannel(ch) + nS, dry->getChannel(ch));

        // Process delay (buffer now holds 100% wet)
        p->processBlock(b, par->ms, par->fb, par->lp, par->hp);

        // Linear dry/wet mix in place
        if (mix < 0.999f) {
            const float wetGain = mix;
            const float dryGain = 1.0f - mix;
            for (int ch = 0; ch < nCh; ++ch) {
                float* wet = b.getChannel(ch);
                const float* dryCh = dry->getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    wet[i] = dryCh[i] * dryGain + wet[i] * wetGain;
            }
        }
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [par](int i, float v) {
        switch(i) {
            case 0: par->ms = v; break;
            case 1: par->fb = v; break;
            case 2: par->lp = v; break;
            case 3: par->hp = v; break;
            case 4: par->mix = v; break;
        }
    };
    return s;
}

inline EffectSlot makeAlgorithmicReverb()
{
    auto p = std::make_shared<dspark::AlgorithmicReverb<float>>();
    EffectSlot s;
    s.name = "Reverb"; s.category = "Spatial";
    s.addChoice("Type", {"Room","Hall","Chamber","Plate","Spring","Cathedral"}, 0); // 0
    s.addSlider("Decay", 0.1f, 10, 1.5f, "s");              // 1
    s.addSlider("Size", 0, 1, 0.5f, "");                     // 2
    s.addSlider("Damping", 0, 1, 0.5f, "");                  // 3
    s.addSlider("Pre-Delay", 0, 100, 10, "ms");              // 4
    s.addSlider("Diffusion", 0, 1, 0.8f, "");                // 5
    s.addSlider("Modulation", 0, 1, 0.15f, "");              // 6
    s.addSlider("HF Decay", 0.1f, 2, 0.5f, "x");            // 7
    s.addSlider("Bass Decay", 0.5f, 2, 1.2f, "x");          // 8
    s.addSlider("HF Crossover", 500, 16000, 5000, "Hz", true); // 9 (log)
    s.addSlider("Bass Crossover", 50, 1000, 200, "Hz", true);  // 10 (log)
    s.addSlider("Tone Low Cut", 10, 500, 20, "Hz", true);    // 11 (log)
    s.addSlider("Tone High Cut", 2000, 20000, 16000, "Hz", true); // 12 (log)
    s.addSlider("ER-Late Gap", 0, 100, 10, "ms");             // 13
    s.addSlider("Mod Rate", 0.1f, 5, 1, "Hz");               // 14
    s.addSlider("Early Level", -20, 6, 0, "dB");             // 15
    s.addSlider("Late Level", -20, 6, 0, "dB");              // 16
    s.addSlider("Width", 0, 2, 1, "");                        // 17
    s.addSlider("Mix", 0, 1, 0, "");                          // 18 — default 0 = dry only
    s.addChoice("Quality", {"Full","Eco"}, 0);                // 19 (Eco = reduced engine)
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0:  p->setType(static_cast<typename dspark::AlgorithmicReverb<float>::Type>(static_cast<int>(v))); break;
            case 1:  p->setDecay(v); break;
            case 2:  p->setSize(v); break;
            case 3:  p->setDamping(v); break;
            case 4:  p->setPreDelay(v); break;
            case 5:  p->setDiffusion(v); break;
            case 6:  p->setModulation(v); break;
            case 7:  p->setHighDecayMultiplier(v); break;
            case 8:  p->setBassDecayMultiplier(v); break;
            case 9:  p->setHighCrossover(v); break;
            case 10: p->setBassCrossover(v); break;
            case 11: p->setToneLowCut(v); break;
            case 12: p->setToneHighCut(v); break;
            case 13: p->setErToLateDelay(v); break;
            case 14: p->setModRate(v); break;
            case 15: p->setEarlyLevel(v); break;
            case 16: p->setLateLevel(v); break;
            case 17: p->setWidth(v); break;
            case 18: p->setMix(v); break;
            case 19: p->setQuality(v > 0.5f
                         ? dspark::AlgorithmicReverb<float>::Quality::Eco
                         : dspark::AlgorithmicReverb<float>::Quality::Full); break;
        }
    };
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setPan(v); break;
            case 1: p->setAlgorithm(static_cast<typename dspark::Panner<float>::Algorithm>(static_cast<int>(v))); break;
            case 2: p->setBinauralMaxITD(v); break;
            case 3: p->setHaasMaxDelay(v); break;
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

    // UI mirror: per band [Freq, Q, Threshold, Amount, Ratio, Attack, Release,
    // Below, Type]. Amount is signed: +dB boosts the band when triggered, -dB
    // cuts it. It is the node's Y axis, so you drag a band UP to boost or DOWN
    // to cut. Type selects the dynamic filter shape (Bell / Low Shelf / High
    // Shelf), matching DynamicEQ::BandShape order.
    auto ui = std::make_shared<std::array<std::array<float, 9>, 2>>();
    (*ui)[0] = {  200.0f, 1.0f, -20.0f, -6.0f, 3.0f, 10.0f, 80.0f, 0.0f, 0.0f };
    (*ui)[1] = { 3000.0f, 1.0f, -20.0f, -6.0f, 3.0f, 10.0f, 80.0f, 0.0f, 0.0f };

    auto buildCfg = [](const std::array<float, 9>& v) {
        DEQ::BandConfig c{};
        c.frequency = v[0];
        c.q         = std::max(0.1f, v[1]);
        c.threshold = v[2];
        c.shape     = static_cast<DEQ::BandShape>(
                          std::clamp(static_cast<int>(v[8]), 0, 2));
        c.enabled   = true;
        const float amount = v[3], ratio = std::max(1.0f, v[4]), atk = v[5], rel = v[6];
        const bool below = v[7] > 0.5f;       // act when below threshold instead of above
        const bool boost = amount > 0.0f;     // sign of Amount = boost / cut
        const float range = std::fabs(amount);
        c.aboveRatio = 1.0f; c.belowRatio = 1.0f;
        c.aboveRangeDb = range; c.belowRangeDb = range;
        c.aboveAttackMs = atk; c.belowAttackMs = atk;
        c.aboveReleaseMs = rel; c.belowReleaseMs = rel;
        c.aboveBoost = false; c.belowBoost = false;
        if (below) { c.belowRatio = ratio; c.belowBoost = boost; }
        else       { c.aboveRatio = ratio; c.aboveBoost = boost; }
        return c;
    };

    // Shape-aware magnitude of one band at gain g (for both curve overlays).
    auto bandMag = [](float freq, float q, int shape, float g, double fHz, double sr) {
        dspark::BiquadCoeffs<float> c;
        switch (shape) {
            default: c = dspark::BiquadCoeffs<float>::makePeak(sr, freq, std::max(0.05f, q), g); break;
            case 1:  c = dspark::BiquadCoeffs<float>::makeLowShelf(sr, freq, g); break;
            case 2:  c = dspark::BiquadCoeffs<float>::makeHighShelf(sr, freq, g); break;
        }
        return static_cast<double>(c.getMagnitude(fHz, sr));
    };

    EffectSlot s;
    s.name = "Dynamic EQ"; s.category = "Filters";
    for (int b = 0; b < 2; ++b)
    {
        const char* pfx = (b == 0) ? "B1 " : "B2 ";
        auto nm = [pfx](const char* n){ static char buf[24]; std::snprintf(buf, sizeof(buf), "%s%s", pfx, n); return buf; };
        const auto& d = (*ui)[b];
        s.addSlider(nm("Freq"),      20, 20000, d[0], "Hz", true);
        s.addSlider(nm("Q"),       0.2f,    10, d[1], "");
        s.addSlider(nm("Threshold"), -60,    0, d[2], "dB");
        s.addSlider(nm("Amount"),    -24,   24, d[3], "dB");   // node Y: +boost / -cut
        s.addSlider(nm("Ratio"),       1,   12, d[4], ":1");
        s.addSlider(nm("Attack"),   0.1f,  100, d[5], "ms");
        s.addSlider(nm("Release"),     5,  500, d[6], "ms");
        s.addToggle(nm("Below Thr"), false);
        s.addChoice(nm("Type"), {"Bell","Low Shelf","High Shelf"}, 0);
    }
    s.prepareFn = [p, ui, buildCfg](auto& sp) {
        p->prepare(sp);
        p->setNumBands(2);
        p->setBand(0, buildCfg((*ui)[0]));
        p->setBand(1, buildCfg((*ui)[1]));
    };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, ui, buildCfg](int i, float v) {
        const int band = i / 9, prm = i % 9;
        if (band < 0 || band >= 2) return;
        (*ui)[static_cast<std::size_t>(band)][static_cast<std::size_t>(prm)] = v;
        p->setBand(band, buildCfg((*ui)[static_cast<std::size_t>(band)]));
    };
    // The orange curve is LIVE (current applied gain via getBandGainDb): it dips for
    // a cut and rises for a boost as the audio triggers each band. The draggable
    // nodes sit at each band's Amount target (X=freq, Y=amount, wheel=Q).
    s.magnitudeFn = [p, bandMag](const float* f, float* mdb, int n, double sr, const float* v) {
        const float g0 = p->getBandGainDb(0);
        const float g1 = p->getBandGainDb(1);
        for (int i = 0; i < n; ++i) {
            const double fHz = static_cast<double>(f[i]);
            const double mag = bandMag(v[0], v[1], static_cast<int>(v[8]),  g0, fHz, sr)
                             * bandMag(v[9], v[10], static_cast<int>(v[17]), g1, fHz, sr);
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    // Faint target curve = each band's Amount (the ceiling the live curve moves to);
    // the draggable nodes sit on it.
    s.targetMagnitudeFn = [bandMag](const float* f, float* mdb, int n, double sr, const float* v) {
        for (int i = 0; i < n; ++i) {
            const double fHz = static_cast<double>(f[i]);
            const double mag = bandMag(v[0], v[1], static_cast<int>(v[8]),  v[3],  fHz, sr)
                             * bandMag(v[9], v[10], static_cast<int>(v[17]), v[12], fHz, sr);
            mdb[i] = static_cast<float>(20.0 * std::log10(std::max(1e-6, mag)));
        }
    };
    s.curveNodes = { {0,3,1}, {9,12,10} };  // X=freq, Y=Amount (drag to boost/cut), wheel=Q
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
        }
    };
    s.gainReductionDbFn = [p]() {
        float gr = 0.0f;
        for (int b = 0; b < 3; ++b) gr = std::max(gr, static_cast<float>(p->getBandGainReductionDb(b)));
        return gr;
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
    s.addSlider("Decay", 0.2f, 6.0f, 1.5f, "s");   // 0 — (re)builds the synthetic IR
    s.addSlider("Pre-Delay", 0, 100, 10, "ms");    // 1
    s.addSlider("Mix", 0, 1, 0, "");               // 2 — default dry
    s.addSlider("Decay Scale", 0.25f, 2.0f, 1.0f, "x"); // 3 (reshapes loaded/synthetic IR)
    s.addSlider("Stretch", 0.5f, 2.0f, 1.0f, "x");      // 4 (tape-speed style)
    s.prepareFn = [p, sr, fileLoaded, regenIR](auto& sp) {
        *sr = sp.sampleRate; p->prepare(sp);
        if (!*fileLoaded) regenIR();   // keep a user-loaded IR across re-prepares
    };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, decayS, regenIR](int i, float v) {
        switch (i) {
            case 0: *decayS = v; regenIR(); break;   // switch to synthetic IR (atomic publish — safe live)
            case 1: p->setPreDelay(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setDecayScale(v); break;      // IR rebuild, atomic publish (safe live)
            case 4: p->setStretch(v); break;         // IR rebuild, atomic publish (safe live)
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
    s.addChoice("Speed", {"7.5 ips","15 ips","30 ips"}, 1);
    s.addChoice("Standard", {"NAB","CCIR"}, 0);
    s.addSlider("Loss", 0, 1, 0.5f, "");
    s.addSlider("Head Bump", 0, 1, 0.5f, "");
    s.addSlider("Wow/Flutter", 0, 1, 0.15f, "");
    s.addSlider("Mix", 0, 1, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
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
        }
    };
    return s;
}

inline EffectSlot makeTubePreamp()
{
    auto p = std::make_shared<dspark::TubePreamp<float>>();
    EffectSlot s;
    s.name = "Tube Preamp"; s.category = "Analog";
    s.addSlider("Drive", -12, 36, 0, "dB");
    s.addChoice("Stages", {"1 (clean)","2 (crunch)","3 (lead)"}, 0);
    s.addSlider("Treble", 0, 1, 0.5f, "");
    s.addSlider("Middle", 0, 1, 0.5f, "");
    s.addSlider("Bass", 0, 1, 0.5f, "");
    s.addSlider("Sag", 0, 1, 0.3f, "");
    s.addSlider("Output", -24, 12, 0, "dB");
    s.addSlider("Mix", 0, 1, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setDrive(v); break;
            case 1: p->setStages(static_cast<int>(v) + 1); break;
            case 2: p->setTreble(v); break;
            case 3: p->setMiddle(v); break;
            case 4: p->setBass(v); break;
            case 5: p->setSag(v); break;
            case 6: p->setOutput(v); break;
            case 7: p->setMix(v); break;
        }
    };
    return s;
}

inline EffectSlot makeTransformerModel()
{
    auto p = std::make_shared<dspark::TransformerModel<float>>();
    EffectSlot s;
    s.name = "Transformer"; s.category = "Analog";
    s.addSlider("Drive", -12, 24, 0, "dB");
    s.addSlider("Core Size", 0, 1, 0.5f, "");
    s.addSlider("Resonance", 0, 1, 0.3f, "");
    s.addSlider("Mix", 0, 1, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setDrive(v); break;
            case 1: p->setCoreSize(v); break;
            case 2: p->setResonance(v); break;
            case 3: p->setMix(v); break;
        }
    };
    return s;
}

// =============================================================================
// PITCH & TEXTURE
// =============================================================================

inline EffectSlot makePitchShifter()
{
    auto p = std::make_shared<dspark::PitchShifter<float>>();
    EffectSlot s;
    s.name = "Pitch Shifter"; s.category = "Pitch";
    s.addSlider("Semitones", -12, 12, 0, "st");
    s.addSlider("Mix", 0, 1, 1, "");
    s.addToggle("Transient Preserve", true);
    s.addToggle("Formant Preserve", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setSemitones(v); break;
            case 1: p->setMix(v); break;
            case 2: p->setTransientPreserve(v > 0.5f); break;
            case 3: p->setFormantPreserve(v > 0.5f); break;
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
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch (i) {
            case 0: p->setLearning(v > 0.5f); break;
            case 1: p->setReduction(v); break;
            case 2: p->setThreshold(v); break;
        }
    };
    return s;
}

inline std::vector<EffectSlot> createAllEffects()
{
    return {
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
        makeClipper(),
        // Analog (physical models)
        makeTapeMachine(),
        makeTubePreamp(),
        makeTransformerModel(),
        // Pitch and texture
        makePitchShifter(),
        makeGranularProcessor(),
        // Modulation
        makeChorus(),
        makePhaser(),
        makeTremolo(),
        makeVibrato(),
        makeRingModulator(),
        makeFrequencyShifter(),
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
    };
}

} // namespace dsplab
