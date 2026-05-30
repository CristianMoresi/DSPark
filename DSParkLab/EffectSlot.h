// DSParkLab — Effect Slot abstraction
// Type-erased wrapper for any DSPark processor with parameter descriptors.

#pragma once

#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dsplab {

// --- Parameter descriptor ---------------------------------------------------

struct ParamDesc
{
    enum Type { Slider, Toggle, Choice };

    std::string name;
    float       min        = 0.0f;
    float       max        = 1.0f;
    float       defaultVal = 0.0f;
    Type        type       = Slider;
    std::string unit;                       // "dB", "ms", "Hz", ":1", "%"
    std::vector<std::string> choices;       // Only for Choice type
    bool        logarithmic = false;        // Log-scale slider
};

// --- Effect slot (one per processor instance) --------------------------------

class EffectSlot
{
public:
    std::string name;
    std::string category;
    std::vector<ParamDesc> params;
    std::vector<float>     values;   // Current parameter values
    bool enabled  = false;
    bool selected = false;           // UI: which panel to show

    // Type-erased processor interface
    std::function<void(const dspark::AudioSpec&)>        prepareFn;
    std::function<void(dspark::AudioBufferView<float>)>  processFn;
    std::function<void()>                                resetFn;
    std::function<void(int, float)>                      setParamFn;

    // Optional metering getter — returns gain reduction in dB (positive value
    // means the processor is attenuating the signal by that amount).
    // Leave empty if the processor does not expose gain reduction.
    std::function<float()> gainReductionDbFn;

    // --- Optional rich-GUI hooks (filled by factories that support them) -------

    // Frequency-response magnitude (dB) for the interactive analyzer/EQ display.
    // Fills magsDb[0..n) for the given frequencies, derived from the current
    // parameter values (pure: no processor state needed).
    std::function<void(const float* freqsHz, float* magsDb, int n,
                       double sampleRate, const float* paramValues)> magnitudeFn;

    // Optional faint "target" curve drawn behind the live one (e.g. a Dynamic EQ
    // band's max boost/cut, where the draggable node lives). Same signature.
    std::function<void(const float* freqsHz, float* magsDb, int n,
                       double sampleRate, const float* paramValues)> targetMagnitudeFn;

    // Draggable curve nodes (parameter indices; -1 = that axis is fixed).
    // X drag -> freqParam (log), Y drag -> gainParam, mouse wheel -> qParam.
    struct CurveNode { int freqParam = -1; int gainParam = -1; int qParam = -1; };
    std::vector<CurveNode> curveNodes;

    // Impulse-response loader (Convolution Reverb): receives a WAV file path.
    std::function<void(const char* wavPath)> loadIRFn;

    // Multiband display: fills crossover frequencies (Hz) and per-band gain
    // reduction (dB, positive = attenuation); returns the active band count.
    std::function<int(float* crossoverHz, float* bandGrDb, int maxBands)> multibandFn;

    // --- Builder helpers ---

    void addSlider(const char* n, float mn, float mx, float def,
                   const char* u = "", bool log = false)
    {
        params.push_back({ n, mn, mx, def, ParamDesc::Slider, u, {}, log });
        values.push_back(def);
    }

    void addToggle(const char* n, bool def = false)
    {
        params.push_back({ n, 0.0f, 1.0f, def ? 1.0f : 0.0f, ParamDesc::Toggle });
        values.push_back(def ? 1.0f : 0.0f);
    }

    void addChoice(const char* n, std::vector<std::string> opts, int def = 0)
    {
        params.push_back({ n, 0.0f, static_cast<float>(opts.size() - 1),
                           static_cast<float>(def), ParamDesc::Choice, "", std::move(opts) });
        values.push_back(static_cast<float>(def));
    }

    // --- Runtime interface ---

    void prepare(const dspark::AudioSpec& spec)
    {
        if (prepareFn) prepareFn(spec);
    }

    void process(dspark::AudioBufferView<float> buf)
    {
        if (enabled && processFn) processFn(buf);
    }

    void reset()
    {
        if (resetFn) resetFn();
    }

    void applyParam(int idx, float value)
    {
        if (idx >= 0 && idx < static_cast<int>(values.size()))
        {
            values[idx] = value;
            if (setParamFn) setParamFn(idx, value);
        }
    }

    void applyAllDefaults()
    {
        for (int i = 0; i < static_cast<int>(params.size()); ++i)
        {
            values[i] = params[i].defaultVal;
            if (setParamFn) setParamFn(i, values[i]);
        }
    }
};

} // namespace dsplab
