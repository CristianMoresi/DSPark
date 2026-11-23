// DSParkLab - Effect Slot abstraction
// Type-erased wrapper for any DSPark processor with parameter descriptors.
//
// Threading
// ---------
// The interface thread owns every member of a slot except one: process() runs
// on the audio thread and reads `enabled`, which the interface writes whenever
// the user clicks the slot's checkbox. That one word is therefore published
// rather than plain (see PublishedFlag). Everything else here -- the parameter
// values, the descriptors, the callables -- is written and read by the
// interface thread alone, or installed once by a factory before any device
// exists; applyParam() reaches the processor through setParamFn, and it is the
// processor that publishes its own parameters to the audio thread.

#pragma once

#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dsplab {

// --- A word shared by the interface and the audio thread ---------------------

/// A boolean one thread writes and another reads.
///
/// A plain bool would not do: a value written by one thread and read by
/// another with no synchronisation is a data race, and a compiler that may
/// assume nobody else is looking is free to fold, duplicate or reorder the
/// accesses. Nothing else is ordered against this flag -- a processor publishes
/// its own parameters -- so a single relaxed store/load pair is the whole
/// contract, and it costs the audio thread exactly what the plain read cost.
///
/// The wrapper exists because std::atomic is neither copyable nor movable,
/// which would make every slot unmovable and break the factories that return
/// one by value. Copying transfers the value; it happens at setup, where a
/// relaxed access is all the ordering a copy needs.
///
/// The word must also be lock-free, or the audio thread would take a lock to
/// read it. No assertion is made here because this is a concrete width the
/// framework's own suite already sweeps, alongside every other word type this
/// application publishes; a header that pins a width the sweep cannot reach --
/// one named by a template parameter -- is the case that asserts locally.
class PublishedFlag
{
public:
    PublishedFlag() = default;
    explicit PublishedFlag(bool v) noexcept : value_(v) {}
    ~PublishedFlag() = default;

    PublishedFlag(const PublishedFlag& other) noexcept : value_(other.get()) {}
    PublishedFlag(PublishedFlag&& other) noexcept : value_(other.get()) {}
    PublishedFlag& operator=(const PublishedFlag& other) noexcept
    {
        set(other.get());
        return *this;
    }
    PublishedFlag& operator=(PublishedFlag&& other) noexcept
    {
        set(other.get());
        return *this;
    }

    PublishedFlag& operator=(bool v) noexcept { set(v); return *this; }

    [[nodiscard]] bool get() const noexcept
    {
        return value_.load(std::memory_order_relaxed);
    }
    void set(bool v) noexcept { value_.store(v, std::memory_order_relaxed); }

    explicit operator bool() const noexcept { return get(); }

private:
    std::atomic<bool> value_ { false };
};

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
    std::string group;                      // Tab the control sits on (empty: the first tab)
};

/// One control of a preset: its name in the interface and the value it takes.
struct PresetValue
{
    const char* control;
    float       value;
};

// --- Effect slot (one per processor instance) --------------------------------

class EffectSlot
{
public:
    std::string name;
    std::string category;
    std::vector<ParamDesc> params;
    std::vector<float>     values;   // Current parameter values
    PublishedFlag enabled;           // Read by the audio thread in process()
    bool selected = false;           // UI: which panel to show

    // Type-erased processor interface
    std::function<void(const dspark::AudioSpec&)>        prepareFn;
    std::function<void(dspark::AudioBufferView<float>)>  processFn;
    std::function<void()>                                resetFn;
    std::function<void(int, float)>                      setParamFn;

    // Optional metering getter - returns the gain change in dB (negative: the
    // processor is attenuating the signal by that amount); the interface shows
    // its history under the display. Leave empty if the processor has none.
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

    // A line of live state shown over the gain-reduction history (e.g. the
    // threshold a compressor is using right now).
    std::function<std::string()> statusFn;

    // A second live curve, drawn thinner in another colour (e.g. the second
    // channel of a processor that treats the two apart). Same signature.
    std::function<void(const float* freqsHz, float* magsDb, int n,
                       double sampleRate, const float* paramValues)> secondaryMagnitudeFn;

    // What the processor finds, drawn as a filled area from 0 dB (e.g. the
    // resonances a suppressor detects over its threshold). Same signature.
    std::function<void(const float* freqsHz, float* magsDb, int n,
                       double sampleRate, const float* paramValues)> activityFn;

    // Legend of the display: what each curve shows (empty: not listed).
    std::string liveLabel = "Response", targetLabel, secondaryLabel, activityLabel;

    // Vertical scale of the response display, +/- dB.
    float displayRangeDb = 24.0f;

    // Draggable curve nodes (parameter indices; -1 = that axis is fixed).
    // X drag -> freqParam (log), Y drag -> gainParam, mouse wheel -> qParam;
    // with enableParam, the node shows only while that control is not 0 (a
    // band whose shape is Off has nothing to drag).
    struct CurveNode { int freqParam = -1; int gainParam = -1; int qParam = -1; int enableParam = -1; };
    std::vector<CurveNode> curveNodes;

    // Tabs of the control panel, in order (see group()).
    std::vector<std::string> groups;

    // The control drawn first on its tab, wherever it sits in params (-1: none).
    int leadParam = -1;

    // Parameter read-back: refreshes values[] from the processor's getters, for
    // processors whose own logic moves parameters (a reverb Type loads a whole
    // preset, a damping knob drives the HF decay). Called by the interface
    // every frame while the slot is shown and no widget is being dragged.
    std::function<void(std::vector<float>& values)> readbackFn;

    // Impulse-response loader (Convolution Reverb): receives a WAV file path.
    std::function<void(const char* wavPath)> loadIRFn;

    // Multiband display: fills crossover frequencies (Hz) and per-band gain
    // reduction (dB, positive = attenuation); returns the active band count.
    std::function<int(float* crossoverHz, float* bandGrDb, int maxBands)> multibandFn;

    // --- Presets: settings to start listening from --------------------------------

    /// A named setting: every control at its default, with these on top.
    struct Preset
    {
        std::string name;
        std::string note;                              // what it shows, and what to compare
        std::vector<std::pair<int, float>> values;     // control index and value, in the order given
    };
    std::vector<Preset> presets;
    int currentPreset = -1;          // UI: the preset last applied (-1: none)
    int presetSettle = 0;            // UI frames until the preset's values go in a second time
    std::string presetErrors;        // Controls a preset named that this slot does not have

    // --- Builder helpers ---

    void addSlider(const char* n, float mn, float mx, float def,
                   const char* u = "", bool log = false)
    {
        params.push_back({ n, mn, mx, def, ParamDesc::Slider, u, {}, log });
        values.push_back(def);
    }

    void addToggle(const char* n, bool def = false)
    {
        params.push_back({ n, 0.0f, 1.0f, def ? 1.0f : 0.0f, ParamDesc::Toggle,
                           "", {}, false });
        values.push_back(def ? 1.0f : 0.0f);
    }

    void addChoice(const char* n, std::vector<std::string> opts, int def = 0)
    {
        params.push_back({ n, 0.0f, static_cast<float>(opts.size() - 1),
                           static_cast<float>(def), ParamDesc::Choice, "",
                           std::move(opts), false });
        values.push_back(static_cast<float>(def));
    }

    /// Puts the controls at these indices on a tab of the control panel.
    /// Tabs appear in the order of the calls; controls on no tab go to the
    /// first one ("Main" when no call names it first).
    void group(const char* tab, std::initializer_list<int> indices)
    {
        if (std::find(groups.begin(), groups.end(), tab) == groups.end()) groups.emplace_back(tab);
        for (int i : indices)
            if (i >= 0 && i < static_cast<int>(params.size())) params[static_cast<size_t>(i)].group = tab;
    }

    /// The tab of a control: its own, or the first tab.
    [[nodiscard]] std::string groupOf(int index) const
    {
        const auto& g = params[static_cast<size_t>(index)].group;
        if (!g.empty()) return g;
        return groups.empty() ? std::string("Main") : groups.front();
    }

    /// The control drawn at a place: the lead first, the others in their order.
    [[nodiscard]] int controlAt(int place) const
    {
        if (leadParam < 0) return place;
        if (place == 0) return leadParam;
        return place <= leadParam ? place - 1 : place;
    }

    /// Every tab, in order ("Main" alone when no control is on a tab).
    [[nodiscard]] std::vector<std::string> tabs() const
    {
        if (groups.empty()) return { std::string("Main") };
        return groups;
    }

    // --- Runtime interface ---

    void prepare(const dspark::AudioSpec& spec)
    {
        if (prepareFn) prepareFn(spec);
    }

    /// Audio thread. One published load of the enabled flag decides the call.
    void process(dspark::AudioBufferView<float> buf)
    {
        if (enabled.get() && processFn) processFn(buf);
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
        currentPreset = -1;
        presetSettle = 0;
    }

    // --- Presets ---

    /// The index of the control with this name, or -1.
    [[nodiscard]] int indexOf(const char* control) const
    {
        for (int i = 0; i < static_cast<int>(params.size()); ++i)
            if (params[static_cast<size_t>(i)].name == control) return i;
        return -1;
    }

    /// Adds a preset by the names of its controls (call it once every control
    /// of the slot exists). A name the slot does not have is left out and
    /// noted in presetErrors, which the interface shows.
    void addPreset(const char* presetName, const char* note,
                   std::initializer_list<PresetValue> controls)
    {
        Preset p;
        p.name = presetName;
        p.note = note;
        for (const PresetValue& control : controls)
        {
            const int index = indexOf(control.control);
            if (index < 0)
            {
                presetErrors += std::string(presetName) + ": no control named " + control.control + "\n";
                continue;
            }
            p.values.emplace_back(index, control.value);
        }
        presets.push_back(std::move(p));
    }

    /// Every control to its default, then the preset's own, in order.
    void applyPreset(int index)
    {
        if (index < 0 || index >= static_cast<int>(presets.size())) return;
        applyAllDefaults();
        applyPresetValues(index);
        currentPreset = index;
        // A processor that loads a set of its own when a mode changes (a reverb
        // type) does so a moment later, on the audio thread: the preset's
        // values go in again once it has.
        presetSettle = 12;
    }

    /// Call once per interface frame while the slot is shown.
    void settlePreset()
    {
        if (presetSettle > 0 && --presetSettle == 0 && currentPreset >= 0) applyPresetValues(currentPreset);
    }

private:
    void applyPresetValues(int index)
    {
        for (const auto& value : presets[static_cast<size_t>(index)].values)
        {
            const ParamDesc& d = params[static_cast<size_t>(value.first)];
            applyParam(value.first, std::clamp(value.second, d.min, d.max));
        }
    }
};

} // namespace dsplab
