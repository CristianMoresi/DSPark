// DSParkLab — Effect Slot abstraction
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

#include <atomic>
#include <functional>
#include <memory>
#include <string>
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
    }
};

} // namespace dsplab
