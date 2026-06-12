// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkPlugin.h
 * @brief Format-agnostic plugin layer: write ONE class, ship every format.
 *
 * The developer describes the plugin declaratively (a Descriptor and a
 * constexpr parameter table) and implements the familiar DSPark contract
 * (`prepare` / `processBlock` plus a `setParameter` switch). The format
 * backends (plugin/vst3/..., later CLAP and AU) translate that single class
 * into each plugin ABI — no external SDK to install, no base class to
 * inherit from.
 *
 * ```cpp
 * struct MyPlugin
 * {
 *     static constexpr auto descriptor = dspark::plugin::Descriptor {
 *         .name      = "My Saturator",
 *         .vendor    = "My Company",
 *         .url       = "https://example.com",
 *         .email     = "mailto:dev@example.com",
 *         .productId = "com.mycompany.mysaturator",   // NEVER change after release
 *         .version   = "1.0.0",
 *     };
 *     static constexpr auto parameters = dspark::plugin::params(
 *         dspark::plugin::param("drive", "Drive", -12.0f, 24.0f, 0.0f, "dB"),
 *         dspark::plugin::param("mix",   "Mix",     0.0f,  1.0f, 1.0f, ""));
 *
 *     void prepare(const dspark::AudioSpec& spec);
 *     void setParameter(int index, float plainValue) noexcept;  // any thread
 *     void processBlock(dspark::AudioBufferView<float> io) noexcept;
 *     // Optional (auto-detected): reset(), getLatency(), getTailSeconds(),
 *     // getState(), setState(data, size).
 * };
 * ```
 *
 * Capability detection is by C++20 concepts: implement only what applies.
 * The full optional menu (each one detected structurally, see the concepts
 * below and docs/plugins.md):
 *   - reset / getLatency / getTailSeconds / getState / setState
 *   - processBlock(io, sidechain)      -> a host-routable sidechain bus
 *   - handleMidiEvent(const MidiEvent&) -> note/CC/pitch-bend input
 *   - setTransport(const TransportInfo&) -> host tempo & timeline, per block
 *   - setOfflineRendering(bool)        -> realtime vs bounce quality switch
 *   - channels (ChannelSupport)        -> mono/stereo bus configurations
 *   - factoryPresets                   -> host-browsable factory programs
 *   - sampleAccurateAutomation         -> opt out of sub-block automation
 *   - hasEditor + editorHtml()         -> the WebView editor layer
 *
 * State: the wrapper always serialises the parameter table itself (stable
 * text ids hashed to 32 bits, version-tolerant); a user getState/setState
 * blob — e.g. DSPark's StateBlob — rides along as an extra section.
 */

#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"

#include <array>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace dspark::plugin {

// -- Descriptor ---------------------------------------------------------------

/** @brief Plugin category (reflected into each format's class metadata).
 *
 * `Fx` processes audio in place (optionally keyed by a sidechain or driven
 * by MIDI — a vocoder, a MIDI-gated effect). `Instrument` GENERATES audio:
 * it has no main audio input in any format (VST3 instrument class, CLAP
 * "instrument" feature, AU `aumu` music device) and almost always pairs
 * with `handleMidiEvent` (see HasMidi). The wrapper clears the output
 * buffers before calling an instrument's processBlock, so voices ADD into
 * `io` without reading it. */
enum class Category
{
    Fx,           ///< Audio effect (audio in -> audio out).
    Instrument    ///< Audio generator (MIDI in -> audio out, no audio input).
};

/**
 * @brief Static identity of a plugin. All fields are plain literals so the
 * whole descriptor can live in a `static constexpr` of the user class.
 *
 * `productId` (reverse-domain string) is the STABLE identity: the format
 * UIDs (VST3 class id, ...) derive from it deterministically. Changing it
 * after a release orphans every saved project — treat it like an ABI.
 */
struct Descriptor
{
    const char* name      = "DSPark Plugin";
    const char* vendor    = "DSPark";
    const char* url       = "";
    const char* email     = "";
    const char* productId = "com.dspark.plugin";
    const char* version   = "1.0.0";
    Category    category  = Category::Fx;
};

// -- Parameters ---------------------------------------------------------------

/**
 * @brief One automatable parameter. Plain values run [min, max]; hosts see
 * the normalized [0, 1] projection. `steps == 0` means continuous,
 * `steps == 1` a toggle, `steps == N` an N+1-position discrete control.
 */
struct Param
{
    const char* id   = "";     ///< Stable text id (state + automation identity).
    const char* name = "";     ///< Display name.
    float minValue   = 0.0f;
    float maxValue   = 1.0f;
    float defValue   = 0.0f;
    const char* unit = "";     ///< Display unit ("dB", "Hz", "%", ...).
    int steps        = 0;      ///< 0 continuous, 1 toggle, N discrete.
};

/** @brief Continuous parameter helper. */
constexpr Param param(const char* id, const char* name,
                      float minValue, float maxValue, float defValue,
                      const char* unit) noexcept
{
    return Param { id, name, minValue, maxValue, defValue, unit, 0 };
}

/** @brief On/off parameter helper. */
constexpr Param toggle(const char* id, const char* name, bool defaultOn) noexcept
{
    return Param { id, name, 0.0f, 1.0f, defaultOn ? 1.0f : 0.0f, "", 1 };
}

/** @brief Builds the parameter table (use inside `static constexpr auto`). */
template <typename... Ps>
constexpr std::array<Param, sizeof...(Ps)> params(Ps... ps) noexcept
{
    return { ps... };
}

// -- Normalisation ------------------------------------------------------------

/** @brief Plain [min, max] -> normalized [0, 1] (steps snap on the way back). */
constexpr double toNormalized(const Param& p, double plain) noexcept
{
    const double range = static_cast<double>(p.maxValue) - p.minValue;
    if (range <= 0.0) return 0.0;
    double n = (plain - p.minValue) / range;
    return n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
}

/** @brief Normalized [0, 1] -> plain [min, max], snapped for stepped params. */
constexpr double toPlain(const Param& p, double normalized) noexcept
{
    double n = normalized < 0.0 ? 0.0 : (normalized > 1.0 ? 1.0 : normalized);
    if (p.steps > 0)
    {
        // Snap to the nearest of steps+1 positions.
        const double scaled = n * p.steps + 0.5;
        const double idx = static_cast<double>(static_cast<long long>(scaled));
        n = idx / p.steps;
        n = n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
    }
    return p.minValue + n * (static_cast<double>(p.maxValue) - p.minValue);
}

/** @brief Case-insensitive "On"/"Off" recognition — the inverse of the toggle
 *  display below. Returns 1 for On, 0 for Off, -1 for anything else. Hosts
 *  round-trip displayed strings through text-to-value (automation lanes,
 *  typed values), so toggles must parse their own output. */
inline int parseToggleText(const char* text) noexcept
{
    auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c; };
    if (text == nullptr) return -1;
    if (lower(text[0]) == 'o' && lower(text[1]) == 'n' && text[2] == '\0') return 1;
    if (lower(text[0]) == 'o' && lower(text[1]) == 'f' && lower(text[2]) == 'f'
        && text[3] == '\0') return 0;
    return -1;
}

/** @brief Formats a plain value for host display ("3.5 dB", "On", "440 Hz"). */
inline void formatValue(const Param& p, double plain, char* out, int outSize) noexcept
{
    if (p.steps == 1)
    {
        std::snprintf(out, static_cast<size_t>(outSize), "%s",
                      plain >= 0.5 * (p.minValue + p.maxValue) ? "On" : "Off");
        return;
    }
    if (p.unit && p.unit[0] != '\0')
        std::snprintf(out, static_cast<size_t>(outSize), "%.2f %s", plain, p.unit);
    else
        std::snprintf(out, static_cast<size_t>(outSize), "%.2f", plain);
}

// -- Stable hashing (parameter ids, class UIDs) -------------------------------

/** @brief FNV-1a 32-bit over a C string — the per-parameter host id. */
constexpr uint32_t hash32(const char* s) noexcept
{
    uint32_t h = 2166136261u;
    while (*s != '\0')
    {
        h ^= static_cast<uint8_t>(*s++);
        h *= 16777619u;
    }
    return h;
}

/** @brief FNV-1a 64-bit with a salt (two runs build a 128-bit class UID). */
constexpr uint64_t hash64(const char* s, uint64_t salt) noexcept
{
    uint64_t h = 14695981039346656037ull ^ salt;
    while (*s != '\0')
    {
        h ^= static_cast<uint8_t>(*s++);
        h *= 1099511628211ull;
    }
    return h;
}

/**
 * @brief Deterministic 16-byte class UID from the stable productId. Each
 * backend salts differently so VST3/CLAP/AU ids never collide.
 */
constexpr std::array<uint8_t, 16> makeUid(const char* productId, uint64_t salt) noexcept
{
    const uint64_t a = hash64(productId, salt);
    const uint64_t b = hash64(productId, salt ^ 0x9E3779B97F4A7C15ull);
    std::array<uint8_t, 16> uid {};
    for (int i = 0; i < 8; ++i)
    {
        uid[static_cast<size_t>(i)]     = static_cast<uint8_t>(a >> (8 * i));
        uid[static_cast<size_t>(i + 8)] = static_cast<uint8_t>(b >> (8 * i));
    }
    return uid;
}

// -- Capability detection -----------------------------------------------------

template <typename P>
concept HasReset = requires(P p) { p.reset(); };

template <typename P>
concept HasLatency = requires(const P p) { { p.getLatency() } -> std::convertible_to<int>; };

template <typename P>
concept HasTail = requires(const P p) { { p.getTailSeconds() } -> std::convertible_to<double>; };

template <typename P>
concept HasGetState = requires(const P p) {
    { p.getState() } -> std::convertible_to<std::vector<uint8_t>>;
};

template <typename P>
concept HasSetState = requires(P p, const uint8_t* d, size_t n) {
    { p.setState(d, n) } -> std::convertible_to<bool>;
};

/** @brief Custom-editor switch. With `hasEditor = false` (or absent) hosts
 *  show their generic parameter UI. With `hasEditor = true` AND
 *  plugin/webview/DSParkWebViewEditor.h included BEFORE the format headers,
 *  the WebView editor layer serves `editorHtml()` inside the host window. */
template <typename P>
concept HasEditor = requires { P::hasEditor; } && P::hasEditor;

/**
 * @brief Sidechain capability. Implement the two-buffer process — the same
 * shape DSPark's own dynamics take:
 *
 * ```cpp
 * void processBlock(dspark::AudioBufferView<float> io,
 *                   dspark::AudioBufferView<float> sidechain) noexcept;
 * ```
 *
 * and every format backend grows a second input the host can route into:
 * a VST3 aux bus, a CLAP non-main port, an AU input element — all named
 * "Sidechain". The key view always mirrors the main width (mono main,
 * mono key) and the wrapper guarantees it is valid and frame-aligned with
 * `io` (pre-allocated silence when the host has nothing connected), so
 * the plugin never branches on availability. Treat the sidechain as
 * read-only. Replaces the single-buffer `processBlock` — implement one or
 * the other, not both.
 */
template <typename P>
concept HasSidechain = requires(P p, AudioBufferView<float> io,
                                AudioBufferView<float> sc) {
    p.processBlock(io, sc);
};

// -- Transport (host tempo & timeline) ------------------------------------------

/**
 * @brief Host transport snapshot, delivered once per audio block.
 *
 * Implement `void setTransport(const dspark::plugin::TransportInfo&)
 * noexcept` (see HasTransport) and every backend feeds it from the native
 * source before each processBlock: the VST3 ProcessContext, the CLAP
 * transport event, the AU host callbacks. Check the `*Valid` flags — hosts
 * differ in what they provide (an offline renderer may have no timeline at
 * all). Fields hold their defaults when the matching flag is false.
 */
struct TransportInfo
{
    double tempoBpm = 120.0;          ///< Current tempo (when tempoValid).
    double ppqPosition = 0.0;         ///< Musical position of frame 0, in quarter notes.
    double barStartPpq = 0.0;         ///< Quarter-note position of the current bar start.
    int    timeSigNumerator = 4;      ///< e.g. 3 in 3/4 (when timeSigValid).
    int    timeSigDenominator = 4;    ///< e.g. 4 in 3/4 (when timeSigValid).
    double loopStartPpq = 0.0;        ///< Cycle start (when loopValid).
    double loopEndPpq = 0.0;          ///< Cycle end (when loopValid).
    bool   playing = false;           ///< Transport is rolling.
    bool   recording = false;         ///< Transport is recording.
    bool   looping = false;           ///< Cycle/loop mode is engaged.
    bool   tempoValid = false;        ///< tempoBpm came from the host.
    bool   positionValid = false;     ///< ppqPosition/barStartPpq came from the host.
    bool   timeSigValid = false;      ///< Time signature came from the host.
    bool   loopValid = false;         ///< Loop points came from the host.

    /** @brief Seconds per quarter note at the current tempo. */
    [[nodiscard]] double secondsPerBeat() const noexcept
    {
        return 60.0 / (tempoBpm > 1.0 ? tempoBpm : 120.0);
    }

    /** @brief Samples per quarter note — the basis for tempo-synced delays/LFOs. */
    [[nodiscard]] double samplesPerBeat(double sampleRate) const noexcept
    {
        return secondsPerBeat() * sampleRate;
    }
};

/**
 * @brief Transport capability: `void setTransport(const TransportInfo&)
 * noexcept`. Called on the audio thread, before processBlock, whenever the
 * host supplied transport data for the block (no call means "same as last
 * block"). This is how tempo-synced delays, LFOs and gates follow the song.
 */
template <typename P>
concept HasTransport = requires(P p, const TransportInfo& t) {
    p.setTransport(t);
};

// -- MIDI ------------------------------------------------------------------------

/**
 * @brief One incoming MIDI-ish event, normalised across formats.
 *
 * VST3 delivers notes as native events and pitch bend / CC / pressure
 * through its controller-mapping scheme; CLAP delivers note events plus raw
 * MIDI; AU delivers raw MIDI bytes. The wrapper translates ALL of them into
 * this one struct so the plugin handles a single vocabulary.
 *
 * `sampleOffset` is the event's position in frames RELATIVE TO THE NEXT
 * processBlock call: events are always delivered (in time order) right
 * before the block that contains them. A simple plugin may ignore the
 * offset (worst error: one automation quantum); a sample-accurate synth
 * starts its voice exactly `sampleOffset` frames into the block.
 */
struct MidiEvent
{
    enum class Type : uint8_t
    {
        NoteOn,            ///< note = key, value = velocity 0..1.
        NoteOff,           ///< note = key, value = release velocity 0..1.
        PitchBend,         ///< value = bend -1..+1 (note unused).
        ControlChange,     ///< note = controller number, value = 0..1.
        ChannelPressure,   ///< value = pressure 0..1 (note unused).
        PolyPressure       ///< note = key, value = pressure 0..1.
    };

    Type    type = Type::NoteOn;
    uint8_t channel = 0;      ///< 0..15.
    uint8_t note = 0;         ///< Key 0..127, or controller number for CC.
    float   value = 0.0f;     ///< Velocity / CC / pressure 0..1; bend -1..+1.
    int     sampleOffset = 0; ///< Frames into the NEXT processBlock call.
};

/**
 * @brief MIDI capability: `void handleMidiEvent(const MidiEvent&) noexcept`.
 * Its presence grows a note/event input in every format (VST3 event bus,
 * CLAP note port, AU music-device selectors) — which is also why PluginBase
 * deliberately ships no default for it. Required for Category::Instrument,
 * optional for MIDI-driven effects. Audio thread; allocation-free.
 */
template <typename P>
concept HasMidi = requires(P p, const MidiEvent& e) {
    p.handleMidiEvent(e);
};

// -- Offline rendering ------------------------------------------------------------

/**
 * @brief Render-mode capability: `void setOfflineRendering(bool) noexcept`.
 * The host flips it to true for non-realtime bounces (VST3 kOffline setup,
 * the CLAP render extension, AU OfflineRender) — the moment to switch to
 * more expensive algorithms (higher oversampling, longer lookahead). Called
 * outside the audio thread, before processing (re)starts. Default: assume
 * realtime.
 */
template <typename P>
concept HasOfflineMode = requires(P p, bool offline) {
    p.setOfflineRendering(offline);
};

// -- Channel support ----------------------------------------------------------------

/**
 * @brief Bus widths a plugin offers to the host. The DSPark contract is
 * channel-agnostic by construction (prepare() receives numChannels and
 * processBlock honours io.numChannels()), so the DEFAULT is mono+stereo:
 * the plugin appears on every track type. Declare
 * `static constexpr auto channels = dspark::plugin::ChannelSupport::...;`
 * only to restrict it (e.g. StereoOnly for M/S wideners). All buses of an
 * instance run the same width — a sidechain follows the main pair.
 */
enum class ChannelSupport
{
    MonoAndStereo,   ///< Negotiates 1->1 or 2->2 with the host (default).
    StereoOnly,      ///< Strictly 2->2 (inherently stereo DSP).
    MonoOnly         ///< Strictly 1->1 (rare; metering/utility plugins).
};

template <typename P>
concept HasChannelSupport = requires {
    { P::channels } -> std::convertible_to<ChannelSupport>;
};

/** @brief The declared channel support of @p P (default: mono+stereo). */
template <typename P>
constexpr ChannelSupport channelSupportOf() noexcept
{
    if constexpr (HasChannelSupport<P>)
        return P::channels;
    else
        return ChannelSupport::MonoAndStereo;
}

/** @brief True when @p P accepts a bus width of @p numChannels (1 or 2). */
template <typename P>
constexpr bool supportsChannelCount(int numChannels) noexcept
{
    constexpr ChannelSupport s = channelSupportOf<P>();
    if (numChannels == 1) return s != ChannelSupport::StereoOnly;
    if (numChannels == 2) return s != ChannelSupport::MonoOnly;
    return false;
}

/** @brief The width every backend starts in before the host negotiates. */
template <typename P>
constexpr int defaultChannelCount() noexcept
{
    return channelSupportOf<P>() == ChannelSupport::MonoOnly ? 1 : 2;
}

// -- Factory presets ----------------------------------------------------------------

/**
 * @brief One factory preset: a name plus a PLAIN value for every parameter,
 * in parameter-table order. Build the table with preset()/presets() so the
 * value count is checked at compile time against the parameter count.
 */
template <size_t NumValues>
struct PresetDef
{
    const char* name = "";
    std::array<float, NumValues> values {};
};

/** @brief Builds one factory preset: `preset("Warm", 6.0f, 0.8f, ...)` —
 *  one PLAIN value per parameter, in table order. */
template <typename... Vs>
constexpr PresetDef<sizeof...(Vs)> preset(const char* name, Vs... values) noexcept
{
    return PresetDef<sizeof...(Vs)> { name, { static_cast<float>(values)... } };
}

/** @brief Builds the factory preset table (all presets must cover the same
 *  parameter count — enforced here; matched against the parameter table by
 *  the backends). Declare as `static constexpr auto factoryPresets = ...`. */
template <typename First, typename... Rest>
constexpr std::array<First, 1 + sizeof...(Rest)> presets(First first, Rest... rest) noexcept
{
    static_assert((std::is_same_v<First, Rest> && ...),
                  "every preset must list one value per parameter");
    return { first, rest... };
}

/**
 * @brief Factory-preset capability: a `static constexpr auto factoryPresets`
 * table built with presets(). The backends publish it natively — a VST3
 * program list, CLAP preset-load + preset-discovery, AU factory presets —
 * so the host's own preset browser offers them. No PluginBase default on
 * purpose: the table's presence changes what hosts display.
 */
template <typename P>
concept HasFactoryPresets = requires {
    { P::factoryPresets.size() } -> std::convertible_to<size_t>;
    { P::factoryPresets[0].name } -> std::convertible_to<const char*>;
    { P::factoryPresets[0].values.size() } -> std::convertible_to<size_t>;
};

/** @brief Number of factory presets of @p P (0 when none declared). */
template <typename P>
constexpr int factoryPresetCountOf() noexcept
{
    if constexpr (HasFactoryPresets<P>)
    {
        static_assert(P::factoryPresets[0].values.size() == P::parameters.size(),
                      "factoryPresets must list one PLAIN value per parameter, "
                      "in parameter-table order");
        return static_cast<int>(P::factoryPresets.size());
    }
    else
        return 0;
}

/** @brief Normalized value of parameter @p paramIndex in factory preset
 *  @p presetIndex (bounds already validated by the caller). */
template <typename P>
constexpr double presetNormalized(int presetIndex, size_t paramIndex) noexcept
{
    if constexpr (HasFactoryPresets<P>)
        return toNormalized(P::parameters[paramIndex],
                            P::factoryPresets[static_cast<size_t>(presetIndex)]
                                .values[paramIndex]);
    else
    {
        (void) presetIndex;
        (void) paramIndex;
        return 0.0;
    }
}

// -- Sample-accurate automation -------------------------------------------------------

/**
 * @brief Sample-accurate opt-out: `static constexpr bool
 * sampleAccurateAutomation = false;`. By default the wrappers split each
 * audio block at automation points (snapped to kAutomationQuantum frames)
 * so fast curves land where the host drew them instead of stepping once
 * per block. Opt out when your plugin has a high fixed cost per
 * processBlock CALL and block-rate automation is acceptable.
 */
template <typename P>
concept HasSampleAccurateOptOut = requires {
    { P::sampleAccurateAutomation } -> std::convertible_to<bool>;
};

template <typename P>
constexpr bool sampleAccurateOf() noexcept
{
    if constexpr (HasSampleAccurateOptOut<P>)
        return P::sampleAccurateAutomation;
    else
        return true;
}

/** @brief Sub-block grain: automation/event splits snap to this many frames
 *  (bounds the per-call overhead; ~0.7 ms at 48 kHz is inaudible). */
inline constexpr int kAutomationQuantum = 32;

/** @brief Hard cap of timestamped events handled per block; the (very
 *  unlikely) excess applies at the position of the last captured event. */
inline constexpr int kMaxBlockEvents = 256;

/**
 * @brief One timestamped in-block event, normalised across formats: a
 * parameter point, a bypass point, or a MIDI event. The backends collect
 * them per block, sort them, and either split processing at quantum
 * boundaries (sample-accurate mode) or apply them all up front.
 */
struct BlockEvent
{
    enum class Kind : uint8_t { Param, Bypass, Midi };
    int32_t  offset = 0;        ///< Frame position within the block.
    Kind     kind = Kind::Param;
    uint32_t paramId = 0;       ///< hash32 id (Kind::Param only).
    double   value = 0.0;       ///< Normalized value / bypass >= 0.5.
    MidiEvent midi {};          ///< Kind::Midi payload.
};

/** @brief Stable insertion sort by offset (tiny N, allocation-free —
 *  audio-thread safe; equal offsets keep arrival order). */
inline void sortBlockEvents(BlockEvent* events, int count) noexcept
{
    for (int i = 1; i < count; ++i)
    {
        BlockEvent key = events[i];
        int j = i - 1;
        while (j >= 0 && events[j].offset > key.offset)
        {
            events[j + 1] = events[j];
            --j;
        }
        events[j + 1] = key;
    }
}

// -- Editor contract (used by plugin/webview/DSParkWebViewEditor.h) -------------

/** @brief Editor window size in logical pixels (physical = logical x host scale). */
struct EditorSize
{
    int width  = 480;
    int height = 320;
};

/**
 * @brief How the host may resize the editor window.
 * Declare `static constexpr EditorResize editorResize = ...;` in the plugin
 * class; without it the window is Fixed (or Free if the simpler
 * `editorResizable = true` shorthand is present).
 */
enum class EditorResize
{
    Fixed,        ///< The window is exactly editorSize; hosts cannot drag-resize it.
    Free,         ///< Drag-resizable between 0.5x and 3x the declared size.
    KeepAspect    ///< Drag-resizable, locked to the declared width:height ratio.
};

/** @brief `static const char* editorHtml()` — the editor page (HTML/CSS/JS),
 *  usually a raw string literal. Required when `hasEditor` is true. */
template <typename P>
concept HasEditorHtml = requires {
    { P::editorHtml() } -> std::convertible_to<const char*>;
};

/** @brief Optional `static constexpr EditorSize editorSize { w, h };`. */
template <typename P>
concept HasEditorSize = requires {
    { P::editorSize.width } -> std::convertible_to<int>;
    { P::editorSize.height } -> std::convertible_to<int>;
};

/** @brief The declared (logical) editor size, or the 480x320 default. */
template <typename P>
constexpr EditorSize editorSizeOf() noexcept
{
    if constexpr (HasEditorSize<P>)
        return EditorSize { static_cast<int>(P::editorSize.width),
                            static_cast<int>(P::editorSize.height) };
    else
        return EditorSize {};
}

/** @brief Optional `static constexpr bool editorResizable = true;` — shorthand
 *  for `editorResize = EditorResize::Free` (default: fixed size). */
template <typename P>
concept HasEditorResizable = requires { P::editorResizable; } && P::editorResizable;

/** @brief Optional `static constexpr EditorResize editorResize = ...;` —
 *  full resize policy; takes precedence over `editorResizable`. */
template <typename P>
concept HasEditorResize = requires {
    { P::editorResize } -> std::convertible_to<EditorResize>;
};

/** @brief The effective resize policy for plugin class @p P. */
template <typename P>
constexpr EditorResize editorResizeOf() noexcept
{
    if constexpr (HasEditorResize<P>)
        return P::editorResize;
    else if constexpr (HasEditorResizable<P>)
        return EditorResize::Free;
    else
        return EditorResize::Fixed;
}

/** @brief Resize bounds: hosts may shrink to half and grow to 3x the declared size. */
inline constexpr double kEditorMinSizeFactor = 0.5;
inline constexpr double kEditorMaxSizeFactor = 3.0;

/**
 * @brief Applies the resize policy of @p P to a size the host proposes:
 * Fixed pins it, Free clamps each axis to the 0.5x..3x window, KeepAspect
 * fits the declared ratio INSIDE the proposal — never exceeding it on
 * either axis. That last property is essential: hosts size the plugin area
 * inside a window the user controls, so answering "larger" on any axis
 * pushes the plugin outside its own window (clipped bottom in REAPER).
 */
template <typename P>
constexpr void constrainEditorSize(double& width, double& height, double scale) noexcept
{
    const EditorSize logical = editorSizeOf<P>();
    constexpr EditorResize mode = editorResizeOf<P>();
    if constexpr (mode == EditorResize::Fixed)
    {
        width  = logical.width * scale;
        height = logical.height * scale;
    }
    else
    {
        const double minW = logical.width * scale * kEditorMinSizeFactor;
        const double maxW = logical.width * scale * kEditorMaxSizeFactor;
        const double minH = logical.height * scale * kEditorMinSizeFactor;
        const double maxH = logical.height * scale * kEditorMaxSizeFactor;
        width  = width < minW ? minW : (width > maxW ? maxW : width);
        height = height < minH ? minH : (height > maxH ? maxH : height);
        if constexpr (mode == EditorResize::KeepAspect)
        {
            const double ratio = static_cast<double>(logical.width) / logical.height;
            double fitW = width;
            if (height * ratio < fitW) fitW = height * ratio;   // inside-fit
            fitW   = fitW < minW ? minW : (fitW > maxW ? maxW : fitW);
            width  = fitW;
            height = fitW / ratio;
        }
    }
}

/** @brief Optional `static constexpr bool editorDebug = true;` — enables the
 *  browser DevTools in the editor (WebView2; development builds only). */
template <typename P>
concept HasEditorDebug = requires { P::editorDebug; } && P::editorDebug;

/** @brief Optional `static const char* editorDevFile()` — absolute path of an
 *  HTML file to load INSTEAD of editorHtml() while developing: edit the file,
 *  reopen the editor, no recompile. Falls back to editorHtml() when the file
 *  is missing, so the same build still works elsewhere. Strip from releases. */
template <typename P>
concept HasEditorDevFile = requires {
    { P::editorDevFile() } -> std::convertible_to<const char*>;
};

// -- PluginBase: the whole contract, visible in one place ----------------------

/**
 * @brief Optional convenience base that makes EVERY contract method visible
 * and overridable from one place — IDE-discoverable, without the virtuals.
 *
 * ```cpp
 * struct MyPlugin : dspark::plugin::PluginBase<MyPlugin> { ... };
 * ```
 *
 * Each method below ships a safe default; define the same signature in your
 * class to replace it (plain C++ shadowing — resolved at compile time, zero
 * dispatch cost, no `override` keyword involved). Delete nothing, implement
 * what applies, and your IDE's autocomplete shows you the full menu.
 *
 * Inheriting is OPTIONAL: a free-standing struct with the same members works
 * identically (the wrappers detect capabilities structurally either way).
 * You must still provide the two identity members yourself — `descriptor`
 * and `parameters` have no safe default — plus `prepare`, `setParameter`
 * and `processBlock`.
 *
 * Three capabilities have no default here on purpose, because their mere
 * presence changes what the host sees:
 *  - a sidechain input: define `void processBlock(AudioBufferView<float> io,
 *    AudioBufferView<float> sidechain) noexcept` INSTEAD of the
 *    single-buffer form and every format grows a host-routable "Sidechain"
 *    bus (see HasSidechain);
 *  - MIDI input: define `void handleMidiEvent(const MidiEvent&) noexcept`
 *    and every format grows a note/event input (see HasMidi);
 *  - factory presets: declare `static constexpr auto factoryPresets =
 *    presets(...)` and every format publishes a host-browsable program
 *    list (see HasFactoryPresets).
 * Channel support is declarative too (`static constexpr auto channels`) and
 * already defaults to mono+stereo without any member here.
 */
template <typename Derived>
struct PluginBase
{
    /**
     * Clears processing history (delay lines, envelopes, reverb tails) on
     * transport jumps. CLAP hosts call this directly; VST3 hosts re-activate
     * (your `prepare()` runs again) instead.
     * Default: nothing to clear.
     */
    void reset() noexcept {}

    /**
     * Samples of delay your chain introduces (lookahead limiters,
     * linear-phase EQ, oversampling, FFT processing). The host shifts other
     * tracks to compensate — report it accurately or parallel paths phase.
     * Read after `prepare()`; sum your DSPark effects' `getLatency()`.
     * Maps to VST3 `getLatencySamples` and the CLAP latency extension.
     * Default: zero latency.
     */
    [[nodiscard]] int getLatency() const noexcept { return 0; }

    /**
     * How long sound continues after the input stops (reverb decay, delay
     * feedback). Hosts keep processing you that long instead of cutting the
     * tail. Maps to VST3 `getTailSamples` and the CLAP tail extension.
     * Default: no tail.
     */
    [[nodiscard]] double getTailSeconds() const noexcept { return 0.0; }

    /**
     * Extra state BEYOND the parameters (learned profiles, loaded IRs,
     * editor layout). The wrapper already saves/restores every parameter by
     * its stable id on its own — most plugins never touch this pair.
     * DSPark's StateBlob (Core/StateBlob.h) is the natural serializer.
     * Default: no extra state.
     */
    [[nodiscard]] std::vector<uint8_t> getState() const { return {}; }

    /** @copydoc getState — restore side. Return false on a foreign blob. */
    bool setState(const uint8_t*, size_t) { return false; }

    /**
     * Host transport snapshot, delivered on the audio thread before each
     * processBlock whenever the host supplies timeline data. Read tempoBpm
     * and ppqPosition (checking the *Valid flags) to sync delays, LFOs and
     * gates to the song. Maps to the VST3 ProcessContext, the CLAP
     * transport event and the AU host callbacks.
     * Default: transport ignored.
     */
    void setTransport(const TransportInfo&) noexcept {}

    /**
     * Render-mode switch: hosts flip it to true for non-realtime bounces —
     * the moment to raise oversampling or other quality/cost trade-offs.
     * Called outside the audio thread, before processing (re)starts. Maps
     * to the VST3 kOffline process mode, the CLAP render extension and the
     * AU OfflineRender property.
     * Default: render mode ignored (always the realtime path).
     */
    void setOfflineRendering(bool) noexcept {}

    /**
     * Custom editor flag. While false, hosts show their generic parameter UI.
     * Set it to true, implement `static const char* editorHtml()` and include
     * plugin/webview/DSParkWebViewEditor.h before the format headers to get a
     * WebView editor embedded in the host window (see docs/plugins.md).
     */
    static constexpr bool hasEditor = false;
};

// -- Wrapper-side state serialisation ------------------------------------------
//
// Every backend stores plugin state in ONE container format so presets stay
// portable across formats and tolerant across versions (same philosophy as
// Core/StateBlob): a magic/version header, the parameter table keyed by
// hash32(id) — unknown ids are skipped, missing ids keep defaults — and an
// optional opaque user-state section appended verbatim.

inline constexpr uint32_t kStateMagic   = 0x4453504Bu;   // "DSPK"
inline constexpr uint32_t kStateVersion = 1u;

template <typename P>
inline std::vector<uint8_t> buildState(const P& user, const double* normalized,
                                       size_t numParams)
{
    auto push32 = [](std::vector<uint8_t>& v, uint32_t x) {
        v.push_back(static_cast<uint8_t>(x));
        v.push_back(static_cast<uint8_t>(x >> 8));
        v.push_back(static_cast<uint8_t>(x >> 16));
        v.push_back(static_cast<uint8_t>(x >> 24));
    };
    std::vector<uint8_t> blob;
    push32(blob, kStateMagic);
    push32(blob, kStateVersion);
    push32(blob, static_cast<uint32_t>(numParams));
    for (size_t i = 0; i < numParams; ++i)
    {
        push32(blob, hash32(P::parameters[i].id));
        uint64_t bits = 0;
        const double v = normalized[i];
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        push32(blob, static_cast<uint32_t>(bits));
        push32(blob, static_cast<uint32_t>(bits >> 32));
    }
    if constexpr (HasGetState<P>)
    {
        const std::vector<uint8_t> userBlob = user.getState();
        push32(blob, static_cast<uint32_t>(userBlob.size()));
        blob.insert(blob.end(), userBlob.begin(), userBlob.end());
    }
    else
        push32(blob, 0u);
    return blob;
}

/** @brief Parses a state blob; fills `normalized` (defaults pre-loaded by the
 *  caller) and forwards the user section. Returns false on a foreign blob. */
template <typename P>
inline bool applyState(P& user, const uint8_t* data, size_t size, double* normalized)
{
    auto read32 = [&](size_t& pos, uint32_t& out) {
        if (pos + 4 > size) return false;
        out = static_cast<uint32_t>(data[pos])
            | static_cast<uint32_t>(data[pos + 1]) << 8
            | static_cast<uint32_t>(data[pos + 2]) << 16
            | static_cast<uint32_t>(data[pos + 3]) << 24;
        pos += 4;
        return true;
    };
    size_t pos = 0;
    uint32_t magic = 0, version = 0, count = 0;
    if (!read32(pos, magic) || magic != kStateMagic) return false;
    if (!read32(pos, version) || version == 0) return false;
    if (!read32(pos, count)) return false;

    constexpr size_t numParams = P::parameters.size();
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t id = 0, lo = 0, hi = 0;
        if (!read32(pos, id) || !read32(pos, lo) || !read32(pos, hi)) return false;
        const uint64_t bits = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        for (size_t k = 0; k < numParams; ++k)
            if (hash32(P::parameters[k].id) == id)
            {
                normalized[k] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                break;
            }
        // Unknown ids are skipped: forward-compatible by construction.
    }

    uint32_t userSize = 0;
    if (read32(pos, userSize) && userSize > 0 && pos + userSize <= size)
    {
        if constexpr (HasSetState<P>)
            user.setState(data + pos, userSize);
    }
    return true;
}

} // namespace dspark::plugin
