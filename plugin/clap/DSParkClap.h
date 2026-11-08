// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DSParkClap.h
 * @brief Native CLAP backend: the same plugin class, one more macro.
 *
 * CLAP (https://cleveraudio.org, MIT, vendored under plugin/clap/clap/) is a
 * plain-C plugin ABI - the cleanest of the desktop formats. This backend
 * exposes the exact same user class as the VST3 backend:
 *
 * ```cpp
 * #include "plugin/vst3/DSParkVst3.h"
 * #include "plugin/clap/DSParkClap.h"
 * struct MyPlugin { ... };           // see plugin/DSParkPlugin.h
 * DSPARK_VST3_PLUGIN(MyPlugin)
 * DSPARK_CLAP_PLUGIN(MyPlugin)       // both ABIs can live in ONE binary:
 * ```                                // ship it as .vst3 AND as .clap
 *
 * Implementation notes:
 * - CLAP parameters travel in PLAIN values (min..max), so clap_param_info
 *   maps 1:1 onto the declarative table; the wrapper still keeps its
 *   normalized shadows so state blobs stay byte-identical with the VST3
 *   backend (presets are portable across formats by construction).
 * - Parameter events apply SAMPLE-ACCURATELY by default: every timestamped
 *   event splits processing at kAutomationQuantum boundaries (plugins may
 *   opt out, see DSParkPlugin.h). DSPark setters smooth internally either
 *   way.
 * - The wrapper-owned Bypass uses CLAP_PARAM_IS_BYPASS and the same short
 *   crossfade against the kept dry signal.
 * - Every parameter is CLAP_PARAM_IS_MODULATABLE: a host's global
 *   modulation (CLAP_EVENT_PARAM_MOD, plain units) is added to the value
 *   the plugin's setParameter receives, clamped and snapped like any other
 *   value, sample-accurately like automation. The parameter's own value
 *   (get_value, state, the editor) stays unmodulated, as CLAP requires.
 * - perVoice() parameters also declare per-note-id, per-key, per-channel
 *   and per-port modulation, and per-note-id, per-key and per-channel
 *   automation. Each addressed modulation amount or value reaches the
 *   plugin as one MidiEvent::Type::NoteModulation per voice it matches,
 *   carrying that voice's plain value (re-sent when a value it follows
 *   moves), and the host hears when each voice ends (NOTE_END): when
 *   isVoiceActive() turns false, or at the voice's NoteOff without that
 *   query. Past 128 simultaneous voices a new one retires the oldest (a
 *   released one first) with its NOTE_END, so the host never keeps a
 *   voice's modulators alive forever.
 * - Note expressions (volume, pan, tuning, vibrato, expression, brightness,
 *   pressure) reach the plugin as MidiEvent::Type::NoteExpression, one per
 *   matching voice (only the declared ones, when the class declares
 *   noteExpressions); NoteOn/NoteOff carry the host's note id.
 * - start_processing() clears the processing history through the plugin's
 *   reset(), so a host that stops and restarts processing (a sleeping
 *   device waking up, a transport restart) never hears a stale tail.
 * - Buses follow the declared ChannelSupport through audio-ports-config
 *   (Mono / Stereo configurations); Category::Instrument drops the audio
 *   inputs and HasMidi adds a note port accepting both CLAP note events
 *   and raw MIDI (notes, pitch bend, CC, pressure).
 * - HasTransport plugins receive the per-block clap transport as a
 *   TransportInfo; HasOfflineMode plugins get the render extension; a
 *   latency change after parameter motion asks the host for a restart
 *   (request_callback, then request_restart on the main thread, as the
 *   CLAP latency extension requires while active) and the new value is
 *   reported through clap_host_latency from the next activate().
 * - Threading: while active, main-thread edits (editor, state, presets)
 *   are handed to the audio thread, which calls the user's setParameter at
 *   its next process()/flush() - one writer, as DSPark setters require.
 * - Notes: a NOTE_OFF/NOTE_CHOKE addressed with wildcards (-1) becomes one
 *   NoteOff per matching note the plugin was told is sounding; a choke
 *   arrives as a NoteOff with MidiEvent::choke set (end the voice now).
 * - Factory presets publish through preset-load + preset-discovery (the
 *   plugin DSO itself is the preset container; load keys are indices).
 * - process() runs under a DenormalGuard (FTZ/DAZ) so user DSP outside
 *   DSPark's own classes is covered in hosts that do not set it.
 * - GUI: when plugin/webview/DSParkWebViewEditor.h is included before this
 *   header and the class declares `hasEditor = true`, the gui extension
 *   embeds the WebView editor (otherwise hosts show their generic editor).
 *   Editor edits reach the host as gesture begin/end plus
 *   CLAP_EVENT_PARAM_VALUE through a lock-free queue drained by
 *   process()/flush(), so automation recording works. A resizable editor
 *   reopens at the size the user left, also in the next session (the size
 *   rides in the state); a state loaded while the editor is open asks the
 *   host to resize it (clap_host_gui request_resize).
 */

#define DSPARK_PLUGIN_CLAP_INCLUDED 1

#include "../DSParkPlugin.h"

#include "clap/clap.h"

// Survive a <windows.h> included earlier without NOMINMAX (restored at the end).
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace dspark::plugin::clap_backend {

inline constexpr uint32_t kBypassParamId = 0x42595053u;   // matches the VST3 backend
inline constexpr int      kBypassRampSamples = 256;

/**
 * @brief The voices this backend started (one per NoteOn it delivered), with
 * the host's modulation amount of every perVoice() parameter for each one.
 *
 * CLAP addresses voices by (port, channel, key, note id), any part -1 for
 * "any". MidiEvent names one voice, so the backend resolves each per-voice
 * event (note expression, polyphonic modulation, per-note automation)
 * against this table and delivers it once per matching voice - which also
 * covers the per-key, per-channel and per-port forms. Audio thread only;
 * fixed capacity, no allocation. When every slot is in use, a new voice
 * takes the slot of the oldest one (a released voice first): the backend
 * reports that voice's end to the host first, so every voice the host
 * started still receives exactly one NOTE_END.
 */
template <size_t NumPerVoice>
class VoiceTable
{
public:
    static constexpr int kCapacity = 128;

    /// A voice's identity (the part every scan reads; kept small).
    struct Voice
    {
        int16_t channel = 0;
        int16_t key = 0;
        int32_t noteId = -1;
        uint32_t serial = 0;    ///< start order, for choosing the oldest
        bool    used = false;
        bool    held = false;   ///< false once its NoteOff arrived
    };

    /// What the host set for each perVoice() parameter of one voice.
    struct VoiceParams
    {
        static constexpr size_t kSlots = NumPerVoice == 0 ? 1 : NumPerVoice;
        std::array<double, kSlots> amount {};
        std::array<bool, kSlots> modulated {};
        /// Per-note automation: the voice's own plain value of the parameter.
        std::array<double, kSlots> value {};
        std::array<bool, kSlots> valued {};
    };

    void clear() noexcept
    {
        for (auto& voice : voices_) voice.used = false;
        end_ = 0;
    }

    /** Starts a voice; returns its slot, or -1 when every slot is in use. */
    int start(int channel, int key, int32_t noteId) noexcept
    {
        for (int i = 0; i < kCapacity; ++i)
        {
            Voice& voice = voices_[static_cast<size_t>(i)];
            if (voice.used) continue;
            voice.channel = static_cast<int16_t>(channel);
            voice.key = static_cast<int16_t>(key);
            voice.noteId = noteId;
            voice.serial = nextSerial_++;
            voice.used = true;
            voice.held = true;
            if constexpr (NumPerVoice > 0)
            {
                // A new voice starts unmodulated (the amounts and values only
                // count once their flag is set).
                auto& params = params_[static_cast<size_t>(i)];
                params.modulated.fill(false);
                params.valued.fill(false);
            }
            if (i >= end_) end_ = i + 1;
            return i;
        }
        return -1;
    }

    /** The slot a new voice takes when the table is full: the oldest
     *  released voice, else the oldest one; -1 when no voice is in use. */
    [[nodiscard]] int oldest() const noexcept
    {
        int best = -1;
        bool bestHeld = true;
        uint32_t bestAge = 0;
        for (int i = 0; i < end_; ++i)
        {
            const Voice& voice = voices_[static_cast<size_t>(i)];
            if (!voice.used) continue;
            const uint32_t age = nextSerial_ - voice.serial;   // wrap-safe
            if (best < 0 || (bestHeld && !voice.held)
                || (bestHeld == voice.held && age > bestAge))
            {
                best = i;
                bestHeld = voice.held;
                bestAge = age;
            }
        }
        return best;
    }

    /** One past the highest slot in use (scans stop there). */
    [[nodiscard]] int end() const noexcept { return end_; }

    /** Calls @p fn(slot) for every voice the address matches (each part -1
     *  for "any"; the one note port has index 0). */
    template <typename Fn>
    void forEachMatching(int port, int channel, int key, int32_t noteId, Fn&& fn) noexcept
    {
        if (port > 0) return;
        for (int i = 0; i < end_; ++i)
        {
            const Voice& voice = voices_[static_cast<size_t>(i)];
            if (!voice.used
                || (channel >= 0 && voice.channel != channel)
                || (key >= 0 && voice.key != key)
                || (noteId >= 0 && voice.noteId != noteId))
                continue;
            fn(i);
        }
    }

    Voice& operator[](int slot) noexcept { return voices_[static_cast<size_t>(slot)]; }

    /** The perVoice() parameter state of voice @p slot. */
    VoiceParams& paramsOf(int slot) noexcept
    {
        return params_[NumPerVoice > 0 ? static_cast<size_t>(slot) : 0u];
    }

    /** Frees voice @p slot. */
    void release(int slot) noexcept
    {
        voices_[static_cast<size_t>(slot)].used = false;
        while (end_ > 0 && !voices_[static_cast<size_t>(end_ - 1)].used) --end_;
    }

private:
    std::array<Voice, kCapacity> voices_ {};
    std::array<VoiceParams, (NumPerVoice > 0 ? kCapacity : 1)> params_ {};
    int end_ = 0;   // one past the highest slot in use
    uint32_t nextSerial_ = 0;
};

template <typename P>
struct Plugin
{
    static constexpr size_t kNumParams = P::parameters.size();
    static constexpr bool kIsInstrument =
        P::descriptor.category == Category::Instrument;
    static constexpr int kNumPresets = factoryPresetCountOf<P>();
    static_assert(!(kIsInstrument && HasSidechain<P>),
                  "an Instrument has no audio inputs; remove the sidechain "
                  "processBlock or use Category::Fx");
    static_assert(!kIsInstrument || HasMidi<P>,
                  "an Instrument needs handleMidiEvent (see HasMidi): it has "
                  "no audio input to process");
    static_assert(paramIdsUnique<P>(),
                  "two parameter ids share a hash32 (or collide with the "
                  "reserved PRGM/BYPS/EDSW/EDSH state ids): automation and state "
                  "would cross-wire. Rename one id.");
    static constexpr size_t kNumPerVoice = perVoiceParamCount<P>();
    static constexpr bool kHasPerVoice = kNumPerVoice > 0;
    static_assert(!kHasPerVoice || HasMidi<P>,
                  "a perVoice() parameter needs handleMidiEvent (see HasMidi): "
                  "only a plugin with a note input has voices to modulate");

    clap_plugin_t plugin {};            // the C-facing object (plugin_data = this)
    const clap_host_t* host = nullptr;
    const clap_host_latency_t* hostLatency = nullptr;
    const clap_host_params_t* hostParams = nullptr;
    const clap_host_preset_load_t* hostPresetLoad = nullptr;

    P user {};
    double sampleRate = 48000.0;
    uint32_t maxFrames = 0;
    bool prepared = false;
    /// Latency in samples, as last reported to the host. The processing call
    /// re-reads it after parameter motion while the host fetches it from its
    /// own thread, so it is a published word rather than a plain int. It
    /// stands alone -- nothing else is ordered against it, and the host is
    /// told to re-fetch through the notification path below -- so the plain
    /// store/load idiom for a single independent value is the whole contract.
    std::atomic<int> cachedLatency { 0 };
    int currentChannels = defaultChannelCount<P>();   // selected ports config
    bool offlineRender = false;          // last render-extension setting
    std::atomic<bool> latencyDirty { false };   // audio -> main thread notify
    int latencyAtActivate = 0;           // what the host last learned in activate()

    /// True between activate() and deactivate(): process()/flush() may then
    /// run on the audio thread at any moment, so main-thread edits (editor,
    /// state, presets) go through `pending` and the audio thread calls the
    /// user's setParameter - DSPark setters such as Equalizer::setBand allow
    /// exactly one writer.
    std::atomic<bool> active { false };
    PendingParams<kNumParams> pending;
    BlockEventList blockEvents;          // one block of host events (audio thread)
    ActiveNotes sounding;                // for CLAP's wildcard note-off / choke

    /// The voices delivered so far, for per-voice events (audio thread).
    VoiceTable<kNumPerVoice> voices;
    /// Voices that ended this block; reported to the host as NOTE_END when the
    /// plugin has perVoice() parameters (the host then retires the voice's
    /// modulators). Sized for the most one call can end: every voice in the
    /// table plus one per note the block can carry (each NoteOn into a full
    /// table ends the oldest voice), so no NOTE_END is ever dropped.
    struct EndedVoice
    {
        int16_t channel;
        int16_t key;
        int32_t noteId;
    };
    static constexpr int kMaxEndedVoices =
        VoiceTable<kNumPerVoice>::kCapacity + kMaxBlockEvents;
    EndedVoice endedVoices[kMaxEndedVoices] {};
    int endedCount = 0;

    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    /// Host modulation amount per parameter, plain units (audio thread).
    double modulation[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<bool>   bypass { false };
    std::atomic<int>    currentProgram { 0 };   // last factory preset loaded
    float bypassMix = 0.0f;
    std::vector<float> dryL, dryR;
    std::vector<float> silence;   // stand-in sidechain when not connected
    DryDelay dryDelay;            // latency-aligned dry path of the bypass
    /// The editor size the user left (also without the editor layer, so a
    /// session saved where the editor ran keeps its size here).
    EditorSizeMemory<P> editorMemory;

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- editor state + UI -> host event queue (single producer: main thread;
    // single consumer: process() on audio or flush() on main - never both).
    webview_ui::Editor<P> guiEditor;
    bool   guiActive = false;
    double guiScale = 1.0;
    int    guiWidth = 0, guiHeight = 0;
    bool   guiEditActive[kNumParams == 0 ? 1 : kNumParams] {};
    const clap_host_gui_t* hostGui = nullptr;   // resize requests (state loads)
#if defined(__linux__)
    // GTK is driven from the host's run loop: a timer-support tick pumps it.
    const clap_host_timer_support_t* hostTimer = nullptr;
    clap_id guiTimerId = CLAP_INVALID_ID;
#endif

    enum : uint8_t { kUiGestureBegin = 0, kUiValue = 1, kUiGestureEnd = 2 };
    struct UiEvent
    {
        uint32_t paramId;
        uint8_t  kind;
        double   value;
    };
    static constexpr uint32_t kUiQueueSize = 256;   // power of two
    UiEvent uiQueue[kUiQueueSize] {};
    std::atomic<uint32_t> uiHead { 0 }, uiTail { 0 };
#endif

    explicit Plugin(const clap_host_t* h) noexcept : host(h)
    {
        for (size_t i = 0; i < kNumParams; ++i)
            shadow[i].store(toNormalized(P::parameters[i], P::parameters[i].defValue),
                            std::memory_order_relaxed);
    }

    static Plugin* self(const clap_plugin_t* p) noexcept
    {
        return static_cast<Plugin*>(p->plugin_data);
    }

    /** The plain value the DSP hears for a parameter at @p normalized: the
     *  value plus the host's modulation amount, clamped to the range and
     *  snapped like any other value (CLAP: value heard = value + mod). */
    double heardPlain(int index, double normalized) const noexcept
    {
        const auto& spec = P::parameters[static_cast<size_t>(index)];
        const double value = toPlain(spec, normalized);
        const double mod = modulation[static_cast<size_t>(index)];
        return (mod == 0.0) ? value : toPlain(spec, toNormalized(spec, value + mod));
    }

    void applyNormalized(int index, double normalized) noexcept
    {
        // Host values are untrusted doubles: a NaN would pass both range
        // clamps, poison the shadow and reach the user's setter. Ignore it.
        if (normalized != normalized) return;
        shadow[static_cast<size_t>(index)].store(normalized, std::memory_order_relaxed);
        user.setParameter(index, static_cast<float>(heardPlain(index, normalized)));
    }

    static int indexOfParamId(uint32_t id) noexcept
    {
        return paramIndexOf<P>(id);
    }

    /**
     * A change made on the main thread (editor, state load, preset). While
     * active only the shadow is written and the parameter is marked; the
     * audio thread applies it at its next process() or flush(). The old code
     * called setParameter here, concurrently with process(). Returns true
     * when the change reached the DSP immediately.
     */
    bool applyFromUi(int index, double normalized) noexcept
    {
        if (normalized != normalized) return false;
        if (active.load(std::memory_order_acquire))
        {
            shadow[static_cast<size_t>(index)].store(normalized, std::memory_order_relaxed);
            pending.mark(static_cast<size_t>(index));
            return false;
        }
        applyNormalized(index, normalized);
        return true;
    }

    bool drainPending() noexcept
    {
        return pending.drain([this](int i) {
            user.setParameter(i, static_cast<float>(heardPlain(
                i, shadow[static_cast<size_t>(i)].load(std::memory_order_relaxed))));
            resendVoiceModulation(i, 0, false);
        });
    }

    // --- per-voice events (polyphonic modulation and automation, expressions) ---

    /** The plain value one voice hears for perVoice() parameter @p index, as
     *  CLAP defines it: the voice's own value when the host automates the
     *  parameter per note (else the parameter's value), plus the voice's own
     *  modulation amount, which already includes the global signal (else the
     *  global amount) - clamped and snapped like any value. */
    double voiceHeardPlain(int index, bool valued, double value, bool modulated,
                           double amount) const noexcept
    {
        const auto& spec = P::parameters[static_cast<size_t>(index)];
        const double base = valued
            ? value
            : toPlain(spec, shadow[static_cast<size_t>(index)].load(std::memory_order_relaxed));
        const double mod = modulated ? amount : modulation[static_cast<size_t>(index)];
        return toPlain(spec, toNormalized(spec, base + mod));
    }

    /** What a voice with the per-voice state @p params hears for perVoice()
     *  parameter @p index (its per-voice slot @p slot). */
    template <typename Params>
    double voiceHeardPlain(int index, int slot, const Params& params) const noexcept
    {
        const auto s = static_cast<size_t>(slot);
        return voiceHeardPlain(index, params.valued[s], params.value[s],
                               params.modulated[s], params.amount[s]);
    }

    /** One per-voice event to the plugin, addressed to the exact voice
     *  (@p channel, @p key, @p noteId). */
    void deliverToVoice(MidiEvent midi, int channel, int key, int32_t noteId) noexcept
    {
        if constexpr (HasMidi<P>)
        {
            midi.channel = static_cast<uint8_t>(channel & 0x0F);
            midi.note = static_cast<uint8_t>(key & 0x7F);
            midi.noteId = noteId;
            user.handleMidiEvent(midi);
        }
        else
        {
            (void) midi;
            (void) channel;
            (void) key;
            (void) noteId;
        }
    }

    /** A perVoice() parameter's global value moved (@p modulationMoved
     *  false) or its global modulation amount did (true): every voice whose
     *  heard value follows the one that moved, and that the plugin hears
     *  through NoteModulation, is told its new value. A voice with its own
     *  value ignores the global value; one with its own amount ignores the
     *  global amount. */
    void resendVoiceModulation(int index, int sampleOffset, bool modulationMoved) noexcept
    {
        if constexpr (kHasPerVoice)
        {
            const int slot = perVoiceSlotTable<P>[static_cast<size_t>(index)];
            if (slot < 0) return;
            const auto s = static_cast<size_t>(slot);
            const auto& spec = P::parameters[static_cast<size_t>(index)];
            // The global value and amount, read once for every voice.
            const double globalValue =
                toPlain(spec, shadow[static_cast<size_t>(index)].load(std::memory_order_relaxed));
            const double globalAmount = modulation[static_cast<size_t>(index)];
            MidiEvent midi {};
            midi.type = MidiEvent::Type::NoteModulation;
            midi.param = static_cast<int16_t>(index);
            midi.sampleOffset = sampleOffset;
            for (int v = 0; v < voices.end(); ++v)
            {
                const auto& voice = voices[v];
                if (!voice.used) continue;
                const auto& params = voices.paramsOf(v);
                const bool valued = params.valued[s];
                const bool modulated = params.modulated[s];
                if (modulationMoved ? (!valued || modulated) : (!modulated || valued)) continue;
                const double base = valued ? params.value[s] : globalValue;
                const double amount = modulated ? params.amount[s] : globalAmount;
                midi.value = static_cast<float>(toPlain(spec, toNormalized(spec, base + amount)));
                deliverToVoice(midi, voice.channel, voice.key, voice.noteId);
            }
        }
        else
        {
            (void) index;
            (void) sampleOffset;
            (void) modulationMoved;
        }
    }

    /** Retires voice @p slot; with perVoice() parameters the host hears of it
     *  (NOTE_END) at the end of the block. */
    void endVoice(int slot) noexcept
    {
        if constexpr (kHasPerVoice)
        {
            if (endedCount < kMaxEndedVoices)   // by construction (see kMaxEndedVoices)
            {
                const auto& voice = voices[slot];
                endedVoices[endedCount++] = EndedVoice { voice.channel, voice.key,
                                                         voice.noteId };
            }
        }
        voices.release(slot);
    }

    /** Voice bookkeeping for a NoteOn / NoteOff the plugin just received. */
    void trackVoice(const MidiEvent& midi) noexcept
    {
        if (midi.type == MidiEvent::Type::NoteOn)
        {
            // A full table retires its oldest voice (a released one first)
            // and reports its end, as a synth that steals a voice does: the
            // old table simply did not track the new voice, which then never
            // received a NOTE_END, and the host kept its modulators forever.
            if (voices.start(midi.channel, midi.note, midi.noteId) < 0)
                if (const int victim = voices.oldest(); victim >= 0)
                {
                    endVoice(victim);
                    (void) voices.start(midi.channel, midi.note, midi.noteId);
                }
            return;
        }
        if (midi.type != MidiEvent::Type::NoteOff) return;
        voices.forEachMatching(-1, midi.channel, midi.note, midi.noteId, [&](int slot) {
            // A choke ends the voice now. A release ends it when the plugin
            // says it fell silent (isVoiceActive) - or, without that query,
            // at the release itself.
            if constexpr (HasVoiceQuery<P>)
            {
                if (!midi.choke)
                {
                    voices[slot].held = false;
                    return;
                }
            }
            endVoice(slot);
        });
    }

    /** Applies one per-voice event (NoteExpression, or NoteModulation from a
     *  per-voice modulation amount or a per-note value) to every voice its
     *  host address matches. An exact channel + key address the table does
     *  not hold (a voice started before activation) is still delivered once. */
    void applyVoiceEvent(const BlockEvent& ev, MidiEvent midi) noexcept
    {
        const bool isModulation = midi.type == MidiEvent::Type::NoteModulation;
        const int index = midi.param;
        const int slot = isModulation ? perVoiceSlotTable<P>[static_cast<size_t>(index)] : -1;
        if (isModulation && slot < 0) return;
        const auto s = static_cast<size_t>(slot < 0 ? 0 : slot);
        bool delivered = false;
        voices.forEachMatching(ev.targetPort, ev.targetChannel, ev.targetKey,
                               ev.targetNoteId, [&](int v) {
            const auto& voice = voices[v];
            if (isModulation)
            {
                auto& params = voices.paramsOf(v);
                if (ev.voiceValue)
                {
                    params.value[s] = ev.value;
                    params.valued[s] = true;
                }
                else
                {
                    params.amount[s] = ev.value;
                    params.modulated[s] = true;
                }
                midi.value = static_cast<float>(voiceHeardPlain(index, slot, params));
            }
            deliverToVoice(midi, voice.channel, voice.key, voice.noteId);
            delivered = true;
        });
        if (!delivered && ev.targetPort <= 0 && ev.targetChannel >= 0 && ev.targetKey >= 0)
        {
            if (isModulation)
                midi.value = static_cast<float>(ev.voiceValue
                    ? voiceHeardPlain(index, true, ev.value, false, 0.0)
                    : voiceHeardPlain(index, false, 0.0, true, ev.value));
            deliverToVoice(midi, ev.targetChannel, ev.targetKey, ev.targetNoteId);
        }
    }

    /** After a processed block: voices the plugin reports silent end (with
     *  perVoice() parameters and the voice query). */
    void pollVoices() noexcept
    {
        if constexpr (kHasPerVoice && HasVoiceQuery<P>)
        {
            for (int v = voices.end() - 1; v >= 0; --v)   // ending shrinks end()
            {
                const auto& voice = voices[v];
                if (voice.used
                    && !static_cast<bool>(user.isVoiceActive(voice.channel, voice.key,
                                                             voice.noteId)))
                    endVoice(v);
            }
        }
    }

    /** Reports this block's ended voices to the host as NOTE_END events. */
    void flushEndedVoices(const clap_output_events_t* out, uint32_t time) noexcept
    {
        if constexpr (kHasPerVoice)
        {
            for (int i = 0; out != nullptr && out->try_push != nullptr && i < endedCount; ++i)
            {
                clap_event_note_t ev {};
                ev.header.size = sizeof(ev);
                ev.header.time = time;
                ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
                ev.header.type = CLAP_EVENT_NOTE_END;
                ev.note_id = endedVoices[i].noteId;
                ev.port_index = 0;
                ev.channel = endedVoices[i].channel;
                ev.key = endedVoices[i].key;
                ev.velocity = 0.0;
                out->try_push(out, &ev.header);
            }
        }
        else
        {
            (void) out;
            (void) time;
        }
        endedCount = 0;
    }

    /** Asks the host for a process()/flush() so deferred values land even
     *  while the transport is stopped. */
    void requestParamFlush() noexcept
    {
        if (hostParams != nullptr && hostParams->request_flush != nullptr && host != nullptr)
            hostParams->request_flush(host);
    }

    /** Applies one factory preset (main thread). */
    void applyFactoryPresetIdx(int idx) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            if (idx < 0 || idx >= kNumPresets) return;
            bool direct = true;
            for (size_t i = 0; i < kNumParams; ++i)
                direct &= applyFromUi(static_cast<int>(i), presetNormalized<P>(idx, i));
            currentProgram.store(idx, std::memory_order_relaxed);
            if (!direct) requestParamFlush();
        }
        else
            (void) idx;
    }

    /** Re-reads the plugin latency after parameter motion. The CLAP host
     *  must hear about it on the main thread, so a change only flags it and
     *  requests a callback; sOnMainThread then asks for a restart (see
     *  there) and activate() reports the new value. */
    void refreshLatency() noexcept
    {
        if constexpr (HasLatency<P>)
        {
            const int now = user.getLatency();
            if (prepared && now != cachedLatency.load(std::memory_order_relaxed))
            {
                cachedLatency.store(now, std::memory_order_relaxed);
                latencyDirty.store(true, std::memory_order_release);
                if (host != nullptr && host->request_callback != nullptr)
                    host->request_callback(host);
            }
        }
    }

    /** Translates one CLAP event into the timestamped block stream. CLAP
     *  param events carry PLAIN values; raw MIDI covers what note events
     *  don't (pitch bend, CC, pressure). */
    void collectEvent(const clap_event_header_t* ev, uint32_t numSamples,
                      BlockEventList& events) noexcept
    {
        if (ev == nullptr || ev->space_id != CLAP_CORE_EVENT_SPACE_ID) return;
        BlockEvent out {};
        out.offset = static_cast<int32_t>(ev->time);
        if (numSamples > 0 && out.offset >= static_cast<int32_t>(numSamples))
            out.offset = static_cast<int32_t>(numSamples) - 1;
        if (out.offset < 0) out.offset = 0;

        if (ev->type == CLAP_EVENT_PARAM_VALUE)
        {
            const auto* pv = reinterpret_cast<const clap_event_param_value_t*>(ev);
            // A NaN value is ignored like a NaN modulation amount: the plain
            // -> normalized mapping sends NaN to 0, so the old code moved the
            // parameter to its minimum.
            if (pv->value != pv->value) return;
            if (pv->param_id == kBypassParamId)
            {
                out.kind = BlockEvent::Kind::Bypass;
                out.paramId = kBypassParamId;
                out.value = pv->value;
            }
            else
            {
                const int idx = indexOfParamId(pv->param_id);
                if (idx < 0) return;
                // Per-note automation: a value addressed by note id, key or
                // channel sets single voices of a perVoice() parameter (the
                // parameter's own value stays as it is) and becomes one
                // NoteModulation per matching voice when applied. The old
                // wrapper offered none and moved the global value instead.
                // A port-only address (one note port: every voice) and any
                // address on another parameter keep the global meaning.
                if (pv->note_id >= 0 || pv->key >= 0 || pv->channel >= 0)
                {
                    if constexpr (HasMidi<P>)
                    {
                        if (P::parameters[static_cast<size_t>(idx)].perVoice)
                        {
                            out.kind = BlockEvent::Kind::Midi;
                            out.midi.type = MidiEvent::Type::NoteModulation;
                            out.midi.param = static_cast<int16_t>(idx);
                            out.value = pv->value;
                            out.voiceValue = true;
                            out.targetNoteId = pv->note_id;
                            out.targetPort = pv->port_index;
                            out.targetChannel = pv->channel;
                            out.targetKey = pv->key;
                            events.add(out);
                            return;
                        }
                    }
                }
                out.kind = BlockEvent::Kind::Param;
                out.paramId = pv->param_id;
                out.value = toNormalized(P::parameters[static_cast<size_t>(idx)],
                                         pv->value);
            }
        }
        else if (ev->type == CLAP_EVENT_PARAM_MOD)
        {
            // The global signal (every address part -1) is added to the value
            // setParameter receives. An addressed one - per note id, key,
            // channel or port - modulates single voices: only a perVoice()
            // parameter declares that, and it becomes one NoteModulation per
            // matching voice when applied. A NaN amount is ignored.
            const auto* pm = reinterpret_cast<const clap_event_param_mod_t*>(ev);
            if (pm->amount != pm->amount) return;
            const int idx = indexOfParamId(pm->param_id);
            if (idx < 0) return;
            if (pm->note_id >= 0 || pm->key >= 0 || pm->channel >= 0 || pm->port_index >= 0)
            {
                if constexpr (HasMidi<P>)
                {
                    if (!P::parameters[static_cast<size_t>(idx)].perVoice) return;
                    out.kind = BlockEvent::Kind::Midi;
                    out.midi.type = MidiEvent::Type::NoteModulation;
                    out.midi.param = static_cast<int16_t>(idx);
                    out.value = pm->amount;
                    out.targetNoteId = pm->note_id;
                    out.targetPort = pm->port_index;
                    out.targetChannel = pm->channel;
                    out.targetKey = pm->key;
                    events.add(out);
                }
                return;
            }
            out.kind = BlockEvent::Kind::ParamMod;
            out.paramId = pm->param_id;
            out.value = pm->amount;
        }
        else if constexpr (HasMidi<P>)
        {
            out.kind = BlockEvent::Kind::Midi;
            if (ev->type == CLAP_EVENT_NOTE_ON || ev->type == CLAP_EVENT_NOTE_OFF
                || ev->type == CLAP_EVENT_NOTE_CHOKE)
            {
                const auto* note = reinterpret_cast<const clap_event_note_t*>(ev);
                if (ev->type == CLAP_EVENT_NOTE_ON)
                {
                    if ((note->key < 0) || (note->key > 127)) return;   // no voice to start
                    const int ch = note->channel >= 0 ? (note->channel & 0x0F) : 0;
                    out.midi.type = MidiEvent::Type::NoteOn;
                    out.midi.channel = static_cast<uint8_t>(ch);
                    out.midi.note = static_cast<uint8_t>(note->key);
                    out.midi.value = static_cast<float>(note->velocity);
                    out.midi.noteId = note->note_id >= 0 ? note->note_id : -1;
                    sounding.noteOn(ch, note->key, note->note_id);
                }
                else
                {
                    // NOTE_OFF and NOTE_CHOKE address notes by (channel, key,
                    // note id), each part possibly the wildcard -1. Every
                    // sounding note the address matches gets its NoteOff. A
                    // choke is a NoteOff with `choke` set: the voice must end
                    // now, without its release phase. Delivered as a plain
                    // NoteOff (the previous behaviour), a choked voice rang
                    // out its whole release; a plugin that ignores the flag
                    // still releases it.
                    const bool choke = ev->type == CLAP_EVENT_NOTE_CHOKE;
                    const float velocity = choke ? 0.0f : static_cast<float>(note->velocity);
                    out.midi.choke = choke;
                    const bool exact = note->channel >= 0 && note->key >= 0;
                    if (exact)
                    {
                        // A precise off is delivered even for a note the
                        // tracker never saw (its on may predate an overflow).
                        sounding.noteOff(note->channel & 0x0F, note->key & 0x7F);
                        out.midi.type = MidiEvent::Type::NoteOff;
                        out.midi.channel = static_cast<uint8_t>(note->channel & 0x0F);
                        out.midi.note = static_cast<uint8_t>(note->key & 0x7F);
                        out.midi.value = velocity;
                        out.midi.noteId = note->note_id >= 0 ? note->note_id : -1;
                    }
                    else
                    {
                        sounding.releaseMatching(note->channel, note->key, note->note_id,
                            [&](int c, int k) {
                                BlockEvent off = out;
                                off.midi.type = MidiEvent::Type::NoteOff;
                                off.midi.channel = static_cast<uint8_t>(c);
                                off.midi.note = static_cast<uint8_t>(k);
                                off.midi.value = velocity;
                                off.midi.noteId = sounding.noteIdOf(c, k);
                                events.add(off);
                            });
                        return;
                    }
                }
            }
            else if (ev->type == CLAP_EVENT_NOTE_EXPRESSION)
            {
                // Per-voice expressions (CLAP numbers them in ExpressionType
                // order). Resolved against the delivered voices when applied;
                // the old code dropped every one of them.
                const auto* ne = reinterpret_cast<const clap_event_note_expression_t*>(ev);
                if ((ne->expression_id < CLAP_NOTE_EXPRESSION_VOLUME)
                    || (ne->expression_id > CLAP_NOTE_EXPRESSION_PRESSURE)
                    || ne->value != ne->value)
                    return;
                const auto expression =
                    static_cast<MidiEvent::ExpressionType>(ne->expression_id);
                // A class that declares its expressions hears only those.
                if ((noteExpressionsOf<P>() & expressionBit(expression)) == 0u) return;
                out.midi.type = MidiEvent::Type::NoteExpression;
                out.midi.expression = expression;
                out.midi.value = static_cast<float>(ne->value);
                out.targetNoteId = ne->note_id;
                out.targetPort = ne->port_index;
                out.targetChannel = ne->channel;
                out.targetKey = ne->key;
            }
            else if (ev->type == CLAP_EVENT_MIDI)
            {
                const auto* midi = reinterpret_cast<const clap_event_midi_t*>(ev);
                const uint8_t status = midi->data[0] & 0xF0u;
                const uint8_t channel = midi->data[0] & 0x0Fu;
                const uint8_t d1 = midi->data[1] & 0x7Fu;
                const uint8_t d2 = midi->data[2] & 0x7Fu;
                out.midi.channel = channel;
                switch (status)
                {
                case 0x90:   // wire convention: velocity 0 means note off
                    out.midi.type = d2 > 0 ? MidiEvent::Type::NoteOn
                                           : MidiEvent::Type::NoteOff;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    if (d2 > 0) sounding.noteOn(channel, d1, -1);
                    else sounding.noteOff(channel, d1);
                    break;
                case 0x80:
                    out.midi.type = MidiEvent::Type::NoteOff;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    sounding.noteOff(channel, d1);
                    break;
                case 0xA0:
                    out.midi.type = MidiEvent::Type::PolyPressure;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    break;
                case 0xB0:
                    out.midi.type = MidiEvent::Type::ControlChange;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    break;
                case 0xD0:
                    out.midi.type = MidiEvent::Type::ChannelPressure;
                    out.midi.value = static_cast<float>(d1) / 127.0f;
                    break;
                case 0xE0:
                    out.midi.type = MidiEvent::Type::PitchBend;
                    out.midi.value = (static_cast<float>((d2 << 7) | d1) - 8192.0f)
                                   / 8192.0f;
                    break;
                default:
                    return;
                }
            }
            else
                return;
        }
        else
            return;

        // Past the capacity an intermediate parameter point gives way; the
        // old code overwrote the last slot, losing every event but the final
        // one - note-offs and final parameter values included.
        events.add(out);
    }

    void collectEvents(const clap_input_events_t* in, uint32_t numSamples,
                       BlockEventList& events) noexcept
    {
        if (in == nullptr) return;
        const uint32_t n = in->size(in);
        for (uint32_t i = 0; i < n; ++i)
            collectEvent(in->get(in, i), numSamples, events);
    }

    /** Applies one event; @p blockStart rebases MIDI offsets onto the next
     *  processBlock call. Returns true when a user parameter changed. */
    bool applyBlockEvent(const BlockEvent& ev, int blockStart) noexcept
    {
        switch (ev.kind)
        {
        case BlockEvent::Kind::Bypass:
            bypass.store(ev.value >= 0.5, std::memory_order_relaxed);
            return false;
        case BlockEvent::Kind::Midi:
            if constexpr (HasMidi<P>)
            {
                MidiEvent midi = ev.midi;
                midi.sampleOffset = ev.offset - blockStart;
                if (midi.sampleOffset < 0) midi.sampleOffset = 0;
                if (midi.type == MidiEvent::Type::NoteExpression
                    || midi.type == MidiEvent::Type::NoteModulation)
                    applyVoiceEvent(ev, midi);
                else
                {
                    user.handleMidiEvent(midi);
                    trackVoice(midi);
                }
            }
            return false;
        case BlockEvent::Kind::ParamMod:
            if (const int idx = indexOfParamId(ev.paramId); idx >= 0)
            {
                modulation[static_cast<size_t>(idx)] = ev.value;
                user.setParameter(idx, static_cast<float>(heardPlain(
                    idx, shadow[static_cast<size_t>(idx)].load(std::memory_order_relaxed))));
                const int offset = ev.offset - blockStart;
                resendVoiceModulation(idx, offset < 0 ? 0 : offset, true);
                return true;
            }
            return false;
        case BlockEvent::Kind::Param:
        default:
            if (const int idx = indexOfParamId(ev.paramId); idx >= 0)
            {
                applyNormalized(idx, ev.value);
                const int offset = ev.offset - blockStart;
                resendVoiceModulation(idx, offset < 0 ? 0 : offset, false);
                return true;
            }
            return false;
        }
    }

    /** Forwards the per-block clap transport as a TransportInfo. */
    void forwardTransport(const clap_event_transport_t* t) noexcept
    {
        if constexpr (HasTransport<P>)
        {
            if (t == nullptr) return;
            constexpr double kBeatFactor = static_cast<double>(CLAP_BEATTIME_FACTOR);
            TransportInfo info {};
            info.playing   = (t->flags & CLAP_TRANSPORT_IS_PLAYING) != 0;
            info.recording = (t->flags & CLAP_TRANSPORT_IS_RECORDING) != 0;
            info.looping   = (t->flags & CLAP_TRANSPORT_IS_LOOP_ACTIVE) != 0;
            if ((t->flags & CLAP_TRANSPORT_HAS_TEMPO) != 0)
            {
                info.tempoBpm = t->tempo;
                info.tempoValid = true;
            }
            if ((t->flags & CLAP_TRANSPORT_HAS_BEATS_TIMELINE) != 0)
            {
                info.ppqPosition = static_cast<double>(t->song_pos_beats) / kBeatFactor;
                info.barStartPpq = static_cast<double>(t->bar_start) / kBeatFactor;
                info.positionValid = true;
                info.loopStartPpq =
                    static_cast<double>(t->loop_start_beats) / kBeatFactor;
                info.loopEndPpq = static_cast<double>(t->loop_end_beats) / kBeatFactor;
                info.loopValid = true;
            }
            if ((t->flags & CLAP_TRANSPORT_HAS_TIME_SIGNATURE) != 0)
            {
                info.timeSigNumerator = t->tsig_num;
                info.timeSigDenominator = t->tsig_denom;
                info.timeSigValid = true;
            }
            user.setTransport(info);
        }
        else
            (void) t;
    }

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- UI -> host parameter events ---------------------------------------------

    void pushUiEvent(uint32_t paramId, uint8_t kind, double value) noexcept
    {
        const uint32_t tail = uiTail.load(std::memory_order_relaxed);
        const uint32_t next = (tail + 1) & (kUiQueueSize - 1);
        if (next == uiHead.load(std::memory_order_acquire)) return;   // full: drop
        uiQueue[tail] = UiEvent { paramId, kind, value };
        uiTail.store(next, std::memory_order_release);
    }

    void requestUiFlush() noexcept
    {
        // Host schedules process()/flush(), which drains the queue below.
        if (hostParams != nullptr && host != nullptr)
            hostParams->request_flush(host);
    }

    void drainUiEvents(const clap_output_events_t* out) noexcept
    {
        if (out == nullptr) return;
        uint32_t head = uiHead.load(std::memory_order_relaxed);
        const uint32_t tail = uiTail.load(std::memory_order_acquire);
        while (head != tail)
        {
            const UiEvent& e = uiQueue[head];
            if (e.kind == kUiValue)
            {
                clap_event_param_value_t ev {};
                ev.header.size = sizeof(ev);
                ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
                ev.header.type = CLAP_EVENT_PARAM_VALUE;
                ev.param_id = e.paramId;
                ev.note_id = -1;
                ev.port_index = -1;
                ev.channel = -1;
                ev.key = -1;
                ev.value = e.value;
                out->try_push(out, &ev.header);
            }
            else
            {
                clap_event_param_gesture_t ev {};
                ev.header.size = sizeof(ev);
                ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
                ev.header.type = static_cast<uint16_t>(
                    e.kind == kUiGestureBegin ? CLAP_EVENT_PARAM_GESTURE_BEGIN
                                              : CLAP_EVENT_PARAM_GESTURE_END);
                ev.param_id = e.paramId;
                out->try_push(out, &ev.header);
            }
            head = (head + 1) & (kUiQueueSize - 1);
        }
        uiHead.store(head, std::memory_order_release);
    }
#endif // DSPARK_PLUGIN_WEBVIEW

    // --- clap_plugin_t callbacks ------------------------------------------------

    static bool sInit(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
        if (s->host != nullptr)
        {
            s->hostParams = static_cast<const clap_host_params_t*>(
                s->host->get_extension(s->host, CLAP_EXT_PARAMS));
            s->hostLatency = static_cast<const clap_host_latency_t*>(
                s->host->get_extension(s->host, CLAP_EXT_LATENCY));
            if constexpr (kNumPresets > 0)
                s->hostPresetLoad = static_cast<const clap_host_preset_load_t*>(
                    s->host->get_extension(s->host, CLAP_EXT_PRESET_LOAD));
#if defined(DSPARK_PLUGIN_WEBVIEW)
            if constexpr (HasEditor<P>)
                s->hostGui = static_cast<const clap_host_gui_t*>(
                    s->host->get_extension(s->host, CLAP_EXT_GUI));
#endif
        }
        return true;
    }

    static void sDestroy(const clap_plugin_t* p) noexcept { delete self(p); }

    static bool sActivate(const clap_plugin_t* p, double sr,
                          uint32_t, uint32_t maxFrames) noexcept
    {
        auto* s = self(p);
        s->sampleRate = sr;
        s->maxFrames = maxFrames;
        dspark::AudioSpec spec { sr, static_cast<int>(maxFrames), s->currentChannels };
        s->user.prepare(spec);
        if constexpr (HasOfflineMode<P>)
            s->user.setOfflineRendering(s->offlineRender);
        s->pending.drain([](int) {});   // every value is re-applied right below
        for (size_t i = 0; i < kNumParams; ++i)
            s->user.setParameter(static_cast<int>(i),
                static_cast<float>(s->heardPlain(static_cast<int>(i),
                                                 s->shadow[i].load(std::memory_order_relaxed))));
        s->dryL.assign(maxFrames, 0.0f);
        s->dryR.assign(maxFrames, 0.0f);
        if constexpr (HasSidechain<P>)
            s->silence.assign(maxFrames, 0.0f);
        s->bypassMix = s->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        s->sounding.clear();
        s->voices.clear();
        s->endedCount = 0;
        if constexpr (HasLatency<P>)
        {
            const int latency = s->user.getLatency();
            s->cachedLatency.store(latency, std::memory_order_relaxed);
            s->dryDelay.prepare(std::max(2 * latency, kDryDelayMinCapacity),
                                static_cast<int>(maxFrames));
            // clap_host_latency.changed is [main-thread & being-activated]:
            // this is the one place it may be called.
            s->latencyDirty.store(false, std::memory_order_relaxed);
            if (latency != s->latencyAtActivate && s->hostLatency != nullptr
                && s->hostLatency->changed != nullptr)
                s->hostLatency->changed(s->host);
            s->latencyAtActivate = latency;
        }
        s->prepared = true;
        s->active.store(true, std::memory_order_release);
        return true;
    }

    static void sDeactivate(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
        s->active.store(false, std::memory_order_release);
        s->voices.clear();   // the host retires its voices with the activation
        s->endedCount = 0;
        s->drainPending();   // no process()/flush() can run now (main thread)
    }

    /** [audio-thread & active]: processing (re)starts from a clean history,
     *  as after reset(). The old no-op kept whatever a stopped processing
     *  run left in delay lines and tails. */
    static bool sStartProcessing(const clap_plugin_t* p) noexcept
    {
        sReset(p);
        return true;
    }
    static void sStopProcessing(const clap_plugin_t*) noexcept {}

    static void sReset(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
        if constexpr (HasReset<P>)
            s->user.reset();
        s->dryDelay.reset();
        s->sounding.clear();
        s->voices.clear();   // reset kills every voice (CLAP reset contract)
        s->endedCount = 0;
        s->bypassMix = s->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
    }

    /** Crossfades @p length output frames (from @p outStart) toward the dry
     *  copy (from @p dryStart) at the current bypass target. */
    void blendBypass(float* const* out, float* const* dry, uint32_t nCh, int outStart,
                     int dryStart, int length) noexcept
    {
        const float target = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (bypassMix == target && target == 0.0f) return;
        const float step = 1.0f / static_cast<float>(kBypassRampSamples);
        float mix = bypassMix;
        for (int i = 0; i < length; ++i)
        {
            mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
            mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
            for (uint32_t ch = 0; ch < nCh; ++ch)
                out[ch][outStart + i] += (dry[ch][dryStart + i] - out[ch][outStart + i]) * mix;
        }
        bypassMix = mix;
    }

    static clap_process_status sProcess(const clap_plugin_t* p,
                                        const clap_process_t* process) noexcept
    {
        auto* s = self(p);

        // FTZ/DAZ for the whole callback: DSPark processors guard their own
        // hot loops, this covers user DSP in hosts that do not set it.
        dspark::DenormalGuard denormalGuard;

#if defined(DSPARK_PLUGIN_WEBVIEW)
        s->drainUiEvents(process->out_events);
#endif

        const uint32_t n = process->frames_count;

        // Main-thread edits handed over since the last block go first, so
        // the host's own events below still win.
        bool paramsChanged = s->drainPending();

        BlockEventList& events = s->blockEvents;
        events.clear();
        s->collectEvents(process->in_events, n, events);
        events.sort();
        const int eventCount = events.size();

        const clap_audio_buffer_t* outPort =
            process->audio_outputs_count >= 1 && process->audio_outputs != nullptr
                ? &process->audio_outputs[0] : nullptr;
        float** out = outPort != nullptr ? outPort->data32 : nullptr;
        const uint32_t width = static_cast<uint32_t>(s->currentChannels);
        uint32_t nCh = out == nullptr ? 0u
                     : (outPort->channel_count < width ? outPort->channel_count : width);
        // Degenerate hosts: stop at the first missing channel pointer (the
        // old trim only looked at the last one and wrote through a null left
        // channel next to a valid right one).
        for (uint32_t ch = 0; ch < nCh; ++ch)
            if (out[ch] == nullptr)
            {
                nCh = ch;
                break;
            }

        const bool canRender = n > 0 && s->prepared && nCh >= 1;
        if (!canRender)
        {
            for (int i = 0; i < eventCount; ++i)
                paramsChanged |= s->applyBlockEvent(events[i], events[i].offset);
            if (paramsChanged) s->refreshLatency();
            s->flushEndedVoices(process->out_events, 0);
            return CLAP_PROCESS_CONTINUE;
        }

        s->forwardTransport(process->transport);

        const bool haveIn = !kIsInstrument && process->audio_inputs_count >= 1
                         && process->audio_inputs != nullptr
                         && process->audio_inputs[0].data32 != nullptr
                         && process->audio_inputs[0].channel_count >= 1;
        float** in = haveIn ? process->audio_inputs[0].data32 : nullptr;
        const uint32_t inCh = haveIn ? process->audio_inputs[0].channel_count : 0;

        // Sidechain: input port 1, pre-allocated silence when not routed.
        float* scHost[2] = { nullptr, nullptr };
        if constexpr (HasSidechain<P>)
        {
            if (process->audio_inputs_count >= 2
                && process->audio_inputs[1].data32 != nullptr)
            {
                const auto& scPort = process->audio_inputs[1];
                const uint32_t scCh = scPort.channel_count < 2
                                    ? scPort.channel_count : 2;
                for (uint32_t ch = 0; ch < scCh; ++ch)
                    scHost[ch] = scPort.data32[ch];
                if (scCh == 1)
                    scHost[1] = scPort.data32[0];   // mono key feeds both ears
            }
        }

        // A block longer than the activated maximum runs as consecutive
        // chunks of at most that size: the user DSP never sees more frames
        // than it prepared for.
        const int total = static_cast<int>(n);
        const int chunkMax = static_cast<int>(s->dryL.size());
        if (chunkMax < 1) return CLAP_PROCESS_CONTINUE;
        float* dry[2] = { s->dryL.data(), s->dryR.data() };
        int evIdx = 0;

        for (int c0 = 0; c0 < total; c0 += chunkMax)
        {
            const int len = total - c0 < chunkMax ? total - c0 : chunkMax;
            const int cEnd = c0 + len;

            // Dry copy for the bypass blend; instruments start cleared (voices
            // ADD) and bypass toward silence (their dry vectors stay zero).
            // Output channels past a narrower input port repeat its last one.
            // An effect whose input is missing processes silence: the old
            // code fell back to the output buffer, feeding the host's stale
            // output back in as input.
            const size_t bytes = sizeof(float) * static_cast<size_t>(len);
            for (uint32_t ch = 0; ch < nCh; ++ch)
            {
                const float* src = nullptr;
                if (!kIsInstrument && haveIn)
                    src = in[ch < inCh ? ch : inCh - 1];
                if (src == nullptr)
                {
                    std::memset(out[ch] + c0, 0, bytes);
                    if (!kIsInstrument) std::memset(dry[ch], 0, bytes);
                    continue;
                }
                std::memcpy(dry[ch], src + c0, bytes);
                if (out[ch] != src)
                    std::memcpy(out[ch] + c0, src + c0, bytes);
            }
            if constexpr (HasLatency<P> && !kIsInstrument)
                s->dryDelay.process(dry, static_cast<int>(nCh), len,
                                    s->cachedLatency.load(std::memory_order_relaxed));

            float* scChunk[2] = { nullptr, nullptr };
            if constexpr (HasSidechain<P>)
                for (int ch = 0; ch < 2; ++ch)
                    scChunk[ch] = scHost[ch] != nullptr ? scHost[ch] + c0
                                                        : s->silence.data();

            // Sub-block processing at quantum-aligned event positions (the
            // sample-accurate default); opted out, the chunk's events apply
            // up front.
            auto processSegment = [&](int start, int length) noexcept {
                float* sub[2] = { out[0] + start,
                                  nCh > 1 ? out[1] + start : out[0] + start };
                dspark::AudioBufferView<float> view(sub, static_cast<int>(nCh), length);
                if constexpr (HasSidechain<P>)
                {
                    // The key view mirrors the main width (mono main, mono key).
                    float* scSub[2] = { scChunk[0] + (start - c0),
                                        scChunk[1] + (start - c0) };
                    dspark::AudioBufferView<float> scView(scSub,
                                                          static_cast<int>(nCh), length);
                    s->user.processBlock(view, scView);
                }
                else
                    s->user.processBlock(view);
                // Soft bypass toward the (latency-aligned) dry signal, per
                // segment: the crossfade starts where the bypass event lands
                // (the old once-per-block blend started it at sample 0).
                s->blendBypass(out, dry, nCh, start, start - c0, length);
            };

            if (!sampleAccurateOf<P>())
            {
                for (; evIdx < eventCount && events[evIdx].offset < cEnd; ++evIdx)
                    paramsChanged |= s->applyBlockEvent(events[evIdx], c0);
                processSegment(c0, len);
            }
            else
            {
                int pos = c0;
                while (pos < cEnd)
                {
                    while (evIdx < eventCount
                           && (events[evIdx].offset / kAutomationQuantum)
                                  * kAutomationQuantum <= pos)
                        paramsChanged |= s->applyBlockEvent(events[evIdx++], pos);
                    int next = cEnd;
                    if (evIdx < eventCount)
                    {
                        const int snapped = (events[evIdx].offset / kAutomationQuantum)
                                          * kAutomationQuantum;
                        if (snapped < next) next = snapped;
                    }
                    if (next <= pos) next = pos + kAutomationQuantum < cEnd
                                          ? pos + kAutomationQuantum : cEnd;
                    processSegment(pos, next - pos);
                    pos = next;
                }
            }
        }
        for (; evIdx < eventCount; ++evIdx)   // safety: events at block end
            paramsChanged |= s->applyBlockEvent(events[evIdx], total);

        if (paramsChanged) s->refreshLatency();
        s->pollVoices();
        s->flushEndedVoices(process->out_events, n - 1);
        return CLAP_PROCESS_CONTINUE;
    }

    static const void* sGetExtension(const clap_plugin_t*, const char* id) noexcept
    {
        if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &kAudioPorts;
        if (std::strcmp(id, CLAP_EXT_PARAMS) == 0)      return &kParams;
        if (std::strcmp(id, CLAP_EXT_STATE) == 0)       return &kState;
        if (std::strcmp(id, CLAP_EXT_LATENCY) == 0)     return &kLatency;
        if (std::strcmp(id, CLAP_EXT_TAIL) == 0)        return &kTail;
        if (channelSupportOf<P>() == ChannelSupport::MonoAndStereo)
        {
            if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS_CONFIG) == 0)
                return &kPortsConfig;
            if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS_CONFIG_INFO) == 0
                || std::strcmp(id, CLAP_EXT_AUDIO_PORTS_CONFIG_INFO_COMPAT) == 0)
                return &kPortsConfigInfo;
        }
        if constexpr (HasMidi<P>)
            if (std::strcmp(id, CLAP_EXT_NOTE_PORTS) == 0) return &kNotePorts;
        if constexpr (HasOfflineMode<P>)
            if (std::strcmp(id, CLAP_EXT_RENDER) == 0) return &kRender;
        if constexpr (kNumPresets > 0)
            if (std::strcmp(id, CLAP_EXT_PRESET_LOAD) == 0
                || std::strcmp(id, CLAP_EXT_PRESET_LOAD_COMPAT) == 0)
                return &kPresetLoad;
#if defined(DSPARK_PLUGIN_WEBVIEW)
        if constexpr (HasEditor<P>)
        {
            if (webview_ui::Editor<P>::available() && std::strcmp(id, CLAP_EXT_GUI) == 0)
                return &kGui;
#if defined(__linux__)
            if (webview_ui::Editor<P>::available()
                && std::strcmp(id, CLAP_EXT_TIMER_SUPPORT) == 0)
                return &kTimer;
#endif
        }
#endif
        return nullptr;
    }

    static void sOnMainThread(const clap_plugin_t* p) noexcept
    {
        // Deferred latency notification (the audio thread only flags it).
        // clap_host_latency.changed is [main-thread & being-activated]: while
        // active the plugin must ask for a restart instead, and the host's
        // deactivate/activate cycle then lets activate() report the new
        // value. The old code called changed() here, outside activation,
        // which a conforming host ignores - the delay compensation never
        // followed a lookahead change.
        auto* s = self(p);
        if (!s->latencyDirty.exchange(false, std::memory_order_acq_rel)) return;
        if (s->active.load(std::memory_order_acquire))
        {
            if (s->host != nullptr && s->host->request_restart != nullptr)
                s->host->request_restart(s->host);
        }
        // Inactive: the next activate() reports the latency itself.
    }

    // --- ext: audio ports ---------------------------------------------------------

    static uint32_t sPortCount(const clap_plugin_t*, bool isInput) noexcept
    {
        if (isInput)
        {
            if (kIsInstrument) return 0;
            return HasSidechain<P> ? 2 : 1;
        }
        return 1;
    }

    static bool sPortGet(const clap_plugin_t* p, uint32_t index, bool isInput,
                         clap_audio_port_info_t* info) noexcept
    {
        if (info == nullptr || index >= sPortCount(p, isInput)) return false;
        const int width = self(p)->currentChannels;
        std::memset(info, 0, sizeof(*info));
        if (isInput && index == 1)   // the sidechain: routable, never the main pair
        {
            info->id = 2;
            std::snprintf(info->name, sizeof(info->name), "Sidechain");
            info->flags = 0;
            info->channel_count = static_cast<uint32_t>(width);
            info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
            info->in_place_pair = CLAP_INVALID_ID;
            return true;
        }
        info->id = isInput ? 0 : 1;
        std::snprintf(info->name, sizeof(info->name), "%s",
                      isInput ? "Input" : "Output");
        info->flags = CLAP_AUDIO_PORT_IS_MAIN;
        info->channel_count = static_cast<uint32_t>(width);
        info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
        info->in_place_pair = kIsInstrument ? CLAP_INVALID_ID
                                            : (isInput ? 1u : 0u);
        return true;
    }

    inline static const clap_plugin_audio_ports_t kAudioPorts = { &sPortCount, &sPortGet };

    // --- ext: audio ports config (mono / stereo selection) --------------------------

    static uint32_t sConfigCount(const clap_plugin_t*) noexcept { return 2; }

    static void fillConfig(clap_audio_ports_config_t* config, int width) noexcept
    {
        std::memset(config, 0, sizeof(*config));
        config->id = static_cast<clap_id>(width);
        std::snprintf(config->name, sizeof(config->name), "%s",
                      width == 1 ? "Mono" : "Stereo");
        config->input_port_count = sPortCount(nullptr, true);
        config->output_port_count = 1;
        config->has_main_input = !kIsInstrument;
        config->main_input_channel_count = static_cast<uint32_t>(width);
        config->main_input_port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
        config->has_main_output = true;
        config->main_output_channel_count = static_cast<uint32_t>(width);
        config->main_output_port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
    }

    static bool sConfigGet(const clap_plugin_t*, uint32_t index,
                           clap_audio_ports_config_t* config) noexcept
    {
        if (config == nullptr || index >= 2) return false;
        fillConfig(config, index == 0 ? 1 : 2);
        return true;
    }

    static bool sConfigSelect(const clap_plugin_t* p, clap_id configId) noexcept
    {
        const int width = static_cast<int>(configId);
        if (!supportsChannelCount<P>(width)) return false;
        self(p)->currentChannels = width;
        return true;
    }

    inline static const clap_plugin_audio_ports_config_t kPortsConfig = {
        &sConfigCount, &sConfigGet, &sConfigSelect
    };

    static clap_id sConfigCurrent(const clap_plugin_t* p) noexcept
    {
        return static_cast<clap_id>(self(p)->currentChannels);
    }

    static bool sConfigInfoGet(const clap_plugin_t* p, clap_id configId,
                               uint32_t portIndex, bool isInput,
                               clap_audio_port_info_t* info) noexcept
    {
        const int width = static_cast<int>(configId);
        if (info == nullptr || !supportsChannelCount<P>(width)
            || portIndex >= sPortCount(p, isInput))
            return false;
        // Same shape as sPortGet, with the width of the asked-about config.
        std::memset(info, 0, sizeof(*info));
        if (isInput && portIndex == 1)
        {
            info->id = 2;
            std::snprintf(info->name, sizeof(info->name), "Sidechain");
            info->channel_count = static_cast<uint32_t>(width);
            info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
            info->in_place_pair = CLAP_INVALID_ID;
            return true;
        }
        info->id = isInput ? 0 : 1;
        std::snprintf(info->name, sizeof(info->name), "%s",
                      isInput ? "Input" : "Output");
        info->flags = CLAP_AUDIO_PORT_IS_MAIN;
        info->channel_count = static_cast<uint32_t>(width);
        info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
        info->in_place_pair = kIsInstrument ? CLAP_INVALID_ID
                                            : (isInput ? 1u : 0u);
        return true;
    }

    inline static const clap_plugin_audio_ports_config_info_t kPortsConfigInfo = {
        &sConfigCurrent, &sConfigInfoGet
    };

    // --- ext: note ports (HasMidi) ----------------------------------------------------

    static uint32_t sNotePortCount(const clap_plugin_t*, bool isInput) noexcept
    {
        return (HasMidi<P> && isInput) ? 1 : 0;
    }

    static bool sNotePortGet(const clap_plugin_t*, uint32_t index, bool isInput,
                             clap_note_port_info_t* info) noexcept
    {
        if (info == nullptr || !isInput || index != 0 || !HasMidi<P>) return false;
        std::memset(info, 0, sizeof(*info));
        info->id = 10;
        info->supported_dialects = CLAP_NOTE_DIALECT_CLAP | CLAP_NOTE_DIALECT_MIDI;
        info->preferred_dialect = CLAP_NOTE_DIALECT_CLAP;
        std::snprintf(info->name, sizeof(info->name), "MIDI In");
        return true;
    }

    inline static const clap_plugin_note_ports_t kNotePorts = {
        &sNotePortCount, &sNotePortGet
    };

    // --- ext: render (HasOfflineMode) ---------------------------------------------------

    static bool sRenderHardRealtime(const clap_plugin_t*) noexcept { return false; }

    static bool sRenderSet(const clap_plugin_t* p, clap_plugin_render_mode mode) noexcept
    {
        auto* s = self(p);
        s->offlineRender = mode == CLAP_RENDER_OFFLINE;
        if constexpr (HasOfflineMode<P>)
            s->user.setOfflineRendering(s->offlineRender);
        return true;
    }

    inline static const clap_plugin_render_t kRender = {
        &sRenderHardRealtime, &sRenderSet
    };

    // --- ext: preset load (factory presets; the DSO is the container) -------------------

    static bool sPresetFromLocation(const clap_plugin_t* p, uint32_t locationKind,
                                    const char* location, const char* loadKey) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            auto* s = self(p);
            if (locationKind != CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN
                || loadKey == nullptr)
                return false;
            const int idx = std::atoi(loadKey);
            if (idx < 0 || idx >= kNumPresets) return false;
            s->applyFactoryPresetIdx(idx);
            s->refreshLatency();
            if (s->hostParams != nullptr && s->hostParams->rescan != nullptr)
                s->hostParams->rescan(s->host, CLAP_PARAM_RESCAN_VALUES);
            if (s->hostPresetLoad != nullptr && s->hostPresetLoad->loaded != nullptr)
                s->hostPresetLoad->loaded(s->host, locationKind, location, loadKey);
            return true;
        }
        else
        {
            (void) p;
            (void) locationKind;
            (void) location;
            (void) loadKey;
            return false;
        }
    }

    inline static const clap_plugin_preset_load_t kPresetLoad = {
        &sPresetFromLocation
    };

    // --- ext: params ----------------------------------------------------------------

    static uint32_t sParamCount(const clap_plugin_t*) noexcept
    {
        return static_cast<uint32_t>(kNumParams) + 1;   // + Bypass
    }

    /** True when every position of a discrete parameter is a whole number
     *  and neighbours are exactly one apart (what CLAP_PARAM_IS_STEPPED
     *  promises a host). */
    static constexpr bool isIntegerGrid(const Param& spec) noexcept
    {
        auto whole = [](double v) {
            const double r = static_cast<double>(static_cast<long long>(v < 0.0 ? v - 0.5
                                                                             : v + 0.5));
            return v - r < 1e-9 && r - v < 1e-9;
        };
        const double step = (static_cast<double>(spec.maxValue) - spec.minValue) / spec.steps;
        return whole(spec.minValue) && step > 1.0 - 1e-9 && step < 1.0 + 1e-9;
    }

    static bool sParamInfo(const clap_plugin_t*, uint32_t index,
                           clap_param_info_t* info) noexcept
    {
        if (info == nullptr || index > kNumParams) return false;
        std::memset(info, 0, sizeof(*info));
        if (index == kNumParams)
        {
            info->id = kBypassParamId;
            std::snprintf(info->name, sizeof(info->name), "Bypass");
            info->flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED
                        | CLAP_PARAM_IS_BYPASS;
            info->min_value = 0.0;
            info->max_value = 1.0;
            info->default_value = 0.0;
            return true;
        }
        const auto& spec = P::parameters[index];
        info->id = hash32(spec.id);
        std::snprintf(info->name, sizeof(info->name), "%s", spec.name);
        info->flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_MODULATABLE;
        // CLAP_PARAM_IS_STEPPED means "integer values only": hosts truncate.
        // Only a grid of whole numbers one apart qualifies (toggles, choices,
        // stepped 1..8). A finer or coarser grid stays continuous for the
        // host and is snapped by the wrapper; flagging it (the old code)
        // made a 5-position 0..1 control offer two positions.
        if (spec.steps > 0 && isIntegerGrid(spec)) info->flags |= CLAP_PARAM_IS_STEPPED;
        if (spec.labels != nullptr) info->flags |= CLAP_PARAM_IS_ENUM;
        // perVoice(): every modulation addressing form CLAP defines, and
        // per-note automation by note id, key and channel - the wrapper
        // resolves each one to the voices it matches. (Per-port automation
        // is not declared: with one note port it would mean every voice,
        // and hosts that fill the port of a global point keep working.)
        if (HasMidi<P> && spec.perVoice)
            info->flags |= CLAP_PARAM_IS_MODULATABLE_PER_NOTE_ID
                         | CLAP_PARAM_IS_MODULATABLE_PER_KEY
                         | CLAP_PARAM_IS_MODULATABLE_PER_CHANNEL
                         | CLAP_PARAM_IS_MODULATABLE_PER_PORT
                         | CLAP_PARAM_IS_AUTOMATABLE_PER_NOTE_ID
                         | CLAP_PARAM_IS_AUTOMATABLE_PER_KEY
                         | CLAP_PARAM_IS_AUTOMATABLE_PER_CHANNEL;
        info->min_value = spec.minValue;
        info->max_value = spec.maxValue;
        info->default_value = spec.defValue;
        return true;
    }

    static bool sParamValue(const clap_plugin_t* p, clap_id id, double* out) noexcept
    {
        auto* s = self(p);
        if (out == nullptr) return false;
        if (id == kBypassParamId)
        {
            *out = s->bypass.load(std::memory_order_relaxed) ? 1.0 : 0.0;
            return true;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return false;
        *out = toPlain(P::parameters[static_cast<size_t>(idx)],
                       s->shadow[static_cast<size_t>(idx)].load(std::memory_order_relaxed));
        return true;
    }

    static bool sParamValueToText(const clap_plugin_t*, clap_id id, double value,
                                  char* out, uint32_t size) noexcept
    {
        if (out == nullptr || size == 0) return false;
        if (id == kBypassParamId)
        {
            std::snprintf(out, size, "%s", value >= 0.5 ? "On" : "Off");
            return true;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return false;
        formatValue(P::parameters[static_cast<size_t>(idx)], value, out,
                    static_cast<int>(size));
        return true;
    }

    static bool sParamTextToValue(const clap_plugin_t*, clap_id id,
                                  const char* text, double* out) noexcept
    {
        if (text == nullptr || out == nullptr) return false;
        const int toggle = parseToggleText(text);
        if (id == kBypassParamId)
        {
            *out = toggle >= 0 ? toggle
                               : (std::strtod(text, nullptr) >= 0.5 ? 1.0 : 0.0);
            return true;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return false;
        // Labels, On/Off or a number, clamped and snapped; "nan" and other
        // garbage are refused rather than passed through to the host.
        return parseValue(P::parameters[static_cast<size_t>(idx)], text, *out);
    }

    static void sParamFlush(const clap_plugin_t* p, const clap_input_events_t* in,
                            const clap_output_events_t* out) noexcept
    {
        auto* s = self(p);
        // [active ? audio-thread : main-thread], never concurrent with
        // process(): the place deferred main-thread edits land while the
        // transport is stopped.
        bool paramsChanged = s->drainPending();
        BlockEventList& events = s->blockEvents;
        events.clear();
        s->collectEvents(in, 0, events);
        events.sort();
        for (int i = 0; i < events.size(); ++i)
            paramsChanged |= s->applyBlockEvent(events[i], events[i].offset);
        if (paramsChanged) s->refreshLatency();
        s->flushEndedVoices(out, 0);
#if defined(DSPARK_PLUGIN_WEBVIEW)
        s->drainUiEvents(out);
#endif
    }

    inline static const clap_plugin_params_t kParams = {
        &sParamCount, &sParamInfo, &sParamValue,
        &sParamValueToText, &sParamTextToValue, &sParamFlush
    };

    // --- ext: state -------------------------------------------------------------------

    static bool sStateSave(const clap_plugin_t* p, const clap_ostream_t* stream) noexcept
    {
        auto* s = self(p);
        double norm[kNumParams == 0 ? 1 : kNumParams];
        for (size_t i = 0; i < kNumParams; ++i)
            norm[i] = s->shadow[i].load(std::memory_order_relaxed);
        EditorSize editorSize {};
        const bool withSize = s->editorMemory.remembered(editorSize);
        const std::vector<uint8_t> blob = buildState(
            s->user, norm, kNumParams,
            kNumPresets > 0 ? s->currentProgram.load(std::memory_order_relaxed) : -1,
            s->bypass.load(std::memory_order_relaxed) ? 1 : 0,
            withSize ? &editorSize : nullptr);
        size_t pos = 0;
        while (pos < blob.size())
        {
            const int64_t put = stream->write(stream, blob.data() + pos,
                                              blob.size() - pos);
            if (put <= 0) return false;
            pos += static_cast<size_t>(put);
        }
        return true;
    }

    static bool sStateLoad(const clap_plugin_t* p, const clap_istream_t* stream) noexcept
    {
        auto* s = self(p);
        std::vector<uint8_t> blob;
        uint8_t chunk[4096];
        for (;;)
        {
            const int64_t got = stream->read(stream, chunk, sizeof(chunk));
            if (got < 0) return false;
            if (got == 0) break;
            blob.insert(blob.end(), chunk, chunk + got);
            // Only an endless hostile stream is refused: the old 16 MiB cap
            // rejected states this plugin's own save() had written.
            if (blob.size() > kMaxStateBytes) return false;
        }
        // Parameters the blob does not carry take their DEFAULTS (the old
        // code kept whatever value the instance happened to hold).
        double norm[kNumParams == 0 ? 1 : kNumParams];
        loadDefaultNormalized<P>(norm);
        int program = -1;
        int bypassState = -1;
        EditorSize editorSize { 0, 0 };
        if (!applyState(s->user, blob.data(), blob.size(), norm, &program,
                        &bypassState, &editorSize))
            return false;
        bool direct = true;
        for (size_t i = 0; i < kNumParams; ++i)
            direct &= s->applyFromUi(static_cast<int>(i), norm[i]);
        if (kNumPresets > 0 && program >= 0 && program < kNumPresets)
            s->currentProgram.store(program, std::memory_order_relaxed);
        if (bypassState >= 0)
            s->bypass.store(bypassState != 0, std::memory_order_relaxed);
        // The editor size the session left: the editor opens at it next time
        // (a state without one keeps the size this instance has), and an
        // editor open right now asks the host for it at once.
        if (editorSize.width > 0)
        {
            s->editorMemory.remember(editorSize.width, editorSize.height);
#if defined(DSPARK_PLUGIN_WEBVIEW)
            s->applyRememberedEditorSize();
#endif
        }
        if (direct)
            s->refreshLatency();   // restored state may imply a new lookahead
        else
            s->requestParamFlush();   // the audio thread applies it, then re-reads
        return true;
    }

    inline static const clap_plugin_state_t kState = { &sStateSave, &sStateLoad };

    // --- ext: latency / tail --------------------------------------------------------

    static uint32_t sLatencyGet(const clap_plugin_t* p) noexcept
    {
        return static_cast<uint32_t>(
            self(p)->cachedLatency.load(std::memory_order_relaxed));
    }

    inline static const clap_plugin_latency_t kLatency = { &sLatencyGet };

    static uint32_t sTailGet(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
        if constexpr (HasTail<P>)
        {
            const double seconds = s->user.getTailSeconds();
            if (!(seconds > 0.0)) return 0;   // negative/NaN -> no tail
            const double samples = seconds * s->sampleRate;
            // CLAP treats anything >= INT32_MAX as an infinite tail; the cap
            // also avoids the double->uint32 overflow (UB).
            if (samples >= 4294967295.0) return 0xFFFFFFFFu;
            return static_cast<uint32_t>(samples);
        }
        else
        {
            (void) s;
            return 0;
        }
    }

    inline static const clap_plugin_tail_t kTail = { &sTailGet };

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- ext: gui (WebView editor layer) ----------------------------------------------
    //
    // CLAP hands the parent window AFTER create(), so create() only arms the
    // state and set_parent() builds the actual web engine. All [main-thread].

    static const char* clapWindowApi() noexcept
    {
#if defined(_WIN32)
        return CLAP_WINDOW_API_WIN32;
#elif defined(__APPLE__)
        return CLAP_WINDOW_API_COCOA;
#else
        return CLAP_WINDOW_API_X11;
#endif
    }

    /** A state loaded while the editor exists carried another size: ask the
     *  host to resize the window to it [main-thread]. The old code only
     *  remembered it, so the editor changed size at its next opening. A
     *  refused request leaves the editor as it is (the next one opens at the
     *  loaded size). */
    void applyRememberedEditorSize() noexcept
    {
        if (!guiActive) return;
        const EditorSize logical = editorMemory.current();
        const int w = toBoundedInt(logical.width * guiScale + 0.5,
                                   1, static_cast<int>(kEditorMaxPixels));
        const int h = toBoundedInt(logical.height * guiScale + 0.5,
                                   1, static_cast<int>(kEditorMaxPixels));
        if (w == guiWidth && h == guiHeight) return;
        webview_ui::debugLog("clap state load: request_resize %dx%d", w, h);
        if (hostGui == nullptr || hostGui->request_resize == nullptr
            || !hostGui->request_resize(host, static_cast<uint32_t>(w),
                                        static_cast<uint32_t>(h)))
            return;
        guiWidth = w;
        guiHeight = h;
        guiEditor.setBounds(w, h);
    }

    static void cbGuiSetParam(void* context, int index, double plain) noexcept
    {
        auto* s = static_cast<Plugin*>(context);
        const Param& spec = P::parameters[static_cast<size_t>(index)];
        const double normalized = toNormalized(spec, plain);
        // While active the audio thread applies the edit (and re-reads the
        // latency there); inactive, it lands now - and the latency must be
        // re-read here, which the old code never did for editor edits.
        if (s->applyFromUi(index, normalized))
            s->refreshLatency();
        const double snapped = toPlain(spec, normalized);
        const uint32_t id = hash32(spec.id);
        if (s->guiEditActive[static_cast<size_t>(index)])
            s->pushUiEvent(id, kUiValue, snapped);
        else
        {
            // Edits outside an explicit gesture still get one (host undo).
            s->pushUiEvent(id, kUiGestureBegin, 0.0);
            s->pushUiEvent(id, kUiValue, snapped);
            s->pushUiEvent(id, kUiGestureEnd, 0.0);
        }
        s->requestUiFlush();
    }

    static void cbGuiBeginEdit(void* context, int index) noexcept
    {
        auto* s = static_cast<Plugin*>(context);
        s->guiEditActive[static_cast<size_t>(index)] = true;
        s->pushUiEvent(hash32(P::parameters[static_cast<size_t>(index)].id),
                       kUiGestureBegin, 0.0);
        s->requestUiFlush();
    }

    static void cbGuiEndEdit(void* context, int index) noexcept
    {
        auto* s = static_cast<Plugin*>(context);
        s->guiEditActive[static_cast<size_t>(index)] = false;
        s->pushUiEvent(hash32(P::parameters[static_cast<size_t>(index)].id),
                       kUiGestureEnd, 0.0);
        s->requestUiFlush();
    }

    static bool sGuiIsApiSupported(const clap_plugin_t*, const char* api,
                                   bool isFloating) noexcept
    {
        return !isFloating && api != nullptr && std::strcmp(api, clapWindowApi()) == 0
            && webview_ui::Editor<P>::available();
    }

    static bool sGuiGetPreferredApi(const clap_plugin_t*, const char** api,
                                    bool* isFloating) noexcept
    {
        if (api == nullptr || isFloating == nullptr) return false;
        *api = clapWindowApi();
        *isFloating = false;
        return webview_ui::Editor<P>::available();
    }

    static bool sGuiCreate(const clap_plugin_t* p, const char* api,
                           bool isFloating) noexcept
    {
        auto* s = self(p);
        if (!sGuiIsApiSupported(p, api, isFloating)) return false;
#if defined(__linux__)
        // GTK breathes only when pumped from the host's run loop; without
        // timer-support the page would freeze - fall back to generic UI.
        s->hostTimer = s->host != nullptr
            ? static_cast<const clap_host_timer_support_t*>(
                  s->host->get_extension(s->host, CLAP_EXT_TIMER_SUPPORT))
            : nullptr;
        if (s->hostTimer == nullptr || s->hostTimer->register_timer == nullptr
            || !s->hostTimer->register_timer(s->host, 33, &s->guiTimerId))
        {
            webview_ui::debugLog("clap gui create: no host timer-support -> no editor");
            s->hostTimer = nullptr;
            s->guiTimerId = CLAP_INVALID_ID;
            return false;
        }
#endif
        // The size the user left the editor at (this session or a saved
        // one), else the declared size.
        const EditorSize logical = s->editorMemory.current();
        s->guiWidth  = toBoundedInt(logical.width * s->guiScale + 0.5,
                                    1, static_cast<int>(kEditorMaxPixels));
        s->guiHeight = toBoundedInt(logical.height * s->guiScale + 0.5,
                                    1, static_cast<int>(kEditorMaxPixels));
        s->guiActive = true;
        webview_ui::debugLog("clap gui create %dx%d scale=%.2f",
                             s->guiWidth, s->guiHeight, s->guiScale);
        return true;
    }

    static void sGuiDestroy(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
#if defined(__linux__)
        if (s->hostTimer != nullptr && s->guiTimerId != CLAP_INVALID_ID)
            s->hostTimer->unregister_timer(s->host, s->guiTimerId);
        s->hostTimer = nullptr;
        s->guiTimerId = CLAP_INVALID_ID;
#endif
        // Keep the size the editor is closed at: some hosts resize their
        // window without a set_size, and the window still exists here.
        int parentW = 0, parentH = 0;
        if (s->guiEditor.created() && s->guiEditor.queryParentSize(parentW, parentH))
        {
            double w = parentW;
            double h = parentH;
            constrainEditorSize<P>(w, h, s->guiScale);
            s->editorMemory.remember(w / s->guiScale, h / s->guiScale);
        }
        s->guiEditor.destroy();
        s->guiActive = false;
    }

    static bool sGuiSetScale(const clap_plugin_t* p, double scale) noexcept
    {
#if defined(__APPLE__)
        (void) p; (void) scale;
        return false;   // cocoa uses logical sizes; scaling is the OS's job
#else
        auto* s = self(p);
        if (!(scale > 0.0)) return false;   // NaN included
        // Only the physical-size negotiation changes; the page itself never
        // zooms (the web engine applies the window DPI to CSS pixels itself).
        const double previous = s->guiScale;
        s->guiScale = scale;
        s->guiWidth  = toBoundedInt(s->guiWidth * (scale / previous) + 0.5,
                                    1, static_cast<int>(kEditorMaxPixels));
        s->guiHeight = toBoundedInt(s->guiHeight * (scale / previous) + 0.5,
                                    1, static_cast<int>(kEditorMaxPixels));
        s->guiEditor.setBounds(s->guiWidth, s->guiHeight);
        return true;
#endif
    }

    static bool sGuiGetSize(const clap_plugin_t* p, uint32_t* width,
                            uint32_t* height) noexcept
    {
        auto* s = self(p);
        if (width == nullptr || height == nullptr || !s->guiActive) return false;
        *width  = static_cast<uint32_t>(s->guiWidth);
        *height = static_cast<uint32_t>(s->guiHeight);
        return true;
    }

    static bool sGuiCanResize(const clap_plugin_t*) noexcept
    {
        return editorResizeOf<P>() != EditorResize::Fixed;
    }

    static bool sGuiGetResizeHints(const clap_plugin_t*,
                                   clap_gui_resize_hints_t* hints) noexcept
    {
        if (hints == nullptr) return false;
        constexpr EditorResize mode = editorResizeOf<P>();
        const EditorSize logical = editorSizeOf<P>();
        hints->can_resize_horizontally = mode != EditorResize::Fixed;
        hints->can_resize_vertically = mode != EditorResize::Fixed;
        hints->preserve_aspect_ratio = mode == EditorResize::KeepAspect;
        hints->aspect_ratio_width = static_cast<uint32_t>(logical.width);
        hints->aspect_ratio_height = static_cast<uint32_t>(logical.height);
        return true;
    }

    static bool sGuiAdjustSize(const clap_plugin_t* p, uint32_t* width,
                               uint32_t* height) noexcept
    {
        auto* s = self(p);
        if (width == nullptr || height == nullptr) return false;
        double w = *width;
        double h = *height;
        constrainEditorSize<P>(w, h, s->guiScale);
        *width  = static_cast<uint32_t>(
            toBoundedInt(w + 0.5, 1, static_cast<int>(kEditorMaxPixels)));
        *height = static_cast<uint32_t>(
            toBoundedInt(h + 0.5, 1, static_cast<int>(kEditorMaxPixels)));
        return true;
    }

    static bool sGuiSetSize(const clap_plugin_t* p, uint32_t width,
                            uint32_t height) noexcept
    {
        auto* s = self(p);
        if (!s->guiActive) return false;
        // Clamp here too: not every host runs the size through adjust_size.
        double w = width;
        double h = height;
        constrainEditorSize<P>(w, h, s->guiScale);
        webview_ui::debugLog("clap gui set_size %ux%u -> %.0fx%.0f", width, height, w, h);
        s->guiWidth  = toBoundedInt(w + 0.5, 1, static_cast<int>(kEditorMaxPixels));
        s->guiHeight = toBoundedInt(h + 0.5, 1, static_cast<int>(kEditorMaxPixels));
        s->guiEditor.setBounds(s->guiWidth, s->guiHeight);
        // The user's size, in logical pixels: the editor reopens at it, and
        // the state saves it for the next session.
        s->editorMemory.remember(s->guiWidth / s->guiScale, s->guiHeight / s->guiScale);
        return true;
    }

    static bool sGuiSetParent(const clap_plugin_t* p,
                              const clap_window_t* window) noexcept
    {
        auto* s = self(p);
        if (window == nullptr || !s->guiActive) return false;
#if defined(__linux__)
        // The X11 window id rides the union's dedicated field.
        void* parentHandle = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(window->x11));
#else
        void* parentHandle = window->ptr;
#endif
        webview_ui::debugLog("clap gui set_parent %p negotiated=%dx%d",
                             parentHandle, s->guiWidth, s->guiHeight);
        const webview_ui::HostCallbacks callbacks {
            s, &cbGuiSetParam, &cbGuiBeginEdit, &cbGuiEndEdit
        };
        if (!s->guiEditor.create(parentHandle, s->shadow, callbacks))
            return false;
        // Fill whatever box the host actually built (call order differs
        // between hosts); fall back to the negotiated size otherwise.
        int parentW = 0, parentH = 0;
        if (s->guiEditor.queryParentSize(parentW, parentH))
        {
            s->guiWidth  = parentW;
            s->guiHeight = parentH;
        }
        s->guiEditor.setBounds(s->guiWidth, s->guiHeight);
        return true;
    }

    static bool sGuiSetTransient(const clap_plugin_t*, const clap_window_t*) noexcept
    {
        return false;   // embedded only; no floating window mode
    }

    static void sGuiSuggestTitle(const clap_plugin_t*, const char*) noexcept {}

    static bool sGuiShow(const clap_plugin_t* p) noexcept
    {
        self(p)->guiEditor.setVisible(true);
        return true;
    }

    static bool sGuiHide(const clap_plugin_t* p) noexcept
    {
        self(p)->guiEditor.setVisible(false);
        return true;
    }

    inline static const clap_plugin_gui_t kGui = {
        &sGuiIsApiSupported, &sGuiGetPreferredApi, &sGuiCreate, &sGuiDestroy,
        &sGuiSetScale, &sGuiGetSize, &sGuiCanResize, &sGuiGetResizeHints,
        &sGuiAdjustSize, &sGuiSetSize, &sGuiSetParent, &sGuiSetTransient,
        &sGuiSuggestTitle, &sGuiShow, &sGuiHide
    };

#if defined(__linux__)

    // --- ext: timer-support (drives the GTK main context for the editor) -----------

    static void sOnTimer(const clap_plugin_t* p, clap_id) noexcept
    {
        self(p)->guiEditor.pump();
    }

    inline static const clap_plugin_timer_support_t kTimer = { &sOnTimer };

#endif // __linux__
#endif // DSPARK_PLUGIN_WEBVIEW

    // --- descriptor & factory ----------------------------------------------------------

    static const clap_plugin_descriptor_t* descriptor() noexcept
    {
        static const char* features[] = {
            kIsInstrument ? CLAP_PLUGIN_FEATURE_INSTRUMENT
                          : CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
            channelSupportOf<P>() == ChannelSupport::MonoOnly
                ? CLAP_PLUGIN_FEATURE_MONO : CLAP_PLUGIN_FEATURE_STEREO,
            nullptr
        };
        static const clap_plugin_descriptor_t desc = {
            CLAP_VERSION_INIT,
            P::descriptor.productId,
            P::descriptor.name,
            P::descriptor.vendor,
            P::descriptor.url,
            "",                       // manual_url
            P::descriptor.url,        // support_url
            P::descriptor.version,
            "Built with DSPark",      // description
            features
        };
        return &desc;
    }

    static const clap_plugin_t* create(const clap_host_t* host) noexcept
    {
        auto* s = new (std::nothrow) Plugin(host);
        if (s == nullptr) return nullptr;
        s->plugin.desc = descriptor();
        s->plugin.plugin_data = s;
        s->plugin.init = &sInit;
        s->plugin.destroy = &sDestroy;
        s->plugin.activate = &sActivate;
        s->plugin.deactivate = &sDeactivate;
        s->plugin.start_processing = &sStartProcessing;
        s->plugin.stop_processing = &sStopProcessing;
        s->plugin.reset = &sReset;
        s->plugin.process = &sProcess;
        s->plugin.get_extension = &sGetExtension;
        s->plugin.on_main_thread = &sOnMainThread;
        return &s->plugin;
    }
};

template <typename P>
struct Factory
{
    static uint32_t sCount(const clap_plugin_factory_t*) noexcept { return 1; }

    static const clap_plugin_descriptor_t* sDescriptor(const clap_plugin_factory_t*,
                                                       uint32_t index) noexcept
    {
        return index == 0 ? Plugin<P>::descriptor() : nullptr;
    }

    static const clap_plugin_t* sCreate(const clap_plugin_factory_t*,
                                        const clap_host_t* host,
                                        const char* pluginId) noexcept
    {
        if (pluginId == nullptr
            || std::strcmp(pluginId, P::descriptor.productId) != 0)
            return nullptr;
        return Plugin<P>::create(host);
    }

    inline static const clap_plugin_factory_t kFactory = {
        &sCount, &sDescriptor, &sCreate
    };

    // --- preset discovery (factory presets live inside the plugin DSO) -------------
    //
    // The provider declares ONE location of kind PLUGIN and reports each
    // factory preset with its table index as the load key; preset-load's
    // from_location() applies it. Hosts with a preset browser (e.g. Bitwig)
    // index these without instantiating the plugin.

    static constexpr int kNumPresets = factoryPresetCountOf<P>();

    static const char* providerId() noexcept
    {
        // Magic static: the discovery factory contract is [thread-safe].
        static const std::array<char, 256> id = [] {
            std::array<char, 256> buf {};
            std::snprintf(buf.data(), buf.size(), "%s.presets",
                          P::descriptor.productId);
            return buf;
        }();
        return id.data();
    }

    static const clap_preset_discovery_provider_descriptor_t* providerDescriptor() noexcept
    {
        static const clap_preset_discovery_provider_descriptor_t desc = {
            CLAP_VERSION_INIT, providerId(), "DSPark Factory Presets",
            P::descriptor.vendor
        };
        return &desc;
    }

    struct ProviderState
    {
        clap_preset_discovery_provider_t provider {};
        const clap_preset_discovery_indexer_t* indexer = nullptr;
    };

    static bool sProviderInit(const clap_preset_discovery_provider_t* provider) noexcept
    {
        const auto* state = static_cast<const ProviderState*>(provider->provider_data);
        if (state->indexer == nullptr || state->indexer->declare_location == nullptr)
            return false;
        clap_preset_discovery_location_t location {};
        location.flags = CLAP_PRESET_DISCOVERY_IS_FACTORY_CONTENT;
        location.name = "Factory Presets";
        location.kind = CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN;
        location.location = nullptr;
        return state->indexer->declare_location(state->indexer, &location);
    }

    static void sProviderDestroy(const clap_preset_discovery_provider_t* provider) noexcept
    {
        delete static_cast<ProviderState*>(provider->provider_data);
    }

    static bool sProviderGetMetadata(const clap_preset_discovery_provider_t*,
                                     uint32_t locationKind, const char*,
                                     const clap_preset_discovery_metadata_receiver_t*
                                         receiver) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            if (locationKind != CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN
                || receiver == nullptr)
                return false;
            const clap_universal_plugin_id_t pluginId {
                "clap", P::descriptor.productId
            };
            for (int i = 0; i < kNumPresets; ++i)
            {
                char key[16];
                std::snprintf(key, sizeof(key), "%d", i);
                if (!receiver->begin_preset(receiver,
                        P::factoryPresets[static_cast<size_t>(i)].name, key))
                    return true;   // the indexer asked to stop; not an error
                receiver->add_plugin_id(receiver, &pluginId);
                receiver->set_flags(receiver,
                                    CLAP_PRESET_DISCOVERY_IS_FACTORY_CONTENT);
            }
            return true;
        }
        else
        {
            (void) locationKind;
            (void) receiver;
            return false;
        }
    }

    static const void* sProviderGetExtension(const clap_preset_discovery_provider_t*,
                                             const char*) noexcept
    { return nullptr; }

    static uint32_t sDiscoveryCount(const clap_preset_discovery_factory_t*) noexcept
    { return kNumPresets > 0 ? 1u : 0u; }

    static const clap_preset_discovery_provider_descriptor_t* sDiscoveryDescriptor(
        const clap_preset_discovery_factory_t*, uint32_t index) noexcept
    {
        return (kNumPresets > 0 && index == 0) ? providerDescriptor() : nullptr;
    }

    static const clap_preset_discovery_provider_t* sDiscoveryCreate(
        const clap_preset_discovery_factory_t*,
        const clap_preset_discovery_indexer_t* indexer,
        const char* providerIdAsked) noexcept
    {
        if (kNumPresets == 0 || indexer == nullptr || providerIdAsked == nullptr
            || std::strcmp(providerIdAsked, providerId()) != 0)
            return nullptr;
        auto* state = new (std::nothrow) ProviderState();
        if (state == nullptr) return nullptr;
        state->indexer = indexer;
        state->provider.desc = providerDescriptor();
        state->provider.provider_data = state;
        state->provider.init = &sProviderInit;
        state->provider.destroy = &sProviderDestroy;
        state->provider.get_metadata = &sProviderGetMetadata;
        state->provider.get_extension = &sProviderGetExtension;
        return &state->provider;
    }

    inline static const clap_preset_discovery_factory_t kDiscoveryFactory = {
        &sDiscoveryCount, &sDiscoveryDescriptor, &sDiscoveryCreate
    };

    static bool sEntryInit(const char*) noexcept { return true; }
    static void sEntryDeinit() noexcept {}

    static const void* sEntryGetFactory(const char* factoryId) noexcept
    {
        if (factoryId == nullptr) return nullptr;
        if (std::strcmp(factoryId, CLAP_PLUGIN_FACTORY_ID) == 0)
            return &kFactory;
        if (kNumPresets > 0
            && (std::strcmp(factoryId, CLAP_PRESET_DISCOVERY_FACTORY_ID) == 0
                || std::strcmp(factoryId, CLAP_PRESET_DISCOVERY_FACTORY_ID_COMPAT) == 0))
            return &kDiscoveryFactory;
        return nullptr;
    }
};

} // namespace dspark::plugin::clap_backend

/**
 * @brief Declares the CLAP module entry for one plugin class. May coexist
 * with DSPARK_VST3_PLUGIN in the same translation unit: the resulting binary
 * is a valid .vst3 AND a valid .clap at the same time.
 */
#define DSPARK_CLAP_PLUGIN(PluginClass)                                            \
    extern "C" CLAP_EXPORT const clap_plugin_entry_t clap_entry = {                \
        CLAP_VERSION_INIT,                                                         \
        &dspark::plugin::clap_backend::Factory<PluginClass>::sEntryInit,           \
        &dspark::plugin::clap_backend::Factory<PluginClass>::sEntryDeinit,         \
        &dspark::plugin::clap_backend::Factory<PluginClass>::sEntryGetFactory      \
    };

#pragma pop_macro("max")
#pragma pop_macro("min")
