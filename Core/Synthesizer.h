// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Synthesizer.h
 * @brief Polyphonic synthesizer: analog-style, FM and wavetable engines, a
 *        sample source, per-voice filter, envelopes, LFOs and a modulation
 *        matrix, with sample-accurate events and block-size invariant output.
 *
 * One voice plays one note. Its sources are summed, filtered and amplified:
 *
 * - **Engines** (parameter `engine`, latched per note): Analog - three
 *   band-limited oscillators (Oscillator: sine, triangle, saw, pulse with
 *   width modulation, hard sync to the voice pitch); FM - four sine
 *   operators in the eight classic four-operator algorithms, feedback on
 *   operator 4, an ADSR per operator; Wavetable - two oscillators morphing
 *   across a WavetableBank (four built-in banks or a user bank).
 * - **Sources common to every engine**: a sub-oscillator one or two octaves
 *   down, white noise, and a sample source that plays the first loaded zone
 *   answering the key and velocity, through the same band-limited reader as
 *   Sampler (SampleZoneSpec zones: one recording over the keyboard, a
 *   multisample, velocity layers, loops).
 * - **Unison**: 1..8 copies of the engine's oscillators (and of the sample
 *   source) per note, detuned symmetrically and spread across the stereo
 *   field, normalised by 1/sqrt(count).
 * - **Filter per voice**: ladder (LP24, LP12, BP12, HP24; saturating
 *   feedback, so resonance 1 self-oscillates at a bounded level) or state
 *   variable (LP, HP, BP at unit peak gain, notch; Q 0.5..50), with key
 *   tracking, an envelope amount in octaves scaled by velocity, and the
 *   cutoff interpolated every sample through the shared zero-delay-feedback
 *   kernels (detail/LadderZdf.h, detail/TptSvf.h).
 * - **Modulation**: envelope 1 drives the amplifier, envelope 2 the filter
 *   (filter.envAmount), envelope 3 is free; two LFOs (Lfo: seven shapes, Hz
 *   or tempo divisions, delay, fade-in, free-running or restarted per note);
 *   an eight-slot matrix (source x via x amount -> destination; amounts are
 *   in the destination's unit: semitones for pitch, octaves for cutoff and
 *   LFO rate, linear units for levels, width, position, resonance, pan and
 *   the amplifier's gain factor).
 * - **Voices**: polyphonic, mono or legato; constant-time glide; Sampler's
 *   stealing policy (the oldest released voice, else the quietest, else the
 *   oldest) with a 3 ms fade in a spare voice, so stealing never clicks;
 *   sustain pedal; analog drift (AnalogRandom) of pitch and cutoff.
 *
 * Timing and determinism: events carry a sample offset into the next block
 * and apply at that sample. Modulation is evaluated every kControlPeriod
 * samples on a grid anchored at each note's start and interpolated linearly
 * every sample, so pitch, levels and cutoff never step; parameter changes
 * are also smoothed per voice (10 ms). Every value a voice computes is a
 * function of the events, the parameters and the seed, never of where a
 * block boundary falls: the same event stream renders bit-identically with
 * any block size. Randomness (per-note values, random start phases, drift,
 * noise, random LFO shapes) derives from `voice.seed` and the note count
 * since reset().
 *
 * Oversampling (parameter `quality.oversampling`): the voices render at 2x
 * or 4x and are decimated by Oversampling's Maximum preset (~-100 dB
 * stopband) through its decimation-only path. It suppresses the aliasing of
 * FM, of the ladder's saturation and of the hard-synced sine at high pitch.
 * The voices start their content a fraction of a sample later so the
 * reported latency (getLatency()) is an integer: 64 samples at 2x and 96 at
 * 4x. Changing the factor stops the sounding voices.
 *
 * Output: stereo on the first two channels of the view (a single channel
 * receives the mean); further channels are silent. No limiting is applied:
 * leave headroom for chords (master.volume, default -6 dB).
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with anything else).
 * - setParameter() / setTempo(): any thread, lock free (relaxed atomics read
 *   at the start of every block).
 * - setSampleZones() / loadSample() / clearSampleZones() /
 *   loadUserWavetable(): control thread, one at a time. They build the new
 *   content off the audio thread and hand it over through a lock-free
 *   exchange, as Sampler does: the audio thread adopts it at a block start
 *   once no voice reads the previous one and hands that back to be freed by
 *   the next control call (or the destructor).
 * - getState() / setState(): setup or UI thread (allocate).
 * - noteOn() ... allNotesOff(), processBlock(), reset(): audio thread, the
 *   stream owner. Events are queued, allocation free.
 * - getActiveVoiceCount() / getLatency(): any thread, lock free.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h,
 * Oscillator.h, EnvelopeGenerator.h, Lfo.h, WavetableBank.h, SampleZone.h,
 * AnalogRandom.h, Oversampling.h, StateBlob.h, detail/LadderZdf.h,
 * detail/TptSvf.h, detail/SampleReader.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"
#include "Oscillator.h"
#include "EnvelopeGenerator.h"
#include "Lfo.h"
#include "WavetableBank.h"
#include "SampleZone.h"
#include "AnalogRandom.h"
#include "Oversampling.h"
#include "StateBlob.h"
#include "detail/LadderZdf.h"
#include "detail/TptSvf.h"
#include "detail/SampleReader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <string_view>
#include <vector>

namespace dspark {

/**
 * @namespace dspark::synthesizer
 * @brief Synthesizer's sample-type-independent vocabulary: choice values,
 *        parameter ids and the parameter table.
 */
namespace synthesizer {

enum class Engine { Analog, Fm, Wavetable };
enum class VoiceMode { Poly, Mono, Legato };
enum class GlideMode { Always, Legato };
enum class PhaseMode { Reset, Random };
enum class OscWave { Sine, Triangle, Saw, Pulse };
enum class SubWave { Square, Triangle, Sine };
enum class FilterType { Off, LadderLp24, LadderLp12, LadderBp12, LadderHp24, SvfLp, SvfHp, SvfBp, SvfNotch };
enum class LfoMode { Free, Retrigger };

/** @brief Modulation sources (values: velocity, pressure, wheel, expression
 *  and breath 0..1; key (key - 60) / 60; bend, LFOs and random -1..1;
 *  envelopes 0..1). */
enum class ModSource
{
    None, Velocity, ReleaseVelocity, Key, ModWheel, Pressure, PitchBend,
    Expression, Breath, Lfo1, Lfo2, Env1, Env2, Env3, Random
};
inline constexpr int kNumModSources = 15;

/** @brief Modulation destinations. OscN Pitch and OscN Level reach analog
 *  oscillator N and wavetable oscillator N; FM operators have their own. */
enum class ModDest
{
    None, Pitch, Osc1Pitch, Osc2Pitch, Osc3Pitch, Osc1Level, Osc2Level, Osc3Level,
    SubLevel, NoiseLevel, SampleLevel, PulseWidth, WtPosition, Wt1Position, Wt2Position,
    Op1Level, Op2Level, Op3Level, Op4Level, FmFeedback, Cutoff, Resonance, Amp, Pan,
    Lfo1Rate, Lfo2Rate
};
inline constexpr int kNumModDests = 26;

/** @brief Every parameter, in the order of the parameter table. */
enum class Param : int
{
    Engine, VoiceMode, Polyphony, Glide, GlideMode, BendRange, Unison, UnisonDetune,
    UnisonSpread, Drift, Seed, PhaseMode, MasterVolume, MasterTune, Oversampling,
    Osc1Wave, Osc1Level, Osc1Octave, Osc1Semitone, Osc1Fine, Osc1PulseWidth, Osc1Sync,
    Osc2Wave, Osc2Level, Osc2Octave, Osc2Semitone, Osc2Fine, Osc2PulseWidth, Osc2Sync,
    Osc3Wave, Osc3Level, Osc3Octave, Osc3Semitone, Osc3Fine, Osc3PulseWidth, Osc3Sync,
    SubLevel, SubOctave, SubWave, NoiseLevel,
    SampleLevel, SampleOctave, SampleSemitone, SampleFine, SampleKeyTrack,
    FmAlgorithm, FmFeedback,
    Op1Ratio, Op1Detune, Op1Fixed, Op1Frequency, Op1Level, Op1Velocity, Op1KeyScale,
    Op1Attack, Op1Decay, Op1Sustain, Op1Release,
    Op2Ratio, Op2Detune, Op2Fixed, Op2Frequency, Op2Level, Op2Velocity, Op2KeyScale,
    Op2Attack, Op2Decay, Op2Sustain, Op2Release,
    Op3Ratio, Op3Detune, Op3Fixed, Op3Frequency, Op3Level, Op3Velocity, Op3KeyScale,
    Op3Attack, Op3Decay, Op3Sustain, Op3Release,
    Op4Ratio, Op4Detune, Op4Fixed, Op4Frequency, Op4Level, Op4Velocity, Op4KeyScale,
    Op4Attack, Op4Decay, Op4Sustain, Op4Release,
    Wt1Bank, Wt1Position, Wt1Level, Wt1Octave, Wt1Semitone, Wt1Fine,
    Wt2Bank, Wt2Position, Wt2Level, Wt2Octave, Wt2Semitone, Wt2Fine,
    FilterType, FilterCutoff, FilterResonance, FilterDrive, FilterKeyTrack,
    FilterEnvAmount, FilterVelocity,
    AmpVelocity, AmpPan,
    Env1Attack, Env1Decay, Env1Sustain, Env1Release, Env1Curve,
    Env2Attack, Env2Decay, Env2Sustain, Env2Release, Env2Curve,
    Env3Attack, Env3Decay, Env3Sustain, Env3Release, Env3Curve,
    Lfo1Shape, Lfo1Rate, Lfo1Sync, Lfo1Division, Lfo1Delay, Lfo1FadeIn, Lfo1Phase, Lfo1Mode,
    Lfo2Shape, Lfo2Rate, Lfo2Sync, Lfo2Division, Lfo2Delay, Lfo2FadeIn, Lfo2Phase, Lfo2Mode,
    Mod1Source, Mod1Via, Mod1Destination, Mod1Amount,
    Mod2Source, Mod2Via, Mod2Destination, Mod2Amount,
    Mod3Source, Mod3Via, Mod3Destination, Mod3Amount,
    Mod4Source, Mod4Via, Mod4Destination, Mod4Amount,
    Mod5Source, Mod5Via, Mod5Destination, Mod5Amount,
    Mod6Source, Mod6Via, Mod6Destination, Mod6Amount,
    Mod7Source, Mod7Via, Mod7Destination, Mod7Amount,
    Mod8Source, Mod8Via, Mod8Destination, Mod8Amount,
    Count
};
inline constexpr int kNumParams = static_cast<int>(Param::Count);

/**
 * @brief Description of one parameter (the shape of the plugin layer's
 *        parameter declarations, so a host can map them one to one).
 *
 * `steps` is 0 for a continuous value, 1 for a toggle and N for N + 1
 * discrete positions; a choice's `labels` name every position and its value
 * is the position index.
 */
struct ParamInfo
{
    Param param;
    const char* id;     ///< Stable text id (state keys, automation identity).
    const char* name;   ///< Display name.
    float minValue;
    float maxValue;
    float defValue;
    const char* unit;   ///< "ms", "Hz", "dB", "cents", "st", "oct", "voices" or "".
    int steps;
    const char* const* labels;
};

inline constexpr int kMaxUnison = 8;
inline constexpr int kModSlots = 8;
inline constexpr int kMaxPolyphony = 64;

inline constexpr const char* kEngineLabels[] = { "Analog", "FM", "Wavetable" };
inline constexpr const char* kVoiceModeLabels[] = { "Poly", "Mono", "Legato" };
inline constexpr const char* kGlideModeLabels[] = { "Always", "Legato" };
inline constexpr const char* kPhaseLabels[] = { "Reset", "Random" };
inline constexpr const char* kOversamplingLabels[] = { "1x", "2x", "4x" };
inline constexpr const char* kOscWaveLabels[] = { "Sine", "Triangle", "Saw", "Pulse" };
inline constexpr const char* kSubOctaveLabels[] = { "-1", "-2" };
inline constexpr const char* kSubWaveLabels[] = { "Square", "Triangle", "Sine" };
inline constexpr const char* kAlgorithmLabels[] = {
    "1: 4>3>2>1", "2: (3+4)>2>1", "3: (4 + 3>2)>1", "4: (4>3 + 2)>1",
    "5: 4>3, 2>1", "6: 4>(1, 2, 3)", "7: 4>3, 2, 1", "8: 1, 2, 3, 4" };
inline constexpr const char* kBankLabels[] = { "Basic", "Harmonic", "Formant", "FM", "User" };
inline constexpr const char* kFilterLabels[] = {
    "Off", "Ladder LP24", "Ladder LP12", "Ladder BP12", "Ladder HP24",
    "SVF LP", "SVF HP", "SVF BP", "SVF Notch" };
inline constexpr const char* kLfoShapeLabels[] = {
    "Sine", "Triangle", "Saw Up", "Saw Down", "Square", "Sample and Hold", "Smooth Random" };
inline constexpr const char* kDivisionLabels[] = {
    "8 bars", "4 bars", "2 bars", "1/1", "1/2.", "1/2", "1/2T", "1/4.", "1/4", "1/4T",
    "1/8.", "1/8", "1/8T", "1/16.", "1/16", "1/16T", "1/32", "1/32T", "1/64" };
inline constexpr const char* kLfoModeLabels[] = { "Free", "Retrigger" };
inline constexpr const char* kModSourceLabels[] = {
    "None", "Velocity", "Release Velocity", "Key", "Mod Wheel", "Pressure", "Pitch Bend",
    "Expression", "Breath", "LFO 1", "LFO 2", "Env 1", "Env 2", "Env 3", "Random" };
inline constexpr const char* kModDestLabels[] = {
    "None", "Pitch", "Osc 1 Pitch", "Osc 2 Pitch", "Osc 3 Pitch", "Osc 1 Level",
    "Osc 2 Level", "Osc 3 Level", "Sub Level", "Noise Level", "Sample Level",
    "Pulse Width", "WT Position", "WT 1 Position", "WT 2 Position", "Op 1 Level",
    "Op 2 Level", "Op 3 Level", "Op 4 Level", "FM Feedback", "Cutoff", "Resonance",
    "Amp", "Pan", "LFO 1 Rate", "LFO 2 Rate" };

namespace table {
constexpr ParamInfo cont(Param p, const char* id, const char* name, float lo, float hi,
                         float def, const char* unit) noexcept
{
    return { p, id, name, lo, hi, def, unit, 0, nullptr };
}
constexpr ParamInfo integer(Param p, const char* id, const char* name, int lo, int hi,
                            int def, const char* unit) noexcept
{
    return { p, id, name, static_cast<float>(lo), static_cast<float>(hi),
             static_cast<float>(def), unit, hi - lo, nullptr };
}
constexpr ParamInfo toggle(Param p, const char* id, const char* name, bool def) noexcept
{
    return { p, id, name, 0.0f, 1.0f, def ? 1.0f : 0.0f, "", 1, nullptr };
}
template <size_t N>
constexpr ParamInfo choice(Param p, const char* id, const char* name,
                           const char* const (&labels)[N], int def) noexcept
{
    return { p, id, name, 0.0f, static_cast<float>(N - 1), static_cast<float>(def), "",
             static_cast<int>(N - 1), labels };
}
} // namespace table

/** @brief The parameter table, indexed by Param. */
inline constexpr std::array<ParamInfo, kNumParams> kParams = {{
    table::choice(Param::Engine, "engine", "Engine", kEngineLabels, 0),
    table::choice(Param::VoiceMode, "voice.mode", "Voice Mode", kVoiceModeLabels, 0),
    table::integer(Param::Polyphony, "voice.polyphony", "Polyphony", 1, kMaxPolyphony, 16, "voices"),
    table::cont(Param::Glide, "voice.glide", "Glide", 0.0f, 10000.0f, 0.0f, "ms"),
    table::choice(Param::GlideMode, "voice.glideMode", "Glide Mode", kGlideModeLabels, 0),
    table::integer(Param::BendRange, "voice.bendRange", "Bend Range", 0, 48, 2, "st"),
    table::integer(Param::Unison, "voice.unison", "Unison", 1, kMaxUnison, 1, ""),
    table::cont(Param::UnisonDetune, "voice.unisonDetune", "Unison Detune", 0.0f, 100.0f, 20.0f, "cents"),
    table::cont(Param::UnisonSpread, "voice.unisonSpread", "Unison Spread", 0.0f, 1.0f, 0.5f, ""),
    table::cont(Param::Drift, "voice.drift", "Analog Drift", 0.0f, 1.0f, 0.0f, ""),
    table::integer(Param::Seed, "voice.seed", "Seed", 0, 65535, 0, ""),
    table::choice(Param::PhaseMode, "voice.phase", "Oscillator Phase", kPhaseLabels, 0),
    table::cont(Param::MasterVolume, "master.volume", "Volume", -60.0f, 12.0f, -6.0f, "dB"),
    table::cont(Param::MasterTune, "master.tune", "Tune", -100.0f, 100.0f, 0.0f, "cents"),
    table::choice(Param::Oversampling, "quality.oversampling", "Oversampling", kOversamplingLabels, 0),

    table::choice(Param::Osc1Wave, "osc1.wave", "Osc 1 Wave", kOscWaveLabels, 2),
    table::cont(Param::Osc1Level, "osc1.level", "Osc 1 Level", 0.0f, 1.0f, 1.0f, ""),
    table::integer(Param::Osc1Octave, "osc1.octave", "Osc 1 Octave", -3, 3, 0, ""),
    table::integer(Param::Osc1Semitone, "osc1.semitone", "Osc 1 Semitone", -12, 12, 0, "st"),
    table::cont(Param::Osc1Fine, "osc1.fine", "Osc 1 Fine", -100.0f, 100.0f, 0.0f, "cents"),
    table::cont(Param::Osc1PulseWidth, "osc1.pulseWidth", "Osc 1 Pulse Width", 0.01f, 0.99f, 0.5f, ""),
    table::toggle(Param::Osc1Sync, "osc1.sync", "Osc 1 Sync", false),
    table::choice(Param::Osc2Wave, "osc2.wave", "Osc 2 Wave", kOscWaveLabels, 2),
    table::cont(Param::Osc2Level, "osc2.level", "Osc 2 Level", 0.0f, 1.0f, 0.0f, ""),
    table::integer(Param::Osc2Octave, "osc2.octave", "Osc 2 Octave", -3, 3, 0, ""),
    table::integer(Param::Osc2Semitone, "osc2.semitone", "Osc 2 Semitone", -12, 12, 0, "st"),
    table::cont(Param::Osc2Fine, "osc2.fine", "Osc 2 Fine", -100.0f, 100.0f, 0.0f, "cents"),
    table::cont(Param::Osc2PulseWidth, "osc2.pulseWidth", "Osc 2 Pulse Width", 0.01f, 0.99f, 0.5f, ""),
    table::toggle(Param::Osc2Sync, "osc2.sync", "Osc 2 Sync", false),
    table::choice(Param::Osc3Wave, "osc3.wave", "Osc 3 Wave", kOscWaveLabels, 2),
    table::cont(Param::Osc3Level, "osc3.level", "Osc 3 Level", 0.0f, 1.0f, 0.0f, ""),
    table::integer(Param::Osc3Octave, "osc3.octave", "Osc 3 Octave", -3, 3, 0, ""),
    table::integer(Param::Osc3Semitone, "osc3.semitone", "Osc 3 Semitone", -12, 12, 0, "st"),
    table::cont(Param::Osc3Fine, "osc3.fine", "Osc 3 Fine", -100.0f, 100.0f, 0.0f, "cents"),
    table::cont(Param::Osc3PulseWidth, "osc3.pulseWidth", "Osc 3 Pulse Width", 0.01f, 0.99f, 0.5f, ""),
    table::toggle(Param::Osc3Sync, "osc3.sync", "Osc 3 Sync", false),

    table::cont(Param::SubLevel, "sub.level", "Sub Level", 0.0f, 1.0f, 0.0f, ""),
    table::choice(Param::SubOctave, "sub.octave", "Sub Octave", kSubOctaveLabels, 0),
    table::choice(Param::SubWave, "sub.wave", "Sub Wave", kSubWaveLabels, 0),
    table::cont(Param::NoiseLevel, "noise.level", "Noise Level", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::SampleLevel, "sample.level", "Sample Level", 0.0f, 1.0f, 0.0f, ""),
    table::integer(Param::SampleOctave, "sample.octave", "Sample Octave", -3, 3, 0, ""),
    table::integer(Param::SampleSemitone, "sample.semitone", "Sample Semitone", -12, 12, 0, "st"),
    table::cont(Param::SampleFine, "sample.fine", "Sample Fine", -100.0f, 100.0f, 0.0f, "cents"),
    table::toggle(Param::SampleKeyTrack, "sample.keyTrack", "Sample Key Track", true),

    table::choice(Param::FmAlgorithm, "fm.algorithm", "FM Algorithm", kAlgorithmLabels, 0),
    table::cont(Param::FmFeedback, "fm.feedback", "FM Feedback", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op1Ratio, "op1.ratio", "Op 1 Ratio", 0.25f, 32.0f, 1.0f, ""),
    table::cont(Param::Op1Detune, "op1.detune", "Op 1 Detune", -100.0f, 100.0f, 0.0f, "cents"),
    table::toggle(Param::Op1Fixed, "op1.fixed", "Op 1 Fixed", false),
    table::cont(Param::Op1Frequency, "op1.frequency", "Op 1 Frequency", 1.0f, 20000.0f, 440.0f, "Hz"),
    table::cont(Param::Op1Level, "op1.level", "Op 1 Level", 0.0f, 1.0f, 1.0f, ""),
    table::cont(Param::Op1Velocity, "op1.velocity", "Op 1 Velocity", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op1KeyScale, "op1.keyScale", "Op 1 Key Scale", -1.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op1Attack, "op1.attack", "Op 1 Attack", 0.0f, 20000.0f, 1.0f, "ms"),
    table::cont(Param::Op1Decay, "op1.decay", "Op 1 Decay", 0.0f, 20000.0f, 500.0f, "ms"),
    table::cont(Param::Op1Sustain, "op1.sustain", "Op 1 Sustain", 0.0f, 1.0f, 1.0f, ""),
    table::cont(Param::Op1Release, "op1.release", "Op 1 Release", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Op2Ratio, "op2.ratio", "Op 2 Ratio", 0.25f, 32.0f, 1.0f, ""),
    table::cont(Param::Op2Detune, "op2.detune", "Op 2 Detune", -100.0f, 100.0f, 0.0f, "cents"),
    table::toggle(Param::Op2Fixed, "op2.fixed", "Op 2 Fixed", false),
    table::cont(Param::Op2Frequency, "op2.frequency", "Op 2 Frequency", 1.0f, 20000.0f, 440.0f, "Hz"),
    table::cont(Param::Op2Level, "op2.level", "Op 2 Level", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op2Velocity, "op2.velocity", "Op 2 Velocity", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op2KeyScale, "op2.keyScale", "Op 2 Key Scale", -1.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op2Attack, "op2.attack", "Op 2 Attack", 0.0f, 20000.0f, 1.0f, "ms"),
    table::cont(Param::Op2Decay, "op2.decay", "Op 2 Decay", 0.0f, 20000.0f, 500.0f, "ms"),
    table::cont(Param::Op2Sustain, "op2.sustain", "Op 2 Sustain", 0.0f, 1.0f, 1.0f, ""),
    table::cont(Param::Op2Release, "op2.release", "Op 2 Release", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Op3Ratio, "op3.ratio", "Op 3 Ratio", 0.25f, 32.0f, 1.0f, ""),
    table::cont(Param::Op3Detune, "op3.detune", "Op 3 Detune", -100.0f, 100.0f, 0.0f, "cents"),
    table::toggle(Param::Op3Fixed, "op3.fixed", "Op 3 Fixed", false),
    table::cont(Param::Op3Frequency, "op3.frequency", "Op 3 Frequency", 1.0f, 20000.0f, 440.0f, "Hz"),
    table::cont(Param::Op3Level, "op3.level", "Op 3 Level", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op3Velocity, "op3.velocity", "Op 3 Velocity", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op3KeyScale, "op3.keyScale", "Op 3 Key Scale", -1.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op3Attack, "op3.attack", "Op 3 Attack", 0.0f, 20000.0f, 1.0f, "ms"),
    table::cont(Param::Op3Decay, "op3.decay", "Op 3 Decay", 0.0f, 20000.0f, 500.0f, "ms"),
    table::cont(Param::Op3Sustain, "op3.sustain", "Op 3 Sustain", 0.0f, 1.0f, 1.0f, ""),
    table::cont(Param::Op3Release, "op3.release", "Op 3 Release", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Op4Ratio, "op4.ratio", "Op 4 Ratio", 0.25f, 32.0f, 1.0f, ""),
    table::cont(Param::Op4Detune, "op4.detune", "Op 4 Detune", -100.0f, 100.0f, 0.0f, "cents"),
    table::toggle(Param::Op4Fixed, "op4.fixed", "Op 4 Fixed", false),
    table::cont(Param::Op4Frequency, "op4.frequency", "Op 4 Frequency", 1.0f, 20000.0f, 440.0f, "Hz"),
    table::cont(Param::Op4Level, "op4.level", "Op 4 Level", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op4Velocity, "op4.velocity", "Op 4 Velocity", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op4KeyScale, "op4.keyScale", "Op 4 Key Scale", -1.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Op4Attack, "op4.attack", "Op 4 Attack", 0.0f, 20000.0f, 1.0f, "ms"),
    table::cont(Param::Op4Decay, "op4.decay", "Op 4 Decay", 0.0f, 20000.0f, 500.0f, "ms"),
    table::cont(Param::Op4Sustain, "op4.sustain", "Op 4 Sustain", 0.0f, 1.0f, 1.0f, ""),
    table::cont(Param::Op4Release, "op4.release", "Op 4 Release", 0.0f, 20000.0f, 300.0f, "ms"),

    table::choice(Param::Wt1Bank, "wt1.bank", "WT 1 Bank", kBankLabels, 0),
    table::cont(Param::Wt1Position, "wt1.position", "WT 1 Position", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Wt1Level, "wt1.level", "WT 1 Level", 0.0f, 1.0f, 1.0f, ""),
    table::integer(Param::Wt1Octave, "wt1.octave", "WT 1 Octave", -3, 3, 0, ""),
    table::integer(Param::Wt1Semitone, "wt1.semitone", "WT 1 Semitone", -12, 12, 0, "st"),
    table::cont(Param::Wt1Fine, "wt1.fine", "WT 1 Fine", -100.0f, 100.0f, 0.0f, "cents"),
    table::choice(Param::Wt2Bank, "wt2.bank", "WT 2 Bank", kBankLabels, 0),
    table::cont(Param::Wt2Position, "wt2.position", "WT 2 Position", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Wt2Level, "wt2.level", "WT 2 Level", 0.0f, 1.0f, 0.0f, ""),
    table::integer(Param::Wt2Octave, "wt2.octave", "WT 2 Octave", -3, 3, 0, ""),
    table::integer(Param::Wt2Semitone, "wt2.semitone", "WT 2 Semitone", -12, 12, 0, "st"),
    table::cont(Param::Wt2Fine, "wt2.fine", "WT 2 Fine", -100.0f, 100.0f, 0.0f, "cents"),

    table::choice(Param::FilterType, "filter.type", "Filter Type", kFilterLabels, 1),
    table::cont(Param::FilterCutoff, "filter.cutoff", "Cutoff", 20.0f, 20000.0f, 20000.0f, "Hz"),
    table::cont(Param::FilterResonance, "filter.resonance", "Resonance", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::FilterDrive, "filter.drive", "Drive", 1.0f, 10.0f, 1.0f, ""),
    table::cont(Param::FilterKeyTrack, "filter.keyTrack", "Key Track", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::FilterEnvAmount, "filter.envAmount", "Env Amount", -10.0f, 10.0f, 0.0f, "oct"),
    table::cont(Param::FilterVelocity, "filter.velocity", "Velocity to Env", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::AmpVelocity, "amp.velocity", "Velocity Range", 0.0f, 60.0f, 24.0f, "dB"),
    table::cont(Param::AmpPan, "amp.pan", "Pan", -1.0f, 1.0f, 0.0f, ""),

    table::cont(Param::Env1Attack, "env1.attack", "Env 1 Attack", 0.0f, 20000.0f, 2.0f, "ms"),
    table::cont(Param::Env1Decay, "env1.decay", "Env 1 Decay", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Env1Sustain, "env1.sustain", "Env 1 Sustain", 0.0f, 1.0f, 1.0f, ""),
    table::cont(Param::Env1Release, "env1.release", "Env 1 Release", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Env1Curve, "env1.curve", "Env 1 Curve", 0.1f, 10.0f, 3.0f, ""),
    table::cont(Param::Env2Attack, "env2.attack", "Env 2 Attack", 0.0f, 20000.0f, 2.0f, "ms"),
    table::cont(Param::Env2Decay, "env2.decay", "Env 2 Decay", 0.0f, 20000.0f, 400.0f, "ms"),
    table::cont(Param::Env2Sustain, "env2.sustain", "Env 2 Sustain", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Env2Release, "env2.release", "Env 2 Release", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Env2Curve, "env2.curve", "Env 2 Curve", 0.1f, 10.0f, 3.0f, ""),
    table::cont(Param::Env3Attack, "env3.attack", "Env 3 Attack", 0.0f, 20000.0f, 2.0f, "ms"),
    table::cont(Param::Env3Decay, "env3.decay", "Env 3 Decay", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Env3Sustain, "env3.sustain", "Env 3 Sustain", 0.0f, 1.0f, 0.0f, ""),
    table::cont(Param::Env3Release, "env3.release", "Env 3 Release", 0.0f, 20000.0f, 300.0f, "ms"),
    table::cont(Param::Env3Curve, "env3.curve", "Env 3 Curve", 0.1f, 10.0f, 3.0f, ""),

    table::choice(Param::Lfo1Shape, "lfo1.shape", "LFO 1 Shape", kLfoShapeLabels, 0),
    table::cont(Param::Lfo1Rate, "lfo1.rate", "LFO 1 Rate", 0.01f, 100.0f, 5.0f, "Hz"),
    table::toggle(Param::Lfo1Sync, "lfo1.sync", "LFO 1 Tempo Sync", false),
    table::choice(Param::Lfo1Division, "lfo1.division", "LFO 1 Division", kDivisionLabels, 8),
    table::cont(Param::Lfo1Delay, "lfo1.delay", "LFO 1 Delay", 0.0f, 10000.0f, 0.0f, "ms"),
    table::cont(Param::Lfo1FadeIn, "lfo1.fadeIn", "LFO 1 Fade In", 0.0f, 10000.0f, 0.0f, "ms"),
    table::cont(Param::Lfo1Phase, "lfo1.phase", "LFO 1 Phase", 0.0f, 1.0f, 0.0f, ""),
    table::choice(Param::Lfo1Mode, "lfo1.mode", "LFO 1 Mode", kLfoModeLabels, 1),
    table::choice(Param::Lfo2Shape, "lfo2.shape", "LFO 2 Shape", kLfoShapeLabels, 0),
    table::cont(Param::Lfo2Rate, "lfo2.rate", "LFO 2 Rate", 0.01f, 100.0f, 5.0f, "Hz"),
    table::toggle(Param::Lfo2Sync, "lfo2.sync", "LFO 2 Tempo Sync", false),
    table::choice(Param::Lfo2Division, "lfo2.division", "LFO 2 Division", kDivisionLabels, 8),
    table::cont(Param::Lfo2Delay, "lfo2.delay", "LFO 2 Delay", 0.0f, 10000.0f, 0.0f, "ms"),
    table::cont(Param::Lfo2FadeIn, "lfo2.fadeIn", "LFO 2 Fade In", 0.0f, 10000.0f, 0.0f, "ms"),
    table::cont(Param::Lfo2Phase, "lfo2.phase", "LFO 2 Phase", 0.0f, 1.0f, 0.0f, ""),
    table::choice(Param::Lfo2Mode, "lfo2.mode", "LFO 2 Mode", kLfoModeLabels, 1),

    table::choice(Param::Mod1Source, "mod1.source", "Mod 1 Source", kModSourceLabels, 0),
    table::choice(Param::Mod1Via, "mod1.via", "Mod 1 Via", kModSourceLabels, 0),
    table::choice(Param::Mod1Destination, "mod1.destination", "Mod 1 Destination", kModDestLabels, 0),
    table::cont(Param::Mod1Amount, "mod1.amount", "Mod 1 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod2Source, "mod2.source", "Mod 2 Source", kModSourceLabels, 0),
    table::choice(Param::Mod2Via, "mod2.via", "Mod 2 Via", kModSourceLabels, 0),
    table::choice(Param::Mod2Destination, "mod2.destination", "Mod 2 Destination", kModDestLabels, 0),
    table::cont(Param::Mod2Amount, "mod2.amount", "Mod 2 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod3Source, "mod3.source", "Mod 3 Source", kModSourceLabels, 0),
    table::choice(Param::Mod3Via, "mod3.via", "Mod 3 Via", kModSourceLabels, 0),
    table::choice(Param::Mod3Destination, "mod3.destination", "Mod 3 Destination", kModDestLabels, 0),
    table::cont(Param::Mod3Amount, "mod3.amount", "Mod 3 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod4Source, "mod4.source", "Mod 4 Source", kModSourceLabels, 0),
    table::choice(Param::Mod4Via, "mod4.via", "Mod 4 Via", kModSourceLabels, 0),
    table::choice(Param::Mod4Destination, "mod4.destination", "Mod 4 Destination", kModDestLabels, 0),
    table::cont(Param::Mod4Amount, "mod4.amount", "Mod 4 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod5Source, "mod5.source", "Mod 5 Source", kModSourceLabels, 0),
    table::choice(Param::Mod5Via, "mod5.via", "Mod 5 Via", kModSourceLabels, 0),
    table::choice(Param::Mod5Destination, "mod5.destination", "Mod 5 Destination", kModDestLabels, 0),
    table::cont(Param::Mod5Amount, "mod5.amount", "Mod 5 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod6Source, "mod6.source", "Mod 6 Source", kModSourceLabels, 0),
    table::choice(Param::Mod6Via, "mod6.via", "Mod 6 Via", kModSourceLabels, 0),
    table::choice(Param::Mod6Destination, "mod6.destination", "Mod 6 Destination", kModDestLabels, 0),
    table::cont(Param::Mod6Amount, "mod6.amount", "Mod 6 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod7Source, "mod7.source", "Mod 7 Source", kModSourceLabels, 0),
    table::choice(Param::Mod7Via, "mod7.via", "Mod 7 Via", kModSourceLabels, 0),
    table::choice(Param::Mod7Destination, "mod7.destination", "Mod 7 Destination", kModDestLabels, 0),
    table::cont(Param::Mod7Amount, "mod7.amount", "Mod 7 Amount", -48.0f, 48.0f, 0.0f, ""),
    table::choice(Param::Mod8Source, "mod8.source", "Mod 8 Source", kModSourceLabels, 0),
    table::choice(Param::Mod8Via, "mod8.via", "Mod 8 Via", kModSourceLabels, 0),
    table::choice(Param::Mod8Destination, "mod8.destination", "Mod 8 Destination", kModDestLabels, 0),
    table::cont(Param::Mod8Amount, "mod8.amount", "Mod 8 Amount", -48.0f, 48.0f, 0.0f, ""),
}};

/** @brief True when the table is in Param order, ids are unique and every
 *  default lies in its range. */
constexpr bool tableIsConsistent() noexcept
{
    std::array<std::string_view, kNumParams> ids {};
    for (int i = 0; i < kNumParams; ++i)
    {
        const ParamInfo& a = kParams[static_cast<size_t>(i)];
        if (static_cast<int>(a.param) != i) return false;
        if (!(a.minValue <= a.defValue && a.defValue <= a.maxValue)) return false;
        ids[static_cast<size_t>(i)] = a.id;
    }
    // Sorted, so duplicates are neighbours: within every compiler's
    // constant-evaluation budget, unlike comparing all pairs.
    std::sort(ids.begin(), ids.end());
    return std::adjacent_find(ids.begin(), ids.end()) == ids.end();
}
static_assert(tableIsConsistent(), "Synthesizer parameter table out of order, duplicated or out of range");

/** @brief Index of the parameter with this id, or -1. */
[[nodiscard]] inline int findParam(std::string_view id) noexcept
{
    for (int i = 0; i < kNumParams; ++i)
        if (id == kParams[static_cast<size_t>(i)].id) return i;
    return -1;
}

} // namespace synthesizer

/**
 * @class Synthesizer
 * @brief Polyphonic synthesizer with analog, FM and wavetable engines and a
 *        sample source.
 *
 * @code
 *   dspark::Synthesizer<float> synth;
 *   synth.prepare({ 48000.0, 512, 2 }, 16);
 *   using P = dspark::Synthesizer<float>::Param;
 *   synth.setParameter(P::Osc1Wave, 3.0f);          // pulse
 *   synth.setParameter(P::FilterCutoff, 1200.0f);
 *   synth.setParameter(P::FilterEnvAmount, 4.0f);   // octaves
 *   // audio thread, per block:
 *   synth.noteOn(48, 0.9f, offsetInBlock);
 *   synth.processBlock(output);
 * @endcode
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Synthesizer
{
public:
    /** Maximum queued events between processBlock calls. Events beyond
     *  this capacity are ignored; hosts must split dense spans beforehand.
     *  Every event kind shares the same queue. */
    static constexpr int kMaxEvents = 1024;

    using Engine = synthesizer::Engine;
    using VoiceMode = synthesizer::VoiceMode;
    using GlideMode = synthesizer::GlideMode;
    using PhaseMode = synthesizer::PhaseMode;
    using OscWave = synthesizer::OscWave;
    using SubWave = synthesizer::SubWave;
    using FilterType = synthesizer::FilterType;
    using LfoMode = synthesizer::LfoMode;
    using ModSource = synthesizer::ModSource;
    using ModDest = synthesizer::ModDest;
    using Param = synthesizer::Param;
    using ParamInfo = synthesizer::ParamInfo;

    static constexpr int kNumParams = synthesizer::kNumParams;
    static constexpr int kMaxUnison = synthesizer::kMaxUnison;
    static constexpr int kModSlots = synthesizer::kModSlots;
    static constexpr int kMaxPolyphony = synthesizer::kMaxPolyphony;
    /// Base-rate samples between modulation evaluations (interpolated between).
    static constexpr int kControlPeriod = 16;

    /** @brief The description of a parameter. */
    [[nodiscard]] static constexpr const ParamInfo& getParamInfo(Param p) noexcept
    {
        return synthesizer::kParams[static_cast<size_t>(std::clamp(static_cast<int>(p), 0, kNumParams - 1))];
    }

    /** @brief Index of the parameter with this id, or -1. */
    [[nodiscard]] static int findParam(std::string_view id) noexcept { return synthesizer::findParam(id); }

    // -- Construction ---------------------------------------------------------------

    Synthesizer() noexcept
    {
        for (int i = 0; i < kNumParams; ++i)
        {
            values_[static_cast<size_t>(i)].store(info(i).defValue, std::memory_order_relaxed);
            params_[static_cast<size_t>(i)] = info(i).defValue;
        }
    }

    ~Synthesizer()
    {
        delete pending_.exchange(nullptr, std::memory_order_acq_rel);
        delete retired_.exchange(nullptr, std::memory_order_acq_rel);
        delete fading_;
        delete active_;
    }

    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;

    // -- Lifecycle ------------------------------------------------------------------

    /**
     * @brief Allocates the voices, buffers and oversamplers and builds the
     *        shared tables (minBLEP, sample kernel, built-in wavetables).
     * @param spec      Sample rate and the largest block (channels are taken
     *                  from the view at processBlock()).
     * @param maxVoices Polyphony available to `voice.polyphony`, 1..kMaxPolyphony.
     *                  Spare voices for stealing fades are added on top.
     */
    void prepare(const AudioSpec& spec, int maxVoices = 16)
    {
        if (!(spec.sampleRate > 0.0) || !std::isfinite(spec.sampleRate) || spec.maxBlockSize < 1)
            return;
        prepared_ = false;
        baseRate_ = spec.sampleRate;
        maxBlock_ = spec.maxBlockSize;
        maxVoices_ = std::clamp(maxVoices, 1, kMaxPolyphony);
        voices_ = std::vector<Voice>(static_cast<size_t>(maxVoices_ + maxVoices_ / 4 + 2));
        for (auto& v : voices_)
            for (auto& u : v.uni) u.reader.allocate();
        weights_.assign(static_cast<size_t>(Reader::kWeightsLength), T(0));
        for (auto& b : mix_) b.assign(static_cast<size_t>(maxBlock_), T(0));
        const AudioSpec osSpec { baseRate_, maxBlock_, 2 };
        os2_.prepare(osSpec);
        os4_.prepare(osSpec);
        // Shared tables, built here and never on the audio thread.
        (void)Reader::polyphaseTable();
        for (int b = 0; b < WavetableBank::kNumBuiltins; ++b)
            (void)WavetableBank::builtin(static_cast<WavetableBank::Builtin>(b));
        (void)MinBlepTable<T>::instance();

        snapshotParameters(true);
        configureRate(oversamplingFactor());
        reset();
        prepared_ = true;
    }

    /** @brief Silences every voice, drops queued events and restarts the
     *  note count, the controllers and the free LFOs. Audio thread. */
    void reset() noexcept
    {
        for (auto& v : voices_) v.active = false;
        eventCount_ = 0;
        sustainPedal_ = false;
        bend_ = 0.0;
        modWheel_ = 0.0;
        pressure_ = 0.0;
        expression_ = 1.0;
        breath_ = 0.0;
        noteCounter_ = 0;
        clock_ = 0;
        timeHi_ = 0;
        heldCount_ = 0;
        lastPitch_ = -1.0;
        haveLastPitch_ = false;
        for (int k = 0; k < 2; ++k)
        {
            freeLfo_[static_cast<size_t>(k)].anchorCycles = p(lfoParam(k, 6));
            freeLfo_[static_cast<size_t>(k)].anchorTime = 0;
        }
        masterGain_ = decibelsToGain(static_cast<double>(p(Param::MasterVolume)), -200.0);
        os2_.reset();
        os4_.reset();
        activeVoices_.store(0, std::memory_order_relaxed);
    }

    /** @brief Output delay in base-rate samples (0 at 1x; 64 at 2x, 96 at 4x). */
    [[nodiscard]] int getLatency() const noexcept { return latency_.load(std::memory_order_relaxed); }

    /** @brief Voices sounding after the last block, stealing fades included. */
    [[nodiscard]] int getActiveVoiceCount() const noexcept
    {
        return activeVoices_.load(std::memory_order_relaxed);
    }

    /** @brief Polyphony prepared (the upper bound of `voice.polyphony`). */
    [[nodiscard]] int getMaxVoices() const noexcept { return maxVoices_; }

    // -- Parameters (any thread) ------------------------------------------------------

    /** @brief Sets a parameter, clamped to its range and rounded to its
     *  steps; NaN is ignored. Takes effect at the next block. */
    void setParameter(Param param, float value) noexcept
    {
        const int i = static_cast<int>(param);
        if (i < 0 || i >= kNumParams || value != value) return;
        const ParamInfo& pinfo = info(i);
        float v = std::clamp(value, pinfo.minValue, pinfo.maxValue);
        if (pinfo.steps > 0) v = std::round(v);
        values_[static_cast<size_t>(i)].store(v, std::memory_order_relaxed);
    }

    /** @brief Sets a parameter by id; false for an unknown id. */
    bool setParameter(std::string_view id, float value) noexcept
    {
        const int i = findParam(id);
        if (i < 0) return false;
        setParameter(static_cast<Param>(i), value);
        return true;
    }

    /** @brief The value last set (any thread). */
    [[nodiscard]] float getParameter(Param param) const noexcept
    {
        const int i = std::clamp(static_cast<int>(param), 0, kNumParams - 1);
        return values_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
    }

    /** @brief Tempo for tempo-synced LFOs, 1..999 BPM (default 120). */
    void setTempo(double bpm) noexcept
    {
        if (!(bpm >= 1.0) || !(bpm <= 999.0)) return;
        tempo_.store(static_cast<float>(bpm), std::memory_order_relaxed);
    }

    // -- Samples and wavetables (control thread) ------------------------------------------

    /**
     * @brief Replaces the zones the sample source plays (sample.level).
     *
     * Validates and copies every zone (Sampler's rules) and publishes them
     * without blocking the audio thread. A note plays the FIRST zone whose key
     * and velocity ranges contain it, so velocity layers and multisamples
     * work, but zones are not layered as in Sampler. The user wavetable is
     * kept.
     * @return False (zones unchanged) when a zone is invalid.
     */
    bool setSampleZones(std::span<const SampleZoneSpec<T>> zones)
    {
        auto built = std::make_shared<std::vector<ZoneData>>();
        built->reserve(zones.size());
        for (const auto& z : zones)
        {
            ZoneData d;
            if (!detail::buildSampleZone(z, d)) return false;
            built->push_back(std::move(d));
        }
        controlZones_ = std::move(built);
        publishContent();
        return true;
    }

    /** @brief One zone over the whole keyboard and velocity range. */
    bool loadSample(AudioBufferView<const T> audio, double sampleRate, int rootKey = 60)
    {
        SampleZoneSpec<T> z;
        z.audio = audio;
        z.sampleRate = sampleRate;
        z.rootKey = rootKey;
        return setSampleZones(std::span<const SampleZoneSpec<T>>(&z, 1));
    }

    /** @brief Removes the sample zones (the user wavetable is kept). */
    void clearSampleZones()
    {
        controlZones_.reset();
        publishContent();
    }

    /**
     * @brief Loads the user wavetable (bank "User"): consecutive single-cycle
     *        frames, analysed as WavetableBank::loadFrames() does.
     * @return False (unchanged) on invalid frames.
     */
    bool loadUserWavetable(const T* frames, int frameSize, int numFrames)
    {
        auto bank = std::make_shared<WavetableBank>();
        if (!bank->loadFrames(frames, frameSize, numFrames)) return false;
        controlWavetable_ = std::move(bank);
        publishContent();
        return true;
    }

    // -- Events (audio thread) -------------------------------------------------------------

    /** @brief Starts a note `offset` samples into the next block; velocity
     *  0..1 (0 is a note-off, as in MIDI). */
    void noteOn(int key, T velocity, int offset = 0) noexcept
    {
        noteOnTuned(key, velocity, T(0), offset);
    }

    /** @brief Starts a note with individual fine tuning in cents, clamped to
     *  -100..100. The tuned pitch is the glide target and combines with
     *  master tuning, oscillators/zones and pitch bend. Key identity,
     *  key modulation/tracking and sample-zone selection use the integer key.
     *  Fixed-frequency FM operators and untracked sample sources ignore it. Mono/legato return
     *  restores the tuning of the most recent key still held.
     *  Non-finite velocity or tuning is ignored. Zero velocity is note-off.
     *  Audio thread; the event applies at offset in the next block. */
    void noteOnTuned(int key, T velocity, T tuningCents, int offset = 0) noexcept
    {
        if (!std::isfinite(velocity) || !std::isfinite(tuningCents)) return;
        if (velocity <= T(0)) { noteOff(key, T(0), offset); return; }
        push({ EventType::NoteOn, std::clamp(key, 0, 127),
               std::clamp(static_cast<double>(velocity), 0.0, 1.0), offset,
               std::clamp(static_cast<double>(tuningCents), -100.0, 100.0) });
    }

    /** @brief Releases a note; releaseVelocity 0..1 is a modulation source. */
    void noteOff(int key, T releaseVelocity = T(0), int offset = 0) noexcept
    {
        const double rv = std::isfinite(releaseVelocity) ? static_cast<double>(releaseVelocity) : 0.0;
        push({ EventType::NoteOff, std::clamp(key, 0, 127), std::clamp(rv, 0.0, 1.0), offset });
    }

    /** @brief Pitch bend, -1..+1 (scaled by voice.bendRange semitones). */
    void setPitchBend(T bend, int offset = 0) noexcept
    {
        if (!std::isfinite(bend)) return;
        push({ EventType::Bend, 0, std::clamp(static_cast<double>(bend), -1.0, 1.0), offset });
    }

    /** @brief Mod wheel (CC 1), 0..1. */
    void setModWheel(T value, int offset = 0) noexcept { pushController(EventType::ModWheel, 0, value, offset); }

    /** @brief Channel pressure, 0..1. */
    void setChannelPressure(T value, int offset = 0) noexcept { pushController(EventType::Pressure, 0, value, offset); }

    /** @brief Polyphonic pressure of one key, 0..1. */
    void setPolyPressure(int key, T value, int offset = 0) noexcept
    {
        pushController(EventType::PolyPressure, std::clamp(key, 0, 127), value, offset);
    }

    /** @brief Sustain pedal: while down, released keys keep sounding. */
    void setSustainPedal(bool down, int offset = 0) noexcept
    {
        push({ EventType::Pedal, 0, down ? 1.0 : 0.0, offset });
    }

    /**
     * @brief A MIDI control change, value 0..1: 1 mod wheel, 2 breath,
     *        11 expression (it also scales the amplifier, as MIDI defines),
     *        64 sustain pedal (down at 0.5 and above), 120 and 123 all notes
     *        off. Other controllers are ignored.
     */
    void setController(int controller, T value, int offset = 0) noexcept
    {
        switch (controller)
        {
            case 1:  setModWheel(value, offset); break;
            case 2:  pushController(EventType::Breath, 0, value, offset); break;
            case 11: pushController(EventType::Expression, 0, value, offset); break;
            case 64: if (std::isfinite(value)) setSustainPedal(value >= T(0.5), offset); break;
            case 120:
            case 123: allNotesOff(offset); break;
            default: break;
        }
    }

    /** @brief Releases every sounding note (and lifts the pedal). */
    void allNotesOff(int offset = 0) noexcept { push({ EventType::AllOff, 0, 0.0, offset }); }

    // -- Processing (audio thread) ------------------------------------------------------

    /**
     * @brief Renders the block, overwriting `out`, and applies the queued
     *        events at their offsets. Allocation free, lock free. A block
     *        longer than prepare()'s maxBlockSize is rendered in pieces.
     */
    void processBlock(AudioBufferView<T> out) noexcept
    {
        DenormalGuard guard;
        const int n = out.getNumSamples();
        const int nCh = out.getNumChannels();
        for (int c = 0; c < nCh; ++c)
            std::fill_n(out.getChannel(c), n, T(0));
        if (!prepared_ || n <= 0) { eventCount_ = 0; return; }

        adoptPending();
        snapshotParameters(false);

        int e = 0;
        for (int start = 0; start < n; start += maxBlock_)
        {
            const int len = std::min(maxBlock_, n - start);
            renderPiece(out, start, len, e, start + len >= n);
        }
        eventCount_ = 0;

        retireFadedContent();
        int count = 0;
        for (const auto& v : voices_) count += v.active ? 1 : 0;
        activeVoices_.store(count, std::memory_order_relaxed);
    }

    // -- State (setup/UI thread) ----------------------------------------------------------

    /** @brief Serializes every parameter by id (allocates). Samples and the
     *  user wavetable are data, not parameters: they are not included. */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("SYNT"), 1);
        for (int i = 0; i < kNumParams; ++i)
            w.write(info(i).id, values_[static_cast<size_t>(i)].load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant: missing ids keep
     *  their defaults, unknown ids are ignored; rejects foreign blobs). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("SYNT")) return false;
        for (int i = 0; i < kNumParams; ++i)
            setParameter(static_cast<Param>(i), r.read(info(i).id, info(i).defValue));
        return true;
    }

private:
    // ===================================================================================
    // Types
    // ===================================================================================

    using Reader = detail::SampleReader<T>;
    using ZoneData = detail::SampleZoneData<T>;

    [[nodiscard]] static constexpr const ParamInfo& info(int i) noexcept
    {
        return synthesizer::kParams[static_cast<size_t>(i)];
    }

    /** Data the control thread publishes: sample zones and the user wavetable. */
    struct Content
    {
        std::shared_ptr<const std::vector<ZoneData>> zones;
        std::shared_ptr<const WavetableBank> wavetable;
    };

    /** A value interpolated linearly across one control period: its value at
     *  position `pos` of the period is start + step * pos, whatever the block
     *  boundaries inside the period. */
    struct Ramp
    {
        double start = 0.0, step = 0.0, target = 0.0;
        [[nodiscard]] double at(int pos) const noexcept { return start + step * static_cast<double>(pos); }
        void to(double next, double invPeriod) noexcept
        {
            start = target;
            target = next;
            step = (next - start) * invPeriod;
        }
        void snap(double value) noexcept { start = target = value; step = 0.0; }
        [[nodiscard]] bool live() const noexcept { return start > 0.0 || target > 0.0; }
    };

    struct UnisonVoice
    {
        std::array<Oscillator<T>, 3> osc;
        std::array<bool, 3> oscOn {};
        std::array<Ramp, 3> freq;          ///< Hz: analog oscillators, wavetable 1 and 2.
        std::array<Ramp, 3> master;        ///< Hz: sync masters.
        std::array<double, 2> wtPhase {};
        std::array<WavetableBank::Mip, 2> mip {};
        std::array<double, 4> opPhase {};
        std::array<Ramp, 4> opFreq;        ///< Hz.
        double fb1 = 0.0, fb2 = 0.0;       ///< Operator 4's last two outputs.
        Reader reader;
        bool readerOn = false;
        Ramp sampleRatio;
        double gainL = 1.0, gainR = 1.0;   ///< Unison spread.
        double detune = 1.0;               ///< Unison detune ratio.
        std::array<double, 3> spread {};   ///< Per-oscillator static drift ratio.
        double driftCents = 0.0;
        AnalogRandom::Generator<double> drift { 1 };
    };

    // Indices of the smoothed values in Voice::smooth.
    enum Smoothed : int
    {
        kOsc1Level, kOsc2Level, kOsc3Level, kSubLevel, kNoiseLevel, kSampleLevel,
        kPw1, kPw2, kPw3, kWt1Pos, kWt2Pos, kWt1Level, kWt2Level,
        kOp1Level, kOp2Level, kOp3Level, kOp4Level, kFeedback,
        kCutoff, kResonance, kDrive, kPan, kNumSmoothed
    };

    struct Voice
    {
        bool active = false;
        bool held = false;
        bool sustained = false;
        bool firstTick = true;
        bool stereo = false;
        int key = 60;
        int unison = 1;
        int stealLeft = -1;
        int tickLeft = 0;
        int tickPos = 0;
        Engine engine = Engine::Analog;
        uint64_t age = 0;
        uint64_t seed = 0;
        int64_t ageHi = 0;                 ///< Samples since note-on at the internal rate.
        double velocity = 1.0;
        double tuningCents = 0.0;         ///< Latched per note, independent of key identity.
        double releaseVelocity = 0.0;
        double polyPressure = 0.0;
        double random = 0.0;
        double velGain = 1.0;
        // Pitch in semitones: a glide from glideFrom to key over glideTotal samples.
        double glideFrom = 60.0;
        double glideLeft = 0.0;
        double glideTotal = 0.0;
        double pitch = 60.0;               ///< Glided note pitch at the last tick.
        const Content* content = nullptr;
        const ZoneData* zone = nullptr;
        double zoneGainL = 1.0, zoneGainR = 1.0;
        uint32_t noise = 1;
        ADSREnvelope<double> amp;
        ADSREnvelope<double> env2, env3;   ///< Control rate.
        std::array<ADSREnvelope<double>, 4> opEnv;
        std::array<Lfo<double>, 2> lfo;
        std::array<double, kNumSmoothed> smooth {};   ///< 10 ms smoothed parameters.
        std::array<Ramp, 3> oscLevel;
        Ramp subLevel, noiseLevel, sampleLevel;
        std::array<Ramp, 3> pulseWidth;
        std::array<Ramp, 2> wtPosition, wtLevel;
        std::array<Ramp, 4> opLevel;
        Ramp feedback, g, resonance, svfK, drive, ampGain, panL, panR, subFreq;
        std::array<std::array<T, 4>, 2> ladder {};
        std::array<detail::TptSvfState<T>, 2> svf {};
        Oscillator<T> sub;
        AnalogRandom::Generator<double> cutoffDrift { 1 };
        std::array<UnisonVoice, kMaxUnison> uni;
    };

    /** One voice's share of a render chunk, gathered before its filter runs. */
    static constexpr int kChunk = 128;
    struct Lane
    {
        Voice* voice = nullptr;
        int start = 0;   ///< First high-rate output index.
        int count = 0;   ///< Samples that reach the output.
        std::array<double, kChunk> l {}, r {}, gain {}, panL {}, panR {};
        std::array<T, kChunk> g {}, q {}, drive {};   ///< q: resonance (ladder) or k (SVF).
    };

    enum class EventType { NoteOn, NoteOff, Pedal, Bend, ModWheel, Pressure, PolyPressure, Expression, Breath, AllOff };
    struct Event { EventType type; int key; double value; int offset; double tuningCents = 0.0; };

    struct FreeLfo
    {
        double anchorCycles = 0.0;
        int64_t anchorTime = 0;
        double increment = 0.0;
    };

    // ===================================================================================
    // Parameter access
    // ===================================================================================

    [[nodiscard]] float p(Param param) const noexcept { return params_[static_cast<size_t>(param)]; }
    [[nodiscard]] int ip(Param param) const noexcept { return static_cast<int>(params_[static_cast<size_t>(param)]); }
    [[nodiscard]] static Param at(Param first, int offset) noexcept
    {
        return static_cast<Param>(static_cast<int>(first) + offset);
    }
    // Field offsets inside each repeated group of the table.
    [[nodiscard]] static Param oscParam(int k, int field) noexcept { return at(Param::Osc1Wave, 7 * k + field); }
    [[nodiscard]] static Param opParam(int k, int field) noexcept { return at(Param::Op1Ratio, 11 * k + field); }
    [[nodiscard]] static Param wtParam(int k, int field) noexcept { return at(Param::Wt1Bank, 6 * k + field); }
    [[nodiscard]] static Param envParam(int k, int field) noexcept { return at(Param::Env1Attack, 5 * k + field); }
    [[nodiscard]] static Param lfoParam(int k, int field) noexcept { return at(Param::Lfo1Shape, 8 * k + field); }
    [[nodiscard]] static Param modParam(int k, int field) noexcept { return at(Param::Mod1Source, 4 * k + field); }
    [[nodiscard]] static ModDest dest(ModDest first, int k) noexcept
    {
        return static_cast<ModDest>(static_cast<int>(first) + k);
    }

    [[nodiscard]] int oversamplingFactor() const noexcept
    {
        const int choice = ip(Param::Oversampling);
        return choice == 2 ? 4 : (choice == 1 ? 2 : 1);
    }

    /** Loads every parameter at a block start and applies what changed. */
    void snapshotParameters(bool force) noexcept
    {
        std::array<float, kNumParams> next {};
        for (int i = 0; i < kNumParams; ++i)
            next[static_cast<size_t>(i)] = values_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
        const float tempo = tempo_.load(std::memory_order_relaxed);
        const bool tempoChanged = tempo != tempoApplied_;

        auto changed = [&](Param a, Param b) noexcept {
            if (force) return true;
            for (int i = static_cast<int>(a); i <= static_cast<int>(b); ++i)
                if (next[static_cast<size_t>(i)] != params_[static_cast<size_t>(i)]) return true;
            return false;
        };
        const bool envs = changed(Param::Env1Attack, Param::Env3Curve);
        const bool ops = changed(Param::Op1Ratio, Param::Op4Release);
        const bool lfos = changed(Param::Lfo1Shape, Param::Lfo2Mode) || tempoChanged;
        const bool rate = !force && next[static_cast<size_t>(Param::Oversampling)]
                                    != params_[static_cast<size_t>(Param::Oversampling)];

        // Free LFOs re-anchor where they are now, so a rate change keeps the
        // phase continuous.
        std::array<double, 2> cyclesNow {};
        for (int k = 0; k < 2; ++k) cyclesNow[static_cast<size_t>(k)] = freeCycles(k, timeHi_);
        params_ = next;
        tempoApplied_ = tempo;

        if (rate && prepared_) configureRate(oversamplingFactor());
        if (envs)
            for (auto& v : voices_) applyEnvelopes(v);
        if (ops)
            for (auto& v : voices_) applyOperatorEnvelopes(v);
        if (lfos)
        {
            for (int k = 0; k < 2; ++k)
            {
                auto& f = freeLfo_[static_cast<size_t>(k)];
                f.anchorCycles = cyclesNow[static_cast<size_t>(k)];
                f.anchorTime = timeHi_;
                f.increment = lfoRateHz(k) / internalRate_;
            }
            for (auto& v : voices_)
                for (int k = 0; k < 2; ++k) applyLfo(v, k);
        }
    }

    // ===================================================================================
    // Rate configuration
    // ===================================================================================

    void configureRate(int factor) noexcept
    {
        factor_ = factor;
        internalRate_ = baseRate_ * factor_;
        periodHi_ = kControlPeriod * factor_;
        invPeriod_ = 1.0 / static_cast<double>(periodHi_);
        const int delay = (factor_ == 2) ? os2_.getDecimationDelay()
                        : (factor_ == 4 ? os4_.getDecimationDelay() : 0);
        const int latency = (delay + factor_ - 1) / factor_;
        shift_ = latency * factor_ - delay;
        latency_.store(latency, std::memory_order_relaxed);
        smoothAlpha_ = 1.0 - std::exp(-static_cast<double>(kControlPeriod) / (0.010 * baseRate_));
        stealFade_ = std::max(1, static_cast<int>(std::lround(0.003 * internalRate_)));
        const double controlRate = baseRate_ / kControlPeriod;
        for (auto& v : voices_)
        {
            v.active = false;
            v.amp.prepare(internalRate_);
            v.env2.prepare(controlRate);
            v.env3.prepare(controlRate);
            for (auto& e : v.opEnv) e.prepare(internalRate_);
            for (auto& l : v.lfo) l.prepare(internalRate_);
            v.sub.prepare(internalRate_);
            v.cutoffDrift.setNoiseType(AnalogRandom::NoiseType::Pink);
            v.cutoffDrift.setRange(-1.0, 1.0);
            v.cutoffDrift.setRateHz(1.5);
            v.cutoffDrift.setSmoothing(true, 150.0);
            v.cutoffDrift.prepare(controlRate);
            for (auto& u : v.uni)
            {
                for (auto& o : u.osc) o.prepare(internalRate_);
                u.drift.setNoiseType(AnalogRandom::NoiseType::Pink);
                u.drift.setRange(-1.0, 1.0);
                u.drift.setRateHz(2.0);
                u.drift.setSmoothing(true, 150.0);
                u.drift.prepare(controlRate);
            }
            applyEnvelopes(v);
            applyOperatorEnvelopes(v);
            for (int k = 0; k < 2; ++k) applyLfo(v, k);
        }
        os2_.reset();
        os4_.reset();
        for (int k = 0; k < 2; ++k)
            freeLfo_[static_cast<size_t>(k)].increment = lfoRateHz(k) / internalRate_;
    }

    // ===================================================================================
    // Content publication (as Sampler's zone sets)
    // ===================================================================================

    /** Control thread: offers the current zones and wavetable to the audio thread. */
    void publishContent()
    {
        delete retired_.exchange(nullptr, std::memory_order_acq_rel);
        auto* content = new Content { controlZones_, controlWavetable_ };
        delete pending_.exchange(content, std::memory_order_acq_rel);
    }

    void adoptPending() noexcept
    {
        if (fading_ != nullptr) return;   // one handover at a time
        Content* next = pending_.exchange(nullptr, std::memory_order_acq_rel);
        if (next == nullptr) return;
        fading_ = active_;
        active_ = next;
    }

    void retireFadedContent() noexcept
    {
        if (fading_ == nullptr) return;
        for (const auto& v : voices_)
            if (v.active && v.content == fading_) return;
        if (retired_.load(std::memory_order_acquire) != nullptr) return;
        retired_.store(fading_, std::memory_order_release);
        fading_ = nullptr;
    }

    // ===================================================================================
    // Events
    // ===================================================================================

    void push(const Event& e) noexcept
    {
        if (!prepared_ || eventCount_ >= kMaxEvents) return;
        // Ordered by offset; equal offsets keep their call order.
        int i = eventCount_++;
        while (i > 0 && events_[static_cast<size_t>(i - 1)].offset > e.offset)
        {
            events_[static_cast<size_t>(i)] = events_[static_cast<size_t>(i - 1)];
            --i;
        }
        events_[static_cast<size_t>(i)] = e;
    }

    void pushController(EventType type, int key, T value, int offset) noexcept
    {
        if (!std::isfinite(value)) return;
        push({ type, key, std::clamp(static_cast<double>(value), 0.0, 1.0), offset });
    }

    void apply(const Event& e) noexcept
    {
        switch (e.type)
        {
            case EventType::NoteOn:     noteOnNow(e.key, e.value, e.tuningCents); break;
            case EventType::NoteOff:    noteOffNow(e.key, e.value); break;
            case EventType::Bend:       bend_ = e.value; break;
            case EventType::ModWheel:   modWheel_ = e.value; break;
            case EventType::Pressure:   pressure_ = e.value; break;
            case EventType::Expression: expression_ = e.value; break;
            case EventType::Breath:     breath_ = e.value; break;
            case EventType::PolyPressure:
                for (auto& v : voices_)
                    if (v.active && v.key == e.key && v.stealLeft < 0) v.polyPressure = e.value;
                break;
            case EventType::Pedal:
                sustainPedal_ = e.value > 0.5;
                if (!sustainPedal_)
                    for (auto& v : voices_)
                        if (v.active && v.sustained && !v.held) release(v);
                break;
            case EventType::AllOff:
                sustainPedal_ = false;
                heldCount_ = 0;
                for (auto& v : voices_)
                    if (v.active && v.stealLeft < 0) release(v);
                break;
        }
    }

    [[nodiscard]] bool monophonic() const noexcept
    {
        return static_cast<VoiceMode>(ip(Param::VoiceMode)) != VoiceMode::Poly;
    }

    void noteOnNow(int key, double velocity, double tuningCents) noexcept
    {
        const bool anyHeldBefore = heldCount_ > 0;
        pushHeld(key);
        heldTuning_[static_cast<size_t>(key)] = tuningCents;
        if (monophonic())
        {
            for (auto& v : voices_)
                if (v.active && v.stealLeft < 0)
                {
                    const bool legato = anyHeldBefore && v.held;
                    const bool retrigger = !(legato && static_cast<VoiceMode>(ip(Param::VoiceMode)) == VoiceMode::Legato);
                    moveMonoVoice(v, key, velocity, tuningCents, retrigger, legato);
                    return;
                }
        }
        else
        {
            // A retriggered key releases what it was playing.
            for (auto& v : voices_)
                if (v.active && v.key == key && v.stealLeft < 0 && (v.held || v.sustained))
                    release(v);
        }
        if (Voice* v = freeVoice()) startVoice(*v, key, velocity, tuningCents, anyHeldBefore);
    }

    void noteOffNow(int key, double releaseVelocity) noexcept
    {
        removeHeld(key);
        for (auto& v : voices_)
        {
            if (!v.active || !v.held || v.key != key || v.stealLeft >= 0) continue;
            if (monophonic() && heldCount_ > 0)
            {
                // Back to the most recent key still held: no retrigger.
                const int nextKey = held_[static_cast<size_t>(heldCount_ - 1)];
                moveMonoVoice(v, nextKey, v.velocity, heldTuning_[static_cast<size_t>(nextKey)], false, true);
                continue;
            }
            v.releaseVelocity = releaseVelocity;
            if (sustainPedal_) { v.held = false; v.sustained = true; }
            else release(v);
        }
    }

    void pushHeld(int key) noexcept
    {
        removeHeld(key);
        if (heldCount_ < 128) held_[static_cast<size_t>(heldCount_++)] = key;
    }

    void removeHeld(int key) noexcept
    {
        int w = 0;
        for (int i = 0; i < heldCount_; ++i)
            if (held_[static_cast<size_t>(i)] != key) held_[static_cast<size_t>(w++)] = held_[static_cast<size_t>(i)];
        heldCount_ = w;
    }

    void release(Voice& v) noexcept
    {
        v.held = false;
        v.sustained = false;
        v.amp.noteOff();
        v.env2.noteOff();
        v.env3.noteOff();
        for (auto& e : v.opEnv) e.noteOff();
        for (int u = 0; u < v.unison; ++u) v.uni[static_cast<size_t>(u)].reader.release();
    }

    /** Mono and legato: the sounding voice moves to a new key. */
    void moveMonoVoice(Voice& v, int key, double velocity, double tuningCents, bool retrigger, bool legato) noexcept
    {
        const double glideMs = static_cast<double>(p(Param::Glide));
        const bool glide = glideMs > 0.0
            && (static_cast<GlideMode>(ip(Param::GlideMode)) == GlideMode::Always || legato);
        const double tunedPitch = static_cast<double>(key) + tuningCents / 100.0;
        v.glideFrom = glide ? v.pitch : tunedPitch;
        v.glideTotal = glide ? glideMs * 0.001 * internalRate_ : 0.0;
        v.glideLeft = v.glideTotal;
        v.key = key;
        v.tuningCents = tuningCents;
        v.held = true;
        v.sustained = false;
        lastPitch_ = tunedPitch;
        haveLastPitch_ = true;
        if (!retrigger) return;
        v.velocity = velocity;
        v.releaseVelocity = 0.0;
        v.velGain = velocityGain(velocity);
        v.amp.noteOn();
        v.env2.noteOn();
        v.env3.noteOn();
        for (auto& e : v.opEnv) e.noteOn();
        for (int k = 0; k < 2; ++k)
            if (static_cast<LfoMode>(ip(lfoParam(k, 7))) == LfoMode::Retrigger)
                v.lfo[static_cast<size_t>(k)].reset();
    }

    [[nodiscard]] double velocityGain(double velocity) const noexcept
    {
        return std::pow(10.0, static_cast<double>(p(Param::AmpVelocity)) * (velocity - 1.0) / 20.0);
    }

    /** A voice for a new note: a free one within the polyphony, else a
     *  stolen one handed to a spare slot to fade out (Sampler's policy). */
    Voice* freeVoice() noexcept
    {
        const int polyphony = monophonic() ? 1 : std::min(ip(Param::Polyphony), maxVoices_);
        int sounding = 0;
        Voice* idle = nullptr;
        for (auto& v : voices_)
        {
            if (!v.active) { if (idle == nullptr) idle = &v; }
            else if (v.stealLeft < 0) ++sounding;
        }
        if (sounding < polyphony && idle != nullptr) return idle;

        // Steal: the oldest released voice, else the quietest, else the oldest.
        Voice* victim = nullptr;
        for (auto& v : voices_)
        {
            if (!v.active || v.stealLeft >= 0) continue;
            if (victim == nullptr) { victim = &v; continue; }
            const bool vr = !v.held && !v.sustained, cr = !victim->held && !victim->sustained;
            if (vr != cr) { if (vr) victim = &v; continue; }
            const double va = v.amp.getCurrentValue(), ca = victim->amp.getCurrentValue();
            if (va < ca - 1e-9 || (std::abs(va - ca) <= 1e-9 && v.age < victim->age)) victim = &v;
        }
        if (victim != nullptr) victim->stealLeft = stealFade_;
        if (idle != nullptr) return idle;
        // No spare slot left: the oldest stealing fade is cut short.
        Voice* oldest = nullptr;
        for (auto& v : voices_)
            if (v.active && v.stealLeft >= 0 && (oldest == nullptr || v.age < oldest->age)) oldest = &v;
        if (oldest != nullptr) oldest->active = false;
        return oldest;
    }

    // ===================================================================================
    // Voice start
    // ===================================================================================

    [[nodiscard]] static uint64_t mix64(uint64_t x) noexcept
    {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    /** Uniform value in [-1, 1) from a voice seed and a stream index. */
    [[nodiscard]] static double voiceRandom(uint64_t seed, int stream) noexcept
    {
        return Lfo<double>::random(seed, stream);
    }

    void startVoice(Voice& v, int key, double velocity, double tuningCents, bool anyHeldBefore) noexcept
    {
        v.active = true;
        v.held = true;
        v.sustained = false;
        v.firstTick = true;
        v.stealLeft = -1;
        v.key = key;
        v.tuningCents = tuningCents;
        v.velocity = velocity;
        v.releaseVelocity = 0.0;
        v.polyPressure = 0.0;
        v.velGain = velocityGain(velocity);
        v.age = ++clock_;
        v.seed = mix64(static_cast<uint64_t>(ip(Param::Seed)) * 0x100000001B3ull + noteCounter_++);
        v.random = voiceRandom(v.seed, 0);
        v.engine = static_cast<Engine>(ip(Param::Engine));
        v.unison = std::clamp(ip(Param::Unison), 1, kMaxUnison);
        v.content = active_;
        v.ageHi = 0;
        v.noise = static_cast<uint32_t>(mix64(v.seed ^ 0x6E6F697365ull)) | 1u;

        // Glide from the previous note.
        const double glideMs = static_cast<double>(p(Param::Glide));
        const bool glide = glideMs > 0.0 && haveLastPitch_
            && (static_cast<GlideMode>(ip(Param::GlideMode)) == GlideMode::Always || anyHeldBefore);
        const double tunedPitch = static_cast<double>(key) + tuningCents / 100.0;
        v.glideFrom = glide ? lastPitch_ : tunedPitch;
        v.glideTotal = glide ? glideMs * 0.001 * internalRate_ : 0.0;
        v.glideLeft = v.glideTotal;
        v.pitch = v.glideFrom;
        lastPitch_ = tunedPitch;
        haveLastPitch_ = true;

        // Sample zone: the first one answering the key and velocity.
        v.zone = nullptr;
        if (v.content != nullptr && v.content->zones != nullptr)
            for (const auto& z : *v.content->zones)
                if (key >= z.keyLow && key <= z.keyHigh && velocity >= z.velLow && velocity <= z.velHigh)
                {
                    v.zone = &z;
                    break;
                }
        if (v.zone != nullptr)
        {
            double mono = 1.0;
            detail::sampleZonePanGains(v.zone->pan, v.zone->channels, v.zone->gain,
                                       v.zoneGainL, v.zoneGainR, mono);
        }

        // Unison layout: symmetric detune, spread across the stereo field,
        // power normalised.
        const int U = v.unison;
        const double spread = static_cast<double>(p(Param::UnisonSpread));
        const double detuneCents = static_cast<double>(p(Param::UnisonDetune));
        const double drift = static_cast<double>(p(Param::Drift));
        const double norm = 1.0 / std::sqrt(static_cast<double>(U));
        v.stereo = (U > 1 && spread > 0.0)
                || (v.zone != nullptr && v.zone->channels > 1 && p(Param::SampleLevel) > 0.0f);
        for (int u = 0; u < U; ++u)
        {
            auto& uv = v.uni[static_cast<size_t>(u)];
            const double x = (U > 1) ? 2.0 * static_cast<double>(u) / static_cast<double>(U - 1) - 1.0 : 0.0;
            uv.detune = std::exp2(0.5 * x * detuneCents / 1200.0);
            // cos on both sides (sin(a) = cos(pi/2 - a)): mirrored positions
            // get exactly mirrored gains, and the centre is exactly mono.
            uv.gainL = norm * std::cos(0.25 * pi<double> * (1.0 + x * spread)) * sqrt2<double>;
            uv.gainR = norm * std::cos(0.25 * pi<double> * (1.0 - x * spread)) * sqrt2<double>;
            for (int k = 0; k < 3; ++k)
                uv.spread[static_cast<size_t>(k)] =
                    std::exp2(drift * 3.0 * voiceRandom(v.seed, 10 + 3 * u + k) / 1200.0);
            uv.driftCents = 0.0;
            uv.drift.reseed(mix64(v.seed + 101u + static_cast<uint64_t>(u)) | 1u);
            uv.fb1 = uv.fb2 = 0.0;
            uv.readerOn = false;
            uv.oscOn = {};
        }
        v.cutoffDrift.reseed(mix64(v.seed + 77u) | 1u);

        // Envelopes and LFOs.
        applyEnvelopes(v);
        applyOperatorEnvelopes(v);
        v.amp.reset();
        v.amp.noteOn();
        v.env2.reset();
        v.env2.noteOn();
        v.env3.reset();
        v.env3.noteOn();
        for (auto& e : v.opEnv) { e.reset(); e.noteOn(); }
        for (int k = 0; k < 2; ++k)
        {
            applyLfo(v, k);
            auto& l = v.lfo[static_cast<size_t>(k)];
            l.setSeed(mix64(v.seed + 200u + static_cast<uint64_t>(k)));
            l.reset();
        }
        v.ladder = {};
        v.svf = {};

        // The first control tick sets every frequency and snaps every ramp;
        // the oscillators are phased after it, at their real frequencies.
        controlTick(v);
        const bool randomPhase = static_cast<PhaseMode>(ip(Param::PhaseMode)) == PhaseMode::Random;
        for (int u = 0; u < U; ++u)
        {
            auto& uv = v.uni[static_cast<size_t>(u)];
            for (int k = 0; k < 3; ++k)
            {
                const double ph = randomPhase ? 0.5 + 0.5 * voiceRandom(v.seed, 40 + 3 * u + k) : 0.0;
                uv.osc[static_cast<size_t>(k)].setPhase(static_cast<T>(ph));
            }
            for (int k = 0; k < 2; ++k)
                uv.wtPhase[static_cast<size_t>(k)] = randomPhase ? 0.5 + 0.5 * voiceRandom(v.seed, 70 + 2 * u + k) : 0.0;
            for (int k = 0; k < 4; ++k)
                uv.opPhase[static_cast<size_t>(k)] = randomPhase ? 0.5 + 0.5 * voiceRandom(v.seed, 90 + 4 * u + k) : 0.0;
            if (v.zone != nullptr)
            {
                uv.reader.start(*v.zone, uv.sampleRatio.target);
                uv.readerOn = true;
            }
        }
        v.sub.reset();
        v.tickLeft = periodHi_;
        v.tickPos = 0;
    }

    void applyEnvelopes(Voice& v) noexcept
    {
        auto set = [&](ADSREnvelope<double>& e, int k) noexcept {
            e.setParameters(p(envParam(k, 0)), p(envParam(k, 1)), p(envParam(k, 2)), p(envParam(k, 3)));
            e.setCurvature(p(envParam(k, 4)));
        };
        set(v.amp, 0);
        set(v.env2, 1);
        set(v.env3, 2);
    }

    void applyOperatorEnvelopes(Voice& v) noexcept
    {
        for (int k = 0; k < 4; ++k)
            v.opEnv[static_cast<size_t>(k)].setParameters(p(opParam(k, 7)), p(opParam(k, 8)),
                                                          p(opParam(k, 9)), p(opParam(k, 10)));
    }

    [[nodiscard]] double lfoRateHz(int k) const noexcept
    {
        if (ip(lfoParam(k, 2)) != 0)
        {
            const auto d = static_cast<typename Lfo<double>::Division>(ip(lfoParam(k, 3)));
            return (static_cast<double>(tempoApplied_) / 60.0) / Lfo<double>::beatsPerCycle(d);
        }
        return static_cast<double>(p(lfoParam(k, 1)));
    }

    void applyLfo(Voice& v, int k) noexcept
    {
        auto& l = v.lfo[static_cast<size_t>(k)];
        l.setShape(static_cast<typename Lfo<double>::Shape>(ip(lfoParam(k, 0))));
        l.setRateHz(static_cast<double>(p(lfoParam(k, 1))));
        l.setTempo(static_cast<double>(tempoApplied_));
        l.setDivision(static_cast<typename Lfo<double>::Division>(ip(lfoParam(k, 3))));
        l.setTempoSync(ip(lfoParam(k, 2)) != 0);
        l.setDelayMs(static_cast<double>(p(lfoParam(k, 4))));
        l.setFadeInMs(static_cast<double>(p(lfoParam(k, 5))));
        l.setStartPhase(static_cast<double>(p(lfoParam(k, 6))));
    }

    [[nodiscard]] double freeCycles(int k, int64_t t) const noexcept
    {
        const auto& f = freeLfo_[static_cast<size_t>(k)];
        return f.anchorCycles + f.increment * static_cast<double>(t - f.anchorTime);
    }

    // ===================================================================================
    // Control tick: modulation and the next period's targets
    // ===================================================================================

    [[nodiscard]] static double noteHz(double semitones) noexcept
    {
        return 440.0 * std::exp2((semitones - 69.0) / 12.0);
    }

    void controlTick(Voice& v) noexcept
    {
        const bool first = v.firstTick;
        v.firstTick = false;
        v.tickPos = 0;
        const int64_t now = timeHi_ + offsetHi_;

        // -- Glide: constant time, linear in semitones.
        const double keyPitch = static_cast<double>(v.key);
        const double tunedPitch = keyPitch + v.tuningCents / 100.0;
        if (v.glideLeft > 0.0)
        {
            v.pitch = tunedPitch + (v.glideFrom - tunedPitch) * (v.glideLeft / std::max(v.glideTotal, 1.0));
            v.glideLeft = std::max(0.0, v.glideLeft - static_cast<double>(periodHi_));
        }
        else
        {
            v.pitch = tunedPitch;
        }

        // -- Sources.
        std::array<double, synthesizer::kNumModSources> src {};
        auto source = [&](ModSource s) noexcept -> double& { return src[static_cast<size_t>(s)]; };
        source(ModSource::Velocity) = v.velocity;
        source(ModSource::ReleaseVelocity) = v.releaseVelocity;
        source(ModSource::Key) = (keyPitch - 60.0) / 60.0;
        source(ModSource::ModWheel) = modWheel_;
        source(ModSource::Pressure) = std::max(pressure_, v.polyPressure);
        source(ModSource::PitchBend) = bend_;
        source(ModSource::Expression) = expression_;
        source(ModSource::Breath) = breath_;
        source(ModSource::Env1) = v.amp.getCurrentValue();
        source(ModSource::Env2) = v.env2.getNextValue();
        source(ModSource::Env3) = v.env3.getNextValue();
        source(ModSource::Random) = v.random;

        // -- Matrix: the LFO rate destinations first, since they shape the
        //    LFOs' advance, then everything else with the LFOs as sources.
        std::array<double, synthesizer::kNumModDests> dst {};
        auto runMatrix = [&](bool rateOnly) noexcept {
            for (int s = 0; s < kModSlots; ++s)
            {
                const auto d = static_cast<ModDest>(ip(modParam(s, 2)));
                const auto a = static_cast<ModSource>(ip(modParam(s, 0)));
                const double amount = static_cast<double>(p(modParam(s, 3)));
                if (d == ModDest::None || a == ModSource::None || amount == 0.0) continue;
                const bool isRate = d == ModDest::Lfo1Rate || d == ModDest::Lfo2Rate;
                if (isRate != rateOnly) continue;
                const auto via = static_cast<ModSource>(ip(modParam(s, 1)));
                const double scale = (via == ModSource::None) ? 1.0 : src[static_cast<size_t>(via)];
                dst[static_cast<size_t>(d)] += src[static_cast<size_t>(a)] * scale * amount;
            }
        };
        auto mod = [&](ModDest d) noexcept { return dst[static_cast<size_t>(d)]; };
        runMatrix(true);
        for (int k = 0; k < 2; ++k)
        {
            auto& l = v.lfo[static_cast<size_t>(k)];
            double value;
            if (static_cast<LfoMode>(ip(lfoParam(k, 7))) == LfoMode::Retrigger)
            {
                value = l.advanceScaled(periodHi_, std::exp2(mod(k == 0 ? ModDest::Lfo1Rate : ModDest::Lfo2Rate)));
            }
            else
            {
                const auto shape = static_cast<typename Lfo<double>::Shape>(ip(lfoParam(k, 0)));
                const uint64_t seed = mix64(static_cast<uint64_t>(ip(Param::Seed)) + 300u + static_cast<uint64_t>(k));
                value = Lfo<double>::valueAt(shape, freeCycles(k, now), seed) * l.depthAt(v.ageHi);
            }
            source(k == 0 ? ModSource::Lfo1 : ModSource::Lfo2) = value;
        }
        runMatrix(false);

        // -- Smoothed base values (10 ms one-pole per control period).
        std::array<double, kNumSmoothed> base {};
        for (int k = 0; k < 3; ++k)
        {
            base[static_cast<size_t>(kOsc1Level + k)] = p(oscParam(k, 1));
            base[static_cast<size_t>(kPw1 + k)] = p(oscParam(k, 5));
        }
        base[kSubLevel] = p(Param::SubLevel);
        base[kNoiseLevel] = p(Param::NoiseLevel);
        base[kSampleLevel] = p(Param::SampleLevel);
        base[kWt1Pos] = p(wtParam(0, 1));
        base[kWt2Pos] = p(wtParam(1, 1));
        base[kWt1Level] = p(wtParam(0, 2));
        base[kWt2Level] = p(wtParam(1, 2));
        for (int k = 0; k < 4; ++k) base[static_cast<size_t>(kOp1Level + k)] = p(opParam(k, 4));
        base[kFeedback] = p(Param::FmFeedback);
        base[kCutoff] = std::log2(static_cast<double>(p(Param::FilterCutoff)));
        base[kResonance] = p(Param::FilterResonance);
        base[kDrive] = p(Param::FilterDrive);
        base[kPan] = p(Param::AmpPan);
        for (int i = 0; i < kNumSmoothed; ++i)
        {
            double& s = v.smooth[static_cast<size_t>(i)];
            s = first ? base[static_cast<size_t>(i)] : s + (base[static_cast<size_t>(i)] - s) * smoothAlpha_;
        }
        auto sm = [&](int i) noexcept { return v.smooth[static_cast<size_t>(i)]; };
        auto setRamp = [&](Ramp& r, double value) noexcept {
            if (first) r.snap(value);
            else r.to(value, invPeriod_);
        };
        auto unit = [](double x) noexcept { return std::clamp(x, 0.0, 1.0); };

        // -- Pitches.
        const double bendSt = bend_ * static_cast<double>(p(Param::BendRange));
        const double notePitch = v.pitch + bendSt + static_cast<double>(p(Param::MasterTune)) / 100.0
                               + mod(ModDest::Pitch);
        const double baseHz = noteHz(notePitch);
        const double limitHz = 0.499 * internalRate_;
        auto clampHz = [&](double hz) noexcept { return std::clamp(hz, 0.0, limitHz); };

        std::array<double, 3> oscRatio {};
        for (int k = 0; k < 3; ++k)
            oscRatio[static_cast<size_t>(k)] = std::exp2((12.0 * p(oscParam(k, 2)) + p(oscParam(k, 3))
                + p(oscParam(k, 4)) / 100.0 + mod(dest(ModDest::Osc1Pitch, k))) / 12.0);
        std::array<double, 2> wtRatio {};
        for (int k = 0; k < 2; ++k)
            wtRatio[static_cast<size_t>(k)] = std::exp2((12.0 * p(wtParam(k, 3)) + p(wtParam(k, 4))
                + p(wtParam(k, 5)) / 100.0 + mod(dest(ModDest::Osc1Pitch, k))) / 12.0);
        std::array<double, 4> opHz {};
        std::array<bool, 4> opFixed {};
        for (int k = 0; k < 4; ++k)
        {
            opFixed[static_cast<size_t>(k)] = ip(opParam(k, 2)) != 0;
            opHz[static_cast<size_t>(k)] = opFixed[static_cast<size_t>(k)]
                ? static_cast<double>(p(opParam(k, 3)))
                : static_cast<double>(p(opParam(k, 0))) * std::exp2(static_cast<double>(p(opParam(k, 1))) / 1200.0);
        }

        const double driftAmount = static_cast<double>(p(Param::Drift));
        constexpr double centRatio = std::numbers::ln2 / 1200.0;   // 2^(c/1200) ~ 1 + c ln2 / 1200
        for (int u = 0; u < v.unison; ++u)
        {
            auto& uv = v.uni[static_cast<size_t>(u)];
            uv.driftCents = driftAmount > 0.0 ? 6.0 * driftAmount * uv.drift.getNextSample() : 0.0;
            const double unisonHz = baseHz * uv.detune * (1.0 + uv.driftCents * centRatio);
            switch (v.engine)
            {
                case Engine::Analog:
                    for (int k = 0; k < 3; ++k)
                    {
                        const double masterHz = unisonHz * uv.spread[static_cast<size_t>(k)];
                        setRamp(uv.freq[static_cast<size_t>(k)], clampHz(masterHz * oscRatio[static_cast<size_t>(k)]));
                        setRamp(uv.master[static_cast<size_t>(k)], clampHz(masterHz));
                    }
                    break;
                case Engine::Wavetable:
                    for (int k = 0; k < 2; ++k)
                        setRamp(uv.freq[static_cast<size_t>(k)],
                                clampHz(unisonHz * uv.spread[static_cast<size_t>(k)] * wtRatio[static_cast<size_t>(k)]));
                    break;
                case Engine::Fm:
                    for (int k = 0; k < 4; ++k)
                        setRamp(uv.opFreq[static_cast<size_t>(k)],
                                clampHz(opFixed[static_cast<size_t>(k)] ? opHz[static_cast<size_t>(k)]
                                                                        : unisonHz * opHz[static_cast<size_t>(k)]));
                    break;
            }
            if (v.zone != nullptr)
            {
                const auto& z = *v.zone;
                const double offset = 12.0 * p(Param::SampleOctave) + p(Param::SampleSemitone)
                                    + p(Param::SampleFine) / 100.0;
                const double played = (ip(Param::SampleKeyTrack) != 0)
                    ? notePitch : static_cast<double>(z.rootKey) + bendSt + mod(ModDest::Pitch);
                const double st = played + offset - static_cast<double>(z.rootKey) + z.tuneCents / 100.0;
                const double ratio = z.sampleRate / internalRate_ * std::exp2(st / 12.0) * uv.detune
                                   * (1.0 + uv.driftCents * centRatio);
                setRamp(uv.sampleRatio, std::clamp(ratio, 1.0 / Reader::kMaxRate, Reader::kMaxRate));
            }
        }
        setRamp(v.subFreq, clampHz(baseHz * (ip(Param::SubOctave) == 1 ? 0.25 : 0.5)));

        // -- Levels and timbre.
        for (int k = 0; k < 3; ++k)
        {
            setRamp(v.oscLevel[static_cast<size_t>(k)], unit(sm(kOsc1Level + k) + mod(dest(ModDest::Osc1Level, k))));
            setRamp(v.pulseWidth[static_cast<size_t>(k)], std::clamp(sm(kPw1 + k) + mod(ModDest::PulseWidth), 0.01, 0.99));
        }
        setRamp(v.subLevel, unit(sm(kSubLevel) + mod(ModDest::SubLevel)));
        setRamp(v.noiseLevel, unit(sm(kNoiseLevel) + mod(ModDest::NoiseLevel)));
        setRamp(v.sampleLevel, unit(sm(kSampleLevel) + mod(ModDest::SampleLevel)));
        for (int k = 0; k < 2; ++k)
        {
            setRamp(v.wtPosition[static_cast<size_t>(k)],
                    unit(sm(kWt1Pos + k) + mod(ModDest::WtPosition) + mod(dest(ModDest::Wt1Position, k))));
            setRamp(v.wtLevel[static_cast<size_t>(k)], unit(sm(kWt1Level + k) + mod(dest(ModDest::Osc1Level, k))));
        }
        const double keyOctaves = (keyPitch - 60.0) / 12.0;
        for (int k = 0; k < 4; ++k)
        {
            const double velSens = static_cast<double>(p(opParam(k, 5)));
            const double keyDb = 6.0 * static_cast<double>(p(opParam(k, 6))) * keyOctaves;
            const double level = unit(sm(kOp1Level + k) + mod(dest(ModDest::Op1Level, k)))
                               * (1.0 - velSens * (1.0 - v.velocity)) * std::pow(10.0, keyDb / 20.0);
            setRamp(v.opLevel[static_cast<size_t>(k)], level);
        }
        setRamp(v.feedback, unit(sm(kFeedback) + mod(ModDest::FmFeedback)));

        // -- Wavetable mip levels for this period's highest frequency.
        if (v.engine == Engine::Wavetable)
            for (int u = 0; u < v.unison; ++u)
                for (int k = 0; k < 2; ++k)
                {
                    auto& uv = v.uni[static_cast<size_t>(u)];
                    const auto& r = uv.freq[static_cast<size_t>(k)];
                    uv.mip[static_cast<size_t>(k)] = bankFor(v, k).selectMip(std::max(r.start, r.target), internalRate_);
                }

        // -- Filter.
        const double cutDrift = driftAmount > 0.0 ? 0.15 * driftAmount * v.cutoffDrift.getNextSample() : 0.0;
        const double envScale = 1.0 - static_cast<double>(p(Param::FilterVelocity)) * (1.0 - v.velocity);
        const double octaves = sm(kCutoff)
                             + static_cast<double>(p(Param::FilterKeyTrack)) * keyOctaves
                             + static_cast<double>(p(Param::FilterEnvAmount)) * envScale * source(ModSource::Env2)
                             + mod(ModDest::Cutoff) + cutDrift;
        const double cutoff = std::clamp(std::exp2(octaves), 10.0, std::min(0.49 * internalRate_, 40000.0));
        setRamp(v.g, std::tan(pi<double> * cutoff / internalRate_));
        const double resonance = unit(sm(kResonance) + mod(ModDest::Resonance));
        setRamp(v.resonance, resonance);
        setRamp(v.svfK, 2.0 / std::pow(100.0, resonance));   // Q from 0.5 to 50
        setRamp(v.drive, std::max(sm(kDrive), 1.01));

        // -- Amplifier and pan (MIDI expression scales the level).
        setRamp(v.ampGain, v.velGain * std::max(0.0, 1.0 + mod(ModDest::Amp)) * expression_);
        const double pan = std::clamp(sm(kPan) + mod(ModDest::Pan), -1.0, 1.0);
        if (v.stereo)
        {
            setRamp(v.panL, std::min(1.0, 1.0 - pan));
            setRamp(v.panR, std::min(1.0, 1.0 + pan));
        }
        else
        {
            // Constant power, symmetric as the unison spread: centred is mono.
            setRamp(v.panL, std::cos(0.25 * pi<double> * (1.0 + pan)) * sqrt2<double>);
            setRamp(v.panR, std::cos(0.25 * pi<double> * (1.0 - pan)) * sqrt2<double>);
        }

        // -- Oscillators at the period start (analog engine) and the sub.
        if (v.engine == Engine::Analog)
            for (int u = 0; u < v.unison; ++u)
                for (int k = 0; k < 3; ++k)
                {
                    auto& uv = v.uni[static_cast<size_t>(u)];
                    auto& o = uv.osc[static_cast<size_t>(k)];
                    const bool on = v.oscLevel[static_cast<size_t>(k)].live();
                    o.setWaveform(waveOf(static_cast<OscWave>(ip(oscParam(k, 0)))));
                    o.setPulseWidth(static_cast<T>(v.pulseWidth[static_cast<size_t>(k)].start));
                    if (ip(oscParam(k, 6)) != 0)
                    {
                        const double m = uv.master[static_cast<size_t>(k)].start;
                        o.setFrequency(static_cast<T>(m));
                        o.setSyncRatio(static_cast<T>(m > 0.0 ? uv.freq[static_cast<size_t>(k)].start / m : 0.0));
                    }
                    else
                    {
                        o.setSyncRatio(T(0));
                        o.setFrequency(static_cast<T>(uv.freq[static_cast<size_t>(k)].start));
                    }
                    if (on && !uv.oscOn[static_cast<size_t>(k)] && !first)
                        o.setPhase(o.getPhase());   // restart cleanly after a silent stretch
                    uv.oscOn[static_cast<size_t>(k)] = on;
                }
        v.sub.setWaveform(subWaveOf(static_cast<SubWave>(ip(Param::SubWave))));
        v.sub.setFrequency(static_cast<T>(v.subFreq.start));
    }

    [[nodiscard]] static typename Oscillator<T>::Waveform waveOf(OscWave w) noexcept
    {
        using W = typename Oscillator<T>::Waveform;
        switch (w)
        {
            case OscWave::Sine:     return W::Sine;
            case OscWave::Triangle: return W::Triangle;
            case OscWave::Saw:      return W::Saw;
            case OscWave::Pulse:    return W::Square;
        }
        return W::Saw;
    }

    [[nodiscard]] static typename Oscillator<T>::Waveform subWaveOf(SubWave w) noexcept
    {
        using W = typename Oscillator<T>::Waveform;
        switch (w)
        {
            case SubWave::Square:   return W::Square;
            case SubWave::Triangle: return W::Triangle;
            case SubWave::Sine:     return W::Sine;
        }
        return W::Square;
    }

    [[nodiscard]] const WavetableBank& bankFor(const Voice& v, int k) const noexcept
    {
        const int b = ip(wtParam(k, 0));
        if (b >= WavetableBank::kNumBuiltins)
        {
            if (v.content != nullptr && v.content->wavetable != nullptr) return *v.content->wavetable;
            return WavetableBank::builtin(WavetableBank::Builtin::Basic);
        }
        return WavetableBank::builtin(static_cast<WavetableBank::Builtin>(b));
    }

    // ===================================================================================
    // Rendering
    // ===================================================================================

    void renderPiece(AudioBufferView<T> out, int start, int len, int& e, bool lastPiece) noexcept
    {
        const int lenHi = len * factor_;
        T* hiL = mix_[0].data();
        T* hiR = mix_[1].data();
        if (factor_ > 1)
        {
            const AudioBufferView<T> hi = (factor_ == 2) ? os2_.getHighRateBlock(len) : os4_.getHighRateBlock(len);
            hiL = hi.getChannel(0);
            hiR = hi.getChannel(1);
        }
        std::fill_n(hiL, lenHi, T(0));
        std::fill_n(hiR, lenHi, T(0));

        int cursor = 0;
        while (e < eventCount_)
        {
            const Event& ev = events_[static_cast<size_t>(e)];
            const int offset = std::max(ev.offset, 0);
            if (!lastPiece && offset >= start + len) break;
            const int local = std::clamp(offset - start, 0, len);
            const int atHi = (local >= len) ? lenHi : local * factor_ + shift_;
            if (atHi > cursor) { renderVoices(hiL, hiR, cursor, atHi); cursor = atHi; }
            offsetHi_ = cursor;
            apply(ev);
            ++e;
        }
        if (cursor < lenHi) renderVoices(hiL, hiR, cursor, lenHi);
        timeHi_ += lenHi;
        offsetHi_ = 0;

        T* baseL = mix_[0].data();
        T* baseR = mix_[1].data();
        if (factor_ > 1)
        {
            T* ch[2] = { baseL, baseR };
            const AudioBufferView<T> base(ch, 2, len);
            if (factor_ == 2) os2_.downsample(base); else os4_.downsample(base);
        }

        // Master volume, smoothed (10 ms) per output sample.
        const double targetGain = decibelsToGain(static_cast<double>(p(Param::MasterVolume)), -200.0);
        const double a = 1.0 - std::exp(-1.0 / (0.010 * baseRate_));
        const int nCh = out.getNumChannels();
        T* outL = nCh > 0 ? out.getChannel(0) : nullptr;
        T* outR = nCh > 1 ? out.getChannel(1) : nullptr;
        for (int i = 0; i < len; ++i)
        {
            masterGain_ += (targetGain - masterGain_) * a;
            const auto g = static_cast<T>(masterGain_);
            if (outR != nullptr)
            {
                outL[start + i] = baseL[i] * g;
                outR[start + i] = baseR[i] * g;
            }
            else if (outL != nullptr)
            {
                outL[start + i] = T(0.5) * (baseL[i] + baseR[i]) * g;
            }
        }
    }

    // Voices render in chunks, two at a time. Each lane first gathers its
    // voice's sources and the per-sample values of its filter and amplifier;
    // then both filter recursions run in one loop. A ZDF ladder or SVF sample
    // is a chain of dependent operations, so a second independent chain in
    // flight nearly halves the filter's cost. The arithmetic per voice is
    // unchanged and every output sample still sums the voices in index order,
    // so the output is identical to rendering the voices one after another.
    // Without a filter there is no recursion to interleave: each voice
    // renders its whole stretch and adds itself to the output as it goes.

    void renderVoices(T* hiL, T* hiR, int from, int to) noexcept
    {
        const auto type = static_cast<FilterType>(ip(Param::FilterType));
        if (type == FilterType::Off)
        {
            for (auto& v : voices_)
                if (v.active) gather(v, lanes_[0], from, to, type, hiL, hiR);
            return;
        }
        for (int c0 = from; c0 < to; c0 += kChunk)
        {
            const int c1 = std::min(to, c0 + kChunk);
            bool pending = false;
            for (auto& v : voices_)
            {
                if (!v.active) continue;
                gather(v, lanes_[pending ? 1 : 0], c0, c1, type, hiL, hiR);
                if (pending) mixLanes(type, hiL, hiR, true);
                pending = !pending;
            }
            if (pending) mixLanes(type, hiL, hiR, false);
        }
    }

    /** Sources, control ticks and amplifier of one voice over [from, to):
     *  with a filter, at most one chunk gathered into the lane; without one,
     *  any stretch, added to the output. Deactivates the voice when it ends. */
    void gather(Voice& v, Lane& lane, int from, int to, FilterType type, T* outL, T* outR) noexcept
    {
        const bool filtered = type != FilterType::Off;
        const bool ladder = type == FilterType::LadderLp24 || type == FilterType::LadderLp12
                         || type == FilterType::LadderBp12 || type == FilterType::LadderHp24;
        lane.voice = &v;
        lane.start = from;
        int i = from;
        while (i < to)
        {
            if (v.tickLeft == 0)
            {
                offsetHi_ = i;
                controlTick(v);
                v.tickLeft = periodHi_;
            }
            const int m = std::min(to - i, v.tickLeft);
            const int at = filtered ? i - from : 0;   // unfiltered: scratch per segment
            double* L = lane.l.data() + at;
            double* R = lane.r.data() + at;
            std::fill_n(L, m, 0.0);
            std::fill_n(R, m, 0.0);
            renderSources(v, L, R, m);

            const int pos0 = v.tickPos;
            for (int j = 0; j < m; ++j)
            {
                const int pos = pos0 + j;
                const auto s = static_cast<size_t>(at + j);
                if (filtered)
                {
                    lane.g[s] = static_cast<T>(v.g.at(pos));
                    lane.q[s] = static_cast<T>(ladder ? v.resonance.at(pos) : v.svfK.at(pos));
                    if (ladder) lane.drive[s] = static_cast<T>(v.drive.at(pos));
                }
                double gain = v.amp.getNextValue() * v.ampGain.at(pos);
                if (v.stealLeft >= 0)
                {
                    gain *= static_cast<double>(v.stealLeft) / static_cast<double>(stealFade_);
                    if (--v.stealLeft < 0)
                    {
                        lane.count = at + j;
                        v.active = false;
                        return;
                    }
                }
                if (filtered)
                {
                    lane.gain[s] = gain;
                    lane.panL[s] = v.panL.at(pos);
                    lane.panR[s] = v.panR.at(pos);
                }
                else
                {
                    const double l = L[j];
                    const double r = v.stereo ? R[j] : l;
                    outL[i + j] += static_cast<T>(l * gain * v.panL.at(pos));
                    outR[i + j] += static_cast<T>(r * gain * v.panR.at(pos));
                }
                if (!v.amp.isActive())
                {
                    lane.count = at + j + 1;
                    v.active = false;
                    return;
                }
            }
            v.tickLeft -= m;
            v.tickPos += m;
            v.ageHi += m;
            i += m;
        }
        lane.count = to - from;
    }

    void mixLanes(FilterType type, T* outL, T* outR, bool pair) noexcept
    {
        using detail::LadderTap;
        switch (type)
        {
            case FilterType::LadderLp24: mixWith<LadderKernel<LadderTap::LP24>>(outL, outR, pair); break;
            case FilterType::LadderLp12: mixWith<LadderKernel<LadderTap::LP12>>(outL, outR, pair); break;
            case FilterType::LadderBp12: mixWith<LadderKernel<LadderTap::BP12>>(outL, outR, pair); break;
            case FilterType::LadderHp24: mixWith<LadderKernel<LadderTap::HP24>>(outL, outR, pair); break;
            case FilterType::SvfHp:      mixWith<SvfKernel<FilterType::SvfHp>>(outL, outR, pair); break;
            case FilterType::SvfBp:      mixWith<SvfKernel<FilterType::SvfBp>>(outL, outR, pair); break;
            case FilterType::SvfNotch:   mixWith<SvfKernel<FilterType::SvfNotch>>(outL, outR, pair); break;
            default:                     mixWith<SvfKernel<FilterType::SvfLp>>(outL, outR, pair); break;
        }
    }

    /** Filters the gathered lanes and adds them to the output; with a pair,
     *  the samples both lanes hold are interleaved. */
    template <class Kernel>
    void mixWith(T* outL, T* outR, bool pair) noexcept
    {
        Lane& a = lanes_[0];
        Lane& b = lanes_[1];
        const int both = pair ? std::min(a.count, b.count) : 0;
        for (int j = 0; j < both; ++j)
        {
            mixSample<Kernel>(a, j, outL, outR);
            mixSample<Kernel>(b, j, outL, outR);
        }
        for (int j = both; j < a.count; ++j) mixSample<Kernel>(a, j, outL, outR);
        if (pair)
            for (int j = both; j < b.count; ++j) mixSample<Kernel>(b, j, outL, outR);
    }

    template <class Kernel>
    static void mixSample(Lane& lane, int j, T* outL, T* outR) noexcept
    {
        const auto s = static_cast<size_t>(j);
        Voice& v = *lane.voice;
        double l = lane.l[s];
        double r = v.stereo ? lane.r[s] : l;
        Kernel::run(v, lane, s, l, r);
        outL[lane.start + j] += static_cast<T>(l * lane.gain[s] * lane.panL[s]);
        outR[lane.start + j] += static_cast<T>(r * lane.gain[s] * lane.panR[s]);
    }

    template <detail::LadderTap Tap> struct LadderKernel
    {
        static void run(Voice& v, const Lane& lane, size_t s, double& l, double& r) noexcept
        {
            const auto c = detail::makeLadderCoeffs(lane.g[s], lane.q[s]);
            const T drive = lane.drive[s];
            l = static_cast<double>(detail::ladderStep<Tap>(static_cast<T>(l), v.ladder[0], c, drive));
            r = v.stereo ? static_cast<double>(detail::ladderStep<Tap>(static_cast<T>(r), v.ladder[1], c, drive)) : l;
        }
    };

    template <FilterType Type> struct SvfKernel
    {
        static T highpass(T x, T k, T bandpass, T lowpass) noexcept { return x - k * bandpass - lowpass; }

        static double step(T x, detail::TptSvfState<T>& state, T a1, T a2, T a3, T k) noexcept
        {
            const auto o = detail::tptSvfStep(x, state, a1, a2, a3);
            if constexpr (Type == FilterType::SvfHp)
                return static_cast<double>(highpass(x, k, o.bandpass, o.lowpass));
            else if constexpr (Type == FilterType::SvfBp)
                return static_cast<double>(k * o.bandpass);   // unit peak gain
            else if constexpr (Type == FilterType::SvfNotch)
                return static_cast<double>(o.lowpass + highpass(x, k, o.bandpass, o.lowpass));
            else
                return static_cast<double>(o.lowpass);
        }

        static void run(Voice& v, const Lane& lane, size_t s, double& l, double& r) noexcept
        {
            const T g = lane.g[s];
            const T k = lane.q[s];
            const T a1 = T(1) / (T(1) + g * (g + k));
            const T a2 = g * a1;
            const T a3 = g * a2;
            l = step(static_cast<T>(l), v.svf[0], a1, a2, a3, k);
            r = v.stereo ? step(static_cast<T>(r), v.svf[1], a1, a2, a3, k) : l;
        }
    };

    void renderSources(Voice& v, double* L, double* R, int m) noexcept
    {
        const int pos0 = v.tickPos;
        switch (v.engine)
        {
            case Engine::Analog:    renderAnalog(v, L, R, m, pos0); break;
            case Engine::Wavetable: renderWavetable(v, L, R, m, pos0); break;
            case Engine::Fm:        renderFm(v, L, R, m, pos0); break;
        }

        // Sample source (every engine). Its readers keep moving while the
        // level is zero, so the sample stays in time if it comes back.
        if (v.zone != nullptr)
        {
            const bool audible = v.sampleLevel.live();
            const bool zoneStereo = v.zone->channels > 1;
            for (int u = 0; u < v.unison; ++u)
            {
                auto& uv = v.uni[static_cast<size_t>(u)];
                for (int j = 0; j < m && uv.readerOn; ++j)
                {
                    double a0 = 0.0, a1 = 0.0;
                    const auto frame = uv.reader.next(uv.sampleRatio.at(pos0 + j), 2, weights_.data(), a0, a1);
                    if (frame == Reader::Frame::Ended) { uv.readerOn = false; break; }
                    if (audible)
                    {
                        const double level = v.sampleLevel.at(pos0 + j);
                        L[j] += a0 * v.zoneGainL * level * uv.gainL;
                        R[j] += (zoneStereo ? a1 : a0) * v.zoneGainR * level * uv.gainR;
                    }
                    if (frame == Reader::Frame::Last) uv.readerOn = false;
                }
            }
        }

        // Sub-oscillator and noise, centred.
        if (v.subLevel.live())
            for (int j = 0; j < m; ++j)
            {
                if (v.subFreq.step != 0.0) v.sub.setFrequency(static_cast<T>(v.subFreq.at(pos0 + j)));
                const double s = static_cast<double>(v.sub.getNextSample()) * v.subLevel.at(pos0 + j);
                L[j] += s;
                R[j] += s;
            }
        if (v.noiseLevel.live())
            for (int j = 0; j < m; ++j)
            {
                uint32_t x = v.noise;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                v.noise = x;
                const double s = (static_cast<double>(x) * (2.0 / 4294967296.0) - 1.0) * v.noiseLevel.at(pos0 + j);
                L[j] += s;
                R[j] += s;
            }
    }

    void renderAnalog(Voice& v, double* L, double* R, int m, int pos0) noexcept
    {
        for (int k = 0; k < 3; ++k)
        {
            const auto& lv = v.oscLevel[static_cast<size_t>(k)];
            if (!lv.live()) continue;
            const auto& pw = v.pulseWidth[static_cast<size_t>(k)];
            const bool sync = ip(oscParam(k, 6)) != 0;
            const bool pulse = static_cast<OscWave>(ip(oscParam(k, 0))) == OscWave::Pulse;
            for (int u = 0; u < v.unison; ++u)
            {
                auto& uv = v.uni[static_cast<size_t>(u)];
                auto& o = uv.osc[static_cast<size_t>(k)];
                const auto& fr = uv.freq[static_cast<size_t>(k)];
                const auto& ms = uv.master[static_cast<size_t>(k)];
                const bool moving = fr.step != 0.0 || ms.step != 0.0;
                for (int j = 0; j < m; ++j)
                {
                    const int pos = pos0 + j;
                    if (moving)
                    {
                        if (sync)
                        {
                            const double mh = ms.at(pos);
                            o.setFrequency(static_cast<T>(mh));
                            o.setSyncRatio(static_cast<T>(mh > 0.0 ? fr.at(pos) / mh : 0.0));
                        }
                        else
                        {
                            o.setFrequency(static_cast<T>(fr.at(pos)));
                        }
                    }
                    if (pulse && pw.step != 0.0) o.setPulseWidth(static_cast<T>(pw.at(pos)));
                    const double s = static_cast<double>(o.getNextSample()) * lv.at(pos);
                    L[j] += s * uv.gainL;
                    R[j] += s * uv.gainR;
                }
            }
        }
    }

    void renderWavetable(Voice& v, double* L, double* R, int m, int pos0) noexcept
    {
        for (int k = 0; k < 2; ++k)
        {
            const auto& lv = v.wtLevel[static_cast<size_t>(k)];
            if (!lv.live()) continue;
            const WavetableBank& bank = bankFor(v, k);
            const double frames = static_cast<double>(std::max(bank.getNumFrames() - 1, 0));
            const auto& wp = v.wtPosition[static_cast<size_t>(k)];
            for (int u = 0; u < v.unison; ++u)
            {
                auto& uv = v.uni[static_cast<size_t>(u)];
                const auto& fr = uv.freq[static_cast<size_t>(k)];
                double& phase = uv.wtPhase[static_cast<size_t>(k)];
                const auto mip = uv.mip[static_cast<size_t>(k)];
                for (int j = 0; j < m; ++j)
                {
                    const int pos = pos0 + j;
                    const double s = static_cast<double>(bank.read<T>(phase, wp.at(pos) * frames, mip)) * lv.at(pos);
                    phase += fr.at(pos) / internalRate_;
                    if (phase >= 1.0) phase -= 1.0;
                    L[j] += s * uv.gainL;
                    R[j] += s * uv.gainR;
                }
            }
        }
    }

    void renderFm(Voice& v, double* L, double* R, int m, int pos0) noexcept
    {
        // Modulation index: an operator at full level and envelope deviates
        // its target's phase by up to 4 pi radians.
        constexpr double kIndex = 4.0 * pi<double>;
        constexpr int kMax = kControlPeriod * 4;
        const int algorithm = std::clamp(ip(Param::FmAlgorithm), 0, 7);

        // Operator amplitudes (level times envelope), shared by the unison copies.
        std::array<std::array<double, kMax>, 4> amp;   // every used entry is written below
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < 4; ++k)
                amp[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                    v.opLevel[static_cast<size_t>(k)].at(pos0 + j) * v.opEnv[static_cast<size_t>(k)].getNextValue();

        for (int u = 0; u < v.unison; ++u)
        {
            auto& uv = v.uni[static_cast<size_t>(u)];
            for (int j = 0; j < m; ++j)
            {
                const int pos = pos0 + j;
                auto op = [&](int k, double pm) noexcept {
                    return amp[static_cast<size_t>(k)][static_cast<size_t>(j)]
                         * fastSin(twoPi<double> * uv.opPhase[static_cast<size_t>(k)] + pm);
                };
                // Operator 4 feeds back the mean of its last two outputs.
                const double o4 = op(3, v.feedback.at(pos) * pi<double> * 0.5 * (uv.fb1 + uv.fb2));
                uv.fb2 = uv.fb1;
                uv.fb1 = o4;
                double out = 0.0;
                switch (algorithm)
                {
                    case 0: { const double o3 = op(2, kIndex * o4); const double o2 = op(1, kIndex * o3);
                              out = op(0, kIndex * o2); break; }
                    case 1: { const double o3 = op(2, 0.0); const double o2 = op(1, kIndex * (o3 + o4));
                              out = op(0, kIndex * o2); break; }
                    case 2: { const double o3 = op(2, 0.0); const double o2 = op(1, kIndex * o3);
                              out = op(0, kIndex * (o2 + o4)); break; }
                    case 3: { const double o3 = op(2, kIndex * o4); const double o2 = op(1, 0.0);
                              out = op(0, kIndex * (o2 + o3)); break; }
                    case 4: { const double o3 = op(2, kIndex * o4); const double o2 = op(1, 0.0);
                              out = 0.5 * (op(0, kIndex * o2) + o3); break; }
                    case 5: out = (op(0, kIndex * o4) + op(1, kIndex * o4) + op(2, kIndex * o4)) / 3.0; break;
                    case 6: out = (op(0, 0.0) + op(1, 0.0) + op(2, kIndex * o4)) / 3.0; break;
                    default: out = 0.25 * (op(0, 0.0) + op(1, 0.0) + op(2, 0.0) + o4); break;
                }
                for (int k = 0; k < 4; ++k)
                {
                    double& ph = uv.opPhase[static_cast<size_t>(k)];
                    ph += uv.opFreq[static_cast<size_t>(k)].at(pos) / internalRate_;
                    if (ph >= 1.0) ph -= 1.0;
                }
                L[j] += out * uv.gainL;
                R[j] += out * uv.gainR;
            }
        }
    }

    // ===================================================================================
    // Members
    // ===================================================================================


    std::array<std::atomic<float>, kNumParams> values_ {};
    std::array<float, kNumParams> params_ {};
    std::atomic<float> tempo_ { 120.0f };
    float tempoApplied_ = 120.0f;

    double baseRate_ = 48000.0;
    double internalRate_ = 48000.0;
    int maxBlock_ = 512;
    int maxVoices_ = 16;
    int factor_ = 1;
    int periodHi_ = kControlPeriod;
    double invPeriod_ = 1.0 / kControlPeriod;
    int shift_ = 0;
    int stealFade_ = 144;
    double smoothAlpha_ = 0.03;
    bool prepared_ = false;

    std::vector<Voice> voices_;
    std::vector<T> weights_;
    std::array<std::vector<T>, 2> mix_;
    std::array<Lane, 2> lanes_ {};
    Oversampling<T, 2> os2_ { 2, Oversampling<T, 2>::Quality::Maximum };
    Oversampling<T, 2> os4_ { 4, Oversampling<T, 2>::Quality::Maximum };

    std::array<Event, kMaxEvents> events_ {};
    int eventCount_ = 0;
    std::array<int, 128> held_ {};
    std::array<double, 128> heldTuning_ {};
    int heldCount_ = 0;
    bool sustainPedal_ = false;
    double bend_ = 0.0;
    double modWheel_ = 0.0;
    double pressure_ = 0.0;
    double expression_ = 1.0;
    double breath_ = 0.0;
    double lastPitch_ = -1.0;
    bool haveLastPitch_ = false;
    double masterGain_ = 0.5;
    uint64_t noteCounter_ = 0;
    uint64_t clock_ = 0;
    int64_t timeHi_ = 0;              ///< Internal-rate samples rendered since reset().
    int offsetHi_ = 0;                ///< Position inside the piece being rendered.
    std::array<FreeLfo, 2> freeLfo_ {};

    // Control thread's current content, shared with what it publishes.
    std::shared_ptr<const std::vector<ZoneData>> controlZones_;
    std::shared_ptr<const WavetableBank> controlWavetable_;

    Content* active_ = nullptr;                  ///< Audio thread's current content.
    Content* fading_ = nullptr;                  ///< Previous content, until its voices end.
    std::atomic<Content*> pending_ { nullptr };  ///< Offered by the control thread.
    std::atomic<Content*> retired_ { nullptr };  ///< Handed back for freeing.

    std::atomic<int> latency_ { 0 };
    std::atomic<int> activeVoices_ { 0 };
};

} // namespace dspark
