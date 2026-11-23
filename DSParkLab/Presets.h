// DSParkLab - Presets
// Settings to start listening from, for the effects with many controls.
//
// A preset is the slot at its defaults with a few controls set: choosing one
// in the header of the effect (or stepping with [ and ]) leaves the effect in
// a state worth hearing, and its note says what it shows and what to compare.
// Controls are named as the interface names them; a name a slot does not have
// is left out and listed in red next to the preset box.
//
// Choice controls take the index of the option (0 is the first).

#pragma once

#include "EffectSlot.h"

#include <cstring>
#include <vector>

namespace dsplab {

/// The slot with this name, or nullptr.
inline EffectSlot* findSlot(std::vector<EffectSlot>& all, const char* name)
{
    for (auto& slot : all)
        if (slot.name == name) return &slot;
    return nullptr;
}

/// Adds the presets of every effect that has them.
inline void addLabPresets(std::vector<EffectSlot>& all)
{
    // --- Filters ------------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Filter"))
    {
        s->addPreset("Resonant low-pass", "A 24 dB/oct low-pass with a peak at its corner. Move Frequency.",
            { { "Type", 0 }, { "Frequency", 800 }, { "Resonance", 6 }, { "Slope", 24 } });
        s->addPreset("Rumble cut, 48 dB/oct", "Everything under 60 Hz goes.",
            { { "Type", 1 }, { "Frequency", 60 }, { "Slope", 48 } });
        s->addPreset("Telephone", "A narrow band-pass around 1.5 kHz.",
            { { "Type", 2 }, { "Frequency", 1500 }, { "Resonance", 1.2f } });
        s->addPreset("Tilt, brighter", "The whole band leans towards the top around 1 kHz.",
            { { "Type", 8 }, { "Frequency", 1000 }, { "Gain", 4 } });
        s->addPreset("Driven ladder structure", "The same low-pass through the ladder structure with its nonlinearity.",
            { { "Type", 0 }, { "Frequency", 1200 }, { "Resonance", 3 }, { "Slope", 24 }, { "Nonlinearity", 0.6f }, { "Structure", 1 } });
    }

    if (EffectSlot* s = findSlot(all, "Equalizer"))
    {
        s->addPreset("Smile", "Shelves up at both ends and a little off the low mids.",
            { { "Band 1 Type", 1 }, { "Band 1 Freq", 90 }, { "Band 1 Gain", 4 },
              { "Band 2 Freq", 400 }, { "Band 2 Gain", -2 }, { "Band 2 Q", 0.8f },
              { "Band 4 Type", 2 }, { "Band 4 Freq", 9000 }, { "Band 4 Gain", 4 } });
        s->addPreset("Smile, linear phase", "The same curve with no phase shift (Phase Mode). Compare with Minimum on transients.",
            { { "Phase Mode", 1 }, { "Band 1 Type", 1 }, { "Band 1 Freq", 90 }, { "Band 1 Gain", 4 },
              { "Band 2 Freq", 400 }, { "Band 2 Gain", -2 }, { "Band 2 Q", 0.8f },
              { "Band 4 Type", 2 }, { "Band 4 Freq", 9000 }, { "Band 4 Gain", 4 } });
        s->addPreset("Resonant low shelf", "A low shelf with Q 2: it rises over its plateau next to the corner. Toggle Resonant Shelves.",
            { { "Analog Response", 1 }, { "Resonant Shelves", 1 }, { "Band 1 Type", 1 }, { "Band 1 Freq", 100 },
              { "Band 1 Gain", 8 }, { "Band 1 Q", 2 } });
        s->addPreset("Steep cuts, 60 dB/oct", "Analog cuts of any slope at both ends. Move the two Slope controls.",
            { { "Analog Response", 1 }, { "Band 1 Type", 4 }, { "Band 1 Freq", 60 }, { "Band 1 Slope", 60 },
              { "Band 4 Type", 3 }, { "Band 4 Freq", 14000 }, { "Band 4 Slope", 60 } });
        s->addPreset("Flat-top bell, order 2.5", "A bell between the second and the third order. Move Band 2 Order from 1 to 6.",
            { { "Analog Response", 1 }, { "Band 2 Freq", 800 }, { "Band 2 Gain", 9 }, { "Band 2 Q", 1.5f }, { "Band 2 Order", 2.5f } });
        s->addPreset("High bell near the top", "A bell at 16 kHz: toggle Matched Bells and Analog Response to hear the top octave.",
            { { "Band 4 Freq", 16000 }, { "Band 4 Gain", 10 }, { "Band 4 Q", 1.2f }, { "Matched Bells", 1 } });
        s->addPreset("Mid weight, wide top", "Lows up in the middle only, highs up on the sides only (Band Channels).",
            { { "Band 1 Type", 1 }, { "Band 1 Freq", 120 }, { "Band 1 Gain", 3 }, { "Band 1 Channels", 3 },
              { "Band 4 Type", 2 }, { "Band 4 Freq", 6000 }, { "Band 4 Gain", 5 }, { "Band 4 Channels", 4 } });
        s->addPreset("Second-order tilt", "A tilt of order 2 around 1 kHz, steeper through the middle.",
            { { "Analog Response", 1 }, { "Band 2 Type", 7 }, { "Band 2 Freq", 1000 }, { "Band 2 Gain", 4 }, { "Band 2 Order", 2 } });
    }

    if (EffectSlot* s = findSlot(all, "Ladder Filter"))
    {
        s->addPreset("Squelch", "A 24 dB/oct low-pass close to self-oscillation, driven at its input.",
            { { "Cutoff", 600 }, { "Resonance", 0.85f }, { "Drive", 9 }, { "Mode", 3 }, { "Saturation", 2 } });
        s->addPreset("Warm low-pass", "Gentle resonance and a little drive. Compare the three Saturation models.",
            { { "Cutoff", 2500 }, { "Resonance", 0.3f }, { "Drive", 4 }, { "Mode", 3 }, { "Saturation", 1 } });
        s->addPreset("Band-pass voice", "The band-pass output with a strong peak. Move Cutoff.",
            { { "Cutoff", 1200 }, { "Resonance", 0.7f }, { "Mode", 4 } });
    }

    if (EffectSlot* s = findSlot(all, "SVF Filter"))
    {
        s->addPreset("Resonant band-pass", "A narrow band-pass. Move Cutoff.",
            { { "Cutoff", 1000 }, { "Q", 6 }, { "Mode", 2 } });
        s->addPreset("Notch", "A narrow cut at 1 kHz.",
            { { "Cutoff", 1000 }, { "Q", 4 }, { "Mode", 3 } });
        s->addPreset("High shelf up", "Six decibels over 5 kHz.",
            { { "Cutoff", 5000 }, { "Gain", 6 }, { "Mode", 7 } });
    }

    if (EffectSlot* s = findSlot(all, "Dynamic EQ"))
    {
        s->addPreset("De-ess, band 2", "Band 2 cuts up to 10 dB at 6.5 kHz when the sibilance passes the threshold.",
            { { "B1 Above Ratio", 1 }, { "B2 Freq", 6500 }, { "B2 Q", 2 }, { "B2 Threshold", -30 },
              { "B2 Above Ratio", 4 }, { "B2 Above Amount", -10 }, { "B2 Above Attack", 1 }, { "B2 Above Release", 60 } });
        s->addPreset("Tame boomy lows", "Band 1 holds the low mids down only when they build up.",
            { { "B2 Above Ratio", 1 }, { "B1 Freq", 140 }, { "B1 Q", 1.2f }, { "B1 Threshold", -28 },
              { "B1 Above Ratio", 3 }, { "B1 Above Amount", -8 } });
        s->addPreset("Low mids give way to the kick", "Band 1 acts at 300 Hz and listens to 45-90 Hz (Own Zone). Set Hear Key to Band 1 to hear the zone.",
            { { "B2 Above Ratio", 1 }, { "B1 Freq", 300 }, { "B1 Q", 1 }, { "B1 Threshold", -30 },
              { "B1 Above Ratio", 4 }, { "B1 Above Amount", -9 }, { "B1 Above Attack", 2 }, { "B1 Above Release", 120 },
              { "B1 Own Zone", 1 }, { "B1 Zone Low", 45 }, { "B1 Zone High", 90 } });
        s->addPreset("Presence lifted when quiet", "Band 2 raises 4 kHz under the threshold and leaves it alone over it.",
            { { "B1 Above Ratio", 1 }, { "B2 Freq", 4000 }, { "B2 Q", 0.8f }, { "B2 Threshold", -35 },
              { "B2 Above Ratio", 1 }, { "B2 Below Ratio", 2 }, { "B2 Below Amount", 6 } });
        s->addPreset("Program timing, relative threshold", "Band 1 times itself from its frequency and counts its threshold from its own recent level.",
            { { "B2 Above Ratio", 1 }, { "B1 Freq", 250 }, { "B1 Q", 1.5f }, { "B1 Timing", 1 }, { "B1 Relative Thr", 1 },
              { "B1 Threshold", -8 }, { "B1 Above Ratio", 4 }, { "B1 Above Amount", -9 }, { "B1 Knee", -1 } });
        s->addPreset("De-ess, mixed bands", "The de-esser with fixed filters mixed by the gain (Mixed Bands). Compare with it off.",
            { { "Mixed Bands", 1 }, { "B1 Above Ratio", 1 }, { "B2 Freq", 6500 }, { "B2 Q", 2 }, { "B2 Threshold", -30 },
              { "B2 Above Ratio", 4 }, { "B2 Above Amount", -10 }, { "B2 Above Attack", 1 }, { "B2 Above Release", 60 } });
        s->addPreset("De-ess, linear phase", "The same cut with no phase shift (it adds latency).",
            { { "Linear Phase", 1 }, { "B1 Above Ratio", 1 }, { "B2 Freq", 6500 }, { "B2 Q", 2 }, { "B2 Threshold", -30 },
              { "B2 Above Ratio", 4 }, { "B2 Above Amount", -10 }, { "B2 Above Attack", 1 }, { "B2 Above Release", 60 } });
        s->addPreset("Sides only: tame harsh width", "Band 2 works on the sides alone (Channels) around 3 kHz.",
            { { "B1 Above Ratio", 1 }, { "B2 Freq", 3000 }, { "B2 Q", 1.2f }, { "B2 Channels", 4 }, { "B2 Threshold", -34 },
              { "B2 Above Ratio", 3 }, { "B2 Above Amount", -8 } });
    }

    // --- Dynamics -----------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Compressor"))
    {
        s->addPreset("Vocal levelling", "3:1 with a wide knee and the analytic makeup. Compare Program Release at 0 and Release Landing at 0.",
            { { "Threshold", -24 }, { "Ratio", 3 }, { "Attack", 8 }, { "Release", 120 }, { "Knee", 12 },
              { "Detector", 4 }, { "Auto Makeup", 3 }, { "Program Release", 0.6f }, { "Release Landing", 200 } });
        s->addPreset("Drum bus punch", "A slow attack lets the hit through; Attack Emphasis deepens the leading edge. Parallel at 70 %.",
            { { "Threshold", -18 }, { "Ratio", 4 }, { "Attack", 25 }, { "Release", 80 }, { "Knee", 4 },
              { "Attack Emphasis", 0.5f }, { "Mix", 0.7f }, { "Auto Makeup", 1 } });
        s->addPreset("FET limiting, with colour", "The FET character at 12:1 with its fastest attack and its second harmonic (Color). Precision High cleans the top.",
            { { "Character", 2 }, { "Threshold", -20 }, { "Ratio", 12 }, { "Attack", 0.3f }, { "Release", 100 },
              { "Knee", 0 }, { "Color", 1 }, { "Precision", 1 }, { "Makeup", 6 } });
        s->addPreset("Opto, three-stage cell", "The optical character with its medium memory: hold a note, then stop, and hear the release in stretches.",
            { { "Character", 1 }, { "Threshold", -22 }, { "Ratio", 4 }, { "Release", 300 }, { "Opto Cell", 1 }, { "Auto Makeup", 2 } });
        s->addPreset("Varimu glue, push-pull colour", "A gentle 2:1 whose ratio grows with level; Color applies the gain through a push-pull pair.",
            { { "Character", 3 }, { "Threshold", -18 }, { "Ratio", 2 }, { "Attack", 30 }, { "Release", 400 },
              { "Knee", 12 }, { "Color", 0.6f }, { "Auto Makeup", 1 } });
        s->addPreset("Layered", "Two layers: one that levels slowly under one that catches peaks.",
            { { "Character", 4 }, { "Threshold", -20 }, { "Ratio", 4 }, { "Attack", 5 }, { "Release", 150 }, { "Auto Makeup", 1 } });
        s->addPreset("Two ways: loud down, quiet up", "Mode Both: 3:1 over -18 dB and a lift of up to 8 dB under -43 dB. Listen to tails and room.",
            { { "Mode", 2 }, { "Threshold", -18 }, { "Ratio", 3 }, { "Attack", 10 }, { "Release", 200 }, { "Detector", 4 },
              { "Lift Below", 25 }, { "Lift Ratio", 2 }, { "Lift Range", 8 } });
        s->addPreset("Leveler", "Mode Leveler: no ratio and no knee, the output settles on the threshold. Range 12 dB with the local bend.",
            { { "Mode", 3 }, { "Threshold", -14 }, { "Attack", 15 }, { "Release", 300 }, { "Detector", 4 },
              { "Range", 12 }, { "Range Shape", 2 }, { "Auto Makeup", 1 } });
        s->addPreset("Reduction that keeps the top", "Tone High at -2 dB: the more it compresses, the more of the top it leaves. Compare -2, 0 and +2.",
            { { "Threshold", -26 }, { "Ratio", 6 }, { "Attack", 5 }, { "Release", 150 }, { "Detector", 4 },
              { "Tone High", -2 }, { "Makeup", 6 } });
        s->addPreset("Vintage control path", "A lead-lag network, a leak of the control and an asymmetric detector. Turn each to 0 to hear what it adds.",
            { { "Threshold", -24 }, { "Ratio", 6 }, { "Attack", 1 }, { "Release", 60 }, { "Knee", 0 },
              { "Network Depth", 6 }, { "Network Corner", 8 }, { "Control Leak", 0.6f }, { "Detector Asymmetry", 0.3f }, { "Makeup", 6 } });
        s->addPreset("Middle only", "Route Mid: the centre is compressed and the sides pass untouched.",
            { { "Route", 2 }, { "Threshold", -22 }, { "Ratio", 4 }, { "Attack", 10 }, { "Release", 150 }, { "Detector", 4 } });
        s->addPreset("Sides ducked by the middle", "Route Mid keys Side: the width closes while the centre is loud.",
            { { "Route", 4 }, { "Threshold", -26 }, { "Ratio", 4 }, { "Attack", 5 }, { "Release", 200 }, { "Detector", 4 } });
        s->addPreset("Key bell on the sides", "The key's bell at 6 kHz, +12 dB, heard only in the sides: wide cymbals pull the gain down, a centred voice with the same brightness does not. Compare Key Bell On at Both channels.",
            { { "Threshold", -24 }, { "Ratio", 4 }, { "Attack", 5 }, { "Release", 150 }, { "Key Bell Freq", 6000 }, { "Key Bell Gain", 12 }, { "Key Bell Q", 1 }, { "Key Bell On", 2 } });
        s->addPreset("Key low cut on the middle", "The key's low cut at 150 Hz only on the middle: a centred kick no longer pumps the mix, a wide bass still does. Compare Key Low Cut On at Both channels.",
            { { "Threshold", -24 }, { "Ratio", 3 }, { "Attack", 10 }, { "Release", 120 }, { "Key Low Cut", 150 }, { "Key Low Cut Slope", 24 }, { "Key Low Cut On", 1 } });
        s->addPreset("Lookahead as a ramp", "A 5 ms lookahead that brings the reduction in along a ramp, with every peak held 20 ms.",
            { { "Threshold", -16 }, { "Ratio", 20 }, { "Attack", 0.1f }, { "Release", 80 }, { "Knee", 0 },
              { "Max Lookahead", 5 }, { "Lookahead", 5 }, { "Lookahead Shape", 1 }, { "Hold", 20 }, { "Hold Mode", 1 } });
        s->addPreset("Parallel crush", "Heavy compression mixed in at 35 %.",
            { { "Threshold", -35 }, { "Ratio", 12 }, { "Attack", 2 }, { "Release", 80 }, { "Mix", 0.35f }, { "Makeup", 6 } });
        s->addPreset("Colour stage after", "A clean 4:1 followed by the analog colour stage.",
            { { "Threshold", -20 }, { "Ratio", 4 }, { "Colour Stage", 2 }, { "Stage Drive", 9 }, { "Stage Emphasis", 6 }, { "Stage Asymmetry", 0.3f } });
    }

    if (EffectSlot* s = findSlot(all, "Limiter"))
    {
        s->addPreset("Transparent master", "4 dB into a true-peak ceiling of -1 dB.",
            { { "Input Gain", 4 }, { "Ceiling", -1 }, { "Release", 100 }, { "Lookahead", 3 }, { "True Peak", 1 }, { "Adaptive Release", 1 } });
        s->addPreset("Loud", "9 dB in, every peak held, the precise true-peak reading. Compare Gain Precision High.",
            { { "Input Gain", 9 }, { "Ceiling", -0.3f }, { "Release", 60 }, { "Lookahead", 5 }, { "True Peak", 1 },
              { "Hold Every Peak", 1 }, { "True Peak High Precision", 1 } });
        s->addPreset("Fast and gritty", "A short lookahead and a fast release: the limiter is heard working.",
            { { "Input Gain", 8 }, { "Release", 20 }, { "Lookahead", 1 }, { "Adaptive Release", 0 } });
    }

    if (EffectSlot* s = findSlot(all, "Noise Gate"))
    {
        s->addPreset("Tight drum gate", "Opens fast, closes 60 dB.",
            { { "Threshold", -30 }, { "Attack", 0.5f }, { "Hold", 40 }, { "Release", 80 }, { "Range", -60 } });
        s->addPreset("Gentle, 10 dB", "Takes 10 dB off what lies under the threshold.",
            { { "Threshold", -45 }, { "Attack", 2 }, { "Hold", 100 }, { "Release", 200 }, { "Range", -10 } });
        s->addPreset("Frequency gate", "Closes the top first instead of the level (Gate Mode).",
            { { "Gate Mode", 1 }, { "Threshold", -40 }, { "Hold", 60 }, { "Release", 150 }, { "Range", -60 } });
        s->addPreset("Keyed to the lows", "The key is cut over 500 Hz: only the low hits open it. Key Listen plays the key.",
            { { "Threshold", -35 }, { "Hold", 60 }, { "Range", -60 }, { "Key High Cut", 500 } });
    }

    if (EffectSlot* s = findSlot(all, "Expander"))
    {
        s->addPreset("Gentle, 2:1", "Quiet passages go down twice as fast, up to 20 dB.",
            { { "Threshold", -40 }, { "Ratio", 2 }, { "Range", -20 } });
        s->addPreset("Tight, 4:1", "A firm expansion with a fast attack.",
            { { "Threshold", -35 }, { "Ratio", 4 }, { "Attack", 1 }, { "Release", 80 }, { "Range", -40 } });
    }

    if (EffectSlot* s = findSlot(all, "Transient Designer"))
    {
        s->addPreset("More snap", "Attacks up, a little less sustain.", { { "Attack", 60 }, { "Sustain", -10 } });
        s->addPreset("Softer and longer", "Attacks down, sustain up.", { { "Attack", -40 }, { "Sustain", 50 } });
        s->addPreset("Room out", "The tails of the hits go down.", { { "Sustain", -70 } });
    }

    if (EffectSlot* s = findSlot(all, "De-Esser"))
    {
        s->addPreset("Vocal, 6.5 kHz", "Up to 9 dB off the sibilance. Hear Removed plays what it takes.",
            { { "Frequency", 6500 }, { "Bandwidth", 1 }, { "Threshold", -30 }, { "Reduction", 9 }, { "Mixed Bell", 1 } });
        s->addPreset("Wide and gentle", "A band two octaves wide and 5 dB at most.",
            { { "Frequency", 5500 }, { "Bandwidth", 2 }, { "Threshold", -28 }, { "Reduction", 5 }, { "Release", 40 } });
        s->addPreset("Automatic threshold", "The threshold follows the level of the programme.",
            { { "Frequency", 6500 }, { "Auto Threshold", 1 }, { "Threshold", -12 }, { "Reduction", 8 } });
    }

    if (EffectSlot* s = findSlot(all, "Multiband Compressor"))
    {
        s->addPreset("Master glue", "Three bands at 2:1.",
            { { "Low Thresh", -24 }, { "Low Ratio", 2 }, { "Mid Thresh", -24 }, { "Mid Ratio", 2 },
              { "High Thresh", -24 }, { "High Ratio", 2 }, { "Attack", 30 }, { "Release", 200 } });
        s->addPreset("Tight lows", "Only the low band works hard.",
            { { "XOver Low", 120 }, { "Low Thresh", -28 }, { "Low Ratio", 5 }, { "Mid Ratio", 1.5f }, { "High Ratio", 1.5f } });
        s->addPreset("Glue, linear-phase split", "The same with a crossover that does not shift the phase.",
            { { "Crossover", 1 }, { "Low Thresh", -24 }, { "Low Ratio", 2 }, { "Mid Thresh", -24 }, { "Mid Ratio", 2 },
              { "High Thresh", -24 }, { "High Ratio", 2 }, { "Attack", 30 }, { "Release", 200 } });
        s->addPreset("De-harsh highs", "Only the top band, from 4 kHz.",
            { { "XOver High", 4000 }, { "Low Ratio", 1 }, { "Mid Ratio", 1 }, { "High Thresh", -30 }, { "High Ratio", 4 } });
    }

    // --- Distortion ---------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Saturation"))
    {
        s->addPreset("Tube warmth", "12 dB into the tube curve at 4x, level kept (Drive Compensation).",
            { { "Algorithm", 0 }, { "Drive", 12 }, { "Oversampling", 2 }, { "Antialiasing", 2 }, { "Drive Compensation", 1 } });
        s->addPreset("Tape glue", "The tape curve.",
            { { "Algorithm", 1 }, { "Drive", 10 }, { "Oversampling", 2 }, { "Drive Compensation", 1 } });
        s->addPreset("Transformer weight", "The lows saturate first.",
            { { "Algorithm", 2 }, { "Drive", 14 }, { "Oversampling", 2 }, { "Drive Compensation", 1 } });
        s->addPreset("Exciter air", "Harmonics of the top, mixed in at 40 %.",
            { { "Algorithm", 5 }, { "Drive", 8 }, { "Mix", 0.4f }, { "Oversampling", 2 } });
        s->addPreset("Wavefolder", "Folds over the top of the wave. Move Drive.",
            { { "Algorithm", 6 }, { "Drive", 18 }, { "Oversampling", 3 }, { "Drive Compensation", 1 } });
        s->addPreset("Short oversampling (audible band)",
            "4x with a first stage that keeps 20 kHz and stops its image: less latency, the same sound under 20 kHz. Switch Oversampling Band to Preset and back: nothing should change to the ear.",
            { { "Algorithm", 0 }, { "Drive", 14 }, { "Oversampling", 2 }, { "Oversampling Band", 1 }, { "Drive Compensation", 1 } });
        s->addPreset("Curve, hard and uneven", "The curve with three controls: hardness, curvature and asymmetry.",
            { { "Algorithm", 10 }, { "Drive", 12 }, { "Curve Hardness", 0.6f }, { "Curve Asymmetry", 0.4f },
              { "Oversampling", 2 }, { "Drive Compensation", 1 } });
        s->addPreset("Level tracking", "The drive follows the level of the programme: the dynamics stay. Compare Level Tracking at 0.",
            { { "Algorithm", 3 }, { "Drive", 18 }, { "Level Tracking", 0.7f }, { "Tracking Corner", 200 }, { "Drive Compensation", 1 } });
        s->addPreset("Comb loop", "A tuned resonance around the curve at 220 Hz. Move Comb Loop Freq.",
            { { "Algorithm", 0 }, { "Drive", 10 }, { "Comb Loop", 0.6f }, { "Comb Loop Freq", 220 }, { "Comb Loop Closing", 1 },
              { "Drive Compensation", 1 } });
        s->addPreset("Minimum-phase oversampling", "The tube curve at 8x with minimum-phase filters (no pre-ringing, no latency of the linear ones).",
            { { "Algorithm", 0 }, { "Drive", 14 }, { "Oversampling", 3 }, { "Oversampling Mode", 1 }, { "Drive Compensation", 1 } });
    }

    if (EffectSlot* s = findSlot(all, "Multiband Saturation"))
    {
        s->addPreset("Thick lows, clean top", "The tube curve on the lows and the mids only.",
            { { "Bands", 3 }, { "Algorithm", 0 }, { "B1 Drive", 14 }, { "B2 Drive", 6 }, { "B3 Drive", 0 } });
        s->addPreset("Mid crunch", "Only the middle band, mixed in at 70 %.",
            { { "Bands", 3 }, { "Algorithm", 3 }, { "B2 Drive", 20 }, { "Mix", 0.7f } });
        s->addPreset("Bright top band", "Four bands; the exciter on the highest, 3 dB down.",
            { { "Bands", 4 }, { "Algorithm", 5 }, { "B4 Drive", 12 }, { "B4 Level", -3 } });
        s->addPreset("All bands into the output clip", "Every band driven, and a clipper at -1 dB after the sum.",
            { { "Bands", 3 }, { "B1 Drive", 12 }, { "B2 Drive", 12 }, { "B3 Drive", 12 }, { "Output Clip", 1 }, { "Output Ceiling", -1 } });
        s->addPreset("Short oversampling (audible band)",
            "Every band's oversampler with a first stage fitted to the rate. Switch Oversampling Band to compare.",
            { { "Bands", 3 }, { "Algorithm", 0 }, { "B1 Drive", 12 }, { "B2 Drive", 10 }, { "B3 Drive", 8 }, { "Oversampling Band", 1 } });
        s->addPreset("Gentle split, 6 dB/oct", "Wide crossovers: the bands overlap and the split is hardly heard.",
            { { "Bands", 3 }, { "Crossover Slope", 0 }, { "B1 Drive", 10 }, { "B3 Drive", 8 } });
    }

    if (EffectSlot* s = findSlot(all, "Lo-Fi"))
    {
        s->addPreset("8 bits", "8 bits at 11 kHz.", { { "Bits", 8 }, { "Rate", 11025 }, { "Rate Filter", 1 } });
        s->addPreset("Dusty sampler", "12 bits at 22 kHz with dither and a little smear.",
            { { "Bits", 12 }, { "Rate", 22050 }, { "Rate Filter", 1 }, { "Dither", 1 }, { "Smear", 0.2f } });
        s->addPreset("Folded", "The wave folds over itself.", { { "Fold", 18 }, { "Output", -6 } });
        s->addPreset("Slowed down", "Five semitones down and smeared.", { { "Pitch", -5 }, { "Smear", 0.4f } });
    }

    if (EffectSlot* s = findSlot(all, "Clipper"))
    {
        s->addPreset("Soft clip", "6 dB into a soft clip at -1 dB, 4x.",
            { { "Mode", 1 }, { "Input Gain", 6 }, { "Ceiling", -1 }, { "Oversampling", 2 } });
        s->addPreset("Hard clip", "9 dB into a hard clip at 8x with the second-order antialiasing. Toggle it.",
            { { "Mode", 0 }, { "Input Gain", 9 }, { "Ceiling", -0.5f }, { "Oversampling", 3 }, { "Antialiasing (2nd order)", 1 } });
        s->addPreset("Three stages", "The same total drive spread over three stages.",
            { { "Mode", 3 }, { "Input Gain", 12 }, { "Stages", 3 }, { "Oversampling", 2 } });
        s->addPreset("Short oversampling (audible band)",
            "A soft clip at 4x with a first stage that keeps 20 kHz and stops its image. Switch Oversampling Band to compare.",
            { { "Mode", 1 }, { "Input Gain", 9 }, { "Ceiling", -1 }, { "Oversampling", 2 }, { "Oversampling Band", 1 } });
    }

    // --- Analog models ------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Tape Machine"))
    {
        s->addPreset("15 ips, aligned", "The machine at reference level: clean, with its head bump.", { { "Drive", 0 } });
        s->addPreset("7.5 ips, hot", "Slow tape driven 9 dB: more bump, less top, some flutter and hiss.",
            { { "Speed", 0 }, { "Drive", 9 }, { "Head Bump", 0.7f }, { "Loss", 0.7f }, { "Wow/Flutter", 0.3f }, { "Hiss", -70 } });
        s->addPreset("30 ips, master", "Fast tape: an open top and a steady transport.",
            { { "Speed", 2 }, { "Drive", 4 }, { "Loss", 0.3f }, { "Head Bump", 0.4f }, { "Wow/Flutter", 0.05f } });
        s->addPreset("3.75 ips, vintage tape", "The slowest speed with the vintage formulation.",
            { { "Speed", 3 }, { "Drive", 8 }, { "Loss", 1 }, { "Wow/Flutter", 0.5f }, { "Hiss", -62 }, { "Formulation", 2 } });
        s->addPreset("Under-biased", "Bias at 0.15: brighter and gritty. Compare with Bias at 0.5.", { { "Bias", 0.15f }, { "Drive", 6 } });
        s->addPreset("Over-biased", "Bias at 0.9: darker and smoother.", { { "Bias", 0.9f }, { "Drive", 6 } });
        s->addPreset("Circuit electronics and iron", "The record and play amplifiers as circuits, an output transformer, even harmonics and hum.",
            { { "Electronics", 1 }, { "Output Iron", 0.7f }, { "Even Harmonics", 0.4f }, { "Drive", 6 }, { "Hum", -75 } });
        s->addPreset("Head contour, quarter track", "The low end from the geometry of the head (Head: Contour).",
            { { "Head", 1 }, { "Track", 2 }, { "Speed", 0 }, { "Drive", 4 } });
        s->addPreset("Record emphasis +4 dB", "The top is recorded hotter and played back flat: it saturates first.",
            { { "Record Emphasis", 4 }, { "Drive", 6 } });
        s->addPreset("Speed transport", "The wow and flutter as changes of the tape's speed (Transport: Speed).",
            { { "Transport", 1 }, { "Wow/Flutter", 0.6f }, { "Drive", 4 } });
        s->addPreset("Two-level bias, as it was", "The earlier carrier: it lifts the loud parts before they saturate. Switch Bias Shape to Four Level to compare.",
            { { "Bias Shape", 0 }, { "Drive", 6 } });
        s->addPreset("Four-level bias, over-biased", "Where the lift was largest (bias at 0.9). Switch Bias Shape to Two Level to compare.",
            { { "Bias Shape", 1 }, { "Bias", 0.9f }, { "Drive", 6 } });
    }

    if (EffectSlot* s = findSlot(all, "Tube Preamp"))
    {
        s->addPreset("Clean and warm", "One stage, 6 dB of drive.", { { "Drive", 6 }, { "Stages", 0 }, { "Sag", 0.2f } });
        s->addPreset("Crunch", "Two stages with a sagging supply.",
            { { "Drive", 18 }, { "Stages", 1 }, { "Sag", 0.5f }, { "Output", -4 } });
        s->addPreset("Lead with a cabinet", "Two stages driven hard into the cabinet. Move Cab Mic Distance.",
            { { "Drive", 28 }, { "Stages", 1 }, { "Treble", 0.6f }, { "Middle", 0.7f }, { "Bass", 0.4f },
              { "Cabinet", 1 }, { "Cab Mic Distance", 0.2f }, { "Output", -8 } });
        s->addPreset("Grid conduction", "The grid draws current when it is driven positive: the note blooms after its attack.",
            { { "Drive", 20 }, { "Stages", 1 }, { "Grid Model", 1 }, { "Output", -4 } });
    }

    if (EffectSlot* s = findSlot(all, "Transformer"))
    {
        s->addPreset("Console iron", "A medium core at 6 dB.", { { "Drive", 6 }, { "Core Size", 0.5f }, { "Resonance", 0.3f } });
        s->addPreset("Small core, driven", "The lows saturate early.", { { "Drive", 14 }, { "Core Size", 0.15f } });
        s->addPreset("Big core, clean", "Headroom in the lows and a soft top.", { { "Drive", 4 }, { "Core Size", 0.9f }, { "Resonance", 0.1f } });
    }

    // --- Pitch and texture --------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Pitch Shifter"))
    {
        s->addPreset("A fifth up", "Seven semitones up.", { { "Semitones", 7 }, { "Quality", 1 } });
        s->addPreset("An octave down", "Twelve semitones down, studio quality.", { { "Semitones", -12 }, { "Quality", 2 } });
        s->addPreset("Voice up, formants kept", "Three semitones up without the small-head effect. Toggle Formant Preserve.",
            { { "Semitones", 3 }, { "Formant Preserve", 1 }, { "Quality", 2 } });
        s->addPreset("Thickener", "A few cents up, half and half with the original.", { { "Semitones", 0.15f }, { "Mix", 0.5f } });
    }

    if (EffectSlot* s = findSlot(all, "Pitch Corrector"))
    {
        s->addPreset("Hard tune, C major", "Instant retune to C major.", { { "Root", 0 }, { "Scale", 1 }, { "Retune Speed", 0 } });
        s->addPreset("Natural, A minor", "A slow retune to A minor with formants kept.",
            { { "Root", 9 }, { "Scale", 2 }, { "Retune Speed", 120 }, { "Formant Preserve", 1 } });
    }

    if (EffectSlot* s = findSlot(all, "Granular"))
    {
        s->addPreset("Cloud", "Many overlapping grains spread across the pair.",
            { { "Grain Size", 120 }, { "Density", 60 }, { "Jitter", 0.6f }, { "Spread", 0.9f }, { "Mix", 0.6f } });
        s->addPreset("Octave shimmer", "Grains an octave up under the original.",
            { { "Pitch", 12 }, { "Grain Size", 90 }, { "Density", 40 }, { "Pitch Jitter", 0.1f }, { "Mix", 0.4f } });
        s->addPreset("Frozen", "The buffer held and scattered.",
            { { "Freeze", 1 }, { "Grain Size", 200 }, { "Density", 80 }, { "Jitter", 1 }, { "Spread", 1 } });
        s->addPreset("Stutter", "A few short grains, in order.", { { "Grain Size", 30 }, { "Density", 8 }, { "Jitter", 0 } });
    }

    // --- Modulation ---------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Chorus"))
    {
        s->addPreset("Lush stereo", "Four voices spread across the pair.",
            { { "Rate", 0.8f }, { "Depth", 6 }, { "Mix", 0.5f }, { "Voices", 4 }, { "Stereo Spread", 1 } });
        s->addPreset("Analog, slow", "The analog model in its slow mode.", { { "Model", 1 }, { "Analog Mode", 0 }, { "Mix", 0.5f } });
        s->addPreset("Analog, fast", "The analog model in its fast mode.", { { "Model", 1 }, { "Analog Mode", 1 }, { "Mix", 0.5f } });
        s->addPreset("Analog, vibrato", "The analog model with no dry signal.", { { "Model", 1 }, { "Analog Mode", 2 }, { "Mix", 1 } });
        s->addPreset("Flanger", "A short delay with a lot of feedback.",
            { { "Centre Delay", 2 }, { "Depth", 1.5f }, { "Rate", 0.3f }, { "Feedback", 0.8f }, { "Mix", 0.5f }, { "Voices", 1 } });
        s->addPreset("Doubler", "A slow, shallow second voice 18 ms behind.",
            { { "Rate", 0.4f }, { "Depth", 2 }, { "Mix", 0.3f }, { "Voices", 2 }, { "Centre Delay", 18 } });
    }

    if (EffectSlot* s = findSlot(all, "Phaser"))
    {
        s->addPreset("Four stages", "The classic sweep.", { { "Rate", 0.4f }, { "Depth", 0.8f }, { "Mix", 0.5f }, { "Stages", 4 }, { "Feedback", 0.4f } });
        s->addPreset("Twelve stages, deep", "A slow, resonant sweep, wide.",
            { { "Rate", 0.15f }, { "Mix", 0.5f }, { "Stages", 12 }, { "Feedback", 0.7f }, { "Stereo Spread", 0.5f } });
        s->addPreset("Fast vibe", "Six stages at 5 Hz.", { { "Rate", 5 }, { "Depth", 0.5f }, { "Mix", 0.5f }, { "Stages", 6 } });
    }

    if (EffectSlot* s = findSlot(all, "Tremolo"))
    {
        s->addPreset("Amp tremolo", "A sine at 5 Hz.", { { "Rate", 5 }, { "Depth", 0.6f } });
        s->addPreset("Auto-pan", "The two channels in opposition.", { { "Rate", 1.5f }, { "Depth", 1 }, { "Stereo (auto-pan)", 1 } });
        s->addPreset("Chop", "A square at 8 Hz.", { { "Rate", 8 }, { "Depth", 1 }, { "Shape", 2 } });
    }

    if (EffectSlot* s = findSlot(all, "Vibrato"))
    {
        s->addPreset("Gentle", "A third of a semitone at 5 Hz.", { { "Rate", 5 }, { "Depth", 0.3f } });
        s->addPreset("Wobble", "Deep and slow, with a rate that drifts.",
            { { "Rate", 2 }, { "Depth", 1.2f }, { "Mod Rate", 0.3f }, { "Mod Depth", 0.4f } });
        s->addPreset("Analog", "The analog model.", { { "Model", 1 }, { "Rate", 5.5f }, { "Depth", 0.4f } });
    }

    if (EffectSlot* s = findSlot(all, "Ring Modulator"))
    {
        s->addPreset("Metallic", "A carrier at 800 Hz, 60 % wet.", { { "Frequency", 800 }, { "Mix", 0.6f } });
        s->addPreset("Robot", "A low carrier, all wet.", { { "Frequency", 60 }, { "Mix", 1 } });
        s->addPreset("Geometric mean", "The other mode of the modulator.", { { "Frequency", 300 }, { "Mix", 1 }, { "Mode", 1 }, { "Soar", 0.05f } });
    }

    if (EffectSlot* s = findSlot(all, "Freq Shifter"))
    {
        s->addPreset("Barber pole", "Two hertz up against the original: an endless slow phase.", { { "Shift", 2 }, { "Mix", 0.5f } });
        s->addPreset("Thirty hertz up", "Every partial moves by the same amount: no longer harmonic.", { { "Shift", 30 }, { "Mix", 1 } });
        s->addPreset("Down, bell-like", "A hundred and twenty hertz down.", { { "Shift", -120 }, { "Mix", 1 } });
    }

    // --- Spatial ------------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Delay"))
    {
        s->addPreset("Slapback", "One short echo.", { { "Time", 110 }, { "Feedback", 0.15f }, { "Mix", 0.35f } });
        s->addPreset("Tape echo", "The tape character: each repeat darker, with wow and flutter.",
            { { "Time", 320 }, { "Feedback", 0.55f }, { "Mix", 0.35f }, { "Character", 1 }, { "Feedback Mode", 1 },
              { "LP Filter", 4500 }, { "Wow & Flutter", 0.6f } });
        s->addPreset("Bucket brigade", "The bucket character: band-limited and grainy.",
            { { "Time", 280 }, { "Feedback", 0.6f }, { "Mix", 0.4f }, { "Character", 2 }, { "LP Filter", 3000 } });
        s->addPreset("Dub", "Long, dark and near to feeding back.",
            { { "Time", 500 }, { "Feedback", 0.75f }, { "LP Filter", 2500 }, { "HP Filter", 250 }, { "Mix", 0.4f }, { "Feedback Mode", 1 } });
        s->addPreset("Clean quarter note at 120", "Half a second, clean.", { { "Time", 500 }, { "Feedback", 0.35f }, { "Mix", 0.3f } });
    }

    if (EffectSlot* s = findSlot(all, "Reverb"))
    {
        s->addPreset("Small room", "A short room, 25 % wet.", { { "Type", 0 }, { "Mix", 0.25f }, { "Decay", 0.5f }, { "Size", 0.2f } });
        s->addPreset("Vocal plate", "A plate with 30 ms of pre-delay and its lows cut.",
            { { "Type", 3 }, { "Mix", 0.3f }, { "Decay", 1.8f }, { "Pre-Delay", 30 }, { "Low Cut (min = off)", 150 } });
        s->addPreset("Large hall", "Almost three seconds.", { { "Type", 1 }, { "Mix", 0.35f }, { "Decay", 2.8f }, { "Size", 0.7f }, { "Pre-Delay", 20 } });
        s->addPreset("Chamber", "A dense chamber.", { { "Type", 2 }, { "Mix", 0.3f }, { "Decay", 1.4f } });
        s->addPreset("Spring", "The spring tank.", { { "Type", 4 }, { "Mix", 0.35f }, { "Decay", 2 } });
        s->addPreset("Cathedral", "Six seconds, as large as it goes.", { { "Type", 5 }, { "Mix", 0.4f }, { "Decay", 6 }, { "Size", 1 } });
        s->addPreset("Gated", "The tail is cut when it falls under -30 dB.",
            { { "Type", 0 }, { "Mix", 0.45f }, { "Decay", 1.5f }, { "Gate (min = off)", -30 }, { "Gate Hold", 120 }, { "Gate Release", 60 } });
        s->addPreset("Out of the way of the dry", "The tail dips 9 dB while the source sounds (Dip Under Dry).",
            { { "Type", 1 }, { "Mix", 0.4f }, { "Decay", 2.5f }, { "Dip Under Dry", 9 } });
        s->addPreset("Shimmer", "An octave up fed back into the tail.", { { "Type", 1 }, { "Mix", 0.4f }, { "Decay", 4 }, { "Shimmer", 0.5f } });
        s->addPreset("Long below, short on top", "A high shelf of the decay: the top dies in 40 % of the time (Decay Band).",
            { { "Type", 1 }, { "Mix", 0.35f }, { "Decay", 3 }, { "Decay Band", 3 }, { "Decay Band Freq", 3000 }, { "Decay Band Time", 0.4f } });
        s->addPreset("Pre-delay of an eighth at 120", "The pre-delay counted in notes.",
            { { "Type", 3 }, { "Mix", 0.3f }, { "Decay", 1.8f }, { "Tempo", 120 }, { "Pre-Delay Note", 4 } });
        s->addPreset("Hall, eco quality", "The lighter engine. Compare with Quality: Full.",
            { { "Type", 1 }, { "Quality", 1 }, { "Mix", 0.35f }, { "Decay", 2.5f } });
        s->addPreset("Held tail", "Play something, then switch Hold Tail on: the tail stays.",
            { { "Type", 1 }, { "Mix", 0.5f }, { "Decay", 4 } });
    }

    if (EffectSlot* s = findSlot(all, "Convolution Reverb"))
    {
        s->addPreset("30 % wet", "The response as it is.", { { "Mix", 0.3f } });
        s->addPreset("Half the decay", "The same response, dying twice as fast (Decay Scale).", { { "Mix", 0.3f }, { "Decay Scale", 0.5f } });
        s->addPreset("Stretched", "The response played slower: a larger, darker space.", { { "Mix", 0.3f }, { "Stretch", 1.6f } });
        s->addPreset("Gated", "The tail is cut under -35 dB.", { { "Mix", 0.4f }, { "Gate (min = off)", -35 } });
    }

    if (EffectSlot* s = findSlot(all, "Panner"))
    {
        s->addPreset("Hard left", "Equal power, all the way.", { { "Pan", -1 } });
        s->addPreset("Binaural, to the right", "Level and arrival time between the ears.", { { "Pan", 0.5f }, { "Algorithm", 1 } });
        s->addPreset("Haas, to the right", "The same level in both ears, one of them late.", { { "Pan", 0.6f }, { "Algorithm", 4 } });
        s->addPreset("Spectral, to the left", "The far ear loses its top.", { { "Pan", -0.6f }, { "Algorithm", 5 } });
    }

    if (EffectSlot* s = findSlot(all, "Stereo Width"))
    {
        s->addPreset("Wider", "The sides up by half.", { { "Width", 1.5f } });
        s->addPreset("Mono check", "No sides.", { { "Width", 0 } });
        s->addPreset("Wide, bass in the middle", "Wider, with the lows kept mono.", { { "Width", 1.6f }, { "Bass Mono", 1 } });
    }

    // --- Spectral -----------------------------------------------------------------

    if (EffectSlot* s = findSlot(all, "Denoiser"))
    {
        s->addPreset("Gentle, 12 dB", "Switch Learn Noise on over a passage of noise alone, then off.",
            { { "Reduction", 12 }, { "Detail", 12 }, { "Gain Release", 80 } });
        s->addPreset("Strong, 30 dB", "A deeper reduction read in bands of a sixth, for a steadier residual.",
            { { "Reduction", 30 }, { "Threshold", 3 }, { "Detail", 6 }, { "Gain Release", 150 } });
        s->addPreset("Finest, each bin", "Detail at the top: down to the space between the partials. Turn it left to compare.",
            { { "Reduction", 18 }, { "Detail", 96 } });
        s->addPreset("The detail wide open", "The noise read by the octave: what plays keeps the noise around it.",
            { { "Reduction", 18 }, { "Detail", 1 } });
        s->addPreset("Shorter pre-echo", "Short frames only around attacks: what is spread ahead of them stays within 5 ms.",
            { { "Reduction", 18 }, { "Filter Mode (5 ms at attacks)", 1 }, { "Detail", 96 } });
    }

    if (EffectSlot* s = findSlot(all, "Tonal Balance"))
    {
        s->addPreset("Gentle", "Half way to an even balance, 6 dB at most.", { { "Balance", 0.5f }, { "Range", 6 } });
        s->addPreset("Strong", "All the way, 12 dB at most, with the level kept.",
            { { "Balance", 1 }, { "Range", 12 }, { "Fill Dips", 0.6f }, { "Trim Peaks", 0.8f }, { "Level Compensation", 1 } });
        s->addPreset("Only tame what sticks out", "No filling of dips.", { { "Balance", 0.8f }, { "Fill Dips", 0 }, { "Trim Peaks", 1 } });
        s->addPreset("Even power as the reference", "The reference is a programme with the same power in every band.",
            { { "Balance", 0.8f }, { "Reference", 1 } });
        s->addPreset("Towards a brighter balance", "The target leans 1.5 dB per octave to the top.", { { "Balance", 0.6f }, { "Tilt", 1.5f } });
        s->addPreset("Low end in detail", "Dips and peaks at full with a small knee: the curve follows the lumps and the holes of the low end. Turn Detail left for the broad shape.",
            { { "Balance", 0.5f }, { "Fill Dips", 1 }, { "Trim Peaks", 1 }, { "Contrast Knee", 1 }, { "Range", 9 }, { "Low Freq", 20 }, { "Detail", 12 } });
        s->addPreset("Only the detail", "No broad balance: only what stands out of its two sides is trimmed and what falls under them filled. Move Detail.",
            { { "Balance", 0 }, { "Fill Dips", 0.8f }, { "Trim Peaks", 0.8f }, { "Range", 9 }, { "Low Freq", 20 }, { "Detail", 6 } });
        s->addPreset("Finest detail", "A 24th of an octave: single partials are trimmed and the space between them is left alone.",
            { { "Balance", 0.5f }, { "Fill Dips", 0.8f }, { "Trim Peaks", 0.8f }, { "Low Freq", 20 }, { "Detail", 24 } });
        s->addPreset("The detail wide open", "The same amounts an octave wide: one broad shape.",
            { { "Balance", 0.5f }, { "Fill Dips", 0.8f }, { "Trim Peaks", 0.8f }, { "Low Freq", 20 }, { "Detail", 1 } });
        s->addPreset("One curve for all, time focus", "A common response instead of one per band, and a shorter window at the top.",
            { { "Balance", 0.8f }, { "Response", 1 }, { "Time Focus", 1 } });
    }

    if (EffectSlot* s = findSlot(all, "Resonance Suppressor"))
    {
        s->addPreset("Polish", "A light hand: 6 dB at most.", { { "Strength", 0.6f }, { "Range", 6 } });
        s->addPreset("Harsh voice", "A lower threshold and more range. Hear Removed plays what it takes.",
            { { "Threshold", 3 }, { "Range", 15 }, { "Width", 0.25f } });
        s->addPreset("Surgical", "The finest detail: narrow cuts, down into the low end (which then follows later).",
            { { "Detail", 24 }, { "Range", 15 }, { "Threshold", 4 } });
        s->addPreset("Broad", "The detail wide open: only what is wide is found, and the cuts are wide.",
            { { "Detail", 2 }, { "Threshold", 3 } });
        s->addPreset("Deep mode", "More detail and stronger: cuts up to half as wide and half as deep again, less under 150 Hz.", { { "Mode", 1 } });
        s->addPreset("Boomy low end", "The sub guard open: a lone low line (hum, boom) is cut like any other.", { { "Sub Guard", 0 } });
        s->addPreset("Knee and slow to act", "A soft threshold, and only what lasts 150 ms is treated.",
            { { "Knee", 6 }, { "Persistence", 150 } });
        s->addPreset("Quiet passages left alone", "Under -50 dBFS it does less and less (Quiet Level).",
            { { "Quiet Level", -50 } });
        s->addPreset("By ear weighting", "The detection weighted as the ear hears.", { { "Ear Weighting", 1 } });
        s->addPreset("Middle and sides", "The pair treated as its middle and its sides.", { { "Stereo", 1 }, { "Link", 0.5f } });
        s->addPreset("Short ring", "Short kernels at the top: a cut there rings for less time after the event, as deep.",
            { { "Threshold", 3 }, { "Short Kernel", 1 } });
    }

    if (EffectSlot* s = findSlot(all, "Spectral Dynamic EQ"))
    {
        s->addPreset("De-harsh at 3 kHz", "A bell that only takes down the partials that stick out.",
            { { "B1 Shape", 1 }, { "B1 Freq", 3000 }, { "B1 Q", 1.5f }, { "B1 Threshold", -30 }, { "B1 Ratio", 4 }, { "B1 Range", -12 } });
        s->addPreset("Air under control", "A high shelf from 8 kHz.",
            { { "B2 Shape", 3 }, { "B2 Freq", 8000 }, { "B2 Threshold", -35 }, { "B2 Ratio", 3 }, { "B2 Range", -9 } });
        s->addPreset("Both bands, finest detail", "The two bands moving a 24th of an octave fine.",
            { { "B1 Shape", 1 }, { "B1 Freq", 3000 }, { "B1 Q", 1.5f }, { "B1 Threshold", -30 }, { "B1 Ratio", 4 }, { "B1 Range", -12 },
              { "B2 Shape", 3 }, { "B2 Freq", 8000 }, { "B2 Threshold", -35 }, { "B2 Ratio", 3 }, { "B2 Range", -9 }, { "Detail", 24 } });
        s->addPreset("Both bands, time focus", "The two bands with a shorter window at the top, read where it is played.",
            { { "B1 Shape", 1 }, { "B1 Freq", 3000 }, { "B1 Q", 1.5f }, { "B1 Threshold", -30 }, { "B1 Ratio", 4 }, { "B1 Range", -12 },
              { "B2 Shape", 3 }, { "B2 Freq", 8000 }, { "B2 Threshold", -35 }, { "B2 Ratio", 3 }, { "B2 Range", -9 }, { "Time Focus", 1 } });
        s->addPreset("Low end, line by line", "A wide bell over the bass that takes the notes that jump out, one by one.",
            { { "B1 Shape", 1 }, { "B1 Freq", 90 }, { "B1 Q", 0.5f }, { "B1 Threshold", -14 }, { "B1 Ratio", 6 }, { "B1 Range", -9 }, { "Detail", 12 } });
        s->addPreset("The detail wide open", "The same bell moving as one broad shape.",
            { { "B1 Shape", 1 }, { "B1 Freq", 3000 }, { "B1 Q", 0.7f }, { "B1 Threshold", -20 }, { "B1 Ratio", 4 }, { "B1 Range", -12 }, { "Detail", 1 } });
        s->addPreset("Presence lifted", "A bell at 4 kHz that raises what passes the threshold.",
            { { "B1 Shape", 1 }, { "B1 Freq", 4000 }, { "B1 Q", 1 }, { "B1 Threshold", -40 }, { "B1 Ratio", 2 }, { "B1 Range", 6 } });
    }
}

} // namespace dsplab
