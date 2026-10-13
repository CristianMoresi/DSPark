// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SampleZone.h
 * @brief Description of one sample zone: a recording mapped over keys and
 *        velocities, with its tuning, level, pan and loop.
 *
 * Shared by Sampler (which layers every matching zone) and Synthesizer (whose
 * sample source plays the first matching zone through the voice's filter,
 * envelopes and modulation). The audio is copied when a zone set is built.
 *
 * Threading: a plain value type; the owner that consumes it states its own
 * contract.
 *
 * Dependencies: AudioBuffer.h, DspMath.h.
 */

#include "AudioBuffer.h"
#include "DspMath.h"

#include <cstdint>

namespace dspark {

/** @brief How a zone loops. */
enum class SampleLoopMode
{
    None,        ///< Play once to the end of the recording.
    Continuous,  ///< Loop for as long as the voice sounds, release included.
    Sustain      ///< Loop while the key is held, then play on past the loop end.
};

/** @brief One zone, as the caller describes it. */
template <FloatType T>
struct SampleZoneSpec
{
    AudioBufferView<const T> audio;  ///< The recording (1 or more channels).
    double sampleRate = 48000.0;     ///< Its sample rate.
    int rootKey = 60;                ///< MIDI key that plays it untransposed.
    double tuneCents = 0.0;          ///< Fine tuning added to every note.
    int keyLow = 0;                  ///< Lowest key it answers (inclusive).
    int keyHigh = 127;               ///< Highest key it answers (inclusive).
    double velocityLow = 0.0;        ///< Lowest velocity it answers, 0..1 (inclusive).
    double velocityHigh = 1.0;       ///< Highest velocity it answers, 0..1 (inclusive).
    double gainDb = 0.0;             ///< Zone gain.
    double pan = 0.0;                ///< -1 left .. +1 right (balance for multichannel).
    int64_t startOffset = 0;         ///< First sample played.
    SampleLoopMode loopMode = SampleLoopMode::None;
    int64_t loopStart = 0;           ///< First sample of the loop.
    int64_t loopEnd = 0;             ///< One past the last sample of the loop.
    int64_t loopCrossfade = 0;       ///< Crossfade length in samples (0 = hard loop).
    bool equalPowerCrossfade = false;///< Equal-power fade (uncorrelated material)
                                     ///< instead of linear (phase-matched material).
};

} // namespace dspark
