// DSPark example — pitch-tracking EQ (low-cut that follows the voice)
// Copyright (c) 2026 Cristian Moresi — MIT License
//
// A high-pass filter whose cutoff rides just below the tonic of the incoming
// voice: low frequencies are cleaned as aggressively as possible without ever
// thinning the fundamental.
//
//   PitchFollower (gated, octave-safe, glided)  ->  FilterEngine high-pass
//
// PitchFollower wraps the raw PitchDetector with everything this use case
// needs: readings outside the range or below the confidence threshold never
// move the cutoff (consonants and silence hold the last pitch), octave errors
// are folded back, and the output glides in semitones — so the filter moves
// musically, with no zipper (FilterEngine::setFrequency also smooths
// internally and is thread-safe).
//
// Input is a synthetic singing voice (vibrato + glide + noise consonants) so
// the example runs standalone; feed your real input in place of the synth.
//
// Build:  g++ -std=c++20 -O2 -I .. pitch_tracking_eq.cpp -o pitch_tracking_eq

#include "DSPark.h"

#include <cmath>
#include <cstdio>

int main()
{
    constexpr double kRate  = 48000.0;
    constexpr int    kBlock = 512;
    const dspark::AudioSpec spec { kRate, kBlock, 2 };

    dspark::PitchFollower<float> follower;
    follower.prepare(spec);
    follower.setRange(70.0f, 800.0f);    // vocal fundamentals
    follower.setConfidence(0.85f);
    follower.setGlide(60.0f);            // ms per octave

    dspark::FilterEngine<float> lowCut;
    lowCut.prepare(spec);
    lowCut.setHighPass(70.0f, 0.707f, 12);   // resting position

    dspark::AudioBuffer<float> buf;
    buf.resize(2, kBlock);

    const int totalBlocks = static_cast<int>(6.0 * kRate / kBlock);
    double phase = 0.0;
    uint32_t rng = 0x2026u;

    for (int b = 0; b < totalBlocks; ++b)
    {
        // --- synthetic voice: A2->A3 glide with vibrato, noise consonants ---
        const double t = b * kBlock / kRate;
        const bool consonant = std::fmod(t, 0.9) < 0.07;
        const double glide = std::clamp(t / 4.0, 0.0, 1.0);
        const double f0 = 110.0 * std::pow(2.0, glide)
                        * (1.0 + 0.02 * std::sin(2.0 * 3.14159265358979 * 5.0 * t));

        float* L = buf.getChannel(0);
        float* R = buf.getChannel(1);
        for (int i = 0; i < kBlock; ++i)
        {
            phase += f0 / kRate;
            if (phase >= 1.0) phase -= 1.0;
            float v = 0.4f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * phase));
            if (consonant)
            {
                rng = rng * 1664525u + 1013904223u;
                v = 0.1f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
            }
            L[i] = v;
            R[i] = v;
        }

        // --- track and ride the low-cut ---------------------------------------
        follower.processBlock(buf.toView());

        const float tracked = follower.getSmoothedHz();
        if (tracked > 0.0f)
            lowCut.setFrequency(tracked * 0.9f);   // sit just under the tonic
        lowCut.processBlock(buf.toView());

        if (b % 24 == 0)
            std::printf("t %4.1f s   tracked %6.1f Hz (%s)   low-cut @ %6.1f Hz\n",
                        t, static_cast<double>(tracked),
                        follower.isTracking() ? "tracking" : "holding ",
                        static_cast<double>(tracked > 0.0f ? tracked * 0.9f : 70.0f));
    }
    return 0;
}
