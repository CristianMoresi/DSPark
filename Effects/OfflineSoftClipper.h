// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineSoftClipper.h
 * @brief Automatic whole-file soft clipping with measured sample-peak reduction.
 */
#include "../Core/detail/OfflineClip.h"
#if DSPARK_HAS_OFFLINE
namespace dspark
{
/**
 * @brief OFFLINE ONLY: calibrates a soft curve to reduce the complete source peak.
 *
 * Options::reductionDb is positive 0..12 dB; zero preserves exact PCM. Sine is
 * the default curve; tanh and golden-ratio curves reuse Clipper's shared shapes.
 * Options::oversamplingFactor is explicitly selectable (1/2/4/8/16, default 4).
 * At factor 1 the clipping residual of a 32-sample windowed-sinc
 * reconstruction is integrated over each sample interval and band-limited by
 * a continuous kernel flat to 0.40 fs and below -65 dB from 0.47 fs on all of
 * its images, so distortion folded across Nyquist is removed wherever it
 * lands: a 10 kHz sine clipped by 6 dB at 48 kHz measures a -97 dB alias floor
 * with the Sine curve and -99 dB with Tanh (both were -25 to -27 dB).
 * The continuous-interval correction uses Core filters and convolution. Its
 * internal delay is fully removed: the result has the source length and clock.
 * Mono/stereo channels share one ceiling. Exclusions preserve exact samples
 * with external 5 ms feathers. If these prevent the target, rendering fails
 * without publishing output. No output gain trim or extra limiter is applied.
 *
 * Threading: synchronous worker only. Sources stay immutable during every pass.
 * Plans own their exclusions and fingerprint; const access is safe while alive
 * and unmoved. This class has no audio-callback processing API.
 */
template <FloatType T> class OfflineSoftClipper final : public detail::OfflineClipProcessor<T, true>
{
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
