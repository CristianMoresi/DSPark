// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineHardClipper.h
 * @brief Automatic whole-file hard clipping with measured sample-peak reduction.
 */
#include "../Core/detail/OfflineClip.h"
#if DSPARK_HAS_OFFLINE
namespace dspark
{
/**
 * @brief OFFLINE ONLY: calibrates an antialiased hard curve on the complete source.
 *
 * Uses the same analysis, numerical integration, calibration and transactional
 * renderer as OfflineSoftClipper. Options::reductionDb is positive 0..12 dB;
 * zero preserves exact PCM. Oversampling is explicit (1/2/4/8/16, default 4).
 * At factor 1 the clipping correction is band-limited: flat to 0.40 fs and
 * below -65 dB from 0.46 fs, so distortion folded from just above Nyquist is
 * removed (a 5 kHz sine clipped by 6 dB at 48 kHz: -62 dB alias floor, was
 * -27 dB); distortion above 0.40 fs is attenuated with it. Aliases landing
 * below 0.40 fs remain: a 10 kHz sine at 48 kHz still leaves -24 dB at
 * factor 1 and -87 dB at factor 2. Higher factors are unchanged.
 * A shared ceiling links mono/stereo channels. Internal latency is compensated.
 * Exclusions preserve exact PCM with external 5 ms feathers. An unreachable
 * target returns a failure and never publishes provisional output. Peak refers
 * to the final samples; true peak is measured separately, without extra limiting.
 *
 * Threading: synchronous worker only; source and plan remain immutable. Owned
 * plans may be reused on the same source. No processSample/processBlock API.
 */
template <FloatType T>
class OfflineHardClipper final : public detail::OfflineClipProcessor<T, false>
{
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
