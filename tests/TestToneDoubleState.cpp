// DSPark Tests - double-precision state round trips (tone and modulation)
//
// getState() of the double instantiations below did not compile: the blob
// writer has float/int32/bool overloads, and a double matched none of them
// unambiguously (MSVC C2668). Kept in its own translation unit because the
// defect is a compile error.

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Effects/Chorus.h"
#include "../Effects/Delay.h"
#include "../Effects/NoiseGenerator.h"
#include "../Effects/Panner.h"
#include "../Effects/Phaser.h"
#include "../Effects/StereoWidth.h"
#include "../Effects/Tremolo.h"

#include <cmath>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {
template <class P, class Set, class Check>
bool roundTripDouble(Set set, Check check)
{
    P a, b;
    a.prepare(spec(48000.0, 512, 2));
    b.prepare(spec(48000.0, 512, 2));
    set(a);
    const std::vector<uint8_t> blob = a.getState();
    return b.setState(blob.data(), blob.size()) && check(b);
}
} // namespace

DSPARK_TEST(Tone_double_instances_serialize)
{
    EXPECT_TRUE((roundTripDouble<Delay<double>>(
        [](auto& p) { p.setDelayMs(125.0); p.setFeedback(0.4); },
        [](auto& p) { return std::abs(p.getCurrentDelaySamples() - 6000.0) < 1e-2; })));
    EXPECT_TRUE((roundTripDouble<NoiseGenerator<double>>(
        [](auto& p) { p.setGain(0.25); },
        [](auto& p) { return std::abs(p.getGain() - 0.25) < 1e-6; })));
    EXPECT_TRUE((roundTripDouble<Chorus<double>>(
        [](auto& p) { p.setMix(0.75); },
        [](auto& p) { return !p.getState().empty(); })));
    EXPECT_TRUE((roundTripDouble<Phaser<double>>(
        [](auto& p) { p.setMix(0.75); },
        [](auto& p) { return !p.getState().empty(); })));
    EXPECT_TRUE((roundTripDouble<Tremolo<double>>(
        [](auto& p) { p.setDepth(0.6); },
        [](auto& p) { return !p.getState().empty(); })));
    EXPECT_TRUE((roundTripDouble<StereoWidth<double>>(
        [](auto& p) { p.setWidth(1.5); },
        [](auto& p) { return !p.getState().empty(); })));
    EXPECT_TRUE((roundTripDouble<Panner<double>>(
        [](auto& p) { p.setPan(0.5); },
        [](auto& p) { return !p.getState().empty(); })));
}
