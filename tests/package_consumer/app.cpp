// DSPark - installed-package consumer: the umbrella header and a processor
// build and run from the installed include/dspark tree.

#include "DSPark.h"

#include <cmath>
#include <cstdio>

int main()
{
    dspark::Gain<float> gain;
    gain.prepare(dspark::AudioSpec { 48000.0, 64, 1 });
    gain.setGainDb(-6.0f);

    dspark::AudioBuffer<float> buffer;
    buffer.resize(1, 64);
    for (int i = 0; i < 64; ++i) buffer.getChannel(0)[i] = 1.0f;
    for (int block = 0; block < 64; ++block)   // let the smoother settle
    {
        for (int i = 0; i < 64; ++i) buffer.getChannel(0)[i] = 1.0f;
        gain.processBlock(buffer.toView());
    }
    const float out = buffer.getChannel(0)[63];
    std::printf("installed DSPark: -6 dB gain -> %.4f\n", static_cast<double>(out));
    return std::fabs(out - 0.5012f) < 1e-3f ? 0 : 1;
}
