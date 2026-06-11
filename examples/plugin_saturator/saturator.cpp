// DSPark example — a complete VST3 saturator plugin in one file, no JUCE.
//
// Build (Windows):
//   cl /std:c++20 /O2 /LD /EHsc /I ..\.. saturator.cpp /Fe:DSParkSaturator.vst3
// Build (Linux):
//   g++ -std=c++20 -O2 -fPIC -shared -I ../.. saturator.cpp -o DSParkSaturator.vst3
// Build (macOS):
//   clang++ -std=c++20 -O2 -fPIC -shared -I ../.. saturator.cpp -o DSParkSaturator
//   (then place inside DSParkSaturator.vst3/Contents/MacOS/ bundle)
//
// Or use the dspark_add_plugin() CMake helper, which builds the proper
// bundle folder layout on every platform.

#include "../../plugin/vst3/DSParkVst3.h"

#include "../../Effects/Saturation.h"
#include "../../Effects/Gain.h"

struct DSParkSaturator
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark Saturator",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.examples.saturator",
        .version   = "1.0.0",
        .category  = dspark::plugin::Category::Fx,
    };

    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("drive",  "Drive",  -12.0f, 36.0f, 0.0f, "dB"),
        dspark::plugin::param("algo",   "Algorithm", 0.0f, 9.0f, 3.0f, ""),  // Saturation::Algorithm
        dspark::plugin::param("mix",    "Mix",      0.0f,  1.0f, 1.0f, ""),
        dspark::plugin::param("output", "Output", -24.0f, 12.0f, 0.0f, "dB"));

    void prepare(const dspark::AudioSpec& spec)
    {
        saturation_.prepare(spec);
        gain_.prepare(spec);
    }

    void setParameter(int index, float value) noexcept
    {
        switch (index)
        {
        case 0: saturation_.setDrive(value); break;
        case 1: saturation_.setAlgorithm(static_cast<dspark::Saturation<float>::Algorithm>(
                    static_cast<int>(value + 0.5f))); break;
        case 2: saturation_.setMix(value); break;
        case 3: gain_.setGainDb(value); break;
        default: break;
        }
    }

    void processBlock(dspark::AudioBufferView<float> io) noexcept
    {
        saturation_.processBlock(io);
        gain_.processBlock(io);
    }

    [[nodiscard]] int getLatency() const noexcept
    {
        return saturation_.getLatencySamples();
    }

private:
    dspark::Saturation<float> saturation_;
    dspark::Gain<float>       gain_;
};

DSPARK_VST3_PLUGIN(DSParkSaturator)
