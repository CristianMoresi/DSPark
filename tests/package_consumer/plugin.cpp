// DSPark - installed-package consumer: a plugin built by dspark_add_plugin()
// against the installed plugin layer (format wrappers + vendored SDKs).

#include "plugin/vst3/DSParkVst3.h"
#include "plugin/clap/DSParkClap.h"

#include "Effects/Gain.h"

struct ConsumerPlugin
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark Package Consumer",
        .vendor    = "DSPark",
        .productId = "com.dspark.test.consumer",
        .version   = "2.1.3",
    };

    static constexpr const char* kModes[] = { "Quiet", "Loud" };

    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("gain", "Gain", -24.0f, 24.0f, 0.0f, "dB"),
        dspark::plugin::choice("mode", "Mode", kModes, 0));

    void prepare(const dspark::AudioSpec& spec) { gain_.prepare(spec); }

    void setParameter(int index, float value) noexcept
    {
        if (index == 0) gain_.setGainDb(value);
    }

    void processBlock(dspark::AudioBufferView<float> io) noexcept { gain_.processBlock(io); }

private:
    dspark::Gain<float> gain_;
};

DSPARK_VST3_PLUGIN(ConsumerPlugin)
DSPARK_CLAP_PLUGIN(ConsumerPlugin)
