// DSPark example — a sidechain ducker: the classic "music dips when the
// voice speaks" compressor, and the reference for the sidechain contract.
//
// Implementing the TWO-BUFFER processBlock is all it takes: every format
// backend then grows a host-routable "Sidechain" input (a VST3 aux bus, a
// CLAP non-main port, an AU input element). Route a voice or kick track
// into it in your DAW and the compressor's detector follows THAT signal
// while the gain reduction lands on the main audio. When the host has
// nothing routed, the wrapper hands the plugin silence — no branches here.
//
// Build (Windows):
//   cl /std:c++20 /O2 /LD /EHsc /I ..\.. ducker.cpp /Fe:DSParkDucker.vst3
// Build (Linux):
//   g++ -std=c++20 -O2 -fPIC -shared -I ../.. ducker.cpp -o DSParkDucker.vst3
// Build (macOS):
//   clang++ -std=c++20 -O2 -fPIC -shared -I ../.. ducker.cpp -o DSParkDucker
//   (AU: compile with -bundle -framework AudioToolbox -framework CoreFoundation
//    and assemble the .component with au/Info.plist, as in CI)
//
// Copy the same binary to .clap for the CLAP build.

#include "../../plugin/vst3/DSParkVst3.h"
#include "../../plugin/clap/DSParkClap.h"
#include "../../plugin/au/DSParkAu.h"

#include "../../Effects/Compressor.h"

struct DSParkDucker
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark Ducker",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.examples.ducker",
        .version   = "1.0.0",
        .category  = dspark::plugin::Category::Fx,
    };

    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("threshold", "Threshold", -60.0f,    0.0f,  -30.0f, "dB"),
        dspark::plugin::param("ratio",     "Ratio",       1.0f,   20.0f,    8.0f, ":1"),
        dspark::plugin::param("attack",    "Attack",      0.1f,   50.0f,    5.0f, "ms"),
        dspark::plugin::param("release",   "Release",    10.0f, 1000.0f,  150.0f, "ms"));

    void prepare(const dspark::AudioSpec& spec) { comp_.prepare(spec); }

    void setParameter(int index, float value) noexcept
    {
        switch (index)
        {
        case 0: comp_.setThreshold(value); break;
        case 1: comp_.setRatio(value); break;
        case 2: comp_.setAttack(value); break;
        case 3: comp_.setRelease(value); break;
        default: break;
        }
    }

    // The two-buffer form IS the sidechain opt-in (see plugin/DSParkPlugin.h,
    // HasSidechain): same shape as DSPark's own dynamics, so it forwards 1:1.
    void processBlock(dspark::AudioBufferView<float> io,
                      dspark::AudioBufferView<float> sidechain) noexcept
    {
        comp_.processBlock(io, sidechain);
    }

private:
    dspark::Compressor<float> comp_;
};

DSPARK_VST3_PLUGIN(DSParkDucker)
DSPARK_CLAP_PLUGIN(DSParkDucker)
DSPARK_AU_PLUGIN(DSParkDucker, "DSdk", "DSpk")
