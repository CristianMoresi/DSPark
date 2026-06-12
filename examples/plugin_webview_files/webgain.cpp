// DSPark example — WebView editor developed as SEPARATE web files.
//
// The interface lives in ui/editor.html + ui/editor.css + ui/editor.js,
// like any web project. At build time dspark_add_plugin(EDITOR_HTML ...)
// inlines them into one generated header (see CMakeLists.txt), so the
// shipped binary stays fully self-contained. Edit a ui/ file and rebuild —
// or point editorDevFile() at the page to skip even the rebuild while
// iterating (see docs/plugins.md).
//
// Build:
//   cmake -S . -B build && cmake --build build --config Release
// Artifacts: build/DSParkWebGain.vst3, build/DSParkWebGain.clap and, on
// macOS, build/DSParkWebGain.component (auval: -v aufx DSwg DSpk).

#include "../../plugin/webview/DSParkWebViewEditor.h"   // FIRST: enables the editor
#include "../../plugin/vst3/DSParkVst3.h"
#include "../../plugin/clap/DSParkClap.h"
#include "../../plugin/au/DSParkAu.h"                   // self-disables off macOS

#include "../../Effects/Gain.h"

#include "DSParkWebGain_editor_html.h"   // generated: kDsparkEditorHtml

struct DSParkWebGain
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark WebGain",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.examples.webgain",
        .version   = "1.0.0",
        .category  = dspark::plugin::Category::Fx,
    };

    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("gain", "Gain", -24.0f, 24.0f, 0.0f, "dB"));

    void prepare(const dspark::AudioSpec& spec) { gain_.prepare(spec); }

    void setParameter(int index, float value) noexcept
    {
        if (index == 0) gain_.setGainDb(value);
    }

    void processBlock(dspark::AudioBufferView<float> io) noexcept
    {
        gain_.processBlock(io);
    }

    static constexpr bool hasEditor = true;
    static constexpr dspark::plugin::EditorSize editorSize { 300, 360 };
    static constexpr auto editorResize = dspark::plugin::EditorResize::KeepAspect;

    static const char* editorHtml() { return kDsparkEditorHtml; }

private:
    dspark::Gain<float> gain_;
};

DSPARK_VST3_PLUGIN(DSParkWebGain)
DSPARK_CLAP_PLUGIN(DSParkWebGain)
DSPARK_AU_PLUGIN(DSParkWebGain, "DSwg", "DSpk")
