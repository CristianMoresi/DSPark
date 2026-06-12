# DSPark WebGain — a WebView editor developed as separate web files

[`plugin_webview_editor`](../plugin_webview_editor/) keeps its whole UI in
one C++ raw string — great for a single-file example, not for real UI work.
This example shows the **production workflow**: the interface lives in
ordinary web files,

```
ui/editor.html      structure (plain <link>/<script> references)
ui/editor.css       styling
ui/editor.js        behaviour (the dspark bridge calls)
```

and the build embeds them automatically. `CMakeLists.txt` is the whole
recipe:

```cmake
dspark_add_plugin(DSParkWebGain
    SOURCES         webgain.cpp
    FORMATS         VST3 CLAP AU
    EDITOR_HTML     ui/editor.html      # <- inlines css/js, generates the header
    AU_SUBTYPE      DSwg
    AU_MANUFACTURER DSpk
)
```

`EDITOR_HTML` inlines the stylesheets/scripts the page references and
generates `DSParkWebGain_editor_html.h` (re-run automatically whenever any
ui/ file changes), which the plugin source includes:

```cpp
#include "DSParkWebGain_editor_html.h"   // defines kDsparkEditorHtml
static const char* editorHtml() { return kDsparkEditorHtml; }
```

## Build

```
cmake -S . -B build
cmake --build build --config Release
```

Artifacts in `build/`: `DSParkWebGain.vst3` (bundle), `DSParkWebGain.clap`
and, on macOS, `DSParkWebGain.component` (validate with
`auval -v aufx DSwg DSpk`).

While designing the UI, add `editorDevFile()` pointing at `ui/editor.html`
to reload it from disk on every editor open — no rebuild at all (inline the
css/js references manually or keep styles in the page while iterating).
