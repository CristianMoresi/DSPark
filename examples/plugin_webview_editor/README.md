# DSPark WebSaturator — a plugin with a custom WebView editor

The same one-file pattern as [`plugin_saturator`](../plugin_saturator/), plus
a **custom GUI written in plain HTML/CSS/JS** — embedded in the host window
by DSPark's WebView editor layer (`plugin/webview/DSParkWebViewEditor.h`).
No GUI framework, no resource files: the page is a string in the same .cpp.

What it demonstrates:

| Piece | Where |
|---|---|
| Declaring an editor | `hasEditor` / `editorSize` / `editorResize` (Fixed, Free or KeepAspect) / `editorHtml()` |
| Initial sync | `dspark.onReady(params)` — parameter table + current values |
| UI -> DSP | `dspark.setParam(id, plainValue)` from drag / wheel / select |
| Automation gestures | `dspark.beginEdit` / `dspark.endEdit` around drags (host undo) |
| DSP/automation -> UI | `dspark.onParam(id, cb)` — knobs follow the host |

## Build

```
Windows:  cl /std:c++20 /O2 /LD /EHsc /I ..\.. webview_saturator.cpp /Fe:DSParkWebSaturator.vst3
macOS:    clang++ -std=c++20 -O2 -fPIC -shared -lobjc -I ../.. webview_saturator.cpp -o DSParkWebSaturator
Linux:    g++ -std=c++20 -O2 -fPIC -shared -I ../.. webview_saturator.cpp -o DSParkWebSaturator.vst3
```

One binary is both formats: install it as `.vst3` and as `.clap`.
On Linux the plugin builds and runs with the host's generic editor (the
WebView embedding path is not wired there yet).

While iterating on a UI, set `editorDebug = true` to get the browser
DevTools (right-click → Inspect in the editor).
