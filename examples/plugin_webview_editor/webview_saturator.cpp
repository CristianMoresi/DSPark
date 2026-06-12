// DSPark example — a VST3/CLAP saturator with a custom WebView editor.
//
// The GUI is plain HTML/CSS/JS embedded in this file and served through
// DSPark's WebView editor layer: one UI codebase for every format, rendered
// by the platform web engine (WebView2 on Windows, WKWebView on macOS; on
// Linux the host's generic editor is used until that path lands). The page
// talks to the DSP through the injected `dspark` bridge — same stable text
// ids as state and automation.
//
// Build (Windows):
//   cl /std:c++20 /O2 /LD /EHsc /I ..\.. webview_saturator.cpp /Fe:DSParkWebSaturator.vst3
// Build (macOS — -lobjc for the WKWebView glue):
//   clang++ -std=c++20 -O2 -fPIC -shared -lobjc -I ../.. webview_saturator.cpp -o DSParkWebSaturator
// Build (Linux — compiles and runs with the generic host UI):
//   g++ -std=c++20 -O2 -fPIC -shared -I ../.. webview_saturator.cpp -o DSParkWebSaturator.vst3
//
// Rename/copy the same binary to .clap for the CLAP build, or use
// dspark_add_plugin(... FORMATS VST3 CLAP) which does both bundles.

#include "../../plugin/webview/DSParkWebViewEditor.h"   // FIRST: enables the editor
#include "../../plugin/vst3/DSParkVst3.h"
#include "../../plugin/clap/DSParkClap.h"

#include "../../Effects/Saturation.h"
#include "../../Effects/Gain.h"

struct DSParkWebSaturator
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark WebSaturator",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.examples.websaturator",
        .version   = "1.0.0",
        .category  = dspark::plugin::Category::Fx,
    };

    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("drive",  "Drive",  -12.0f, 36.0f, 0.0f, "dB"),
        dspark::plugin::Param { "algo", "Algorithm", 0.0f, 9.0f, 0.0f, "", 9 },
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

    // --- the editor ---------------------------------------------------------------

    static constexpr bool hasEditor = true;
    static constexpr dspark::plugin::EditorSize editorSize { 560, 330 };
    static constexpr bool editorResizable = true;
    // static constexpr bool editorDebug = true;   // uncomment for DevTools (F12)

    static const char* editorHtml();

private:
    dspark::Saturation<float> saturation_;
    dspark::Gain<float>       gain_;
};

// The page below shows the whole bridge surface:
//   dspark.onReady(params)  — parameter table + current values (handshake)
//   dspark.onParam(id, cb)  — DSP/automation -> UI (knobs follow the host)
//   dspark.setParam(id, v)  — UI -> DSP, plain values
//   dspark.beginEdit/endEdit — automation gestures around drags (host undo)
// Knobs: vertical drag (Shift = fine), mouse wheel, double-click = default.
const char* DSParkWebSaturator::editorHtml()
{
    return R"dsphtml(<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root { --bg0:#16191f; --bg1:#0c0e12; --panel:#1d2129; --line:#2a2f3a;
          --text:#cfd5e1; --dim:#7d8596; --accent:#ffb454; --accent2:#ff8a3d; }
  * { margin:0; padding:0; box-sizing:border-box; user-select:none;
      -webkit-user-select:none; cursor:default; }
  html,body { width:100%; height:100%; overflow:hidden; }
  body { background:radial-gradient(120% 140% at 50% 0%, var(--bg0), var(--bg1));
         color:var(--text); display:flex; flex-direction:column;
         font:13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; align-items:baseline; gap:10px;
           padding:14px 20px 10px; border-bottom:1px solid var(--line); }
  header h1 { font-size:17px; font-weight:650; letter-spacing:.4px; }
  header .badge { font-size:10px; color:var(--dim); letter-spacing:1.2px;
                  text-transform:uppercase; }
  header .right { margin-left:auto; font-size:10px; color:var(--dim); }
  main { flex:1; display:flex; align-items:center; justify-content:center;
         gap:34px; padding:10px 24px 4px; flex-wrap:wrap; }
  .knob { display:flex; flex-direction:column; align-items:center; gap:2px;
          cursor:ns-resize; touch-action:none; }
  .knob svg { width:92px; height:92px; display:block; }
  .knob .track { fill:none; stroke:var(--line); stroke-width:7; stroke-linecap:round; }
  .knob .val { fill:none; stroke:var(--accent); stroke-width:7; stroke-linecap:round;
               filter:drop-shadow(0 0 5px rgba(255,164,84,.35)); }
  .knob .ptr { stroke:var(--text); stroke-width:3.4; stroke-linecap:round; }
  .knob label { font-size:11px; letter-spacing:1.4px; text-transform:uppercase;
                color:var(--dim); }
  .knob output { font-size:12.5px; color:var(--accent);
                 font-variant-numeric:tabular-nums; }
  .algo { display:flex; flex-direction:column; align-items:center; gap:7px; }
  .algo label { font-size:11px; letter-spacing:1.4px; text-transform:uppercase;
                color:var(--dim); }
  .algo select { background:var(--panel); color:var(--text); cursor:pointer;
                 border:1px solid var(--line); border-radius:7px;
                 padding:7px 12px; font:inherit; outline:none; }
  footer { padding:6px 20px 10px; font-size:10px; color:var(--dim);
           text-align:center; }
</style></head><body>
<header>
  <h1>WebSaturator</h1><span class="badge">DSPark &middot; WebView editor</span>
  <span class="right">drag &middot; shift = fine &middot; double-click = reset</span>
</header>
<main>
  <div class="knob" data-param="drive">
    <svg viewBox="0 0 100 100">
      <path class="track" d="M 21.7 78.3 A 40 40 0 1 1 78.3 78.3"/>
      <path class="val"   d="M 21.7 78.3 A 40 40 0 1 1 78.3 78.3"/>
      <line class="ptr" x1="50" y1="50" x2="50" y2="17"/>
    </svg>
    <label>Drive</label><output>&ndash;</output>
  </div>
  <div class="knob" data-param="mix">
    <svg viewBox="0 0 100 100">
      <path class="track" d="M 21.7 78.3 A 40 40 0 1 1 78.3 78.3"/>
      <path class="val"   d="M 21.7 78.3 A 40 40 0 1 1 78.3 78.3"/>
      <line class="ptr" x1="50" y1="50" x2="50" y2="17"/>
    </svg>
    <label>Mix</label><output>&ndash;</output>
  </div>
  <div class="knob" data-param="output">
    <svg viewBox="0 0 100 100">
      <path class="track" d="M 21.7 78.3 A 40 40 0 1 1 78.3 78.3"/>
      <path class="val"   d="M 21.7 78.3 A 40 40 0 1 1 78.3 78.3"/>
      <line class="ptr" x1="50" y1="50" x2="50" y2="17"/>
    </svg>
    <label>Output</label><output>&ndash;</output>
  </div>
  <div class="algo">
    <label>Algorithm</label>
    <select id="algo">
      <option value="0">Tube</option><option value="1">Tape</option>
      <option value="2">Transformer</option><option value="3">Soft Clip</option>
      <option value="4">Hard Clip</option><option value="5">Exciter</option>
      <option value="6">Wavefolder</option><option value="7">Bitcrusher</option>
      <option value="8">Downsample</option><option value="9">Multi-Stage</option>
    </select>
  </div>
</main>
<footer>same parameter ids as automation &amp; state &mdash; presets stay portable</footer>
<script>
(function () {
  'use strict';
  var ARC = 188.5;   // length of the 270-degree knob arc

  function initKnob(el, p) {
    var val = el.querySelector('.val');
    var ptr = el.querySelector('.ptr');
    var out = el.querySelector('output');
    val.setAttribute('stroke-dasharray', ARC);

    function clamp(v) { return Math.min(p.max, Math.max(p.min, v)); }
    function paint(v) {
      var f = (v - p.min) / (p.max - p.min);
      val.setAttribute('stroke-dashoffset', ARC * (1 - f));
      ptr.setAttribute('transform', 'rotate(' + (-135 + 270 * f) + ' 50 50)');
      out.textContent = v.toFixed(p.max - p.min > 4 ? 1 : 2)
                      + (p.unit ? ' ' + p.unit : '');
    }
    dspark.onParam(p.id, paint);   // covers host automation and our own edits

    var startY = 0, startV = 0, dragging = false;
    el.addEventListener('pointerdown', function (e) {
      dragging = true; startY = e.clientY; startV = dspark.getParam(p.id);
      el.setPointerCapture(e.pointerId);
      dspark.beginEdit(p.id);
      e.preventDefault();
    });
    el.addEventListener('pointermove', function (e) {
      if (!dragging) { return; }
      var pixelsForFullRange = e.shiftKey ? 1600 : 190;
      dspark.setParam(p.id, clamp(
        startV + (startY - e.clientY) * (p.max - p.min) / pixelsForFullRange));
    });
    el.addEventListener('pointerup', function (e) {
      if (!dragging) { return; }
      dragging = false;
      dspark.endEdit(p.id);
      el.releasePointerCapture(e.pointerId);
    });
    el.addEventListener('dblclick', function () { dspark.setParam(p.id, p.def); });
    el.addEventListener('wheel', function (e) {
      e.preventDefault();
      var step = (p.max - p.min) / (e.shiftKey ? 400 : 80);
      dspark.setParam(p.id, clamp(dspark.getParam(p.id) - Math.sign(e.deltaY) * step));
    }, { passive: false });
  }

  function initSelect(sel, p) {
    dspark.onParam(p.id, function (v) { sel.value = String(Math.round(v)); });
    sel.addEventListener('change', function () { dspark.setParam(p.id, +sel.value); });
  }

  dspark.onReady(function (params) {
    var byId = {};
    for (var i = 0; i < params.length; i++) { byId[params[i].id] = params[i]; }
    var knobs = document.querySelectorAll('.knob');
    for (var k = 0; k < knobs.length; k++) {
      initKnob(knobs[k], byId[knobs[k].getAttribute('data-param')]);
    }
    initSelect(document.getElementById('algo'), byId.algo);
  });
})();
</script>
</body></html>)dsphtml";
}

// One translation unit, one binary, two formats — now with a custom editor.
DSPARK_VST3_PLUGIN(DSParkWebSaturator)
DSPARK_CLAP_PLUGIN(DSParkWebSaturator)
