// DSParkLab — GUI
// ImGui interface: transport, effect list, parameter panel, and an interactive
// plugin-style analyzer (log-frequency spectrum + draggable EQ curve + multiband).

#pragma once

#include "AudioEngine.h"
#include "EffectSlot.h"

#include "vendor/imgui/imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Win32 file dialog
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <commdlg.h>

namespace dsplab {

class Gui
{
public:
    // Polished dark theme with an amber accent (call once after StyleColorsDark()).
    static void applyTheme()
    {
        ImGuiStyle& s = ImGui::GetStyle();
        s.WindowRounding = 6.0f; s.ChildRounding = 6.0f; s.FrameRounding = 4.0f;
        s.GrabRounding = 4.0f; s.PopupRounding = 4.0f; s.ScrollbarRounding = 4.0f;
        s.WindowBorderSize = 0.0f; s.FrameBorderSize = 0.0f; s.ChildBorderSize = 1.0f;
        s.WindowPadding = ImVec2(10, 10); s.FramePadding = ImVec2(8, 4);
        s.ItemSpacing = ImVec2(8, 6); s.ScrollbarSize = 12.0f;

        ImVec4* c = s.Colors;
        const ImVec4 accent(1.00f, 0.62f, 0.20f, 1.00f);
        c[ImGuiCol_WindowBg]         = ImVec4(0.09f, 0.10f, 0.12f, 1.00f);
        c[ImGuiCol_ChildBg]          = ImVec4(0.11f, 0.12f, 0.15f, 1.00f);
        c[ImGuiCol_PopupBg]          = ImVec4(0.12f, 0.13f, 0.16f, 0.98f);
        c[ImGuiCol_Border]           = ImVec4(0.22f, 0.24f, 0.29f, 1.00f);
        c[ImGuiCol_FrameBg]          = ImVec4(0.16f, 0.17f, 0.21f, 1.00f);
        c[ImGuiCol_FrameBgHovered]   = ImVec4(0.22f, 0.24f, 0.29f, 1.00f);
        c[ImGuiCol_FrameBgActive]    = ImVec4(0.26f, 0.28f, 0.34f, 1.00f);
        c[ImGuiCol_TitleBg]          = ImVec4(0.09f, 0.10f, 0.12f, 1.00f);
        c[ImGuiCol_TitleBgActive]    = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
        c[ImGuiCol_Button]           = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        c[ImGuiCol_ButtonHovered]    = ImVec4(0.28f, 0.31f, 0.38f, 1.00f);
        c[ImGuiCol_ButtonActive]     = accent;
        c[ImGuiCol_SliderGrab]       = accent;
        c[ImGuiCol_SliderGrabActive] = ImVec4(1.00f, 0.74f, 0.35f, 1.00f);
        c[ImGuiCol_CheckMark]        = accent;
        c[ImGuiCol_Header]           = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        c[ImGuiCol_HeaderHovered]    = ImVec4(0.28f, 0.31f, 0.38f, 1.00f);
        c[ImGuiCol_HeaderActive]     = ImVec4(1.00f, 0.62f, 0.20f, 0.55f);
        c[ImGuiCol_Separator]        = ImVec4(0.24f, 0.26f, 0.31f, 1.00f);
        c[ImGuiCol_Text]             = ImVec4(0.88f, 0.90f, 0.93f, 1.00f);
        c[ImGuiCol_TextDisabled]     = ImVec4(0.50f, 0.53f, 0.60f, 1.00f);
        c[ImGuiCol_ScrollbarBg]      = ImVec4(0.09f, 0.10f, 0.12f, 0.00f);
        c[ImGuiCol_ScrollbarGrab]    = ImVec4(0.26f, 0.28f, 0.34f, 1.00f);
    }

    void render(AudioEngine& engine, std::vector<EffectSlot>& effects)
    {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("DSParkLab", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        drawTransport(engine);
        ImGui::Separator();

        EffectSlot* sel = nullptr;
        for (auto& e : effects)
            if (e.selected) { sel = &e; break; }

        const float listW  = 230.0f;
        const float paramW = 340.0f;

        // Column 1 — effect list
        ImGui::BeginChild("EffectList", ImVec2(listW, 0), true);
        drawEffectList(effects);
        ImGui::EndChild();

        ImGui::SameLine();

        // Column 2 — analyzer (fills the vertical slack) + meters underneath
        float centerW = ImGui::GetContentRegionAvail().x - paramW - 8.0f;
        if (centerW < 220.0f) centerW = 220.0f;
        ImGui::BeginChild("Center", ImVec2(centerW, 0), false);
        {
            const float meterH = 92.0f;
            float analyzerH = ImGui::GetContentRegionAvail().y - meterH;
            if (analyzerH < 200.0f) analyzerH = 200.0f;
            drawAnalyzer(engine, sel, analyzerH);
            drawMeters(engine);
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Column 3 — parameters for the selected effect. NoScrollWithMouse so the
        // wheel adjusts the hovered control (combo/slider) instead of scrolling the
        // list; use the scrollbar to scroll long parameter sets.
        ImGui::BeginChild("ParamCol", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollWithMouse);
        drawParamPanel(sel);
        ImGui::EndChild();

        ImGui::End();
    }

private:
    int   dragNode_ = -1;       // index of the curve node being dragged (-1 = none)
    void* dragOwner_ = nullptr; // which EffectSlot owns the active drag
    float seekValue_ = 0.0f;    // seek-bar value while the user is dragging it
    bool  seekActive_ = false;  // true while the seek bar is grabbed (defer the seek)
    float thdTimer_ = 0.0f;     // live THD reading refresh accumulator
    AudioEngine::ThdReading thdLast_ {};

    // --- Frequency / dB mapping helpers ---------------------------------------

    static constexpr float kFMin = 20.0f, kFMax = 20000.0f;
    static constexpr float kEqRange = 24.0f;  // EQ curve vertical scale (+/- dB)

    static float freqToX(float f, float x0, float w)
    {
        const float lo = std::log10(kFMin), hi = std::log10(kFMax);
        const float t = (std::log10(std::clamp(f, kFMin, kFMax)) - lo) / (hi - lo);
        return x0 + t * w;
    }
    static float xToFreq(float x, float x0, float w)
    {
        const float lo = std::log10(kFMin), hi = std::log10(kFMax);
        const float t = std::clamp((x - x0) / w, 0.0f, 1.0f);
        return std::pow(10.0f, lo + t * (hi - lo));
    }
    static float dbToYspec(float db, float y0, float h)         // -100..0 dB -> bottom..top
    {
        const float t = std::clamp((db + 100.0f) / 100.0f, 0.0f, 1.0f);
        return y0 + h - t * h;
    }
    static float dbToYeq(float db, float y0, float h)           // +range..-range -> top..bottom
    {
        const float t = std::clamp((db + kEqRange) / (2.0f * kEqRange), 0.0f, 1.0f);
        return y0 + h - t * h;
    }
    static float yToDbEq(float y, float y0, float h)
    {
        const float t = std::clamp((y0 + h - y) / h, 0.0f, 1.0f);
        return -kEqRange + t * 2.0f * kEqRange;
    }

    // --- Transport bar ---------------------------------------------------------

    void drawTransport(AudioEngine& engine)
    {
        const bool loading = engine.isLoading();

        ImGui::BeginDisabled(loading);
        if (ImGui::Button("Open File"))
        {
            char path[MAX_PATH] = {};
            OPENFILENAMEA ofn = {};
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "Audio Files\0*.wav;*.mp3\0All Files\0*.*\0";
            ofn.lpstrFile = path;
            ofn.nMaxFile = MAX_PATH;
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetOpenFileNameA(&ofn))
                engine.loadFile(path);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(loading);
        bool playing = engine.isPlaying();
        if (ImGui::Button(playing ? "Pause" : "Play"))
            engine.togglePlay();
        ImGui::SameLine();
        if (ImGui::Button("Stop"))
            engine.stop();
        ImGui::EndDisabled();

        ImGui::SameLine();
        bool loop = engine.isLooping();
        if (ImGui::Checkbox("Loop", &loop))
            engine.setLooping(loop);

        // Seek bar — only jumps the transport when the mouse is RELEASED, not
        // continuously while dragging (which made the position thrash/loop).
        ImGui::SameLine();
        float playPos = engine.getPosition();
        float shown = seekActive_ ? seekValue_ : playPos;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 120.0f);
        ImGui::SliderFloat("##seek", &shown, 0.0f, 1.0f, "");
        if (ImGui::IsItemActivated()) seekActive_ = true;
        if (seekActive_) seekValue_ = shown;
        if (ImGui::IsItemDeactivatedAfterEdit()) { engine.seekTo(seekValue_); seekActive_ = false; }
        else if (seekActive_ && !ImGui::IsItemActive()) seekActive_ = false;

        ImGui::SameLine();
        float dur = engine.getDurationSeconds();
        float cur = shown * dur;
        char timeBuf[32];
        std::snprintf(timeBuf, sizeof(timeBuf), "%02d:%02d / %02d:%02d",
                      static_cast<int>(cur) / 60, static_cast<int>(cur) % 60,
                      static_cast<int>(dur) / 60, static_cast<int>(dur) % 60);
        ImGui::Text("%s", timeBuf);

        if (loading)
        {
            const int dots = static_cast<int>(ImGui::GetTime() * 2.0) % 4;
            const char* dotStrs[4] = { "", ".", "..", "..." };
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Loading%s", dotStrs[dots]);
            ImGui::SameLine();
        }
        else if (engine.hasFile())
        {
            ImGui::TextDisabled("%.0f Hz  |  %dch", engine.getSampleRate(), engine.getChannels());
            ImGui::SameLine();
        }

        bool bypass = engine.isBypassed();
        if (ImGui::Checkbox("Bypass (A/B)", &bypass))
            engine.setBypass(bypass);
    }

    // --- Effect list (left panel) ----------------------------------------------

    void drawEffectList(std::vector<EffectSlot>& effects)
    {
        ImGui::Text("Effects");
        ImGui::Separator();

        std::string currentCat;
        bool catOpen = true;

        for (int i = 0; i < static_cast<int>(effects.size()); ++i)
        {
            auto& e = effects[i];
            if (e.category != currentCat)
            {
                currentCat = e.category;
                catOpen = ImGui::CollapsingHeader(currentCat.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
            }
            if (!catOpen) continue;

            ImGui::PushID(i);
            ImGui::Checkbox("##en", &e.enabled);
            ImGui::SameLine();
            if (ImGui::Selectable(e.name.c_str(), e.selected))
            {
                for (auto& eff : effects) eff.selected = false;
                e.selected = true;
            }
            ImGui::PopID();
        }
    }

    // --- Interactive analyzer (spectrum + EQ curve + nodes + multiband) --------

    void drawAnalyzer(AudioEngine& engine, EffectSlot* sel, float budgetH)
    {
        const bool hasOverlay = sel && (sel->magnitudeFn || sel->multibandFn);
        ImGui::Text("%s", sel ? sel->name.c_str() : "Analyzer");
        if (hasOverlay) {
            ImGui::SameLine();
            ImGui::TextDisabled("— drag nodes: X freq, Y gain | wheel = Q");
        }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        float h = budgetH - 26.0f;             // leave room for the title row
        if (h < 160.0f) h = 160.0f;
        const ImVec2 p1 = ImVec2(p0.x + w, p0.y + h);

        ImGui::InvisibleButton("##analyzer", ImVec2(w, h));
        const bool canvasHovered = ImGui::IsItemHovered();

        dl->AddRectFilled(p0, p1, IM_COL32(18, 20, 26, 255), 4.0f);
        dl->PushClipRect(p0, p1, true);

        // Frequency grid + labels
        const float gridF[]   = { 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
        const char* gridL[]   = { "30","50","100","200","500","1k","2k","5k","10k","20k" };
        for (int i = 0; i < 10; ++i)
        {
            float x = freqToX(gridF[i], p0.x, w);
            dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y), IM_COL32(42, 46, 56, 255));
            dl->AddText(ImVec2(x + 2, p1.y - 14), IM_COL32(110, 116, 130, 255), gridL[i]);
        }
        // dB grid (EQ scale): +12, 0, -12
        for (int d = -12; d <= 12; d += 12)
        {
            float y = dbToYeq(static_cast<float>(d), p0.y, h);
            dl->AddLine(ImVec2(p0.x, y), ImVec2(p1.x, y),
                        d == 0 ? IM_COL32(70, 76, 90, 255) : IM_COL32(38, 42, 50, 255));
            char b[8]; std::snprintf(b, sizeof(b), "%+d", d);
            dl->AddText(ImVec2(p0.x + 3, y - 13), IM_COL32(110, 116, 130, 255), b);
        }

        // Spectrum (faint filled curve)
        const int bins = engine.getSpectrumBins();
        const float* spec = engine.getSpectrumDb();
        if (bins > 1 && spec)
        {
            float prevX = 0, prevY = 0; bool have = false;
            for (int k = 1; k < bins; ++k)
            {
                const float f = engine.binToFrequency(k);
                if (f < kFMin || f > kFMax) continue;
                const float x = freqToX(f, p0.x, w);
                const float y = dbToYspec(spec[k], p0.y, h);
                if (have)
                {
                    dl->AddQuadFilled(ImVec2(prevX, prevY), ImVec2(x, y),
                                      ImVec2(x, p1.y), ImVec2(prevX, p1.y),
                                      IM_COL32(60, 130, 200, 60));
                    dl->AddLine(ImVec2(prevX, prevY), ImVec2(x, y), IM_COL32(90, 170, 240, 160), 1.5f);
                }
                prevX = x; prevY = y; have = true;
            }
        }

        // EQ / filter response curve + draggable nodes
        if (sel && sel->magnitudeFn)
            drawCurveAndNodes(dl, sel, engine, p0, w, h, canvasHovered);

        // Multiband bands + per-band gain reduction
        if (sel && sel->multibandFn)
            drawMultiband(dl, sel, p0, w, h);

        dl->PopClipRect();
        dl->AddRect(p0, p1, IM_COL32(70, 76, 90, 255), 4.0f);
    }

    void drawCurveAndNodes(ImDrawList* dl, EffectSlot* sel, AudioEngine& engine,
                           ImVec2 p0, float w, float h, bool canvasHovered)
    {
        const double sr = engine.getSampleRate() > 0 ? engine.getSampleRate() : 48000.0;
        // One sample per screen pixel so sharp features (notch nulls, high-Q peaks)
        // render smoothly and don't flicker/alias as the frequency is moved.
        constexpr int kMaxN = 1200;
        const int N = std::clamp(static_cast<int>(w), 64, kMaxN);
        static float freqs[kMaxN], mags[kMaxN];
        static ImVec2 pts[kMaxN];
        for (int i = 0; i < N; ++i)
            freqs[i] = xToFreq(p0.x + (static_cast<float>(i) / (N - 1)) * w, p0.x, w);

        // Faint "target" curve behind (e.g. a dynamic-EQ band's max boost/cut, where
        // the draggable node sits); the live curve below animates toward it.
        if (sel->targetMagnitudeFn)
        {
            static float tmags[kMaxN]; static ImVec2 tpts[kMaxN];
            sel->targetMagnitudeFn(freqs, tmags, N, sr, sel->values.data());
            for (int i = 0; i < N; ++i)
            {
                float dbc = std::clamp(tmags[i], -kEqRange, kEqRange);
                tpts[i] = ImVec2(p0.x + (static_cast<float>(i) / (N - 1)) * w, dbToYeq(dbc, p0.y, h));
            }
            dl->AddPolyline(tpts, N, IM_COL32(255, 190, 70, 90), 0, 1.5f);
        }

        sel->magnitudeFn(freqs, mags, N, sr, sel->values.data());

        // Response polyline (clamp dB to the visible range so deep notches read as a
        // clean dip to the floor instead of a 1-pixel spike).
        for (int i = 0; i < N; ++i)
        {
            float dbc = std::clamp(mags[i], -kEqRange, kEqRange);
            pts[i] = ImVec2(p0.x + (static_cast<float>(i) / (N - 1)) * w, dbToYeq(dbc, p0.y, h));
        }
        dl->AddPolyline(pts, N, IM_COL32(255, 190, 70, 255), 0, 2.4f);

        auto curveDbAt = [&](float fHz) -> float {
            float t = (freqToX(fHz, p0.x, w) - p0.x) / w;
            int idx = std::clamp(static_cast<int>(t * (N - 1) + 0.5f), 0, N - 1);
            return mags[idx];
        };

        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const float wheel = ImGui::GetIO().MouseWheel;

        // On press, grab the NEAREST node so edge/hidden nodes are reachable by
        // clicking anywhere on the graph; X is weighted more (bands run along X).
        if (ImGui::IsItemActivated() && dragNode_ < 0)
        {
            int best = -1; float bestD = 1e30f;
            for (int ni = 0; ni < static_cast<int>(sel->curveNodes.size()); ++ni)
            {
                const auto& nd = sel->curveNodes[ni];
                if (nd.freqParam < 0) continue;
                const float fHz  = sel->values[nd.freqParam];
                const float gain = (nd.gainParam >= 0) ? sel->values[nd.gainParam] : curveDbAt(fHz);
                const ImVec2 cc(freqToX(fHz, p0.x, w), dbToYeq(gain, p0.y, h));
                const float d = std::fabs(mouse.x - cc.x) + 0.35f * std::fabs(mouse.y - cc.y);
                if (d < bestD) { bestD = d; best = ni; }
            }
            if (best >= 0) { dragNode_ = best; dragOwner_ = sel; }
        }

        for (int ni = 0; ni < static_cast<int>(sel->curveNodes.size()); ++ni)
        {
            const auto& node = sel->curveNodes[ni];
            if (node.freqParam < 0) continue;
            const float fHz  = sel->values[node.freqParam];
            const float gain = (node.gainParam >= 0) ? sel->values[node.gainParam] : curveDbAt(fHz);
            const ImVec2 c(freqToX(fHz, p0.x, w), dbToYeq(gain, p0.y, h));

            const bool nearNode = std::fabs(mouse.x - c.x) < 12.0f && std::fabs(mouse.y - c.y) < 12.0f;
            const bool active = (dragNode_ == ni && dragOwner_ == sel);

            // Move only while actually dragging (a plain click grabs but doesn't jump)
            if (active && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                const auto& fp = sel->params[node.freqParam];
                sel->applyParam(node.freqParam, std::clamp(xToFreq(mouse.x, p0.x, w), fp.min, fp.max));
                if (node.gainParam >= 0)
                {
                    const auto& gp = sel->params[node.gainParam];
                    sel->applyParam(node.gainParam, std::clamp(yToDbEq(mouse.y, p0.y, h), gp.min, gp.max));
                }
            }
            // Mouse wheel over node -> Q
            if (canvasHovered && nearNode && node.qParam >= 0 && wheel != 0.0f)
            {
                const auto& qp = sel->params[node.qParam];
                float nq = std::clamp(sel->values[node.qParam] * (wheel > 0 ? 1.15f : 0.87f), qp.min, qp.max);
                sel->applyParam(node.qParam, nq);
            }

            const ImU32 col = (active || nearNode) ? IM_COL32(255, 230, 140, 255) : IM_COL32(255, 190, 70, 255);
            dl->AddCircleFilled(c, 6.0f, col);
            dl->AddCircle(c, 6.0f, IM_COL32(30, 30, 30, 255), 0, 1.5f);
            char lbl[8]; std::snprintf(lbl, sizeof(lbl), "%d", ni + 1);
            dl->AddText(ImVec2(c.x - 3, c.y - 7), IM_COL32(20, 20, 20, 255), lbl);

            // Value readout while hovering/dragging a node
            if (active || nearNode)
            {
                char vb[64];
                const bool hasG = node.gainParam >= 0;
                const bool hasQ = node.qParam >= 0;
                const float gv = hasG ? sel->values[node.gainParam] : 0.0f;
                const float qv = hasQ ? sel->values[node.qParam] : 0.0f;
                if (hasG && hasQ)      std::snprintf(vb, sizeof(vb), "%.0f Hz  %+.1f dB  Q %.2f", fHz, gv, qv);
                else if (hasG)         std::snprintf(vb, sizeof(vb), "%.0f Hz  %+.1f dB", fHz, gv);
                else if (hasQ)         std::snprintf(vb, sizeof(vb), "%.0f Hz  Q %.2f", fHz, qv);
                else                   std::snprintf(vb, sizeof(vb), "%.0f Hz", fHz);
                const ImVec2 tsz = ImGui::CalcTextSize(vb);
                ImVec2 tp(c.x + 10, c.y - 9);
                if (tp.x + tsz.x > p0.x + w) tp.x = c.x - 10 - tsz.x;
                dl->AddRectFilled(ImVec2(tp.x - 3, tp.y - 2), ImVec2(tp.x + tsz.x + 3, tp.y + tsz.y + 2),
                                  IM_COL32(20, 22, 28, 220), 3.0f);
                dl->AddText(tp, IM_COL32(255, 235, 180, 255), vb);
            }
        }

        // End drag
        if (dragNode_ >= 0 && !ImGui::IsItemActive())
        {
            dragNode_ = -1; dragOwner_ = nullptr;
        }
    }

    void drawMultiband(ImDrawList* dl, EffectSlot* sel, ImVec2 p0, float w, float h)
    {
        float xo[8] = {}; float gr[8] = {};
        const int nb = sel->multibandFn(xo, gr, 8);
        if (nb <= 0) return;

        const ImU32 bandCols[3] = { IM_COL32(80, 160, 90, 40), IM_COL32(90, 130, 200, 40), IM_COL32(200, 130, 80, 40) };
        float left = p0.x;
        for (int b = 0; b < nb; ++b)
        {
            const float right = (b < nb - 1) ? freqToX(xo[b], p0.x, w) : (p0.x + w);
            dl->AddRectFilled(ImVec2(left, p0.y), ImVec2(right, p0.y + h), bandCols[b % 3]);
            if (b < nb - 1)
                dl->AddLine(ImVec2(right, p0.y), ImVec2(right, p0.y + h), IM_COL32(150, 150, 160, 200), 1.5f);

            // Gain-reduction bar from the top of the band region
            const float grDb = std::clamp(gr[b], 0.0f, 24.0f);
            const float barH = (grDb / 24.0f) * (h * 0.5f);
            dl->AddRectFilled(ImVec2(left + 2, p0.y), ImVec2(right - 2, p0.y + barH),
                              IM_COL32(230, 90, 70, 150));
            char lbl[24]; std::snprintf(lbl, sizeof(lbl), "-%.1f dB", grDb);
            dl->AddText(ImVec2(left + 6, p0.y + 4), IM_COL32(235, 235, 240, 255), lbl);
            left = right;
        }
    }

    // --- Parameter panel -------------------------------------------------------

    void drawParamPanel(EffectSlot* sel)
    {
        if (!sel)
        {
            ImGui::TextDisabled("Select an effect");
            ImGui::Spacing();
            ImGui::TextWrapped("Pick an effect from the list to edit it. Filters/EQs show an "
                               "interactive response curve; drag the nodes on the analyzer.");
            return;
        }

        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f), "%s", sel->name.c_str());
        ImGui::TextDisabled("%s", sel->category.c_str());
        if (ImGui::Button("Reset Defaults", ImVec2(-1, 0)))
            sel->applyAllDefaults();

        // Convolution-reverb IR loader
        if (sel->loadIRFn)
        {
            if (ImGui::Button("Load IR (WAV)...", ImVec2(-1, 0)))
            {
                char path[MAX_PATH] = {};
                OPENFILENAMEA ofn = {};
                ofn.lStructSize = sizeof(ofn);
                ofn.lpstrFilter = "WAV Impulse Response\0*.wav\0All Files\0*.*\0";
                ofn.lpstrFile = path;
                ofn.nMaxFile = MAX_PATH;
                ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
                if (GetOpenFileNameA(&ofn))
                    sel->loadIRFn(path);
            }
            ImGui::TextDisabled("(or use Decay for a synthetic IR)");
        }
        ImGui::Separator();

        // Live gain-reduction meter at the top (most-watched value).
        if (sel->gainReductionDbFn)
        {
            float grv = std::clamp(-sel->gainReductionDbFn(), 0.0f, 24.0f);
            float norm = grv / 24.0f;
            char lbl[32]; std::snprintf(lbl, sizeof(lbl), "GR  -%.1f dB", grv);
            ImVec4 col = norm < 0.25f ? ImVec4(0.2f, 0.8f, 0.2f, 1)
                       : norm < 0.6f  ? ImVec4(0.9f, 0.8f, 0.2f, 1)
                                      : ImVec4(1.0f, 0.3f, 0.2f, 1);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
            ImGui::ProgressBar(norm, ImVec2(-1, 16), lbl);
            ImGui::PopStyleColor();
            ImGui::Separator();
        }

        for (int i = 0; i < static_cast<int>(sel->params.size()); ++i)
        {
            auto& pd = sel->params[i];
            float& val = sel->values[i];
            ImGui::PushID(i);

            switch (pd.type)
            {
            case ParamDesc::Slider:
            {
                float old = val;
                ImGui::TextUnformatted(pd.name.c_str());     // label above (no truncation in a column)
                ImGui::SetNextItemWidth(-1.0f);              // full-width slider
                ImGui::SliderFloat("##s", &val, pd.min, pd.max, formatStr(val, pd.unit),
                                   pd.logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
                // Hover + mouse wheel nudges the value (audio-style).
                if (ImGui::IsItemHovered())
                {
                    const float wh = ImGui::GetIO().MouseWheel;
                    if (wh != 0.0f)
                        val = std::clamp(pd.logarithmic ? val * std::pow(1.06f, wh)
                                                        : val + (pd.max - pd.min) * 0.03f * wh,
                                         pd.min, pd.max);
                }
                if (val != old) sel->applyParam(i, val);
                break;
            }
            case ParamDesc::Toggle:
            {
                bool on = val > 0.5f;
                if (ImGui::Checkbox(pd.name.c_str(), &on)) { val = on ? 1.0f : 0.0f; sel->applyParam(i, val); }
                // Wheel toggles too
                if (ImGui::IsItemHovered() && ImGui::GetIO().MouseWheel != 0.0f)
                {
                    float nv = ImGui::GetIO().MouseWheel > 0.0f ? 1.0f : 0.0f;
                    if (nv != val) { val = nv; sel->applyParam(i, val); }
                }
                break;
            }
            case ParamDesc::Choice:
            {
                int cur = static_cast<int>(val);
                ImGui::TextUnformatted(pd.name.c_str());
                ImGui::SetNextItemWidth(-1.0f);
                if (ImGui::BeginCombo("##c",
                    cur < static_cast<int>(pd.choices.size()) ? pd.choices[cur].c_str() : ""))
                {
                    for (int c = 0; c < static_cast<int>(pd.choices.size()); ++c)
                    {
                        bool isSel = (c == cur);
                        if (ImGui::Selectable(pd.choices[c].c_str(), isSel)) { val = static_cast<float>(c); sel->applyParam(i, val); }
                        if (isSel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                // Hover + mouse wheel cycles options without opening the list.
                if (ImGui::IsItemHovered())
                {
                    const float wh = ImGui::GetIO().MouseWheel;
                    if (wh != 0.0f)
                    {
                        // Wheel up = previous option, wheel down = next (matches list feel).
                        int nc = std::clamp(cur + (wh > 0.0f ? -1 : 1), 0,
                                            static_cast<int>(pd.choices.size()) - 1);
                        if (nc != cur) { val = static_cast<float>(nc); sel->applyParam(i, val); }
                    }
                }
                break;
            }
            }
            ImGui::Spacing();
            ImGui::PopID();
        }
    }

    // --- Compact meters + waveform (bottom) ------------------------------------

    void drawMeters(AudioEngine& engine)
    {
        int waveN = engine.getWaveformSize();
        if (waveN > 0)
            ImGui::PlotLines("##wave", engine.getWaveformL(), waveN, 0, "Waveform", -1.0f, 1.0f, ImVec2(-1, 50));

        float peakL = engine.getPeakL(), peakR = engine.getPeakR();
        float dbL = peakL > 0.0f ? 20.0f * std::log10(peakL) : -100.0f;
        float dbR = peakR > 0.0f ? 20.0f * std::log10(peakR) : -100.0f;
        float normL = std::clamp((dbL + 60.0f) / 60.0f, 0.0f, 1.0f);
        float normR = std::clamp((dbR + 60.0f) / 60.0f, 0.0f, 1.0f);
        char lblL[32], lblR[32];
        std::snprintf(lblL, sizeof(lblL), "L  %.1f dB", dbL);
        std::snprintf(lblR, sizeof(lblR), "R  %.1f dB", dbR);
        ImVec4 colL = normL > 0.9f ? ImVec4(1, 0.2f, 0.2f, 1) : ImVec4(0.2f, 0.8f, 0.2f, 1);
        ImVec4 colR = normR > 0.9f ? ImVec4(1, 0.2f, 0.2f, 1) : ImVec4(0.2f, 0.8f, 0.2f, 1);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colL);
        ImGui::ProgressBar(normL, ImVec2(-1, 16), lblL);
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colR);
        ImGui::ProgressBar(normR, ImVec2(-1, 16), lblR);
        ImGui::PopStyleColor();

        drawThdPanel(engine);
    }

    // --- Test tone + live THD metering panel ------------------------------------

    void drawThdPanel(AudioEngine& engine)
    {
        ImGui::Separator();
        bool tone = engine.getTestTone();
        if (ImGui::Checkbox("Test Tone", &tone))
            engine.setTestTone(tone);
        if (!engine.isDeviceReady())
        {
            ImGui::SameLine();
            ImGui::TextDisabled("(load a file once to initialise audio)");
            return;
        }

        if (tone)
        {
            float freq = engine.getTestToneFreq();
            float db   = engine.getTestToneDb();
            ImGui::SetNextItemWidth(140);
            if (ImGui::SliderFloat("Freq", &freq, 20.0f, 20000.0f, "%.0f Hz",
                                   ImGuiSliderFlags_Logarithmic))
                engine.setTestToneFreq(freq);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            if (ImGui::SliderFloat("Level", &db, -60.0f, 0.0f, "%.1f dB"))
                engine.setTestToneDb(db);

            // Refresh the reading a few times per second: enough for a live
            // meter, cheap enough to never matter on the GUI thread.
            thdTimer_ += ImGui::GetIO().DeltaTime;
            if (thdTimer_ > 0.25f)
            {
                thdTimer_ = 0.0f;
                thdLast_ = engine.computeThd();
            }
            if (thdLast_.valid)
            {
                ImGui::Text("THD+N: %s   Fund: %.1f dBFS   RMS: %.1f dBFS",
                            thdLast_.thdnPercent < 0.001f ? "<0.001 %"
                                : formatStr(thdLast_.thdnPercent, "%"),
                            thdLast_.fundDb, thdLast_.rmsDb);
            }
            else
                ImGui::TextDisabled("THD+N: settling...");
        }
    }

    // --- Helpers ---------------------------------------------------------------

    static const char* formatStr(float val, const std::string& unit)
    {
        static char buf[64];
        if (unit.empty()) std::snprintf(buf, sizeof(buf), "%.2f", val);
        else              std::snprintf(buf, sizeof(buf), "%.1f %s", val, unit.c_str());
        return buf;
    }
};

} // namespace dsplab
