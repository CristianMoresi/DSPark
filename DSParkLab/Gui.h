// DSParkLab - GUI
// ImGui interface built for auditioning processors: a transport bar with the
// A/B switch, an effect browser with search, and, for the selected effect, a
// large display (spectrum, its response or live gain curves, what it detects,
// draggable nodes), an activity strip (gain-reduction history, scope, output
// meters, test tone and THD), and the controls as knobs on tabs, so no long
// list has to be scrolled.
//
// Keys (when no text field has focus): Space plays/pauses, B switches the A/B
// bypass, E enables or disables the selected effect, R resets its controls.

#pragma once

#include "AudioEngine.h"
#include "EffectSlot.h"

#include "vendor/imgui/imgui.h"

#include <algorithm>
#include <array>
#include <cctype>
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
    /// A larger font for titles (set by main() when a system font is found).
    static void setTitleFont(ImFont* f) noexcept { titleFont() = f; }

    /// The scale of the display (1 at 96 dpi): every size of the layout follows it.
    static void setScale(float s) noexcept { uiScale() = std::clamp(s, 0.5f, 4.0f); }

    // Dark theme with an amber accent (call once after StyleColorsDark()).
    static void applyTheme()
    {
        ImGuiStyle& s = ImGui::GetStyle();
        s.WindowRounding = 6.0f; s.ChildRounding = 6.0f; s.FrameRounding = 5.0f;
        s.GrabRounding = 5.0f; s.PopupRounding = 5.0f; s.ScrollbarRounding = 5.0f;
        s.TabRounding = 5.0f;
        s.WindowBorderSize = 0.0f; s.FrameBorderSize = 0.0f; s.ChildBorderSize = 1.0f;
        s.WindowPadding = ImVec2(10, 10); s.FramePadding = ImVec2(8, 5);
        s.ItemSpacing = ImVec2(8, 6); s.ScrollbarSize = 12.0f;

        ImVec4* c = s.Colors;
        const ImVec4 accent(1.00f, 0.62f, 0.20f, 1.00f);
        c[ImGuiCol_WindowBg]           = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);
        c[ImGuiCol_ChildBg]            = ImVec4(0.10f, 0.11f, 0.14f, 1.00f);
        c[ImGuiCol_PopupBg]            = ImVec4(0.12f, 0.13f, 0.16f, 0.98f);
        c[ImGuiCol_Border]             = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        c[ImGuiCol_FrameBg]            = ImVec4(0.15f, 0.16f, 0.20f, 1.00f);
        c[ImGuiCol_FrameBgHovered]     = ImVec4(0.21f, 0.23f, 0.28f, 1.00f);
        c[ImGuiCol_FrameBgActive]      = ImVec4(0.25f, 0.27f, 0.33f, 1.00f);
        c[ImGuiCol_TitleBg]            = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);
        c[ImGuiCol_TitleBgActive]      = ImVec4(0.11f, 0.12f, 0.15f, 1.00f);
        c[ImGuiCol_Button]             = ImVec4(0.19f, 0.21f, 0.26f, 1.00f);
        c[ImGuiCol_ButtonHovered]      = ImVec4(0.27f, 0.30f, 0.37f, 1.00f);
        c[ImGuiCol_ButtonActive]       = accent;
        c[ImGuiCol_SliderGrab]         = accent;
        c[ImGuiCol_SliderGrabActive]   = ImVec4(1.00f, 0.74f, 0.35f, 1.00f);
        c[ImGuiCol_CheckMark]          = accent;
        c[ImGuiCol_Header]             = ImVec4(0.19f, 0.21f, 0.26f, 1.00f);
        c[ImGuiCol_HeaderHovered]      = ImVec4(0.27f, 0.30f, 0.37f, 1.00f);
        c[ImGuiCol_HeaderActive]       = ImVec4(1.00f, 0.62f, 0.20f, 0.55f);
        c[ImGuiCol_Separator]          = ImVec4(0.22f, 0.24f, 0.29f, 1.00f);
        c[ImGuiCol_Tab]                = ImVec4(0.14f, 0.15f, 0.19f, 1.00f);
        c[ImGuiCol_TabHovered]         = ImVec4(0.30f, 0.26f, 0.20f, 1.00f);
        c[ImGuiCol_TabSelected]        = ImVec4(0.36f, 0.26f, 0.12f, 1.00f);
        c[ImGuiCol_TabSelectedOverline]= accent;
        c[ImGuiCol_Text]               = ImVec4(0.89f, 0.91f, 0.94f, 1.00f);
        c[ImGuiCol_TextDisabled]       = ImVec4(0.50f, 0.53f, 0.60f, 1.00f);
        c[ImGuiCol_ScrollbarBg]        = ImVec4(0.08f, 0.09f, 0.11f, 0.00f);
        c[ImGuiCol_ScrollbarGrab]      = ImVec4(0.25f, 0.27f, 0.33f, 1.00f);
    }

    void render(AudioEngine& engine, std::vector<EffectSlot>& effects)
    {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("DSParkLab", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

        EffectSlot* sel = nullptr;
        for (auto& e : effects)
            if (e.selected) { sel = &e; break; }

        handleKeys(engine, sel);
        drawTopBar(engine);
        ImGui::Spacing();

        ImGui::BeginChild("Browser", ImVec2(px(230.0f), 0), ImGuiChildFlags_Borders);
        drawBrowser(effects);
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("Main", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
        drawMain(engine, sel);
        ImGui::EndChild();

        ImGui::End();
    }

private:
    // --- Sizes of the control grid ----------------------------------------------

    static constexpr float kFMin = 20.0f, kFMax = 20000.0f;
    static constexpr int   kHistory = 360;   // frames of gain-reduction history

    static ImFont*& titleFont() noexcept { static ImFont* f = nullptr; return f; }
    static float& uiScale() noexcept { static float s = 1.0f; return s; }
    /// A size in pixels at 96 dpi, in the pixels of this display.
    static float px(float v) noexcept { return v * uiScale(); }
    static float cellW() noexcept { return px(104.0f); }
    static float knobR() noexcept { return px(21.0f); }

    // --- Interface state ----------------------------------------------------------

    int   dragNode_ = -1;        // curve node being dragged (-1 = none)
    void* dragOwner_ = nullptr;  // slot that owns the drag
    float seekValue_ = 0.0f;     // seek bar value while it is held
    bool  seekActive_ = false;
    float thdTimer_ = 0.0f;
    AudioEngine::ThdReading thdLast_ {};
    char  search_[64] = {};
    float range_ = 24.0f;        // vertical scale of the display (+/- dB), per slot
    std::array<float, kHistory> grHistory_ {};
    int   grHead_ = 0;
    const EffectSlot* grOwner_ = nullptr;
    std::array<float, AudioEngine::kBlockSize> waveL_ {};
    std::array<float, AudioEngine::kBlockSize> waveR_ {};

    // --- Mapping helpers -----------------------------------------------------------

    /// A cell of the control grid: two lines of name, the control, a line of value.
    static float cellH() noexcept
    {
        return 3.0f * ImGui::GetTextLineHeight() + 2.0f * knobR() + px(16.0f);
    }
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
    static float dbToYspec(float db, float y0, float h)
    {
        const float t = std::clamp((db + 100.0f) / 100.0f, 0.0f, 1.0f);
        return y0 + h - t * h;
    }
    float dbToYeq(float db, float y0, float h) const
    {
        const float t = std::clamp((db + range_) / (2.0f * range_), 0.0f, 1.0f);
        return y0 + h - t * h;
    }
    float yToDbEq(float y, float y0, float h) const
    {
        const float t = std::clamp((y0 + h - y) / h, 0.0f, 1.0f);
        return -range_ + t * 2.0f * range_;
    }

    static std::string lower(std::string s)
    {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    /// The text cut to a width, with an ellipsis when it does not fit.
    static std::string fitText(const std::string& s, float maxW)
    {
        if (ImGui::CalcTextSize(s.c_str()).x <= maxW) return s;
        std::string t = s;
        while (!t.empty() && ImGui::CalcTextSize((t + "...").c_str()).x > maxW) t.pop_back();
        return t + "...";
    }

    /// The text in at most two lines of a width: broken between words (the
    /// first line as long as it fits), the second cut if it still does not.
    static std::array<std::string, 2> twoLines(const std::string& s, float maxW)
    {
        if (ImGui::CalcTextSize(s.c_str()).x <= maxW) return { std::string(), s };
        size_t best = std::string::npos;
        for (size_t i = s.find(' '); i != std::string::npos; i = s.find(' ', i + 1))
            if (ImGui::CalcTextSize(s.substr(0, i).c_str()).x <= maxW) best = i;
        if (best == std::string::npos) return { std::string(), fitText(s, maxW) };
        return { s.substr(0, best), fitText(s.substr(best + 1), maxW) };
    }

    // --- Keys -------------------------------------------------------------------------

    void handleKeys(AudioEngine& engine, EffectSlot* sel)
    {
        const ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput || ImGui::IsAnyItemActive()) return;
        if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) engine.togglePlay();
        if (ImGui::IsKeyPressed(ImGuiKey_B, false)) engine.setBypass(!engine.isBypassed());
        if (sel && ImGui::IsKeyPressed(ImGuiKey_E, false)) sel->enabled = !sel->enabled.get();
        if (sel && ImGui::IsKeyPressed(ImGuiKey_R, false)) sel->applyAllDefaults();
        // The presets of the effect, one after another.
        if (sel && !sel->presets.empty())
        {
            const int count = static_cast<int>(sel->presets.size());
            if (ImGui::IsKeyPressed(ImGuiKey_RightBracket, false))
                sel->applyPreset(sel->currentPreset < 0 ? 0 : (sel->currentPreset + 1) % count);
            if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, false))
                sel->applyPreset(sel->currentPreset <= 0 ? count - 1 : sel->currentPreset - 1);
        }
    }

    // --- Top bar: file, transport, seek, A/B --------------------------------------------

    void drawTopBar(AudioEngine& engine)
    {
        const bool loading = engine.isLoading();
        if (titleFont()) ImGui::PushFont(titleFont());
        ImGui::TextColored(ImVec4(1.0f, 0.70f, 0.28f, 1.0f), "DSParkLab");
        if (titleFont()) ImGui::PopFont();
        ImGui::SameLine(0, px(18.0f));

        ImGui::BeginDisabled(loading);
        if (ImGui::Button("Open..."))
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
        ImGui::SameLine();
        const bool playing = engine.isPlaying();
        if (ImGui::Button(playing ? "Pause" : "Play", ImVec2(px(64.0f), 0)))
            engine.togglePlay();
        ImGui::SetItemTooltip("Space");
        ImGui::SameLine();
        if (ImGui::Button("Stop"))
            engine.stop();
        ImGui::EndDisabled();
        ImGui::SameLine();
        bool loop = engine.isLooping();
        if (ImGui::Checkbox("Loop", &loop))
            engine.setLooping(loop);

        // A/B switch: wide and coloured, the most used control while listening.
        ImGui::SameLine(0, px(16.0f));
        const bool bypass = engine.isBypassed();
        ImGui::PushStyleColor(ImGuiCol_Button, bypass ? ImVec4(0.55f, 0.18f, 0.16f, 1.0f) : ImVec4(0.16f, 0.36f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bypass ? ImVec4(0.66f, 0.24f, 0.20f, 1.0f) : ImVec4(0.20f, 0.45f, 0.28f, 1.0f));
        if (ImGui::Button(bypass ? "B: Bypassed" : "A: Processed", ImVec2(px(130.0f), 0)))
            engine.setBypass(!bypass);
        ImGui::PopStyleColor(2);
        ImGui::SetItemTooltip("Switch between the processed and the dry signal (B)");

        // Seek bar: the transport jumps when the bar is released.
        ImGui::SameLine(0, px(16.0f));
        const float playPos = engine.getPosition();
        float shown = seekActive_ ? seekValue_ : playPos;
        const float infoW = px(270.0f);
        ImGui::SetNextItemWidth(std::max(px(80.0f), ImGui::GetContentRegionAvail().x - infoW));
        ImGui::SliderFloat("##seek", &shown, 0.0f, 1.0f, "");
        if (ImGui::IsItemActivated()) seekActive_ = true;
        if (seekActive_) seekValue_ = shown;
        if (ImGui::IsItemDeactivatedAfterEdit()) { engine.seekTo(seekValue_); seekActive_ = false; }
        else if (seekActive_ && !ImGui::IsItemActive()) seekActive_ = false;

        ImGui::SameLine();
        const float dur = engine.getDurationSeconds();
        const float cur = shown * dur;
        ImGui::Text("%02d:%02d / %02d:%02d",
                    static_cast<int>(cur) / 60, static_cast<int>(cur) % 60,
                    static_cast<int>(dur) / 60, static_cast<int>(dur) % 60);
        ImGui::SameLine();
        if (loading)
        {
            const int dots = static_cast<int>(ImGui::GetTime() * 2.0) % 4;
            const char* dotStrs[4] = { "", ".", "..", "..." };
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Loading%s", dotStrs[dots]);
        }
        else if (engine.hasFile())
            ImGui::TextDisabled("%.0f Hz | %d ch", engine.getSampleRate(), engine.getChannels());
        else
            ImGui::TextDisabled("No file");
    }

    // --- Browser -----------------------------------------------------------------------

    void drawBrowser(std::vector<EffectSlot>& effects)
    {
        ImGui::TextDisabled("EFFECTS");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##search", "Search...", search_, sizeof(search_));
        const std::string needle = lower(search_);

        int enabledCount = 0;
        for (const auto& e : effects) enabledCount += e.enabled.get() ? 1 : 0;
        if (enabledCount > 0)
        {
            ImGui::TextDisabled("%d on", enabledCount);
            ImGui::SameLine();
            if (ImGui::SmallButton("All off"))
                for (auto& e : effects) e.enabled = false;
        }
        ImGui::Separator();

        ImGui::BeginChild("##list", ImVec2(0, 0));
        std::string currentCat;
        bool catOpen = true;
        for (int i = 0; i < static_cast<int>(effects.size()); ++i)
        {
            auto& e = effects[static_cast<size_t>(i)];
            if (!needle.empty() && lower(e.name).find(needle) == std::string::npos
                && lower(e.category).find(needle) == std::string::npos)
                continue;
            if (e.category != currentCat)
            {
                currentCat = e.category;
                if (!needle.empty()) ImGui::SetNextItemOpen(true);
                catOpen = ImGui::CollapsingHeader(currentCat.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
            }
            if (!catOpen) continue;

            ImGui::PushID(i);
            // The audio thread reads this flag: edit a copy, publish the result.
            bool slotEnabled = e.enabled.get();
            if (ImGui::Checkbox("##en", &slotEnabled))
                e.enabled = slotEnabled;
            ImGui::SetItemTooltip(slotEnabled ? "In the chain (click to take it out)" : "Out of the chain (click to put it in)");
            ImGui::SameLine();
            if (slotEnabled) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.78f, 0.40f, 1.0f));
            if (ImGui::Selectable(e.name.c_str(), e.selected))
            {
                for (auto& eff : effects) eff.selected = false;
                e.selected = true;
            }
            if (slotEnabled) ImGui::PopStyleColor();
            ImGui::PopID();
        }
        ImGui::EndChild();
    }

    // --- Main area ---------------------------------------------------------------------

    void drawMain(AudioEngine& engine, EffectSlot* sel)
    {
        if (!sel)
        {
            drawWelcome(engine);
            return;
        }
        if (sel != grOwner_) { grHistory_.fill(0.0f); grHead_ = 0; grOwner_ = sel; }
        range_ = sel->displayRangeDb > 1.0f ? sel->displayRangeDb : 24.0f;

        drawHeader(sel);

        const float total = ImGui::GetContentRegionAvail().y;
        const float width = ImGui::GetContentRegionAvail().x;
        const float activityH = px(96.0f);
        float controlsH = controlsHeight(*sel, width);
        controlsH = std::clamp(controlsH, cellH() + px(24.0f), std::max(cellH() + px(24.0f), total * 0.5f));
        float displayH = total - controlsH - activityH - 2.0f * ImGui::GetStyle().ItemSpacing.y;
        if (displayH < px(180.0f)) displayH = px(180.0f);

        drawDisplay(engine, sel, displayH);
        drawActivity(engine, sel, activityH);
        ImGui::BeginChild("Controls", ImVec2(0, 0), ImGuiChildFlags_Borders);
        drawControls(sel);
        ImGui::EndChild();
    }

    void drawWelcome(AudioEngine& engine)
    {
        drawDisplay(engine, nullptr, std::max(px(180.0f), ImGui::GetContentRegionAvail().y - px(104.0f)));
        drawActivity(engine, nullptr, px(96.0f));
    }

    void drawHeader(EffectSlot* sel)
    {
        if (titleFont()) ImGui::PushFont(titleFont());
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f), "%s", sel->name.c_str());
        if (titleFont()) ImGui::PopFont();
        ImGui::SameLine(0, px(18.0f));
        const bool on = sel->enabled.get();
        ImGui::PushStyleColor(ImGuiCol_Button, on ? ImVec4(0.75f, 0.45f, 0.12f, 1.0f) : ImVec4(0.19f, 0.21f, 0.26f, 1.0f));
        if (ImGui::Button(on ? "On" : "Off", ImVec2(px(52.0f), 0)))
            sel->enabled = !on;
        ImGui::PopStyleColor();
        ImGui::SetItemTooltip("Put the effect in or out of the chain (E)");
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            sel->applyAllDefaults();
        ImGui::SetItemTooltip("Every control to its default (R)");

        // Presets: a setting to start listening from. Choosing one puts every
        // control at its default and then sets the preset's own.
        sel->settlePreset();
        if (!sel->presets.empty())
        {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(px(300.0f));
            const int current = sel->currentPreset;
            const char* shown = current >= 0 ? sel->presets[static_cast<size_t>(current)].name.c_str() : "Presets...";
            if (ImGui::BeginCombo("##preset", shown, ImGuiComboFlags_HeightLarge))
            {
                for (int i = 0; i < static_cast<int>(sel->presets.size()); ++i)
                {
                    const auto& preset = sel->presets[static_cast<size_t>(i)];
                    if (ImGui::Selectable(preset.name.c_str(), i == current)) sel->applyPreset(i);
                    if (!preset.note.empty()) ImGui::SetItemTooltip("%s", preset.note.c_str());
                }
                ImGui::EndCombo();
            }
            else
            {
                ImGui::SetItemTooltip("A setting to start from: every control to its default, then the preset's own. [ and ] step through them.");
            }
            if (!sel->presetErrors.empty())
            {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.30f, 1.0f), "(!)");
                ImGui::SetItemTooltip("%s", sel->presetErrors.c_str());
            }
            if (sel->currentPreset >= 0 && !sel->presets[static_cast<size_t>(sel->currentPreset)].note.empty())
                ImGui::TextDisabled("%s", sel->presets[static_cast<size_t>(sel->currentPreset)].note.c_str());
        }

        if (sel->loadIRFn)
        {
            ImGui::SameLine();
            if (ImGui::Button("Load IR..."))
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
            ImGui::SetItemTooltip("A WAV file (summed to mono); without one, Decay builds a synthetic IR");
        }

        ImGui::SameLine();
        if (!sel->curveNodes.empty())
            ImGui::TextDisabled("  Drag a node: frequency and gain | wheel on it: width | double-click a knob: default");
        else
            ImGui::TextDisabled("  Drag a knob up or down (Shift: fine) | wheel: step | double-click: default");
    }

    // --- Display: spectrum, curves, what the effect detects, nodes ----------------------

    void drawDisplay(AudioEngine& engine, EffectSlot* sel, float h)
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        const ImVec2 p1 = ImVec2(p0.x + w, p0.y + h);

        ImGui::InvisibleButton("##display", ImVec2(std::max(w, 1.0f), std::max(h, 1.0f)));
        const bool hovered = ImGui::IsItemHovered();

        dl->AddRectFilled(p0, p1, IM_COL32(15, 17, 22, 255), 6.0f);
        dl->PushClipRect(p0, p1, true);

        const float gridF[] = { 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
        const char* gridL[] = { "30","50","100","200","500","1k","2k","5k","10k","20k" };
        for (int i = 0; i < 10; ++i)
        {
            const float x = freqToX(gridF[i], p0.x, w);
            dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y), IM_COL32(38, 42, 52, 255));
            dl->AddText(ImVec2(x + px(3.0f), p1.y - ImGui::GetTextLineHeight() - px(3.0f)), IM_COL32(105, 112, 128, 255), gridL[i]);
        }
        const bool curves = sel && (sel->magnitudeFn || sel->activityFn);
        if (curves)
        {
            const float step = range_ >= 24.0f ? 12.0f : 6.0f;
            for (float d = -range_ + step; d < range_; d += step)
            {
                const float y = dbToYeq(d, p0.y, h);
                dl->AddLine(ImVec2(p0.x, y), ImVec2(p1.x, y),
                            std::fabs(d) < 0.01f ? IM_COL32(72, 78, 94, 255) : IM_COL32(34, 38, 47, 255));
                char b[8]; std::snprintf(b, sizeof(b), std::fabs(d) < 0.01f ? "0" : "%+.0f", d);
                dl->AddText(ImVec2(p0.x + px(4.0f), y - ImGui::GetTextLineHeight()), IM_COL32(105, 112, 128, 255), b);
            }
        }

        drawSpectrum(dl, engine, p0, w, h);

        float* mags = nullptr;
        int n = 0;
        if (sel && curves)
            drawCurves(dl, sel, engine, p0, w, h, mags, n);
        if (sel && sel->magnitudeFn)
            drawNodes(dl, sel, p0, w, h, hovered, mags, n);
        if (sel && sel->multibandFn)
            drawMultiband(dl, sel, p0, w, h);
        if (sel) drawLegend(dl, sel, p0, w);
        // An effect that is not in the chain is not heard: say so where the eye is.
        if (sel && !sel->enabled.get())
        {
            const char* msg = "Off: this effect is not in the chain. Press On (or E) to hear it.";
            const ImVec2 ts = ImGui::CalcTextSize(msg);
            const ImVec2 a(p0.x + px(10.0f), p0.y + px(8.0f));
            dl->AddRectFilled(a, ImVec2(a.x + ts.x + px(16.0f), a.y + ts.y + px(10.0f)), IM_COL32(120, 40, 34, 235), px(5.0f));
            dl->AddText(ImVec2(a.x + px(8.0f), a.y + px(5.0f)), IM_COL32(255, 232, 220, 255), msg);
        }

        // Readout under the cursor: the frequency and the curve there.
        if (hovered && dragNode_ < 0)
        {
            const ImVec2 m = ImGui::GetIO().MousePos;
            dl->AddLine(ImVec2(m.x, p0.y), ImVec2(m.x, p1.y), IM_COL32(120, 128, 145, 90));
            const float f = xToFreq(m.x, p0.x, w);
            char b[64];
            if (mags && n > 1)
            {
                const int idx = std::clamp(static_cast<int>((m.x - p0.x) / w * static_cast<float>(n - 1) + 0.5f), 0, n - 1);
                std::snprintf(b, sizeof(b), f >= 1000.0f ? "%.2f kHz  %+.1f dB" : "%.0f Hz  %+.1f dB",
                              f >= 1000.0f ? f / 1000.0f : f, mags[idx]);
            }
            else
                std::snprintf(b, sizeof(b), f >= 1000.0f ? "%.2f kHz" : "%.0f Hz", f >= 1000.0f ? f / 1000.0f : f);
            const ImVec2 ts = ImGui::CalcTextSize(b);
            float tx = m.x + 8;
            if (tx + ts.x > p1.x - 4) tx = m.x - 8 - ts.x;
            dl->AddRectFilled(ImVec2(tx - 4, p0.y + 6), ImVec2(tx + ts.x + 4, p0.y + 8 + ts.y), IM_COL32(20, 22, 28, 230), 3.0f);
            dl->AddText(ImVec2(tx, p0.y + 7), IM_COL32(235, 238, 245, 255), b);
        }

        if (!sel)
        {
            const char* hint = engine.hasFile() ? "Pick an effect on the left to see and change what it does"
                                                : "Open an audio file (or turn on the test tone) and pick an effect on the left";
            const ImVec2 ts = ImGui::CalcTextSize(hint);
            dl->AddText(ImVec2(p0.x + (w - ts.x) * 0.5f, p0.y + h * 0.45f), IM_COL32(150, 156, 170, 255), hint);
        }

        dl->PopClipRect();
        dl->AddRect(p0, p1, IM_COL32(58, 64, 78, 255), 6.0f);
    }

    static void drawSpectrum(ImDrawList* dl, AudioEngine& engine, ImVec2 p0, float w, float h)
    {
        const int bins = engine.getSpectrumBins();
        const float* spec = engine.getSpectrumDb();
        if (bins <= 1 || !spec) return;
        const float bottom = p0.y + h;
        float prevX = 0, prevY = 0, loudest = bottom;
        bool have = false;
        for (int k = 1; k < bins; ++k)
        {
            const float f = engine.binToFrequency(k);
            if (f < kFMin || f > kFMax) continue;
            const float x = freqToX(f, p0.x, w);
            const float y = dbToYspec(spec[k], p0.y, h);
            if (!have) { prevX = x; prevY = y; have = true; continue; }
            // Where several bins share a pixel the loudest is drawn: one bin
            // in each would leave a line between two of them undrawn.
            loudest = std::min(loudest, y);
            if (x - prevX >= 0.5f)
            {
                dl->AddQuadFilled(ImVec2(prevX, prevY), ImVec2(x, loudest), ImVec2(x, bottom), ImVec2(prevX, bottom),
                                  IM_COL32(55, 120, 190, 50));
                dl->AddLine(ImVec2(prevX, prevY), ImVec2(x, loudest), IM_COL32(85, 160, 230, 140), 1.3f);
                prevX = x; prevY = loudest;
                loudest = bottom;
            }
        }
    }

    void drawCurves(ImDrawList* dl, EffectSlot* sel, AudioEngine& engine, ImVec2 p0, float w, float h,
                    float*& outMags, int& outN)
    {
        const double sr = engine.getSampleRate() > 0 ? engine.getSampleRate() : 48000.0;
        constexpr int kMaxN = 1200;
        // Bounded before the conversion: the region can be zero or negative.
        const int N = !(w > 64.0f) ? 64 : (w > static_cast<float>(kMaxN) ? kMaxN : static_cast<int>(w));
        static float freqs[kMaxN], mags[kMaxN], extra[kMaxN];
        static ImVec2 pts[kMaxN];
        for (int i = 0; i < N; ++i)
            freqs[i] = xToFreq(p0.x + (static_cast<float>(i) / (N - 1)) * w, p0.x, w);
        const auto xAt = [&](int i) { return p0.x + (static_cast<float>(i) / (N - 1)) * w; };
        const float zeroY = dbToYeq(0.0f, p0.y, h);

        // What the effect detects: a filled area from 0 dB.
        if (sel->activityFn)
        {
            sel->activityFn(freqs, extra, N, sr, sel->values.data());
            for (int i = 1; i < N; ++i)
            {
                const float y0 = dbToYeq(std::clamp(extra[i - 1], -range_, range_), p0.y, h);
                const float y1 = dbToYeq(std::clamp(extra[i], -range_, range_), p0.y, h);
                dl->AddQuadFilled(ImVec2(xAt(i - 1), y0), ImVec2(xAt(i), y1), ImVec2(xAt(i), zeroY), ImVec2(xAt(i - 1), zeroY),
                                  IM_COL32(230, 80, 70, 80));
            }
        }
        // The target behind the live curve.
        if (sel->targetMagnitudeFn)
        {
            sel->targetMagnitudeFn(freqs, extra, N, sr, sel->values.data());
            for (int i = 0; i < N; ++i)
                pts[i] = ImVec2(xAt(i), dbToYeq(std::clamp(extra[i], -range_, range_), p0.y, h));
            dl->AddPolyline(pts, N, IM_COL32(170, 182, 210, 150), 0, px(1.6f));
        }
        if (sel->secondaryMagnitudeFn)
        {
            sel->secondaryMagnitudeFn(freqs, extra, N, sr, sel->values.data());
            for (int i = 0; i < N; ++i)
                pts[i] = ImVec2(xAt(i), dbToYeq(std::clamp(extra[i], -range_, range_), p0.y, h));
            dl->AddPolyline(pts, N, IM_COL32(110, 210, 230, 220), 0, 1.6f);
        }
        if (sel->magnitudeFn)
        {
            sel->magnitudeFn(freqs, mags, N, sr, sel->values.data());
            for (int i = 0; i < N; ++i)
                pts[i] = ImVec2(xAt(i), dbToYeq(std::clamp(mags[i], -range_, range_), p0.y, h));
            // A light fill between the curve and 0 dB shows where the effect acts.
            for (int i = 1; i < N; ++i)
                dl->AddQuadFilled(pts[i - 1], pts[i], ImVec2(pts[i].x, zeroY), ImVec2(pts[i - 1].x, zeroY),
                                  IM_COL32(255, 170, 60, 28));
            dl->AddPolyline(pts, N, IM_COL32(255, 186, 70, 255), 0, px(2.4f));
            outMags = mags;
            outN = N;
        }
    }

    void drawNodes(ImDrawList* dl, EffectSlot* sel, ImVec2 p0, float w, float h, bool hovered,
                   const float* mags, int n)
    {
        const auto curveDbAt = [&](float fHz) -> float {
            if (!mags || n < 2) return 0.0f;
            const float t = (freqToX(fHz, p0.x, w) - p0.x) / w;
            const float scaled = t * static_cast<float>(n - 1) + 0.5f;
            const int idx = !(scaled > 0.0f) ? 0 : (scaled >= static_cast<float>(n - 1) ? n - 1 : static_cast<int>(scaled));
            return mags[idx];
        };
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const float wheel = ImGui::GetIO().MouseWheel;

        // On press, grab the nearest node, so nodes at the edges are easy to reach.
        if (ImGui::IsItemActivated() && dragNode_ < 0)
        {
            int best = -1; float bestD = 1e30f;
            for (int ni = 0; ni < static_cast<int>(sel->curveNodes.size()); ++ni)
            {
                const auto& nd = sel->curveNodes[static_cast<size_t>(ni)];
                if (nd.freqParam < 0 || !nodeShown(*sel, nd)) continue;
                const float fHz = sel->values[static_cast<size_t>(nd.freqParam)];
                const float gain = nd.gainParam >= 0 ? sel->values[static_cast<size_t>(nd.gainParam)] : curveDbAt(fHz);
                const ImVec2 cc(freqToX(fHz, p0.x, w), dbToYeq(gain, p0.y, h));
                const float d = std::fabs(mouse.x - cc.x) + 0.35f * std::fabs(mouse.y - cc.y);
                if (d < bestD) { bestD = d; best = ni; }
            }
            if (best >= 0) { dragNode_ = best; dragOwner_ = sel; }
        }

        for (int ni = 0; ni < static_cast<int>(sel->curveNodes.size()); ++ni)
        {
            const auto& node = sel->curveNodes[static_cast<size_t>(ni)];
            if (node.freqParam < 0 || !nodeShown(*sel, node)) continue;
            const float fHz = sel->values[static_cast<size_t>(node.freqParam)];
            const float gain = node.gainParam >= 0 ? sel->values[static_cast<size_t>(node.gainParam)] : curveDbAt(fHz);
            const ImVec2 c(freqToX(fHz, p0.x, w), dbToYeq(gain, p0.y, h));
            const bool nearNode = std::fabs(mouse.x - c.x) < px(12.0f) && std::fabs(mouse.y - c.y) < px(12.0f);
            const bool active = dragNode_ == ni && dragOwner_ == sel;

            if (active && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                const auto& fp = sel->params[static_cast<size_t>(node.freqParam)];
                sel->applyParam(node.freqParam, std::clamp(xToFreq(mouse.x, p0.x, w), fp.min, fp.max));
                if (node.gainParam >= 0)
                {
                    const auto& gp = sel->params[static_cast<size_t>(node.gainParam)];
                    sel->applyParam(node.gainParam, std::clamp(yToDbEq(mouse.y, p0.y, h), gp.min, gp.max));
                }
            }
            if (hovered && nearNode && node.qParam >= 0 && wheel != 0.0f)
            {
                const auto& qp = sel->params[static_cast<size_t>(node.qParam)];
                const float nq = std::clamp(sel->values[static_cast<size_t>(node.qParam)] * (wheel > 0 ? 1.15f : 0.87f), qp.min, qp.max);
                sel->applyParam(node.qParam, nq);
            }

            const ImU32 col = (active || nearNode) ? IM_COL32(255, 232, 150, 255) : IM_COL32(255, 186, 70, 255);
            dl->AddCircleFilled(c, px(8.0f), col);
            dl->AddCircle(c, px(8.0f), IM_COL32(25, 25, 25, 255), 0, px(1.5f));
            char lbl[8]; std::snprintf(lbl, sizeof(lbl), "%d", ni + 1);
            const ImVec2 ls = ImGui::CalcTextSize(lbl);
            dl->AddText(ImVec2(c.x - ls.x * 0.5f, c.y - ls.y * 0.5f), IM_COL32(20, 20, 20, 255), lbl);

            if (active || nearNode)
            {
                char vb[64];
                const bool hasG = node.gainParam >= 0, hasQ = node.qParam >= 0;
                const float gv = hasG ? sel->values[static_cast<size_t>(node.gainParam)] : 0.0f;
                const float qv = hasQ ? sel->values[static_cast<size_t>(node.qParam)] : 0.0f;
                if (hasG && hasQ)  std::snprintf(vb, sizeof(vb), "%.0f Hz  %+.1f dB  Q %.2f", fHz, gv, qv);
                else if (hasG)     std::snprintf(vb, sizeof(vb), "%.0f Hz  %+.1f dB", fHz, gv);
                else if (hasQ)     std::snprintf(vb, sizeof(vb), "%.0f Hz  Q %.2f", fHz, qv);
                else               std::snprintf(vb, sizeof(vb), "%.0f Hz", fHz);
                const ImVec2 tsz = ImGui::CalcTextSize(vb);
                ImVec2 tp(c.x + px(12.0f), c.y - tsz.y * 0.5f);
                if (tp.x + tsz.x > p0.x + w) tp.x = c.x - px(12.0f) - tsz.x;
                dl->AddRectFilled(ImVec2(tp.x - 3, tp.y - 2), ImVec2(tp.x + tsz.x + 3, tp.y + tsz.y + 2), IM_COL32(20, 22, 28, 225), 3.0f);
                dl->AddText(tp, IM_COL32(255, 236, 185, 255), vb);
            }
        }
        if (dragNode_ >= 0 && !ImGui::IsItemActive()) { dragNode_ = -1; dragOwner_ = nullptr; }
    }

    static bool nodeShown(const EffectSlot& s, const EffectSlot::CurveNode& n)
    {
        return n.enableParam < 0 || s.values[static_cast<size_t>(n.enableParam)] > 0.5f;
    }

    static void drawMultiband(ImDrawList* dl, EffectSlot* sel, ImVec2 p0, float w, float h)
    {
        float xo[8] = {}; float gr[8] = {};
        const int nb = sel->multibandFn(xo, gr, 8);
        if (nb <= 0) return;
        const ImU32 bandCols[3] = { IM_COL32(80, 160, 90, 36), IM_COL32(90, 130, 200, 36), IM_COL32(200, 130, 80, 36) };
        float left = p0.x;
        for (int b = 0; b < nb; ++b)
        {
            const float right = b < nb - 1 ? freqToX(xo[b], p0.x, w) : p0.x + w;
            dl->AddRectFilled(ImVec2(left, p0.y), ImVec2(right, p0.y + h), bandCols[b % 3]);
            if (b < nb - 1)
                dl->AddLine(ImVec2(right, p0.y), ImVec2(right, p0.y + h), IM_COL32(150, 150, 160, 200), 1.5f);
            const float grDb = std::clamp(gr[b], 0.0f, 24.0f);
            const float barH = grDb / 24.0f * (h * 0.5f);
            dl->AddRectFilled(ImVec2(left + 2, p0.y), ImVec2(right - 2, p0.y + barH), IM_COL32(230, 90, 70, 150));
            char lbl[24]; std::snprintf(lbl, sizeof(lbl), "-%.1f dB", grDb);
            dl->AddText(ImVec2(left + px(6.0f), p0.y + px(30.0f)), IM_COL32(235, 235, 240, 255), lbl);
            left = right;
        }
    }

    static void drawLegend(ImDrawList* dl, const EffectSlot* sel, ImVec2 p0, float w)
    {
        struct Item { const std::string* text; ImU32 col; bool filled; };
        Item items[4];
        int n = 0;
        if (sel->magnitudeFn && !sel->liveLabel.empty()) items[n++] = { &sel->liveLabel, IM_COL32(255, 186, 70, 255), false };
        if (sel->secondaryMagnitudeFn && !sel->secondaryLabel.empty()) items[n++] = { &sel->secondaryLabel, IM_COL32(110, 210, 230, 255), false };
        if (sel->targetMagnitudeFn && !sel->targetLabel.empty()) items[n++] = { &sel->targetLabel, IM_COL32(170, 182, 210, 200), false };
        if (sel->activityFn && !sel->activityLabel.empty()) items[n++] = { &sel->activityLabel, IM_COL32(230, 80, 70, 170), true };
        float x = p0.x + w - px(10.0f);
        const float y = p0.y + px(8.0f);
        for (int i = n - 1; i >= 0; --i)
        {
            const ImVec2 ts = ImGui::CalcTextSize(items[i].text->c_str());
            x -= ts.x;
            dl->AddText(ImVec2(x, y), IM_COL32(200, 205, 215, 255), items[i].text->c_str());
            x -= px(22.0f);
            const float my = y + ts.y * 0.5f;
            if (items[i].filled) dl->AddRectFilled(ImVec2(x, my - px(5.0f)), ImVec2(x + px(16.0f), my + px(5.0f)), items[i].col, px(2.0f));
            else dl->AddLine(ImVec2(x, my), ImVec2(x + px(16.0f), my), items[i].col, px(2.5f));
            x -= px(14.0f);
        }
    }

    // --- Activity strip: gain reduction over time, scope, meters, test tone -------------

    void drawActivity(AudioEngine& engine, EffectSlot* sel, float h)
    {
        ImGui::BeginChild("Activity", ImVec2(0, h), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
        const float avail = ImGui::GetContentRegionAvail().x;
        const float innerH = ImGui::GetContentRegionAvail().y;
        const float toneW = px(300.0f), meterW = px(118.0f);
        const bool hasGr = sel && sel->gainReductionDbFn;
        const float graphW = std::max(px(120.0f), avail - toneW - meterW - 3.0f * ImGui::GetStyle().ItemSpacing.x);

        if (hasGr)
        {
            // + 0.0f: a reduction of zero reads as 0, not as -0.
            const float gr = std::clamp(-sel->gainReductionDbFn(), 0.0f, 48.0f) + 0.0f;
            grHistory_[static_cast<size_t>(grHead_)] = gr;
            grHead_ = (grHead_ + 1) % kHistory;
            drawGrHistory(gr, graphW, innerH, sel->statusFn ? sel->statusFn() : std::string());
        }
        else
            drawScope(engine, graphW, innerH);

        ImGui::SameLine();
        drawMeters(engine, meterW, innerH);
        ImGui::SameLine();
        ImGui::BeginChild("Tone", ImVec2(0, innerH));
        drawTone(engine);
        ImGui::EndChild();
        ImGui::EndChild();
    }

    void drawGrHistory(float now, float w, float h, const std::string& status)
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(w, h));
        const ImVec2 p1(p0.x + w, p0.y + h);
        dl->AddRectFilled(p0, p1, IM_COL32(15, 17, 22, 255), 4.0f);
        float peak = 6.0f;
        for (float v : grHistory_) peak = std::max(peak, v);
        const float scale = std::ceil(peak / 6.0f) * 6.0f;
        for (float d = 6.0f; d < scale; d += 6.0f)
        {
            const float y = p0.y + d / scale * h;
            dl->AddLine(ImVec2(p0.x, y), ImVec2(p1.x, y), IM_COL32(34, 38, 47, 255));
        }
        const float dx = w / static_cast<float>(kHistory - 1);
        for (int i = 1; i < kHistory; ++i)
        {
            const float a = grHistory_[static_cast<size_t>((grHead_ + i - 1) % kHistory)];
            const float b = grHistory_[static_cast<size_t>((grHead_ + i) % kHistory)];
            const float x0 = p0.x + (i - 1) * dx, x1 = p0.x + i * dx;
            const float ya = p0.y + a / scale * h, yb = p0.y + b / scale * h;
            dl->AddQuadFilled(ImVec2(x0, p0.y), ImVec2(x1, p0.y), ImVec2(x1, yb), ImVec2(x0, ya), IM_COL32(230, 95, 70, 110));
            dl->AddLine(ImVec2(x0, ya), ImVec2(x1, yb), IM_COL32(250, 130, 95, 230), 1.5f);
        }
        char b[48];
        std::snprintf(b, sizeof(b), "Gain reduction  %.1f dB", now);
        dl->AddText(ImVec2(p0.x + px(6.0f), p1.y - ImGui::GetTextLineHeight() - px(4.0f)), IM_COL32(215, 220, 230, 255), b);
        std::snprintf(b, sizeof(b), "%.0f dB", scale);
        dl->AddText(ImVec2(p1.x - ImGui::CalcTextSize(b).x - px(6.0f), p1.y - ImGui::GetTextLineHeight() - px(4.0f)), IM_COL32(120, 126, 140, 255), b);
        if (!status.empty())
        {
            const ImVec2 ts = ImGui::CalcTextSize(status.c_str());
            const ImVec2 a(p1.x - ts.x - px(12.0f), p0.y + px(5.0f));
            dl->AddRectFilled(a, ImVec2(p1.x - px(4.0f), a.y + ts.y + px(6.0f)), IM_COL32(20, 22, 28, 220), px(3.0f));
            dl->AddText(ImVec2(a.x + px(4.0f), a.y + px(3.0f)), IM_COL32(255, 214, 140, 255), status.c_str());
        }
        dl->AddRect(p0, p1, IM_COL32(58, 64, 78, 255), 4.0f);
    }

    void drawScope(AudioEngine& engine, float w, float h)
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(w, h));
        const ImVec2 p1(p0.x + w, p0.y + h);
        dl->AddRectFilled(p0, p1, IM_COL32(15, 17, 22, 255), 4.0f);
        const float mid = p0.y + h * 0.5f;
        dl->AddLine(ImVec2(p0.x, mid), ImVec2(p1.x, mid), IM_COL32(40, 44, 54, 255));
        // The engine overwrites its snapshot on the audio thread: draw a copy.
        const int n = engine.getWaveform(waveL_.data(), waveR_.data(), static_cast<int>(waveL_.size()));
        if (n > 1)
        {
            static ImVec2 pts[AudioEngine::kBlockSize];
            for (int i = 0; i < n; ++i)
                pts[i] = ImVec2(p0.x + w * static_cast<float>(i) / static_cast<float>(n - 1),
                                mid - std::clamp(waveL_[static_cast<size_t>(i)], -1.0f, 1.0f) * h * 0.46f);
            dl->AddPolyline(pts, n, IM_COL32(110, 200, 150, 230), 0, 1.3f);
        }
        dl->AddText(ImVec2(p0.x + px(6.0f), p1.y - ImGui::GetTextLineHeight() - px(4.0f)), IM_COL32(150, 156, 170, 255), "Output");
        dl->AddRect(p0, p1, IM_COL32(58, 64, 78, 255), 4.0f);
    }

    static void drawMeters(AudioEngine& engine, float w, float h)
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(w, h));
        // Two thin bars (-60 to 0 dBFS, ticks every 12 dB) and, beside them,
        // the peak of each channel in figures: nothing overlaps.
        const float peaks[2] = { engine.getPeakL(), engine.getPeakR() };
        const char* names[2] = { "L", "R" };
        const float barW = px(11.0f), gap = px(5.0f);
        const float top = p0.y + px(3.0f), bottom = p0.y + h - px(3.0f);
        for (int c = 0; c < 2; ++c)
        {
            const float db = peaks[c] > 0.0f ? 20.0f * std::log10(peaks[c]) : -100.0f;
            const float t = std::clamp((db + 60.0f) / 60.0f, 0.0f, 1.0f);
            const float x0 = p0.x + px(2.0f) + static_cast<float>(c) * (barW + gap);
            dl->AddRectFilled(ImVec2(x0, top), ImVec2(x0 + barW, bottom), IM_COL32(24, 27, 34, 255), px(2.0f));
            const float y = bottom - t * (bottom - top);
            const ImU32 col = db > -1.0f ? IM_COL32(235, 70, 60, 255) : db > -9.0f ? IM_COL32(235, 200, 70, 255) : IM_COL32(80, 200, 110, 255);
            dl->AddRectFilled(ImVec2(x0, y), ImVec2(x0 + barW, bottom), col, px(2.0f));
            for (int d = 12; d < 60; d += 12)
            {
                const float ty = bottom - (1.0f - static_cast<float>(d) / 60.0f) * (bottom - top);
                dl->AddLine(ImVec2(x0, ty), ImVec2(x0 + barW, ty), IM_COL32(15, 17, 22, 200), 1.0f);
            }
        }
        const float tx = p0.x + px(2.0f) + 2.0f * barW + gap + px(8.0f);
        const float lh = ImGui::GetTextLineHeight();
        for (int c = 0; c < 2; ++c)
        {
            const float db = peaks[c] > 0.0f ? std::max(-99.9f, 20.0f * std::log10(peaks[c])) : -99.9f;
            char b[24];
            std::snprintf(b, sizeof(b), "%s %5.1f", names[c], db);
            dl->AddText(ImVec2(tx, p0.y + px(4.0f) + static_cast<float>(c) * lh), IM_COL32(205, 210, 220, 255), b);
        }
        dl->AddText(ImVec2(tx, p0.y + px(4.0f) + 2.0f * lh), IM_COL32(110, 116, 130, 255), "dBFS peak");
    }

    void drawTone(AudioEngine& engine)
    {
        bool tone = engine.getTestTone();
        if (ImGui::Checkbox("Test tone", &tone))
            engine.setTestTone(tone);
        ImGui::SetItemTooltip("A sine through the chain instead of the file, with THD+N read at its frequency");
        if (!engine.isDeviceReady())
        {
            ImGui::TextDisabled("Load a file once to start the audio");
            return;
        }
        if (!tone) { ImGui::TextDisabled("THD+N reads with the tone on"); return; }

        float freq = engine.getTestToneFreq();
        float db = engine.getTestToneDb();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(px(96.0f));
        if (ImGui::SliderFloat("##f", &freq, 20.0f, 20000.0f, "%.0f Hz", ImGuiSliderFlags_Logarithmic))
            engine.setTestToneFreq(freq);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(px(84.0f));
        if (ImGui::SliderFloat("##l", &db, -60.0f, 0.0f, "%.0f dB"))
            engine.setTestToneDb(db);

        // A few readings per second: enough for a live meter, cheap on the GUI thread.
        thdTimer_ += ImGui::GetIO().DeltaTime;
        if (thdTimer_ > 0.25f) { thdTimer_ = 0.0f; thdLast_ = engine.computeThd(); }
        if (thdLast_.valid)
        {
            if (thdLast_.thdnPercent < 0.001f) ImGui::Text("THD+N < 0.001 %%");
            else ImGui::Text("THD+N %.3f %%", thdLast_.thdnPercent);
            ImGui::TextDisabled("Fundamental %.1f dBFS  RMS %.1f dBFS", thdLast_.fundDb, thdLast_.rmsDb);
        }
        else
            ImGui::TextDisabled("THD+N: settling...");
    }

    // --- Controls: knobs, switches and choices on tabs ----------------------------------

    /// Width of the segmented buttons of a choice (0 when it has too many options).
    static float segmentWidth(const ParamDesc& pd)
    {
        if (pd.choices.empty() || pd.choices.size() > 3) return 0.0f;
        float w = 0.0f;
        for (const auto& c : pd.choices) w += ImGui::CalcTextSize(c.c_str()).x + px(16.0f);
        return w;
    }

    /// Cells a control takes: knobs and switches one; a choice two when its
    /// buttons or its list need the room.
    static int spanOf(const ParamDesc& pd)
    {
        if (pd.type != ParamDesc::Choice) return 1;
        const float inner = cellW() - px(12.0f);
        const float seg = segmentWidth(pd);
        if (seg > 0.0f) return seg <= inner ? 1 : 2;
        float widest = 0.0f;
        for (const auto& c : pd.choices) widest = std::max(widest, ImGui::CalcTextSize(c.c_str()).x + px(36.0f));
        return widest > inner ? 2 : 1;
    }

    /// The rows of one tab laid out in a width, as the height they need.
    static float tabHeight(const EffectSlot& s, const std::string& tab, float width)
    {
        const int cols = std::max(1, static_cast<int>(width / cellW()));
        int col = 0, rows = 0;
        for (int i = 0; i < static_cast<int>(s.params.size()); ++i)
        {
            if (s.groupOf(i) != tab) continue;
            const int sp = std::min(cols, spanOf(s.params[static_cast<size_t>(i)]));
            if (rows == 0) rows = 1;
            if (col + sp > cols) { ++rows; col = 0; }
            col += sp;
        }
        return static_cast<float>(rows) * cellH();
    }

    static float controlsHeight(const EffectSlot& s, float width)
    {
        const auto tabs = s.tabs();
        float tallest = 0.0f;
        for (const auto& t : tabs) tallest = std::max(tallest, tabHeight(s, t, width - px(24.0f)));
        const float tabBar = tabs.size() > 1 ? ImGui::GetFrameHeight() + px(10.0f) : 0.0f;
        return tallest + tabBar + 2.0f * ImGui::GetStyle().WindowPadding.y + px(6.0f);
    }

    void drawControls(EffectSlot* sel)
    {
        // Mirror what the processor moved itself (a preset), unless a widget is held.
        if (sel->readbackFn && !ImGui::IsAnyItemActive())
            sel->readbackFn(sel->values);

        const auto tabs = sel->tabs();
        ImGui::PushID(sel->name.c_str());
        if (tabs.size() > 1)
        {
            if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_FittingPolicyScroll))
            {
                for (const auto& t : tabs)
                    if (ImGui::BeginTabItem(t.c_str()))
                    {
                        drawGrid(sel, t);
                        ImGui::EndTabItem();
                    }
                ImGui::EndTabBar();
            }
        }
        else
            drawGrid(sel, tabs.front());
        ImGui::PopID();
    }

    void drawGrid(EffectSlot* sel, const std::string& tab)
    {
        const float width = ImGui::GetContentRegionAvail().x;
        const int cols = std::max(1, static_cast<int>(width / cellW()));
        const ImVec2 origin = ImGui::GetCursorPos();
        int col = 0, row = 0;
        for (int i = 0; i < static_cast<int>(sel->params.size()); ++i)
        {
            if (sel->groupOf(i) != tab) continue;
            const auto& pd = sel->params[static_cast<size_t>(i)];
            const int sp = std::min(cols, spanOf(pd));
            if (col + sp > cols) { ++row; col = 0; }
            const ImVec2 cell(origin.x + static_cast<float>(col) * cellW(), origin.y + static_cast<float>(row) * cellH());
            ImGui::PushID(i);
            const float cellWidth = static_cast<float>(sp) * cellW();
            switch (pd.type)
            {
                case ParamDesc::Slider: knob(sel, i, cell, cellWidth); break;
                case ParamDesc::Toggle: toggle(sel, i, cell, cellWidth); break;
                case ParamDesc::Choice: choice(sel, i, cell, cellWidth); break;
            }
            ImGui::PopID();
            col += sp;
        }
        ImGui::SetCursorPos(ImVec2(origin.x, origin.y + static_cast<float>(row + 1) * cellH()));
        ImGui::Dummy(ImVec2(1, 1));
    }

    static bool integral(const ParamDesc& pd)
    {
        const auto whole = [](float v) { return std::fabs(v - std::round(v)) < 1e-4f; };
        return whole(pd.min) && whole(pd.max) && whole(pd.defaultVal) && pd.max - pd.min >= 3.0f
            && (pd.unit.empty() || pd.unit == "st" || pd.unit == "BPM");
    }

    static std::string valueText(const ParamDesc& pd, float v)
    {
        char b[48];
        if (pd.unit == "Hz" && std::fabs(v) >= 1000.0f) std::snprintf(b, sizeof(b), "%.2f kHz", v / 1000.0f);
        else if (pd.unit == "Hz") std::snprintf(b, sizeof(b), std::fabs(v) < 100.0f ? "%.1f Hz" : "%.0f Hz", v);
        else if (pd.unit == ":1") std::snprintf(b, sizeof(b), "%.1f:1", v);
        else if (integral(pd)) std::snprintf(b, sizeof(b), pd.unit.empty() ? "%.0f" : "%.0f %s", v, pd.unit.c_str());
        else if (pd.unit.empty()) std::snprintf(b, sizeof(b), "%.2f", v);
        else if (pd.unit == "ms" && std::fabs(v) >= 100.0f) std::snprintf(b, sizeof(b), "%.0f ms", v);
        else std::snprintf(b, sizeof(b), "%.1f %s", v, pd.unit.c_str());
        // A value that rounds to zero reads as 0, not as -0.
        std::string out = b;
        if (out.rfind("-0", 0) == 0 && out.find_first_of("123456789") == std::string::npos) out.erase(0, 1);
        return out;
    }

    static float toNorm(const ParamDesc& pd, float v)
    {
        if (pd.max <= pd.min) return 0.0f;
        if (pd.logarithmic && pd.min > 0.0f) return std::clamp(std::log(v / pd.min) / std::log(pd.max / pd.min), 0.0f, 1.0f);
        return std::clamp((v - pd.min) / (pd.max - pd.min), 0.0f, 1.0f);
    }
    static float fromNorm(const ParamDesc& pd, float t)
    {
        t = std::clamp(t, 0.0f, 1.0f);
        if (pd.logarithmic && pd.min > 0.0f) return pd.min * std::pow(pd.max / pd.min, t);
        return pd.min + t * (pd.max - pd.min);
    }

    /// The name of a control over its cell, in up to two centred lines (a
    /// short name sits on the lower one, next to the control).
    static void drawName(ImDrawList* dl, const std::string& name, ImVec2 topLeft, float w, ImU32 col)
    {
        const float lh = ImGui::GetTextLineHeight();
        const auto lines = twoLines(name, w - px(8.0f));
        for (int k = 0; k < 2; ++k)
        {
            const std::string& t = lines[static_cast<size_t>(k)];
            if (t.empty()) continue;
            const float tw = ImGui::CalcTextSize(t.c_str()).x;
            dl->AddText(ImVec2(topLeft.x + (w - tw) * 0.5f, topLeft.y + static_cast<float>(k) * lh), col, t.c_str());
        }
    }

    /// Screen position of a point of the window's content.
    static ImVec2 toScreen(ImVec2 local)
    {
        const ImVec2 wp = ImGui::GetWindowPos();
        return ImVec2(wp.x + local.x - ImGui::GetScrollX(), wp.y + local.y - ImGui::GetScrollY());
    }

    void knob(EffectSlot* sel, int i, ImVec2 cell, float w)
    {
        const auto& pd = sel->params[static_cast<size_t>(i)];
        float val = sel->values[static_cast<size_t>(i)];
        const float lh = ImGui::GetTextLineHeight(), r = knobR();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 s0 = toScreen(cell);
        drawName(dl, pd.name, s0, w, ImGui::GetColorU32(ImGuiCol_Text));

        const float knobTop = cell.y + 2.0f * lh + px(4.0f);
        ImGui::SetCursorPos(ImVec2(cell.x, knobTop));
        ImGui::InvisibleButton("##k", ImVec2(w, 2.0f * r + px(4.0f)));
        const bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
        const ImGuiIO& io = ImGui::GetIO();
        float t = toNorm(pd, val);
        bool changed = false;
        if (active && io.MouseDelta.y != 0.0f)
        {
            t -= io.MouseDelta.y / uiScale() * (io.KeyShift ? 0.0012f : 0.0060f);
            changed = true;
        }
        if (hovered && io.MouseWheel != 0.0f)
        {
            t += io.MouseWheel * (io.KeyShift ? 0.005f : 0.025f);
            changed = true;
        }
        if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        {
            val = pd.defaultVal;
            sel->applyParam(i, val);
        }
        else if (changed)
        {
            val = fromNorm(pd, t);
            if (val != sel->values[static_cast<size_t>(i)]) sel->applyParam(i, val);
        }
        if (hovered || active)
            ImGui::SetTooltip("%s\n%s   (%s to %s)\nDrag: change, Shift: fine, double-click: %s",
                              pd.name.c_str(), valueText(pd, val).c_str(), valueText(pd, pd.min).c_str(),
                              valueText(pd, pd.max).c_str(), valueText(pd, pd.defaultVal).c_str());

        const ImVec2 centre(s0.x + w * 0.5f, toScreen(ImVec2(0.0f, knobTop)).y + r + px(2.0f));
        const float a0 = 0.75f * 3.14159265f, a1 = 2.25f * 3.14159265f;
        const float tv = toNorm(pd, val);
        const bool bipolar = !pd.logarithmic && pd.min < 0.0f && pd.max > 0.0f;
        const float tz = bipolar ? toNorm(pd, 0.0f) : 0.0f;
        const float thick = px(4.0f);
        dl->PathArcTo(centre, r, a0, a1, 48);
        dl->PathStroke(IM_COL32(48, 53, 65, 255), ImDrawFlags_None, thick);
        const float from = a0 + (a1 - a0) * std::min(tz, tv), to = a0 + (a1 - a0) * std::max(tz, tv);
        if (to - from > 0.01f)
        {
            dl->PathArcTo(centre, r, from, to, 40);
            dl->PathStroke(active ? IM_COL32(255, 214, 130, 255) : IM_COL32(255, 160, 50, 255), ImDrawFlags_None, thick);
        }
        dl->AddCircleFilled(centre, r - px(6.0f), hovered || active ? IM_COL32(52, 57, 70, 255) : IM_COL32(40, 44, 55, 255), 32);
        const float av = a0 + (a1 - a0) * tv;
        dl->AddLine(ImVec2(centre.x + std::cos(av) * (r - px(14.0f)), centre.y + std::sin(av) * (r - px(14.0f))),
                    ImVec2(centre.x + std::cos(av) * (r - px(6.0f)), centre.y + std::sin(av) * (r - px(6.0f))),
                    IM_COL32(240, 242, 246, 255), px(2.5f));

        const std::string vt = fitText(valueText(pd, val), w - px(4.0f));
        const float tw = ImGui::CalcTextSize(vt.c_str()).x;
        dl->AddText(ImVec2(s0.x + (w - tw) * 0.5f, centre.y + r + px(4.0f)),
                    active ? IM_COL32(255, 216, 140, 255) : IM_COL32(184, 191, 209, 255), vt.c_str());
    }

    void toggle(EffectSlot* sel, int i, ImVec2 cell, float w)
    {
        const auto& pd = sel->params[static_cast<size_t>(i)];
        const bool on = sel->values[static_cast<size_t>(i)] > 0.5f;
        const float lh = ImGui::GetTextLineHeight();
        // A switch: a rounded box carrying the name in up to two lines,
        // lit when on, with its state under it.
        const float bw = w - px(12.0f), bh = 2.0f * lh + px(14.0f);
        const ImVec2 local(cell.x + px(6.0f), cell.y + lh + px(2.0f));
        ImGui::SetCursorPos(local);
        if (ImGui::InvisibleButton("##t", ImVec2(bw, bh)))
            sel->applyParam(i, on ? 0.0f : 1.0f);
        const bool hovered = ImGui::IsItemHovered();
        ImGui::SetItemTooltip("%s: %s (click to switch)", pd.name.c_str(), on ? "on" : "off");
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 a = toScreen(local), b(a.x + bw, a.y + bh);
        const ImU32 fill = on ? (hovered ? IM_COL32(222, 136, 44, 255) : IM_COL32(190, 112, 34, 255))
                              : (hovered ? IM_COL32(60, 66, 82, 255) : IM_COL32(44, 49, 61, 255));
        dl->AddRectFilled(a, b, fill, px(6.0f));
        const auto lines = twoLines(pd.name, bw - px(10.0f));
        const int n = lines[0].empty() ? 1 : 2;
        const float textTop = a.y + (bh - static_cast<float>(n) * lh) * 0.5f;
        for (int k = 2 - n; k < 2; ++k)
        {
            const std::string& t = lines[static_cast<size_t>(k)];
            const float tw = ImGui::CalcTextSize(t.c_str()).x;
            dl->AddText(ImVec2(a.x + (bw - tw) * 0.5f, textTop + static_cast<float>(k - (2 - n)) * lh),
                        on ? IM_COL32(25, 18, 8, 255) : IM_COL32(220, 224, 232, 255), t.c_str());
        }
        const char* state = on ? "On" : "Off";
        const float sw = ImGui::CalcTextSize(state).x;
        dl->AddText(ImVec2(a.x + (bw - sw) * 0.5f, b.y + px(4.0f)), IM_COL32(150, 156, 170, 255), state);
    }

    void choice(EffectSlot* sel, int i, ImVec2 cell, float w)
    {
        const auto& pd = sel->params[static_cast<size_t>(i)];
        const int nOpts = static_cast<int>(pd.choices.size());
        const int cur = std::clamp(static_cast<int>(sel->values[static_cast<size_t>(i)] + 0.5f), 0, std::max(0, nOpts - 1));
        const float lh = ImGui::GetTextLineHeight();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        drawName(dl, pd.name, toScreen(cell), w, ImGui::GetColorU32(ImGuiCol_Text));

        const float inner = w - px(12.0f);
        const float seg = segmentWidth(pd);
        const float controlTop = cell.y + 2.0f * lh + px(4.0f) + knobR() - ImGui::GetFrameHeight() * 0.5f;
        ImGui::SetCursorPos(ImVec2(cell.x + px(6.0f), controlTop));
        if (seg > 0.0f && seg <= inner)
        {
            // Few short options: segmented buttons, the chosen one lit.
            const float bw = inner / static_cast<float>(nOpts);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(2.0f), 0.0f));
            for (int c = 0; c < nOpts; ++c)
            {
                if (c > 0) ImGui::SameLine();
                const bool isSel = c == cur;
                ImGui::PushStyleColor(ImGuiCol_Button, isSel ? ImVec4(0.75f, 0.44f, 0.13f, 1.0f) : ImVec4(0.17f, 0.19f, 0.24f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, isSel ? ImVec4(0.10f, 0.07f, 0.03f, 1.0f) : ImGui::GetStyle().Colors[ImGuiCol_Text]);
                if (ImGui::Button((pd.choices[static_cast<size_t>(c)] + "##o" + std::to_string(c)).c_str(), ImVec2(bw - px(2.0f), 0.0f)) && !isSel)
                    sel->applyParam(i, static_cast<float>(c));
                ImGui::PopStyleColor(2);
            }
            ImGui::PopStyleVar();
            return;
        }
        ImGui::SetNextItemWidth(inner);
        if (ImGui::BeginCombo("##c", nOpts > 0 ? pd.choices[static_cast<size_t>(cur)].c_str() : ""))
        {
            for (int c = 0; c < nOpts; ++c)
            {
                const bool isSel = c == cur;
                if (ImGui::Selectable(pd.choices[static_cast<size_t>(c)].c_str(), isSel))
                    sel->applyParam(i, static_cast<float>(c));
                if (isSel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
        {
            const float wh = ImGui::GetIO().MouseWheel;
            if (wh != 0.0f)
            {
                const int nc = std::clamp(cur + (wh > 0.0f ? -1 : 1), 0, nOpts - 1);
                if (nc != cur) sel->applyParam(i, static_cast<float>(nc));
            }
            ImGui::SetTooltip("%s: %s (wheel: next or previous)", pd.name.c_str(),
                              nOpts > 0 ? pd.choices[static_cast<size_t>(cur)].c_str() : "");
        }
    }
};

} // namespace dsplab
