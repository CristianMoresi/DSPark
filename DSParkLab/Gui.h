// DSParkLab — GUI
// ImGui-based interface: transport, effect list, parameter panel, visualization.

#pragma once

#include "AudioEngine.h"
#include "EffectSlot.h"

#include "vendor/imgui/imgui.h"

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

        float listW = 220.0f;

        ImGui::BeginChild("EffectList", ImVec2(listW, 0), true);
        drawEffectList(effects);
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("RightPanel", ImVec2(0, 0), false);
        drawParamPanel(effects);
        ImGui::Separator();
        drawVisualization(engine);
        ImGui::EndChild();

        ImGui::End();
    }

private:
    // --- Transport bar ---------------------------------------------------------

    void drawTransport(AudioEngine& engine)
    {
        // Open file
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

        ImGui::SameLine();

        // Play/Pause/Stop
        bool playing = engine.isPlaying();
        if (ImGui::Button(playing ? "Pause" : "Play"))
            engine.togglePlay();
        ImGui::SameLine();
        if (ImGui::Button("Stop"))
            engine.stop();

        ImGui::SameLine();
        bool loop = engine.isLooping();
        if (ImGui::Checkbox("Loop", &loop))
            engine.setLooping(loop);

        // Seek bar
        ImGui::SameLine();
        float pos = engine.getPosition();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 120.0f);
        if (ImGui::SliderFloat("##seek", &pos, 0.0f, 1.0f, ""))
            engine.seekTo(pos);

        // Time display
        ImGui::SameLine();
        float dur = engine.getDurationSeconds();
        float cur = pos * dur;
        char timeBuf[32];
        std::snprintf(timeBuf, sizeof(timeBuf), "%02d:%02d / %02d:%02d",
                      static_cast<int>(cur) / 60, static_cast<int>(cur) % 60,
                      static_cast<int>(dur) / 60, static_cast<int>(dur) % 60);
        ImGui::Text("%s", timeBuf);

        // Second row: file info + bypass
        if (engine.hasFile())
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

            // Category header
            if (e.category != currentCat)
            {
                currentCat = e.category;
                catOpen = ImGui::CollapsingHeader(currentCat.c_str(),
                                                  ImGuiTreeNodeFlags_DefaultOpen);
            }

            if (!catOpen) continue;

            ImGui::PushID(i);

            // Enable checkbox
            ImGui::Checkbox("##en", &e.enabled);
            ImGui::SameLine();

            // Selectable name
            bool sel = e.selected;
            if (ImGui::Selectable(e.name.c_str(), sel))
            {
                // Deselect all, select this one
                for (auto& eff : effects) eff.selected = false;
                e.selected = true;
            }

            ImGui::PopID();
        }
    }

    // --- Parameter panel (right top) -------------------------------------------

    void drawParamPanel(std::vector<EffectSlot>& effects)
    {
        EffectSlot* sel = nullptr;
        for (auto& e : effects)
            if (e.selected) { sel = &e; break; }

        float panelH = ImGui::GetContentRegionAvail().y * 0.55f;
        ImGui::BeginChild("Params", ImVec2(0, panelH), true);

        if (!sel)
        {
            ImGui::TextDisabled("Select an effect from the list");
            ImGui::EndChild();
            return;
        }

        ImGui::Text("%s", sel->name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", sel->category.c_str());
        ImGui::SameLine(ImGui::GetWindowWidth() - 140);
        if (ImGui::Button("Reset Defaults"))
            sel->applyAllDefaults();
        ImGui::Separator();

        // Auto-generate controls from ParamDesc
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
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 180.0f);

                if (pd.logarithmic)
                {
                    ImGui::SliderFloat(pd.name.c_str(), &val, pd.min, pd.max,
                                       formatStr(val, pd.unit),
                                       ImGuiSliderFlags_Logarithmic);
                }
                else
                {
                    ImGui::SliderFloat(pd.name.c_str(), &val, pd.min, pd.max,
                                       formatStr(val, pd.unit));
                }

                if (val != old)
                    sel->applyParam(i, val);
                break;
            }
            case ParamDesc::Toggle:
            {
                bool on = val > 0.5f;
                if (ImGui::Checkbox(pd.name.c_str(), &on))
                {
                    val = on ? 1.0f : 0.0f;
                    sel->applyParam(i, val);
                }
                break;
            }
            case ParamDesc::Choice:
            {
                int cur = static_cast<int>(val);
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 180.0f);

                if (ImGui::BeginCombo(pd.name.c_str(),
                    cur < static_cast<int>(pd.choices.size()) ? pd.choices[cur].c_str() : ""))
                {
                    for (int c = 0; c < static_cast<int>(pd.choices.size()); ++c)
                    {
                        bool isSel = (c == cur);
                        if (ImGui::Selectable(pd.choices[c].c_str(), isSel))
                        {
                            val = static_cast<float>(c);
                            sel->applyParam(i, val);
                        }
                        if (isSel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                break;
            }
            }

            ImGui::PopID();
        }

        // --- Gain reduction meter (if processor exposes it) -------------------
        if (sel->gainReductionDbFn)
        {
            ImGui::Separator();
            ImGui::Text("Gain Reduction");

            float gr = sel->gainReductionDbFn();           // Positive value = attenuation in dB
            if (gr < 0.0f) gr = 0.0f;
            if (gr > 24.0f) gr = 24.0f;

            float norm = gr / 24.0f;                       // 0..24 dB -> 0..1

            char lbl[32];
            std::snprintf(lbl, sizeof(lbl), "-%.1f dB", gr);

            // Color gradient: green (idle) -> yellow (light) -> red (heavy)
            ImVec4 col = norm < 0.25f ? ImVec4(0.2f, 0.8f, 0.2f, 1)
                       : norm < 0.6f  ? ImVec4(0.9f, 0.8f, 0.2f, 1)
                                      : ImVec4(1.0f, 0.3f, 0.2f, 1);

            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
            ImGui::ProgressBar(norm, ImVec2(-1, 18), lbl);
            ImGui::PopStyleColor();

            ImGui::TextDisabled("0 dB                                                  -24 dB");
        }

        ImGui::EndChild();
    }

    // --- Visualization (right bottom) ------------------------------------------

    void drawVisualization(AudioEngine& engine)
    {
        float availH = ImGui::GetContentRegionAvail().y;

        // Waveform
        int waveN = engine.getWaveformSize();
        if (waveN > 0)
        {
            ImGui::Text("Waveform");
            ImGui::PlotLines("##waveL", engine.getWaveformL(), waveN,
                             0, nullptr, -1.0f, 1.0f, ImVec2(-1, availH * 0.25f));
        }

        // Spectrum
        int bins = engine.getSpectrumBins();
        const float* specDb = engine.getSpectrumDb();
        if (bins > 0 && specDb)
        {
            ImGui::Text("Spectrum");
            // Show first half (up to Nyquist)
            int displayBins = std::min(bins / 2, 512);
            ImGui::PlotHistogram("##spec", specDb, displayBins,
                                 0, nullptr, -100.0f, 0.0f, ImVec2(-1, availH * 0.25f));
        }

        // Level meters
        ImGui::Text("Level");
        float peakL = engine.getPeakL();
        float peakR = engine.getPeakR();

        // Convert to dB for display, clamp
        float dbL = peakL > 0.0f ? 20.0f * std::log10(peakL) : -100.0f;
        float dbR = peakR > 0.0f ? 20.0f * std::log10(peakR) : -100.0f;
        float normL = (dbL + 60.0f) / 60.0f;  // -60..0 dB -> 0..1
        float normR = (dbR + 60.0f) / 60.0f;
        normL = std::max(0.0f, std::min(1.0f, normL));
        normR = std::max(0.0f, std::min(1.0f, normR));

        char lblL[32], lblR[32];
        std::snprintf(lblL, sizeof(lblL), "L  %.1f dB", dbL);
        std::snprintf(lblR, sizeof(lblR), "R  %.1f dB", dbR);

        ImVec4 colL = normL > 0.9f ? ImVec4(1,0.2f,0.2f,1) : ImVec4(0.2f,0.8f,0.2f,1);
        ImVec4 colR = normR > 0.9f ? ImVec4(1,0.2f,0.2f,1) : ImVec4(0.2f,0.8f,0.2f,1);

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colL);
        ImGui::ProgressBar(normL, ImVec2(-1, 18), lblL);
        ImGui::PopStyleColor();

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colR);
        ImGui::ProgressBar(normR, ImVec2(-1, 18), lblR);
        ImGui::PopStyleColor();
    }

    // --- Helpers ---------------------------------------------------------------

    static const char* formatStr(float val, const std::string& unit)
    {
        static char buf[64];
        if (unit.empty())
            std::snprintf(buf, sizeof(buf), "%.2f", val);
        else
            std::snprintf(buf, sizeof(buf), "%.1f %s", val, unit.c_str());
        return buf;
    }
};

} // namespace dsplab
