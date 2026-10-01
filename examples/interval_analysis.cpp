// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License
// cl /std:c++20 /EHsc /O2 /I . examples\interval_analysis.cpp
// c++ -std=c++20 -O2 -I . examples/interval_analysis.cpp -o interval_analysis

#include "Analysis/AudioIntervalAnalyzer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>

int main()
{
    using Analyzer = dspark::AudioIntervalAnalyzer<double>;
    constexpr int rate = 48000, blockSize = 512, frames = 6 * rate;
    const std::array<Analyzer::Interval, 3> regions {{
        { 13, 2 * rate + 29 }, { rate + 7, 5 * rate + 19 }, { 3 * rate, 3 * rate + 100 }
    }};
    std::array<Analyzer, 3> analyzers;
    for (size_t i = 0; i < analyzers.size(); ++i)
        if (!analyzers[i].prepare({ rate, blockSize, 1 }, regions[i],
                                 Analyzer::Context::IncludeContinuous)) return 1;
    std::array<double, blockSize> samples{};
    const double* channel[] = { samples.data() };
    // Replace this generator with a decoder/read callback. Only one block of PCM
    // is held; the regions overlap and start/end inside decoder blocks.
    for (int first = 0; first < frames; first += blockSize)
    {
        const int count = std::min(blockSize, frames - first);
        for (int n = 0; n < count; ++n)
        {
            const int frame = first + n;
            const double amplitude = frame < 3 * rate ? .1 : .01;
            samples[static_cast<size_t>(n)] = amplitude * std::sin(
                2 * std::numbers::pi * 1000 * frame / rate);
        }
        for (auto& analyzer : analyzers)
            if (!analyzer.processBlock({ channel, 1, count }, first)) return 1;
    }
    const auto print = [](const char* label, const Analyzer::Measurement& metric)
    {
        std::cout << "  " << label << ": ";
        if (metric.hasValue()) std::cout << metric.value;
        else std::cout << "unavailable (validity " << static_cast<int>(metric.validity) << ")";
        std::cout << '\n';
    };
    for (auto& analyzer : analyzers)
    {
        const auto result = analyzer.finish();
        if (!result.complete()) return 1;
        std::cout << '[' << result.measured.begin << ',' << result.measured.end << ")\n";
        print("RMS, linear", result.rms);
        print("sample peak, linear", result.samplePeak);
        print("finite true peak, dBTP", result.truePeakDbtp);
        print("region integrated, LUFS", result.region.integrated);
        print("region LRA, LU", result.region.range);
        print("region momentary, LUFS", result.region.momentary);
        print("region short-term, LUFS", result.region.shortTerm);
        print("continuous integrated, LUFS", result.continuous.integrated);
        std::cout << "  continuous input: [" << result.continuous.input.begin << ','
                  << result.continuous.input.end << ")\n";
    }
}
