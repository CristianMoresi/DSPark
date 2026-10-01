// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// Cross-toolchain numerical regression, using only the public reverb API.
// Emits identical dyadic input and full output for the x64 SSE2 CI builds.
// This checks numerical agreement; it is not a perceptual quality test.

#include "../Effects/AlgorithmicReverb.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

template <class T>
bool writeFixture(const std::filesystem::path& directory, int sampleRate, int mode)
{
    static_assert(std::numeric_limits<T>::is_iec559);
    static_assert(std::endian::native == std::endian::little);
    constexpr int blockSize = 512;
    const int frames = 4 * sampleRate;
    const std::string name = std::string(sizeof(T) == 4 ? "float" : "double") +
        "-" + std::to_string(sampleRate) + "-" + std::to_string(mode);
    dspark::AlgorithmicReverb<T> reverb;
    reverb.setType(dspark::AlgorithmicReverb<T>::Type::Chamber);
    reverb.setDecay(T(2.5));
    reverb.setSize(T(0.7f));
    reverb.setDamping(T(0.4f));
    reverb.setPreDelay(T(20));
    reverb.setMix(T(1));
    if (mode != 1) reverb.setModulation(mode == 0 ? T(0) : T(1));
    reverb.prepare({ static_cast<double>(sampleRate), blockSize, 2 });

    std::ofstream input(directory / (name + ".input"), std::ios::binary);
    std::ofstream output(directory / (name + ".output"), std::ios::binary);
    if (!input || !output) return false;
    std::array<T, blockSize> left{}, right{};
    T* channels[] = { left.data(), right.data() };
    std::array<T, 2 * blockSize> interleaved{};
    std::uint32_t state = 0x1badb002u;
    for (int offset = 0; offset < frames; offset += blockSize)
    {
        const int count = std::min(blockSize, frames - offset);
        for (int n = 0; n < count; ++n)
            for (int ch = 0; ch < 2; ++ch)
            {
                state = state * 1664525u + 1013904223u;
                const int value = static_cast<int>(state >> 16) - 32768;
                channels[ch][n] = static_cast<T>(value) * T(1.0 / 262144.0);
                interleaved[static_cast<size_t>(2 * n + ch)] = channels[ch][n];
            }
        const auto bytes = static_cast<std::streamsize>(2 * count * sizeof(T));
        input.write(reinterpret_cast<const char*>(interleaved.data()), bytes);
        reverb.processBlock({ channels, 2, count });
        for (int n = 0; n < count; ++n)
            for (int ch = 0; ch < 2; ++ch)
            {
                if (!std::isfinite(channels[ch][n])) return false;
                interleaved[static_cast<size_t>(2 * n + ch)] = channels[ch][n];
            }
        output.write(reinterpret_cast<const char*>(interleaved.data()), bytes);
    }
    input.close();
    output.close();
    return input.good() && output.good();
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    std::error_code error;
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory, error);
    if (error) return 1;
    for (int rate : { 44100, 48000, 96000 })
        for (int mode : { 0, 1, 2 }) // Static, preset and maximum modulation.
            if (!writeFixture<float>(directory, rate, mode) ||
                !writeFixture<double>(directory, rate, mode)) return 1;
    std::cout << "Wrote 18 finite stereo reverb fixtures.\n";
}
