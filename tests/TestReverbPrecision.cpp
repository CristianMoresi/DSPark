// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

// Public-API regressions for low-frequency output-filter roundoff.
// Dependencies: DSPark reverb and the test harness. Single-threaded.
#include "dspark_test.h"
#include "../Effects/AlgorithmicReverb.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
struct Residual {
    double peak = 0.0;
    double sumSquares = 0.0;
    std::uint64_t count = 0;
    bool finite = true;
    void add(double first, double second) {
        finite = finite && std::isfinite(first) && std::isfinite(second);
        const double error = first - second;
        peak = std::max(peak, std::abs(error));
        sumSquares += error * error;
        ++count;
    }
    double rms() const { return std::sqrt(sumSquares / static_cast<double>(count)); }
};

template<class T>
Residual sensitivity(int rate, int type, int quality, bool endpoints = false) {
    constexpr int block = 512;
    using Reverb = dspark::AlgorithmicReverb<T>;
    Reverb reference, perturbed;
    for (auto* room : { &reference, &perturbed }) {
        room->setType(static_cast<typename Reverb::Type>(type));
        room->setQuality(static_cast<typename Reverb::Quality>(quality));
        room->setDecay(endpoints ? T(20) : T(1.05f));
        room->setMix(T(0.145f));
        room->setSize(endpoints ? T(1) : T(0.52f));
        room->setDamping(endpoints ? T(0) : T(0.5f));
        room->setPreDelay(T(18));
        room->setWidth(endpoints ? T(2) : T(1.6f));
        if (endpoints) room->setModulation(T(1));
        room->prepare({ static_cast<double>(rate), block, 2 });
    }
    std::array<std::array<T, block>, 4> data{};
    std::array<T*, 2> first{ data[0].data(), data[1].data() };
    std::array<T*, 2> second{ data[2].data(), data[3].data() };
    std::uint32_t seed = 0x2641825u;
    Residual residual;
    // Exactly represented dyadic noise, then silence. Every 997th scalar
    // sample is moved by one ULP; there is no platform-dependent oscillator.
    // The old float output filters turned a -156.5 dBFS input perturbation
    // into a -105.2 dBFS output residual at 48 kHz with these Room settings.
    for (int at = 0; at < 4 * rate; at += block) {
        const int count = std::min(block, 4 * rate - at);
        for (int i = 0; i < count; ++i) for (int c = 0; c < 2; ++c) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            const int code = static_cast<int>(seed & 65535u) - 32768;
            const T value = at + i < 3 * rate ? T(code) / T(131072) : T(0);
            first[c][i] = value;
            const auto ordinal = static_cast<std::uint64_t>(at + i) * 2u + static_cast<unsigned>(c);
            second[c][i] = ordinal % 997u == 123u && value != T(0)
                ? std::nextafter(value, T(1)) : value;
        }
        reference.processBlock({ first.data(), 2, count });
        perturbed.processBlock({ second.data(), 2, count });
        for (int i = 0; i < count; ++i) for (int c = 0; c < 2; ++c)
            residual.add(static_cast<double>(first[c][i]), static_cast<double>(second[c][i]));
    }
    return residual;
}
} // namespace

DSPARK_TEST(AlgoReverb_float_small_input_perturbation_stays_small) {
    for (int rate : { 44100, 48000, 96000, 192000 }) {
        const auto result = sensitivity<float>(rate, 0, 0);
        EXPECT_TRUE(result.finite);
        EXPECT_LT(result.peak, 1e-6);
        EXPECT_LT(result.rms(), 1e-7);
    }
}

DSPARK_TEST(AlgoReverb_double_small_input_perturbation_stays_small) {
    for (int rate : { 44100, 48000, 96000, 192000 }) {
        const auto result = sensitivity<double>(rate, 0, 0);
        EXPECT_TRUE(result.finite);
        EXPECT_LT(result.peak, 1e-12);
        EXPECT_LT(result.rms(), 1e-13);
    }
}

DSPARK_TEST(AlgoReverb_output_precision_covers_presets_and_qualities) {
    for (int type = 0; type < 6; ++type) for (int quality = 0; quality < 2; ++quality) {
        const auto result = sensitivity<float>(48000, type, quality);
        EXPECT_TRUE(result.finite);
        EXPECT_LT(result.peak, 1e-6);
        EXPECT_LT(result.rms(), 1e-7);
    }
}

DSPARK_TEST(AlgoReverb_output_precision_at_long_bright_wide_modulated_endpoints) {
    for (int rate : { 48000, 192000 }) {
        const auto result = sensitivity<float>(rate, 0, 0, true);
        EXPECT_TRUE(result.finite);
        EXPECT_LT(result.peak, 1e-6);
        EXPECT_LT(result.rms(), 1e-7);
    }
}
