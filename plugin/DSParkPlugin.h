// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkPlugin.h
 * @brief Format-agnostic plugin layer: write ONE class, ship every format.
 *
 * The developer describes the plugin declaratively (a Descriptor and a
 * constexpr parameter table) and implements the familiar DSPark contract
 * (`prepare` / `processBlock` plus a `setParameter` switch). The format
 * backends (plugin/vst3/..., later CLAP and AU) translate that single class
 * into each plugin ABI — no JUCE, no external SDK to install, no base class
 * to inherit from.
 *
 * ```cpp
 * struct MyPlugin
 * {
 *     static constexpr auto descriptor = dspark::plugin::Descriptor {
 *         .name      = "My Saturator",
 *         .vendor    = "My Company",
 *         .url       = "https://example.com",
 *         .email     = "mailto:dev@example.com",
 *         .productId = "com.mycompany.mysaturator",   // NEVER change after release
 *         .version   = "1.0.0",
 *     };
 *     static constexpr auto parameters = dspark::plugin::params(
 *         dspark::plugin::param("drive", "Drive", -12.0f, 24.0f, 0.0f, "dB"),
 *         dspark::plugin::param("mix",   "Mix",     0.0f,  1.0f, 1.0f, ""));
 *
 *     void prepare(const dspark::AudioSpec& spec);
 *     void setParameter(int index, float plainValue) noexcept;  // any thread
 *     void processBlock(dspark::AudioBufferView<float> io) noexcept;
 *     // Optional (auto-detected): reset(), getLatency(), getTailSeconds(),
 *     // getState(), setState(data, size).
 * };
 * ```
 *
 * Capability detection is by C++20 concepts: implement only what applies.
 * State: the wrapper always serialises the parameter table itself (stable
 * text ids hashed to 32 bits, version-tolerant); a user getState/setState
 * blob — e.g. DSPark's StateBlob — rides along as an extra section.
 *
 * The editor story is deliberately deferred: every format shows the host's
 * generic parameter UI in v1. The layer reserves the hook (`hasEditor`)
 * so a future webview-based editor can plug in without API breakage.
 */

#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dspark::plugin {

// -- Descriptor ---------------------------------------------------------------

/** @brief Plugin category (reflected into each format's class metadata). */
enum class Category
{
    Fx,           ///< Audio effect (stereo in -> stereo out).
    Instrument    ///< Reserved for a future instrument contract (events/voices).
};

/**
 * @brief Static identity of a plugin. All fields are plain literals so the
 * whole descriptor can live in a `static constexpr` of the user class.
 *
 * `productId` (reverse-domain string) is the STABLE identity: the format
 * UIDs (VST3 class id, ...) derive from it deterministically. Changing it
 * after a release orphans every saved project — treat it like an ABI.
 */
struct Descriptor
{
    const char* name      = "DSPark Plugin";
    const char* vendor    = "DSPark";
    const char* url       = "";
    const char* email     = "";
    const char* productId = "com.dspark.plugin";
    const char* version   = "1.0.0";
    Category    category  = Category::Fx;
};

// -- Parameters ---------------------------------------------------------------

/**
 * @brief One automatable parameter. Plain values run [min, max]; hosts see
 * the normalized [0, 1] projection. `steps == 0` means continuous,
 * `steps == 1` a toggle, `steps == N` an N+1-position discrete control.
 */
struct Param
{
    const char* id   = "";     ///< Stable text id (state + automation identity).
    const char* name = "";     ///< Display name.
    float minValue   = 0.0f;
    float maxValue   = 1.0f;
    float defValue   = 0.0f;
    const char* unit = "";     ///< Display unit ("dB", "Hz", "%", ...).
    int steps        = 0;      ///< 0 continuous, 1 toggle, N discrete.
};

/** @brief Continuous parameter helper. */
constexpr Param param(const char* id, const char* name,
                      float minValue, float maxValue, float defValue,
                      const char* unit) noexcept
{
    return Param { id, name, minValue, maxValue, defValue, unit, 0 };
}

/** @brief On/off parameter helper. */
constexpr Param toggle(const char* id, const char* name, bool defaultOn) noexcept
{
    return Param { id, name, 0.0f, 1.0f, defaultOn ? 1.0f : 0.0f, "", 1 };
}

/** @brief Builds the parameter table (use inside `static constexpr auto`). */
template <typename... Ps>
constexpr std::array<Param, sizeof...(Ps)> params(Ps... ps) noexcept
{
    return { ps... };
}

// -- Normalisation ------------------------------------------------------------

/** @brief Plain [min, max] -> normalized [0, 1] (steps snap on the way back). */
constexpr double toNormalized(const Param& p, double plain) noexcept
{
    const double range = static_cast<double>(p.maxValue) - p.minValue;
    if (range <= 0.0) return 0.0;
    double n = (plain - p.minValue) / range;
    return n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
}

/** @brief Normalized [0, 1] -> plain [min, max], snapped for stepped params. */
constexpr double toPlain(const Param& p, double normalized) noexcept
{
    double n = normalized < 0.0 ? 0.0 : (normalized > 1.0 ? 1.0 : normalized);
    if (p.steps > 0)
    {
        // Snap to the nearest of steps+1 positions.
        const double scaled = n * p.steps + 0.5;
        const double idx = static_cast<double>(static_cast<long long>(scaled));
        n = idx / p.steps;
        n = n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
    }
    return p.minValue + n * (static_cast<double>(p.maxValue) - p.minValue);
}

/** @brief Formats a plain value for host display ("3.5 dB", "On", "440 Hz"). */
inline void formatValue(const Param& p, double plain, char* out, int outSize) noexcept
{
    if (p.steps == 1)
    {
        std::snprintf(out, static_cast<size_t>(outSize), "%s",
                      plain >= 0.5 * (p.minValue + p.maxValue) ? "On" : "Off");
        return;
    }
    if (p.unit && p.unit[0] != '\0')
        std::snprintf(out, static_cast<size_t>(outSize), "%.2f %s", plain, p.unit);
    else
        std::snprintf(out, static_cast<size_t>(outSize), "%.2f", plain);
}

// -- Stable hashing (parameter ids, class UIDs) -------------------------------

/** @brief FNV-1a 32-bit over a C string — the per-parameter host id. */
constexpr uint32_t hash32(const char* s) noexcept
{
    uint32_t h = 2166136261u;
    while (*s != '\0')
    {
        h ^= static_cast<uint8_t>(*s++);
        h *= 16777619u;
    }
    return h;
}

/** @brief FNV-1a 64-bit with a salt (two runs build a 128-bit class UID). */
constexpr uint64_t hash64(const char* s, uint64_t salt) noexcept
{
    uint64_t h = 14695981039346656037ull ^ salt;
    while (*s != '\0')
    {
        h ^= static_cast<uint8_t>(*s++);
        h *= 1099511628211ull;
    }
    return h;
}

/**
 * @brief Deterministic 16-byte class UID from the stable productId. Each
 * backend salts differently so VST3/CLAP/AU ids never collide.
 */
constexpr std::array<uint8_t, 16> makeUid(const char* productId, uint64_t salt) noexcept
{
    const uint64_t a = hash64(productId, salt);
    const uint64_t b = hash64(productId, salt ^ 0x9E3779B97F4A7C15ull);
    std::array<uint8_t, 16> uid {};
    for (int i = 0; i < 8; ++i)
    {
        uid[static_cast<size_t>(i)]     = static_cast<uint8_t>(a >> (8 * i));
        uid[static_cast<size_t>(i + 8)] = static_cast<uint8_t>(b >> (8 * i));
    }
    return uid;
}

// -- Capability detection -----------------------------------------------------

template <typename P>
concept HasReset = requires(P p) { p.reset(); };

template <typename P>
concept HasLatency = requires(const P p) { { p.getLatency() } -> std::convertible_to<int>; };

template <typename P>
concept HasTail = requires(const P p) { { p.getTailSeconds() } -> std::convertible_to<double>; };

template <typename P>
concept HasGetState = requires(const P p) {
    { p.getState() } -> std::convertible_to<std::vector<uint8_t>>;
};

template <typename P>
concept HasSetState = requires(P p, const uint8_t* d, size_t n) {
    { p.setState(d, n) } -> std::convertible_to<bool>;
};

/** @brief Editor hook — v1 backends report "no editor" and hosts show their
 *  generic parameter UI. A webview-based editor layer can claim this later. */
template <typename P>
concept HasEditor = requires { P::hasEditor; } && P::hasEditor;

// -- Wrapper-side state serialisation ------------------------------------------
//
// Every backend stores plugin state in ONE container format so presets stay
// portable across formats and tolerant across versions (same philosophy as
// Core/StateBlob): a magic/version header, the parameter table keyed by
// hash32(id) — unknown ids are skipped, missing ids keep defaults — and an
// optional opaque user-state section appended verbatim.

inline constexpr uint32_t kStateMagic   = 0x4453504Bu;   // "DSPK"
inline constexpr uint32_t kStateVersion = 1u;

template <typename P>
inline std::vector<uint8_t> buildState(const P& user, const double* normalized,
                                       size_t numParams)
{
    auto push32 = [](std::vector<uint8_t>& v, uint32_t x) {
        v.push_back(static_cast<uint8_t>(x));
        v.push_back(static_cast<uint8_t>(x >> 8));
        v.push_back(static_cast<uint8_t>(x >> 16));
        v.push_back(static_cast<uint8_t>(x >> 24));
    };
    std::vector<uint8_t> blob;
    push32(blob, kStateMagic);
    push32(blob, kStateVersion);
    push32(blob, static_cast<uint32_t>(numParams));
    for (size_t i = 0; i < numParams; ++i)
    {
        push32(blob, hash32(P::parameters[i].id));
        uint64_t bits = 0;
        const double v = normalized[i];
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        push32(blob, static_cast<uint32_t>(bits));
        push32(blob, static_cast<uint32_t>(bits >> 32));
    }
    if constexpr (HasGetState<P>)
    {
        const std::vector<uint8_t> userBlob = user.getState();
        push32(blob, static_cast<uint32_t>(userBlob.size()));
        blob.insert(blob.end(), userBlob.begin(), userBlob.end());
    }
    else
        push32(blob, 0u);
    return blob;
}

/** @brief Parses a state blob; fills `normalized` (defaults pre-loaded by the
 *  caller) and forwards the user section. Returns false on a foreign blob. */
template <typename P>
inline bool applyState(P& user, const uint8_t* data, size_t size, double* normalized)
{
    auto read32 = [&](size_t& pos, uint32_t& out) {
        if (pos + 4 > size) return false;
        out = static_cast<uint32_t>(data[pos])
            | static_cast<uint32_t>(data[pos + 1]) << 8
            | static_cast<uint32_t>(data[pos + 2]) << 16
            | static_cast<uint32_t>(data[pos + 3]) << 24;
        pos += 4;
        return true;
    };
    size_t pos = 0;
    uint32_t magic = 0, version = 0, count = 0;
    if (!read32(pos, magic) || magic != kStateMagic) return false;
    if (!read32(pos, version) || version == 0) return false;
    if (!read32(pos, count)) return false;

    constexpr size_t numParams = P::parameters.size();
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t id = 0, lo = 0, hi = 0;
        if (!read32(pos, id) || !read32(pos, lo) || !read32(pos, hi)) return false;
        const uint64_t bits = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        for (size_t k = 0; k < numParams; ++k)
            if (hash32(P::parameters[k].id) == id)
            {
                normalized[k] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                break;
            }
        // Unknown ids are skipped: forward-compatible by construction.
    }

    uint32_t userSize = 0;
    if (read32(pos, userSize) && userSize > 0 && pos + userSize <= size)
    {
        if constexpr (HasSetState<P>)
            user.setState(data + pos, userSize);
    }
    return true;
}

} // namespace dspark::plugin
