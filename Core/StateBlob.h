// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file StateBlob.h
 * @brief Versioned key-value state serialization for processor presets.
 *
 * The uniform getState()/setState() backbone: processors serialize their
 * parameters as readable key/value entries into a compact, versioned binary
 * blob, and restore tolerantly -- unknown keys are ignored, missing keys keep
 * their defaults, so presets survive both directions of version drift.
 *
 * Layout (little-endian):
 *
 *   "DSPK" | format u16 | processorId u32 | processorVersion u16 | count u16
 *   count x ( keyLen u8 | key bytes | type u8 | payload 4 bytes )
 *
 * Types: 0 = float, 1 = int32, 2 = bool. Keys are short ASCII strings
 * (letters, digits, '.', '_' -- no quotes or backslashes, so the JSON
 * helpers never need escaping), which keeps blobs self-describing:
 * `stateToJson()` renders any blob as a flat JSON object and
 * `stateFromJson()` parses that same subset back -- no external libraries.
 * Both are locale-independent: a host process running with a decimal-comma
 * LC_NUMERIC (the classic plugin bug) cannot corrupt the number formatting
 * or parsing.
 *
 * Serialization is a setup/UI-thread operation (it allocates); never call it
 * from the audio callback. That matches every plugin host's preset flow.
 *
 * Dependencies: C++20 standard library only.
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dspark {

// ============================================================================
// StateWriter
// ============================================================================

/** @brief Serializes key/value parameters into a versioned blob. */
class StateWriter
{
public:
    /**
     * @param processorId      FOURCC-style identity of the processor type.
     * @param processorVersion Schema version of that processor's state.
     */
    StateWriter(uint32_t processorId, uint16_t processorVersion)
    {
        blob_.reserve(256);
        append("DSPK", 4);
        appendU16(kFormatVersion);
        appendU32(processorId);
        appendU16(processorVersion);
        countPos_ = blob_.size();
        appendU16(0);   // patched by blob()
    }

    /** @brief Writes a float parameter. */
    void write(const char* key, float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, 4);
        writeEntry(key, 0, bits);
    }

    /** @brief Writes an integer parameter (enums, counts, modes). */
    void write(const char* key, int32_t value)
    {
        writeEntry(key, 1, static_cast<uint32_t>(value));
    }

    /** @brief Writes a boolean parameter. */
    void write(const char* key, bool value)
    {
        writeEntry(key, 2, value ? 1u : 0u);
    }

    /** @brief Writes a nested blob (composite processors: bands, slots...). */
    void write(const char* key, const std::vector<uint8_t>& nested)
    {
        const size_t len = std::min<size_t>(std::strlen(key), 255);
        assert(len == std::strlen(key) && "state key longer than 255 chars is truncated");
        assert(count_ < 0xFFFF && "state entry count overflows u16");
        blob_.push_back(static_cast<uint8_t>(len));
        append(key, len);
        blob_.push_back(3);
        appendU32(static_cast<uint32_t>(nested.size()));
        blob_.insert(blob_.end(), nested.begin(), nested.end());
        ++count_;
    }

    /** @brief Finalizes and returns the blob. */
    [[nodiscard]] std::vector<uint8_t> blob() const
    {
        std::vector<uint8_t> out = blob_;
        out[countPos_] = static_cast<uint8_t>(count_ & 0xFF);
        out[countPos_ + 1] = static_cast<uint8_t>((count_ >> 8) & 0xFF);
        return out;
    }

private:
    static constexpr uint16_t kFormatVersion = 1;

    void writeEntry(const char* key, uint8_t type, uint32_t payload)
    {
        const size_t len = std::min<size_t>(std::strlen(key), 255);
        assert(len == std::strlen(key) && "state key longer than 255 chars is truncated");
        assert(count_ < 0xFFFF && "state entry count overflows u16");
        blob_.push_back(static_cast<uint8_t>(len));
        append(key, len);
        blob_.push_back(type);
        appendU32(payload);
        ++count_;
    }

    void append(const char* data, size_t n)
    {
        blob_.insert(blob_.end(), data, data + n);
    }
    void appendU16(uint16_t v)
    {
        blob_.push_back(static_cast<uint8_t>(v & 0xFF));
        blob_.push_back(static_cast<uint8_t>(v >> 8));
    }
    void appendU32(uint32_t v)
    {
        for (int i = 0; i < 4; ++i)
            blob_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    }

    std::vector<uint8_t> blob_;
    size_t countPos_ = 0;
    uint16_t count_ = 0;
};

// ============================================================================
// StateReader
// ============================================================================

/**
 * @brief Tolerant reader: missing keys yield defaults, unknown keys are skipped.
 *
 * A truncated or corrupt blob parses as far as it stays consistent: entries
 * decoded before the damage remain readable, and isValid() reports whether
 * the FULL blob parsed. Bounds are checked entry by entry, so no input can
 * read outside [data, data+size).
 */
class StateReader
{
public:
    StateReader(const uint8_t* data, size_t size)
    {
        if (data == nullptr || size < 14 || std::memcmp(data, "DSPK", 4) != 0)
            return;
        const uint16_t fmt = readU16(data + 4);
        if (fmt != 1) return;
        processorId_ = readU32(data + 6);
        processorVersion_ = readU16(data + 10);
        const uint16_t count = readU16(data + 12);

        size_t pos = 14;
        for (uint16_t i = 0; i < count; ++i)
        {
            if (pos + 1 > size) return;
            const size_t keyLen = data[pos++];
            if (pos + keyLen + 5 > size) return;
            Entry e;
            e.key.assign(reinterpret_cast<const char*>(data + pos), keyLen);
            pos += keyLen;
            e.type = data[pos++];
            e.payload = readU32(data + pos);
            pos += 4;
            if (e.type == 3)   // nested blob: payload is its byte length
            {
                // Subtraction form: `pos + e.payload > size` would wrap on
                // 32-bit size_t (WASM) for a corrupt length near UINT32_MAX
                // and pass the check into a giant out-of-bounds read.
                if (e.payload > size - pos) return;
                e.bytes.assign(data + pos, data + pos + e.payload);
                pos += e.payload;
            }
            entries_.push_back(std::move(e));
        }
        valid_ = true;
    }

    [[nodiscard]] bool isValid() const noexcept { return valid_; }
    [[nodiscard]] uint32_t processorId() const noexcept { return processorId_; }
    [[nodiscard]] uint16_t processorVersion() const noexcept { return processorVersion_; }

    /** @brief Reads a float, or `defaultValue` when the key is absent. */
    [[nodiscard]] float read(const char* key, float defaultValue) const
    {
        if (const Entry* e = find(key, 0))
        {
            float v = 0.0f;
            std::memcpy(&v, &e->payload, 4);
            return v;
        }
        return defaultValue;
    }

    /** @brief Reads an int32, or `defaultValue` when the key is absent. */
    [[nodiscard]] int32_t read(const char* key, int32_t defaultValue) const
    {
        if (const Entry* e = find(key, 1))
            return static_cast<int32_t>(e->payload);
        return defaultValue;
    }

    /** @brief Reads a bool, or `defaultValue` when the key is absent. */
    [[nodiscard]] bool read(const char* key, bool defaultValue) const
    {
        if (const Entry* e = find(key, 2))
            return e->payload != 0;
        return defaultValue;
    }

    /** @brief Reads a nested blob; empty when the key is absent. */
    [[nodiscard]] std::vector<uint8_t> readBlob(const char* key) const
    {
        if (const Entry* e = find(key, 3))
            return e->bytes;
        return {};
    }

    /** @brief Access for generic tooling (JSON rendering, preset browsers). */
    struct Entry
    {
        std::string key;
        uint8_t type = 0;
        uint32_t payload = 0;
        std::vector<uint8_t> bytes;   ///< Nested blob content (type 3).
    };
    [[nodiscard]] const std::vector<Entry>& entries() const noexcept { return entries_; }

private:
    [[nodiscard]] const Entry* find(const char* key, uint8_t type) const
    {
        for (const auto& e : entries_)
            if (e.type == type && e.key == key)
                return &e;
        return nullptr;
    }

    static uint16_t readU16(const uint8_t* p)
    {
        return static_cast<uint16_t>(p[0] | (p[1] << 8));
    }
    static uint32_t readU32(const uint8_t* p)
    {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
             | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    }

    bool valid_ = false;
    uint32_t processorId_ = 0;
    uint16_t processorVersion_ = 0;
    std::vector<Entry> entries_;
};

// ============================================================================
// JSON helpers (the blob's flat-object subset, zero dependencies)
// ============================================================================

namespace detail {

/// Nesting cap for the JSON round-trip: real processor states nest 2-3
/// levels; a crafted input must not be able to overflow the stack.
constexpr int kMaxStateJsonDepth = 32;

/** @brief Locale-independent float rendering ("%.9g" with a forced dot). */
inline void appendStateJsonNumber(std::string& out, float v)
{
    char num[48];
    std::snprintf(num, sizeof(num), "%.9g", static_cast<double>(v));
    for (char* c = num; *c != '\0'; ++c)
        if (*c == ',') *c = '.';   // locale decimal comma -> JSON dot
    out += num;
}

/** @brief Locale-independent number parse over [s, end); nullptr on failure. */
inline const char* parseStateJsonNumber(const char* s, const char* end, double& out) noexcept
{
    bool negative = false;
    if (s < end && (*s == '-' || *s == '+'))
    {
        negative = (*s == '-');
        ++s;
    }
    double v = 0.0;
    bool any = false;
    while (s < end && *s >= '0' && *s <= '9')
    {
        v = v * 10.0 + (*s - '0');
        ++s;
        any = true;
    }
    if (s < end && *s == '.')
    {
        ++s;
        double f = 0.1;
        while (s < end && *s >= '0' && *s <= '9')
        {
            v += (*s - '0') * f;
            f *= 0.1;
            ++s;
            any = true;
        }
    }
    if (!any) return nullptr;
    if (s < end && (*s == 'e' || *s == 'E'))
    {
        ++s;
        bool expNegative = false;
        if (s < end && (*s == '-' || *s == '+'))
        {
            expNegative = (*s == '-');
            ++s;
        }
        int e = 0;
        bool eAny = false;
        while (s < end && *s >= '0' && *s <= '9')
        {
            e = e * 10 + (*s - '0');
            ++s;
            eAny = true;
            if (e > 308) { e = 309; }   // saturate
        }
        if (!eAny) return nullptr;
        double scale = 1.0;
        for (int k = 0; k < (e > 309 ? 309 : e); ++k)
            scale *= 10.0;
        v = expNegative ? v / scale : v * scale;
    }
    out = negative ? -v : v;
    return s;
}

inline std::string stateToJsonImpl(const std::vector<uint8_t>& blob, int depth)
{
    if (depth >= kMaxStateJsonDepth) return "{}";

    StateReader r(blob.data(), blob.size());
    if (!r.isValid()) return "{}";

    std::string out = "{\"id\":" + std::to_string(r.processorId())
                    + ",\"version\":" + std::to_string(r.processorVersion())
                    + ",\"params\":{";
    bool first = true;
    for (const auto& e : r.entries())
    {
        if (!first) out += ',';
        first = false;
        out += '"' + e.key + "\":";
        if (e.type == 0)
        {
            float v = 0.0f;
            std::memcpy(&v, &e.payload, 4);
            appendStateJsonNumber(out, v);
        }
        else if (e.type == 1)
        {
            out += std::to_string(static_cast<int32_t>(e.payload));
        }
        else if (e.type == 3)
        {
            out += stateToJsonImpl(e.bytes, depth + 1);   // nested state: recurse
        }
        else
        {
            out += (e.payload != 0) ? "true" : "false";
        }
    }
    out += "}}";
    return out;
}

inline std::vector<uint8_t> stateFromJsonImpl(const std::string& json, int depth)
{
    if (depth >= kMaxStateJsonDepth) return {};

    auto skipWs = [&](size_t& p) { while (p < json.size() && (json[p] == ' ' || json[p] == '\n'
                                          || json[p] == '\t' || json[p] == '\r')) ++p; };
    auto expect = [&](size_t& p, char c) {
        skipWs(p);
        if (p >= json.size() || json[p] != c) return false;
        ++p;
        return true;
    };
    auto parseString = [&](size_t& p, std::string& out) {
        skipWs(p);
        if (p >= json.size() || json[p] != '"') return false;
        ++p;
        out.clear();
        while (p < json.size() && json[p] != '"') out += json[p++];
        if (p >= json.size()) return false;
        ++p;
        return true;
    };
    auto parseNumber = [&](size_t& p, double& out) {
        skipWs(p);
        const char* start = json.c_str() + p;
        const char* endOfJson = json.c_str() + json.size();
        const char* stop = parseStateJsonNumber(start, endOfJson, out);
        if (stop == nullptr) return false;
        p = static_cast<size_t>(stop - json.c_str());
        return true;
    };

    size_t p = 0;
    std::string key;
    double num = 0.0;
    uint32_t id = 0;
    uint16_t version = 0;

    if (!expect(p, '{')) return {};
    struct Param { std::string key; uint8_t type; uint32_t payload;
                   std::vector<uint8_t> bytes; };
    std::vector<Param> params;

    while (true)
    {
        if (!parseString(p, key) || !expect(p, ':')) return {};
        if (key == "id" && parseNumber(p, num))
        {
            id = static_cast<uint32_t>(num);
        }
        else if (key == "version" && parseNumber(p, num))
        {
            version = static_cast<uint16_t>(num);
        }
        else if (key == "params")
        {
            if (!expect(p, '{')) return {};
            skipWs(p);
            if (p < json.size() && json[p] == '}') { ++p; }
            else
            {
                while (true)
                {
                    std::string pk;
                    if (!parseString(p, pk) || !expect(p, ':')) return {};
                    skipWs(p);
                    Param prm { pk, 0, 0, {} };
                    if (p < json.size() && json[p] == '{')
                    {
                        // Nested state object: balance braces, recurse.
                        size_t braceDepth = 0, end = p;
                        for (; end < json.size(); ++end)
                        {
                            if (json[end] == '{') ++braceDepth;
                            else if (json[end] == '}' && --braceDepth == 0) { ++end; break; }
                        }
                        if (braceDepth != 0) return {};
                        prm.type = 3;
                        prm.bytes = stateFromJsonImpl(json.substr(p, end - p), depth + 1);
                        if (prm.bytes.empty()) return {};
                        p = end;
                    }
                    else if (json.compare(p, 4, "true") == 0)
                    {
                        prm.type = 2; prm.payload = 1; p += 4;
                    }
                    else if (json.compare(p, 5, "false") == 0)
                    {
                        prm.type = 2; prm.payload = 0; p += 5;
                    }
                    else if (parseNumber(p, num))
                    {
                        // Integers without fraction round-trip as int32 too;
                        // store as float (type 0) plus int mirror when exact.
                        const auto f = static_cast<float>(num);
                        std::memcpy(&prm.payload, &f, 4);
                        prm.type = 0;
                        if (num == static_cast<double>(static_cast<int32_t>(num)))
                        {
                            params.push_back({ pk, 1,
                                static_cast<uint32_t>(static_cast<int32_t>(num)), {} });
                        }
                    }
                    else return {};
                    params.push_back(std::move(prm));
                    skipWs(p);
                    if (p < json.size() && json[p] == ',') { ++p; continue; }
                    break;
                }
                if (!expect(p, '}')) return {};
            }
        }
        else return {};
        skipWs(p);
        if (p < json.size() && json[p] == ',') { ++p; continue; }
        break;
    }
    if (!expect(p, '}')) return {};

    StateWriter w(id, version);
    for (const auto& prm : params)
    {
        if (prm.type == 0)
        {
            float v = 0.0f;
            std::memcpy(&v, &prm.payload, 4);
            w.write(prm.key.c_str(), v);
        }
        else if (prm.type == 1)
        {
            w.write(prm.key.c_str(), static_cast<int32_t>(prm.payload));
        }
        else if (prm.type == 3)
        {
            w.write(prm.key.c_str(), prm.bytes);
        }
        else
        {
            w.write(prm.key.c_str(), prm.payload != 0);
        }
    }
    return w.blob();
}

} // namespace detail

/** @brief Renders a state blob as a flat JSON object string. */
[[nodiscard]] inline std::string stateToJson(const std::vector<uint8_t>& blob)
{
    return detail::stateToJsonImpl(blob, 0);
}

/**
 * @brief Parses the JSON produced by stateToJson() back into a blob.
 *
 * Accepts exactly that flat subset (id/version/params of numbers and bools);
 * returns an empty vector on malformed input. Number-typed params re-enter
 * as floats; integer-coded modes survive because setState() readers request
 * the type they stored, and stateToJson tags ints without decimals -- both
 * int and float lookups are attempted by the float/int read pair below.
 */
[[nodiscard]] inline std::vector<uint8_t> stateFromJson(const std::string& json)
{
    return detail::stateFromJsonImpl(json, 0);
}

/** @brief Builds a FOURCC processor id, e.g. dspark::stateId("COMP"). */
[[nodiscard]] constexpr uint32_t stateId(const char (&tag)[5]) noexcept
{
    // Bytes, not chars: a (theoretical) non-ASCII tag char would sign-extend
    // and smear across the other lanes.
    return static_cast<uint32_t>(static_cast<uint8_t>(tag[0]))
         | (static_cast<uint32_t>(static_cast<uint8_t>(tag[1])) << 8)
         | (static_cast<uint32_t>(static_cast<uint8_t>(tag[2])) << 16)
         | (static_cast<uint32_t>(static_cast<uint8_t>(tag[3])) << 24);
}

} // namespace dspark
