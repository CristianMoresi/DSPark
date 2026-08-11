// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file MidiFile.h
 * @brief Bounded Standard MIDI File reader and writer.
 *
 * Reads SMF formats 0, 1, and 2 and writes formats 0 and 1. Events are
 * represented semantically: channel running status is accepted on input but
 * output always carries explicit status bytes, while SysEx packet boundaries,
 * velocity-zero note-ons, and opaque meta payloads are retained exactly.
 * Timing uses PPQN division only; SMPTE division is rejected.
 *
 * File operations allocate and block. They are intended for offline or worker
 * threads, never the real-time audio thread. The header is omitted from the
 * umbrella when DSPARK_NO_FILE_IO is defined.
 *
 * Threading: owner-managed. One instance is used by one thread at a time.
 *
 * Dependencies: C++20 standard library only.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace dspark {

/** @brief Semantic kind of a Standard MIDI File event. */
enum class MidiEventKind : uint8_t
{
    Channel,
    SysExF0,
    SysExF7,
    Meta
};

/**
 * @brief One semantic SMF event with a delta time.
 *
 * Channel events use status/data1/data2. Meta events use metaType and payload.
 * SysEx events use payload. Fields not used by the selected kind are zero.
 */
struct MidiEvent
{
    uint32_t deltaTicks = 0;
    MidiEventKind kind = MidiEventKind::Channel;
    uint8_t status = 0;
    uint8_t data1 = 0;
    uint8_t data2 = 0;
    uint8_t metaType = 0;
    std::vector<uint8_t> payload;

    friend bool operator==(const MidiEvent&, const MidiEvent&) = default;
};

/** @brief Ordered events belonging to one SMF track chunk. */
struct MidiTrack
{
    std::vector<MidiEvent> events;

    friend bool operator==(const MidiTrack&, const MidiTrack&) = default;
};

/** @brief Effective tempo at an absolute tick. */
struct MidiTempoChange
{
    uint64_t tick = 0;
    uint32_t microsecondsPerQuarter = 500000;
};

/**
 * @class MidiFile
 * @brief Transactional, resource-bounded SMF parser and authoring API.
 *
 * A failed read closes the logical file and leaves the object empty. Failed
 * authoring operations leave the existing document unchanged. write()
 * validates and serializes the complete document before opening its target.
 *
 * Threading: owner-managed offline use. tracks() returns an
 * owner-thread reference view; no thread may read that view while another
 * thread accesses the same MidiFile instance mutably.
 */
class MidiFile
{
public:
    /** @brief Maximum accepted or produced file size. */
    static constexpr uint64_t kMaxInputBytes = 256ull * 1024 * 1024;
    /** @brief Maximum number of track chunks. */
    static constexpr uint32_t kMaxTracks = 4096;
    /** @brief Maximum aggregate event count. */
    static constexpr uint64_t kMaxEvents = 2000000;
    /** @brief Maximum aggregate SysEx and meta payload bytes. */
    static constexpr uint64_t kMaxAggregatePayloadBytes = 128ull * 1024 * 1024;
    /** @brief Maximum bytes in one track or skipped alien chunk. */
    static constexpr uint64_t kMaxTrackChunkBytes = 128ull * 1024 * 1024;

    /**
     * @brief Replaces the document with an empty writable format 0 or 1 file.
     * @param format SMF format, either 0 or 1.
     * @param ppqn Ticks per quarter note in the range 1..32767.
     * @param trackCount Initial track count. Format 0 requires exactly one.
     * @return True on success; false leaves the current document unchanged.
     */
    [[nodiscard]] bool create(uint16_t format, uint16_t ppqn,
                              size_t trackCount = 1)
    {
        if ((format != 0 && format != 1) || ppqn == 0 || ppqn > 0x7fffu)
            return false;
        if (trackCount == 0 || trackCount > kMaxTracks)
            return false;
        if (format == 0 && trackCount != 1)
            return false;

        std::vector<MidiTrack> replacement(trackCount);
        std::vector<TrackAuthoringState> replacementStates(trackCount);
        format_ = static_cast<int>(format);
        ppqn_ = ppqn;
        tracks_ = std::move(replacement);
        validationTotals_ = {};
        trackAuthoringStates_ = std::move(replacementStates);
        return true;
    }

    /**
     * @brief Adds an empty track to a format 1 document.
     * @return The new track index, or nullopt when the operation is invalid.
     */
    [[nodiscard]] std::optional<size_t> addTrack()
    {
        if (format_ != 1 || tracks_.size() >= kMaxTracks)
            return std::nullopt;
        trackAuthoringStates_.emplace_back();
        try
        {
            tracks_.emplace_back();
        }
        catch (...)
        {
            trackAuthoringStates_.pop_back();
            throw;
        }
        return tracks_.size() - 1;
    }

    /**
     * @brief Adds a validated MIDI channel event.
     *
     * Status must be 0x80..0xef. Program-change and channel-pressure events
     * use one data byte and require data2 == 0; all other channel events use
     * two 7-bit data bytes.
     */
    [[nodiscard]] bool addChannelEvent(size_t track, uint32_t delta,
                                       uint8_t status, uint8_t data1,
                                       uint8_t data2 = 0)
    {
        MidiEvent event;
        event.deltaTicks = delta;
        event.kind = MidiEventKind::Channel;
        event.status = status;
        event.data1 = data1;
        event.data2 = data2;
        return appendEvent(track, std::move(event));
    }

    /**
     * @brief Adds an F0 or F7 SysEx packet event.
     * @param f0OrF7 Must be MidiEventKind::SysExF0 or SysExF7.
     */
    [[nodiscard]] bool addSysExEvent(size_t track, uint32_t delta,
                                     MidiEventKind f0OrF7,
                                     std::span<const uint8_t> payload)
    {
        if (!preflightPayloadEvent(track, delta, f0OrF7, 0, payload))
            return false;
        MidiEvent event;
        event.deltaTicks = delta;
        event.kind = f0OrF7;
        event.payload.assign(payload.begin(), payload.end());
        return appendEvent(track, std::move(event));
    }

    /** @brief Adds an opaque meta event. */
    [[nodiscard]] bool addMetaEvent(size_t track, uint32_t delta, uint8_t type,
                                    std::span<const uint8_t> payload = {})
    {
        if (!preflightPayloadEvent(track, delta, MidiEventKind::Meta,
                                   type, payload))
            return false;
        MidiEvent event;
        event.deltaTicks = delta;
        event.kind = MidiEventKind::Meta;
        event.metaType = type;
        event.payload.assign(payload.begin(), payload.end());
        return appendEvent(track, std::move(event));
    }

    /** @brief Restores the empty-state sentinel. */
    void clear() noexcept
    {
        format_ = -1;
        ppqn_ = 0;
        tracks_.clear();
        validationTotals_ = {};
        trackAuthoringStates_.clear();
    }

    /**
     * @brief Reads and validates an SMF format 0, 1, or 2 file.
     * @return True on success. Any failure leaves this object empty.
     */
    [[nodiscard]] bool read(const std::filesystem::path& path)
    {
        clear();

        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input.is_open())
            return false;
        const std::streampos end = input.tellg();
        if (end <= std::streampos(0))
            return false;
        const auto fileSize = static_cast<uint64_t>(end);
        if (fileSize > kMaxInputBytes
            || fileSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            return false;

        std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
        input.seekg(0, std::ios::beg);
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
            return false;

        MidiFile parsed;
        if (!parseFile(bytes, parsed))
            return false;
        *this = std::move(parsed);
        return true;
    }

    /**
     * @brief Writes a validated format 0 or 1 document.
     *
     * A missing EOT is emitted with delta zero without changing tracks(). The
     * complete output is built before the destination is opened.
     */
    [[nodiscard]] bool write(const std::filesystem::path& path) const
    {
        ValidationTotals totals;
        if (format_ != 0 && format_ != 1)
            return false;
        if (!validateDocument(format_, ppqn_, tracks_, false, false, totals))
            return false;

        uint64_t implicitEotCount = 0;
        for (const MidiTrack& track : tracks_)
            if (track.events.empty() || !isEot(track.events.back()))
                ++implicitEotCount;
        if (implicitEotCount > kMaxEvents - totals.events)
            return false;

        std::vector<uint8_t> output;
        output.reserve(14);
        appendTag(output, "MThd");
        appendBE32(output, 6);
        appendBE16(output, static_cast<uint16_t>(format_));
        appendBE16(output, static_cast<uint16_t>(tracks_.size()));
        appendBE16(output, ppqn_);

        for (const MidiTrack& track : tracks_)
        {
            std::vector<uint8_t> data;
            for (const MidiEvent& event : track.events)
            {
                appendVlq(data, event.deltaTicks);
                switch (event.kind)
                {
                    case MidiEventKind::Channel:
                        data.push_back(event.status);
                        data.push_back(event.data1);
                        if (channelDataCount(event.status) == 2)
                            data.push_back(event.data2);
                        break;
                    case MidiEventKind::SysExF0:
                    case MidiEventKind::SysExF7:
                        data.push_back(event.kind == MidiEventKind::SysExF0
                                           ? uint8_t { 0xf0 } : uint8_t { 0xf7 });
                        appendVlq(data, static_cast<uint32_t>(event.payload.size()));
                        data.insert(data.end(), event.payload.begin(), event.payload.end());
                        break;
                    case MidiEventKind::Meta:
                        data.push_back(0xff);
                        data.push_back(event.metaType);
                        appendVlq(data, static_cast<uint32_t>(event.payload.size()));
                        data.insert(data.end(), event.payload.begin(), event.payload.end());
                        break;
                }
            }

            if (track.events.empty() || !isEot(track.events.back()))
            {
                data.push_back(0x00);
                data.push_back(0xff);
                data.push_back(0x2f);
                data.push_back(0x00);
            }

            if (data.size() > kMaxTrackChunkBytes
                || data.size() > std::numeric_limits<uint32_t>::max())
                return false;
            if (output.size() > kMaxInputBytes - 8
                || data.size() > kMaxInputBytes - output.size() - 8)
                return false;
            appendTag(output, "MTrk");
            appendBE32(output, static_cast<uint32_t>(data.size()));
            output.insert(output.end(), data.begin(), data.end());
        }

        std::ofstream destination(path, std::ios::binary | std::ios::trunc);
        if (!destination.is_open())
            return false;
        destination.write(reinterpret_cast<const char*>(output.data()),
                          static_cast<std::streamsize>(output.size()));
        return destination.good();
    }

    /** @brief Returns 0, 1, or 2; returns -1 for an empty object. */
    [[nodiscard]] int format() const noexcept { return format_; }

    /** @brief Returns PPQN, or zero for an empty object. */
    [[nodiscard]] uint16_t ticksPerQuarter() const noexcept { return ppqn_; }

    /**
     * @brief Returns immutable semantic tracks as an owner-thread reference view.
     *
     * The reference is valid only while this MidiFile remains alive and until
     * the next non-const operation that can replace or mutate its document. No
     * thread may read the view while another thread accesses this instance
     * mutably. This accessor does not provide atomic publication.
     */
    [[nodiscard]] const std::vector<MidiTrack>& tracks() const noexcept
    {
        return tracks_;
    }

    /**
     * @brief Builds the effective tempo map for a track.
     *
     * Formats 0 and 1 always use track zero. Format 2 uses the requested
     * independent track. The returned map always starts at tick zero with the
     * default 500000 us/qn unless a tick-zero tempo replaces it.
     */
    [[nodiscard]] std::optional<std::vector<MidiTempoChange>>
    tempoMap(size_t track = 0) const
    {
        const std::optional<size_t> source = tempoTrack(track);
        if (!source)
            return std::nullopt;

        std::vector<MidiTempoChange> result;
        result.push_back({ 0, 500000 });
        uint64_t absoluteTick = 0;
        for (const MidiEvent& event : tracks_[*source].events)
        {
            if (absoluteTick > std::numeric_limits<uint64_t>::max() - event.deltaTicks)
                return std::nullopt;
            absoluteTick += event.deltaTicks;
            uint32_t tempo = 0;
            if (!getTempo(event, tempo))
                continue;
            if (result.back().tick == absoluteTick)
                result.back().microsecondsPerQuarter = tempo;
            else
                result.push_back({ absoluteTick, tempo });
        }
        return result;
    }

    /**
     * @brief Converts an absolute tick to floor(exact elapsed microseconds).
     * @return nullopt for an empty/invalid track or uint64 overflow.
     */
    [[nodiscard]] std::optional<uint64_t>
    tickToMicroseconds(uint64_t tick, size_t track = 0) const noexcept
    {
        uint64_t whole = 0;
        uint64_t remainder = 0;
        if (!integrateTicks(tick, track, whole, remainder))
            return std::nullopt;
        return whole;
    }

    /**
     * @brief Converts an absolute tick directly from the shared rational sum.
     * @return nullopt for an empty/invalid track or elapsed-time overflow.
     */
    [[nodiscard]] std::optional<double>
    tickToSeconds(uint64_t tick, size_t track = 0) const noexcept
    {
        uint64_t whole = 0;
        uint64_t remainder = 0;
        if (!integrateTicks(tick, track, whole, remainder))
            return std::nullopt;
        return secondsFromRationalMicroseconds(whole, remainder, ppqn_);
    }

private:
    struct ValidationTotals
    {
        uint64_t events = 0;
        uint64_t payloadBytes = 0;
    };

    struct TrackAuthoringState
    {
        uint64_t absoluteTick = 0;
        bool sysexOpen = false;
        bool sawEot = false;
    };

    class ByteCursor
    {
    public:
        ByteCursor(const std::vector<uint8_t>& bytes, size_t begin, size_t end)
            : bytes_(bytes), pos_(begin), end_(end)
        {
        }

        [[nodiscard]] size_t remaining() const noexcept { return end_ - pos_; }
        [[nodiscard]] size_t position() const noexcept { return pos_; }

        bool readU8(uint8_t& value) noexcept
        {
            if (pos_ >= end_) return false;
            value = bytes_[pos_++];
            return true;
        }

        bool readBE16(uint16_t& value) noexcept
        {
            uint8_t a = 0, b = 0;
            if (!readU8(a) || !readU8(b)) return false;
            value = static_cast<uint16_t>((static_cast<uint16_t>(a) << 8)
                                        | static_cast<uint16_t>(b));
            return true;
        }

        bool readBE32(uint32_t& value) noexcept
        {
            uint8_t a = 0, b = 0, c = 0, d = 0;
            if (!readU8(a) || !readU8(b) || !readU8(c) || !readU8(d))
                return false;
            value = (static_cast<uint32_t>(a) << 24)
                  | (static_cast<uint32_t>(b) << 16)
                  | (static_cast<uint32_t>(c) << 8)
                  | static_cast<uint32_t>(d);
            return true;
        }

        bool readTag(char tag[4]) noexcept
        {
            if (remaining() < 4) return false;
            for (size_t i = 0; i < 4; ++i)
                tag[i] = static_cast<char>(bytes_[pos_++]);
            return true;
        }

        bool readSpan(size_t length, const uint8_t*& data) noexcept
        {
            if (length > remaining()) return false;
            data = bytes_.data() + pos_;
            pos_ += length;
            return true;
        }

        bool skip(size_t length) noexcept
        {
            const uint8_t* ignored = nullptr;
            return readSpan(length, ignored);
        }

    private:
        const std::vector<uint8_t>& bytes_;
        size_t pos_ = 0;
        size_t end_ = 0;
    };

    int format_ = -1;
    uint16_t ppqn_ = 0;
    std::vector<MidiTrack> tracks_;
    ValidationTotals validationTotals_;
    std::vector<TrackAuthoringState> trackAuthoringStates_;

    static bool sameTag(const char tag[4], const char (&expected)[5]) noexcept
    {
        return std::memcmp(tag, expected, 4) == 0;
    }

    static int channelDataCount(uint8_t status) noexcept
    {
        if (status < 0x80 || status > 0xef)
            return 0;
        const uint8_t high = status & 0xf0u;
        return (high == 0xc0u || high == 0xd0u) ? 1 : 2;
    }

    static bool isEot(const MidiEvent& event) noexcept
    {
        return event.kind == MidiEventKind::Meta && event.metaType == 0x2f;
    }

    static bool payloadEndsF7(const MidiEvent& event) noexcept
    {
        return !event.payload.empty() && event.payload.back() == 0xf7;
    }

    static bool validTempoPayload(const MidiEvent& event) noexcept
    {
        if (event.kind != MidiEventKind::Meta || event.metaType != 0x51
            || event.payload.size() != 3)
            return true;
        return event.payload[0] != 0 || event.payload[1] != 0
            || event.payload[2] != 0;
    }

    static bool validateEventShape(const MidiEvent& event) noexcept
    {
        if (event.deltaTicks > 0x0fffffffu)
            return false;
        switch (event.kind)
        {
            case MidiEventKind::Channel:
            {
                const int count = channelDataCount(event.status);
                if (count == 0 || event.data1 >= 0x80 || event.data2 >= 0x80)
                    return false;
                if (count == 1 && event.data2 != 0)
                    return false;
                return event.metaType == 0 && event.payload.empty();
            }
            case MidiEventKind::SysExF0:
            case MidiEventKind::SysExF7:
                return event.status == 0 && event.data1 == 0 && event.data2 == 0
                    && event.metaType == 0;
            case MidiEventKind::Meta:
                if (event.status != 0 || event.data1 != 0 || event.data2 != 0
                    || event.metaType >= 0x80 || !validTempoPayload(event))
                    return false;
                return event.metaType != 0x2f || event.payload.empty();
        }
        return false;
    }

    static bool advanceTrackAuthoringState(
        const MidiEvent& event, TrackAuthoringState& state) noexcept
    {
        if (state.sawEot
            || state.absoluteTick
                > std::numeric_limits<uint64_t>::max() - event.deltaTicks)
            return false;
        state.absoluteTick += event.deltaTicks;

        switch (event.kind)
        {
            case MidiEventKind::Channel:
                return !state.sysexOpen;
            case MidiEventKind::Meta:
                if (state.sysexOpen)
                    return false;
                if (isEot(event))
                    state.sawEot = true;
                return true;
            case MidiEventKind::SysExF0:
                if (state.sysexOpen)
                    return false;
                state.sysexOpen = !payloadEndsF7(event);
                return true;
            case MidiEventKind::SysExF7:
                if (state.sysexOpen)
                    state.sysexOpen = !payloadEndsF7(event);
                return true;
        }
        return false;
    }

    static bool validateTrack(const MidiTrack& track, bool requireEot,
                              bool allowOpenSysEx,
                              ValidationTotals& totals,
                              TrackAuthoringState* authoringState = nullptr) noexcept
    {
        TrackAuthoringState state;
        for (const MidiEvent& event : track.events)
        {
            if (!validateEventShape(event))
                return false;
            if (totals.events >= kMaxEvents)
                return false;
            ++totals.events;
            if (event.payload.size() > kMaxAggregatePayloadBytes
                || totals.payloadBytes > kMaxAggregatePayloadBytes - event.payload.size())
                return false;
            totals.payloadBytes += event.payload.size();
            if (!advanceTrackAuthoringState(event, state))
                return false;
        }
        if ((!allowOpenSysEx && state.sysexOpen)
            || (requireEot && !state.sawEot))
            return false;
        if (authoringState != nullptr)
            *authoringState = state;
        return true;
    }

    static bool validateDocument(int format, uint16_t ppqn,
                                 const std::vector<MidiTrack>& tracks,
                                 bool requireEot,
                                 bool allowOpenSysEx,
                                 ValidationTotals& totals,
                                 std::span<TrackAuthoringState> authoringStates = {}) noexcept
    {
        if (format < 0 || format > 2 || ppqn == 0 || ppqn > 0x7fff)
            return false;
        if (tracks.empty() || tracks.size() > kMaxTracks)
            return false;
        if (format == 0 && tracks.size() != 1)
            return false;
        if (!authoringStates.empty() && authoringStates.size() != tracks.size())
            return false;
        totals = {};
        for (size_t index = 0; index < tracks.size(); ++index)
        {
            TrackAuthoringState* state = authoringStates.empty()
                ? nullptr : &authoringStates[index];
            if (!validateTrack(tracks[index], requireEot, allowOpenSysEx,
                               totals, state))
                return false;
        }
        return true;
    }

    [[nodiscard]] bool appendEvent(size_t track, MidiEvent event)
    {
        if (format_ != 0 && format_ != 1)
            return false;
        if (track >= tracks_.size()
            || trackAuthoringStates_.size() != tracks_.size()
            || !validateEventShape(event))
            return false;
        if (validationTotals_.events >= kMaxEvents
            || event.payload.size() > kMaxAggregatePayloadBytes
            || validationTotals_.payloadBytes
                > kMaxAggregatePayloadBytes - event.payload.size())
            return false;

        TrackAuthoringState nextState = trackAuthoringStates_[track];
        if (!advanceTrackAuthoringState(event, nextState))
            return false;
        const size_t payloadBytes = event.payload.size();
        tracks_[track].events.push_back(std::move(event));
        ++validationTotals_.events;
        validationTotals_.payloadBytes += payloadBytes;
        trackAuthoringStates_[track] = nextState;
        return true;
    }

    [[nodiscard]] bool preflightPayloadEvent(
        size_t track, uint32_t delta, MidiEventKind kind, uint8_t metaType,
        std::span<const uint8_t> payload) const noexcept
    {
        if ((format_ != 0 && format_ != 1) || track >= tracks_.size()
            || delta > 0x0fffffffu || payload.size() > kMaxAggregatePayloadBytes)
            return false;
        if (kind != MidiEventKind::SysExF0
            && kind != MidiEventKind::SysExF7
            && kind != MidiEventKind::Meta)
            return false;
        if (kind == MidiEventKind::Meta)
        {
            if (metaType >= 0x80 || (metaType == 0x2f && !payload.empty()))
                return false;
            if (metaType == 0x51 && payload.size() == 3
                && payload[0] == 0 && payload[1] == 0 && payload[2] == 0)
                return false;
        }

        if (trackAuthoringStates_.size() != tracks_.size()
            || validationTotals_.events >= kMaxEvents
            || validationTotals_.payloadBytes
                > kMaxAggregatePayloadBytes - payload.size())
            return false;

        const TrackAuthoringState& state = trackAuthoringStates_[track];
        if (state.sawEot
            || state.absoluteTick > std::numeric_limits<uint64_t>::max() - delta)
            return false;
        if (kind == MidiEventKind::SysExF0)
            return !state.sysexOpen;
        if (kind == MidiEventKind::SysExF7)
            return true;
        return !state.sysexOpen;
    }

    static bool readVlq(ByteCursor& cursor, uint32_t& value) noexcept
    {
        value = 0;
        for (unsigned count = 0; count < 4; ++count)
        {
            uint8_t byte = 0;
            if (!cursor.readU8(byte))
                return false;
            if (count == 0 && (byte & 0x80u) != 0 && (byte & 0x7fu) == 0)
                return false;
            value = static_cast<uint32_t>((value << 7) | (byte & 0x7fu));
            if ((byte & 0x80u) == 0)
                return true;
        }
        return false;
    }

    static bool copyPayload(ByteCursor& cursor, uint32_t length,
                            ValidationTotals& totals,
                            std::vector<uint8_t>& payload)
    {
        if (length > cursor.remaining()
            || totals.payloadBytes > kMaxAggregatePayloadBytes - length)
            return false;
        const uint8_t* data = nullptr;
        if (!cursor.readSpan(length, data))
            return false;
        payload.assign(data, data + length);
        totals.payloadBytes += length;
        return true;
    }

    static bool parseTrack(const std::vector<uint8_t>& bytes, size_t begin,
                           size_t end, ValidationTotals& totals,
                           MidiTrack& result)
    {
        ByteCursor cursor(bytes, begin, end);
        uint8_t runningStatus = 0;
        bool sysexOpen = false;
        bool sawEot = false;
        uint64_t absoluteTick = 0;

        while (cursor.remaining() > 0)
        {
            if (sawEot || totals.events >= kMaxEvents)
                return false;
            MidiEvent event;
            if (!readVlq(cursor, event.deltaTicks))
                return false;
            if (absoluteTick > std::numeric_limits<uint64_t>::max() - event.deltaTicks)
                return false;
            absoluteTick += event.deltaTicks;

            uint8_t first = 0;
            if (!cursor.readU8(first))
                return false;
            if (first < 0x80)
            {
                if (runningStatus == 0 || sysexOpen)
                    return false;
                event.kind = MidiEventKind::Channel;
                event.status = runningStatus;
                event.data1 = first;
                if (channelDataCount(runningStatus) == 2)
                {
                    if (!cursor.readU8(event.data2) || event.data2 >= 0x80)
                        return false;
                }
            }
            else if (first >= 0x80 && first <= 0xef)
            {
                if (sysexOpen)
                    return false;
                event.kind = MidiEventKind::Channel;
                event.status = first;
                runningStatus = first;
                if (!cursor.readU8(event.data1) || event.data1 >= 0x80)
                    return false;
                if (channelDataCount(first) == 2)
                {
                    if (!cursor.readU8(event.data2) || event.data2 >= 0x80)
                        return false;
                }
            }
            else if (first == 0xf0 || first == 0xf7)
            {
                runningStatus = 0;
                event.kind = first == 0xf0 ? MidiEventKind::SysExF0
                                           : MidiEventKind::SysExF7;
                uint32_t length = 0;
                if (!readVlq(cursor, length)
                    || !copyPayload(cursor, length, totals, event.payload))
                    return false;
                if (event.kind == MidiEventKind::SysExF0)
                {
                    if (sysexOpen) return false;
                    sysexOpen = !payloadEndsF7(event);
                }
                else if (sysexOpen)
                {
                    sysexOpen = !payloadEndsF7(event);
                }
            }
            else if (first == 0xff)
            {
                runningStatus = 0;
                if (sysexOpen)
                    return false;
                event.kind = MidiEventKind::Meta;
                if (!cursor.readU8(event.metaType) || event.metaType >= 0x80)
                    return false;
                uint32_t length = 0;
                if (!readVlq(cursor, length)
                    || !copyPayload(cursor, length, totals, event.payload)
                    || !validTempoPayload(event))
                    return false;
                if (event.metaType == 0x2f)
                {
                    if (!event.payload.empty()) return false;
                    sawEot = true;
                }
            }
            else
            {
                return false;
            }

            if (!validateEventShape(event))
                return false;
            ++totals.events;
            result.events.push_back(std::move(event));
        }
        return sawEot && !sysexOpen;
    }

    static bool parseFile(const std::vector<uint8_t>& bytes, MidiFile& result)
    {
        if (bytes.size() < 14 || bytes.size() > kMaxInputBytes)
            return false;
        ByteCursor cursor(bytes, 0, bytes.size());
        char tag[4] = {};
        uint32_t headerLength = 0;
        if (!cursor.readTag(tag) || !sameTag(tag, "MThd")
            || !cursor.readBE32(headerLength) || headerLength < 6
            || headerLength > cursor.remaining())
            return false;

        uint16_t format = 0, trackCount = 0, division = 0;
        if (!cursor.readBE16(format) || !cursor.readBE16(trackCount)
            || !cursor.readBE16(division))
            return false;
        if (headerLength > 6 && !cursor.skip(headerLength - 6))
            return false;
        if (format > 2 || trackCount == 0 || trackCount > kMaxTracks
            || (format == 0 && trackCount != 1)
            || division == 0 || (division & 0x8000u) != 0)
            return false;

        std::vector<MidiTrack> tracks;
        tracks.reserve(trackCount);
        ValidationTotals totals;
        while (cursor.remaining() > 0)
        {
            if (cursor.remaining() < 8)
                return false;
            uint32_t length = 0;
            if (!cursor.readTag(tag) || !cursor.readBE32(length)
                || length > cursor.remaining())
                return false;
            const size_t begin = cursor.position();
            const size_t end = begin + static_cast<size_t>(length);

            if (sameTag(tag, "MTrk"))
            {
                if (tracks.size() >= trackCount || length > kMaxTrackChunkBytes)
                    return false;
                MidiTrack track;
                if (!parseTrack(bytes, begin, end, totals, track))
                    return false;
                tracks.push_back(std::move(track));
            }
            else
            {
                if (sameTag(tag, "MThd") || length > kMaxTrackChunkBytes)
                    return false;
            }
            if (!cursor.skip(length))
                return false;
        }
        if (tracks.size() != trackCount)
            return false;

        ValidationTotals verified;
        std::vector<TrackAuthoringState> authoringStates(tracks.size());
        if (!validateDocument(format, division, tracks, true, false, verified,
                              authoringStates))
            return false;
        result.format_ = format;
        result.ppqn_ = division;
        result.tracks_ = std::move(tracks);
        result.validationTotals_ = verified;
        result.trackAuthoringStates_ = std::move(authoringStates);
        return true;
    }

    static void appendTag(std::vector<uint8_t>& bytes, const char (&tag)[5])
    {
        bytes.insert(bytes.end(), tag, tag + 4);
    }

    static void appendBE16(std::vector<uint8_t>& bytes, uint16_t value)
    {
        bytes.push_back(static_cast<uint8_t>(value >> 8));
        bytes.push_back(static_cast<uint8_t>(value));
    }

    static void appendBE32(std::vector<uint8_t>& bytes, uint32_t value)
    {
        bytes.push_back(static_cast<uint8_t>(value >> 24));
        bytes.push_back(static_cast<uint8_t>(value >> 16));
        bytes.push_back(static_cast<uint8_t>(value >> 8));
        bytes.push_back(static_cast<uint8_t>(value));
    }

    static void appendVlq(std::vector<uint8_t>& bytes, uint32_t value)
    {
        uint8_t encoded[4] = {};
        unsigned count = 1;
        encoded[3] = static_cast<uint8_t>(value & 0x7fu);
        while ((value >>= 7) != 0)
            encoded[3 - count++] = static_cast<uint8_t>((value & 0x7fu) | 0x80u);
        bytes.insert(bytes.end(), encoded + (4 - count), encoded + 4);
    }

    [[nodiscard]] std::optional<size_t> tempoTrack(size_t track) const noexcept
    {
        if (format_ < 0 || format_ > 2 || ppqn_ == 0 || track >= tracks_.size())
            return std::nullopt;
        return format_ == 2 ? track : size_t { 0 };
    }

    static bool getTempo(const MidiEvent& event, uint32_t& tempo) noexcept
    {
        if (event.kind != MidiEventKind::Meta || event.metaType != 0x51
            || event.payload.size() != 3)
            return false;
        tempo = (static_cast<uint32_t>(event.payload[0]) << 16)
              | (static_cast<uint32_t>(event.payload[1]) << 8)
              | static_cast<uint32_t>(event.payload[2]);
        return tempo != 0;
    }

    static bool addInterval(uint64_t ticks, uint32_t tempo, uint16_t ppqn,
                            uint64_t& whole, uint64_t& remainder) noexcept
    {
        const uint64_t quotient = ticks / ppqn;
        const uint64_t tickRemainder = ticks % ppqn;
        if (quotient != 0
            && quotient > std::numeric_limits<uint64_t>::max() / tempo)
            return false;
        const uint64_t wholePart = quotient * tempo;
        if (whole > std::numeric_limits<uint64_t>::max() - wholePart)
            return false;
        whole += wholePart;

        const uint64_t fractional = tickRemainder * tempo + remainder;
        const uint64_t carry = fractional / ppqn;
        remainder = fractional % ppqn;
        if (whole > std::numeric_limits<uint64_t>::max() - carry)
            return false;
        whole += carry;
        return true;
    }

    static double secondsFromRationalMicroseconds(
        uint64_t wholeMicroseconds, uint64_t remainder,
        uint16_t ppqn) noexcept
    {
        constexpr uint64_t microsecondsPerSecond = 1000000;
        const uint64_t wholeSeconds = wholeMicroseconds / microsecondsPerSecond;
        const uint64_t fractionNumerator =
            (wholeMicroseconds % microsecondsPerSecond) * ppqn + remainder;
        const uint64_t fractionDenominator = microsecondsPerSecond * ppqn;
        if (fractionNumerator == 0)
            return static_cast<double>(wholeSeconds);
        if (wholeSeconds == 0)
            return static_cast<double>(fractionNumerator)
                 / static_cast<double>(fractionDenominator);

        unsigned exponent = 0;
        for (uint64_t value = wholeSeconds; value > 1; value >>= 1)
            ++exponent;
        const unsigned fractionalBits = 52u - exponent;
        uint64_t units = 0;
        uint64_t residual = fractionNumerator;
        for (unsigned bit = 0; bit < fractionalBits; ++bit)
        {
            residual *= 2;
            units <<= 1;
            if (residual >= fractionDenominator)
            {
                residual -= fractionDenominator;
                ++units;
            }
        }
        const uint64_t twiceResidual = residual * 2;
        if (twiceResidual > fractionDenominator
            || (twiceResidual == fractionDenominator && (units & 1u) != 0))
            ++units;
        const uint64_t scale = uint64_t { 1 } << fractionalBits;
        return static_cast<double>(wholeSeconds)
             + static_cast<double>(units) / static_cast<double>(scale);
    }

    bool integrateTicks(uint64_t targetTick, size_t track, uint64_t& whole,
                        uint64_t& remainder) const noexcept
    {
        const std::optional<size_t> source = tempoTrack(track);
        if (!source)
            return false;
        whole = 0;
        remainder = 0;
        uint64_t eventTick = 0;
        uint64_t integratedTick = 0;
        uint32_t tempo = 500000;

        for (const MidiEvent& event : tracks_[*source].events)
        {
            if (eventTick > std::numeric_limits<uint64_t>::max() - event.deltaTicks)
                return false;
            eventTick += event.deltaTicks;
            uint32_t nextTempo = 0;
            if (!getTempo(event, nextTempo))
                continue;
            if (eventTick > targetTick)
                break;
            if (!addInterval(eventTick - integratedTick, tempo, ppqn_, whole, remainder))
                return false;
            integratedTick = eventTick;
            tempo = nextTempo;
        }
        return addInterval(targetTick - integratedTick, tempo, ppqn_, whole, remainder);
    }
};

} // namespace dspark
