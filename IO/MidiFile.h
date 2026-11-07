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
 *
 * The default read is strict: every structural defect is fatal. A caller can
 * opt in, per read, to named recoveries for defects common in real files
 * (padding, redundant VLQ bytes, wrong track lengths or counts, missing
 * end-of-track events, unterminated SysEx) through MidiReadOptions; the
 * recoveries actually applied are reported by recoveries(). Resource limits
 * and every other malformation stay fatal in all modes.
 *
 * Timing uses PPQN division by default. SMPTE division (24, 25, 29.97 drop
 * frame or 30 frames per second) is read when MidiReadOptions allows it, and
 * can be authored with createSmpte(); its ticks are fixed fractions of a
 * second and tempo events do not affect timing.
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
#include <type_traits>
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
 * @brief Recoveries a tolerant MidiFile::read() may apply, as bit flags.
 *
 * Each flag names one class of defect found in real files that the strict
 * reader rejects. A read applies a recovery only when MidiReadOptions allows
 * it, and MidiFile::recoveries() reports every recovery the last read applied.
 * Resource limits, illegal status bytes, out-of-range data bytes, a zero
 * tempo, unknown formats and every other malformation stay fatal.
 */
enum class MidiReadRecovery : uint32_t
{
    None = 0,
    /** A variable-length quantity carried redundant leading 0x80 bytes. The
        value is kept; its significant part must still fit four bytes. */
    NoncanonicalVlq = 1u << 0,
    /** Bytes followed the end-of-track event inside its chunk (padding). They
        are ignored. */
    DataAfterEndOfTrack = 1u << 1,
    /** A track chunk length disagreed with its events: the events continued
        past the declared end, the next track chunk started before it, or it
        ran past the end of the file. The events up to the end-of-track event
        are used and chunk parsing resumes after them. */
    TrackLengthMismatch = 1u << 2,
    /** A track ended without an end-of-track event, possibly inside an
        incomplete final event, which is dropped. An end-of-track event with
        delta zero is appended. */
    MissingEndOfTrack = 1u << 3,
    /** A SysEx packet chain was interrupted by another event, or by the end of
        the track, without its closing 0xF7. 0xF7 is appended to the last
        packet of the chain. */
    UnterminatedSysEx = 1u << 4,
    /** The header track count disagreed with the track chunks present, or a
        format 0 file did not hold exactly one track. Formats 1 and 2 keep
        every track chunk present; format 0 keeps the first one. */
    TrackCountMismatch = 1u << 5,
    /** Bytes after the last chunk did not form a complete chunk (zero or 0x1A
        padding, a truncated or overlong non-track chunk). They are ignored. */
    TrailingBytes = 1u << 6,
    /** Every recovery above. */
    All = (1u << 7) - 1u
};

/** @brief Union of two recovery sets. */
[[nodiscard]] constexpr MidiReadRecovery operator|(MidiReadRecovery a,
                                                   MidiReadRecovery b) noexcept
{
    return static_cast<MidiReadRecovery>(static_cast<uint32_t>(a)
                                         | static_cast<uint32_t>(b));
}

/** @brief Intersection of two recovery sets. */
[[nodiscard]] constexpr MidiReadRecovery operator&(MidiReadRecovery a,
                                                   MidiReadRecovery b) noexcept
{
    return static_cast<MidiReadRecovery>(static_cast<uint32_t>(a)
                                         & static_cast<uint32_t>(b));
}

/** @brief True when @p set contains every flag of @p flags (and flags != None). */
[[nodiscard]] constexpr bool hasRecovery(MidiReadRecovery set,
                                         MidiReadRecovery flags) noexcept
{
    return flags != MidiReadRecovery::None && (set & flags) == flags;
}

/**
 * @brief Options for one MidiFile::read() call.
 *
 * The default is the strict contract: no recovery and PPQN division only, so
 * a successful default read always reports ticksPerQuarter() > 0.
 */
struct MidiReadOptions
{
    /** Recoveries the read may apply; None keeps every defect fatal. */
    MidiReadRecovery tolerate = MidiReadRecovery::None;
    /** Accept SMPTE time division. A file read with it reports
        ticksPerQuarter() == 0 and isSmpteDivision() == true. */
    bool acceptSmpteDivision = false;

    /** @brief The default strict options. */
    [[nodiscard]] static constexpr MidiReadOptions strict() noexcept
    {
        return {};
    }

    /** @brief Every recovery and SMPTE division allowed. */
    [[nodiscard]] static constexpr MidiReadOptions tolerant() noexcept
    {
        return { MidiReadRecovery::All, true };
    }
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
        if (ppqn == 0 || ppqn > 0x7fffu)
            return false;
        return replaceWithEmpty(format, ppqn, trackCount);
    }

    /**
     * @brief Replaces the document with an empty format 0 or 1 file that uses
     *        SMPTE time division.
     * @param format SMF format, either 0 or 1.
     * @param framesPerSecond 24, 25, 29 (30 drop frame, 29.97 frames per
     *        second) or 30.
     * @param ticksPerFrame Ticks per frame in the range 1..255.
     * @param trackCount Initial track count. Format 0 requires exactly one.
     * @return True on success; false leaves the current document unchanged.
     */
    [[nodiscard]] bool createSmpte(uint16_t format, int framesPerSecond,
                                   uint8_t ticksPerFrame, size_t trackCount = 1)
    {
        if (framesPerSecond != 24 && framesPerSecond != 25
            && framesPerSecond != 29 && framesPerSecond != 30)
            return false;
        const auto division = static_cast<uint16_t>(
            (static_cast<unsigned>(256 - framesPerSecond) << 8) | ticksPerFrame);
        if (!validDivision(division))
            return false;
        return replaceWithEmpty(format, division, trackCount);
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
     * @param track Destination track index.
     * @param delta Delta time in ticks from the preceding event.
     * @param f0OrF7 Must be MidiEventKind::SysExF0 or SysExF7.
     * @param payload SysEx payload bytes, excluding the status byte.
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
        division_ = 0;
        tracks_.clear();
        validationTotals_ = {};
        trackAuthoringStates_.clear();
        recoveries_ = MidiReadRecovery::None;
    }

    /**
     * @brief Reads and validates an SMF format 0, 1, or 2 file.
     *
     * With the default options every structural defect is fatal and only PPQN
     * division is accepted. Recoveries allowed in @p options are applied only
     * when needed and reported by recoveries(); a recovered document always
     * satisfies the strict model (each track ends with one end-of-track event
     * and no SysEx chain is left open), so it can be written back unchanged.
     *
     * @param path File to read.
     * @param options Allowed recoveries and division kinds.
     * @return True on success. Any failure leaves this object empty.
     */
    [[nodiscard]] bool read(const std::filesystem::path& path,
                            const MidiReadOptions& options = {})
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
        if (!parseFile(bytes, options, parsed))
            return false;
        *this = std::move(parsed);
        return true;
    }

    /**
     * @brief Writes a validated format 0 or 1 document.
     *
     * A missing EOT is emitted with delta zero without changing tracks(). The
     * complete output is built before the destination is opened, and false is
     * returned when writing, flushing or closing the destination fails.
     */
    [[nodiscard]] bool write(const std::filesystem::path& path) const
    {
        ValidationTotals totals;
        if (format_ != 0 && format_ != 1)
            return false;
        if (!validateDocument(format_, division_, tracks_, false, false, totals))
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
        appendBE16(output, division_);

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
        // The stream buffers: a short file is only written by the flush in
        // close(), so good() right after write() reported success even when
        // nothing reached the disk.
        destination.flush();
        if (!destination.good())
            return false;
        destination.close();
        return !destination.fail();
    }

    /** @brief Returns 0, 1, or 2; returns -1 for an empty object. */
    [[nodiscard]] int format() const noexcept { return format_; }

    /** @brief Returns PPQN, or zero for an empty object or SMPTE division. */
    [[nodiscard]] uint16_t ticksPerQuarter() const noexcept
    {
        return isSmpteDivision() ? uint16_t { 0 } : division_;
    }

    /** @brief Returns the raw header division word, or zero for an empty object. */
    [[nodiscard]] uint16_t division() const noexcept { return division_; }

    /** @brief True when the document uses SMPTE time division. */
    [[nodiscard]] bool isSmpteDivision() const noexcept
    {
        return (division_ & 0x8000u) != 0;
    }

    /**
     * @brief Returns the SMPTE frame-rate code: 24, 25, 29 or 30.
     *
     * 29 is the SMF code for 30 drop frame, whose real rate is 30000/1001
     * (29.97) frames per second; timing uses that exact rate. Returns 0 for
     * PPQN division or an empty object.
     */
    [[nodiscard]] int smpteFramesPerSecond() const noexcept
    {
        return isSmpteDivision() ? 256 - static_cast<int>(division_ >> 8) : 0;
    }

    /** @brief Returns SMPTE ticks per frame, or 0 for PPQN division. */
    [[nodiscard]] uint8_t ticksPerFrame() const noexcept
    {
        return isSmpteDivision() ? static_cast<uint8_t>(division_ & 0xffu)
                                 : uint8_t { 0 };
    }

    /**
     * @brief Returns the recoveries the last successful read applied.
     *
     * None after a strict read, a failed read, create(), createSmpte() or
     * clear().
     */
    [[nodiscard]] MidiReadRecovery recoveries() const noexcept
    {
        return recoveries_;
    }

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
     * default 500000 us/qn unless a tick-zero tempo replaces it. Returns
     * nullopt for an SMPTE-division document: tempo events do not affect its
     * timing.
     */
    [[nodiscard]] std::optional<std::vector<MidiTempoChange>>
    tempoMap(size_t track = 0) const
    {
        const std::optional<size_t> source = tempoTrack(track);
        if (!source || isSmpteDivision())
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
     *
     * PPQN documents integrate the tempo map. SMPTE documents ignore tempo
     * events: a tick is 1 / (framesPerSecond * ticksPerFrame) seconds, with
     * 30000/1001 frames per second for the 29 (30 drop frame) code.
     *
     * @return nullopt for an empty/invalid track or uint64 overflow.
     */
    [[nodiscard]] std::optional<uint64_t>
    tickToMicroseconds(uint64_t tick, size_t track = 0) const noexcept
    {
        uint64_t whole = 0;
        uint64_t remainder = 0;
        uint16_t denominator = 0;
        if (!integrateTicks(tick, track, whole, remainder, denominator))
            return std::nullopt;
        return whole;
    }

    /**
     * @brief Converts an absolute tick directly from the shared rational sum.
     *
     * Same time base as tickToMicroseconds(), rounded once to double.
     *
     * @return nullopt for an empty/invalid track or elapsed-time overflow.
     */
    [[nodiscard]] std::optional<double>
    tickToSeconds(uint64_t tick, size_t track = 0) const noexcept
    {
        uint64_t whole = 0;
        uint64_t remainder = 0;
        uint16_t denominator = 0;
        if (!integrateTicks(tick, track, whole, remainder, denominator))
            return std::nullopt;
        return secondsFromRationalMicroseconds(whole, remainder, denominator);
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

        /** Moves to an absolute position, clamped to the end. */
        void seek(size_t position) noexcept
        {
            pos_ = std::min(position, end_);
        }

        /** Moves the end bound; the caller keeps it inside the byte vector. */
        void setEnd(size_t end) noexcept
        {
            end_ = end;
            pos_ = std::min(pos_, end_);
        }

    private:
        const std::vector<uint8_t>& bytes_;
        size_t pos_ = 0;
        size_t end_ = 0;
    };

    int format_ = -1;
    uint16_t division_ = 0;
    std::vector<MidiTrack> tracks_;
    ValidationTotals validationTotals_;
    std::vector<TrackAuthoringState> trackAuthoringStates_;
    MidiReadRecovery recoveries_ = MidiReadRecovery::None;

    [[nodiscard]] bool replaceWithEmpty(uint16_t format, uint16_t division,
                                        size_t trackCount)
    {
        if (format != 0 && format != 1)
            return false;
        if (trackCount == 0 || trackCount > kMaxTracks)
            return false;
        if (format == 0 && trackCount != 1)
            return false;

        std::vector<MidiTrack> replacement(trackCount);
        std::vector<TrackAuthoringState> replacementStates(trackCount);
        format_ = static_cast<int>(format);
        division_ = division;
        tracks_ = std::move(replacement);
        validationTotals_ = {};
        trackAuthoringStates_ = std::move(replacementStates);
        recoveries_ = MidiReadRecovery::None;
        return true;
    }

    /** PPQN 1..32767, or SMPTE -24/-25/-29/-30 with 1..255 ticks per frame. */
    static bool validDivision(uint16_t division) noexcept
    {
        if ((division & 0x8000u) == 0)
            return division != 0;
        const unsigned rate = 256u - (static_cast<unsigned>(division) >> 8);
        return (rate == 24 || rate == 25 || rate == 29 || rate == 30)
            && (division & 0xffu) != 0;
    }

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

    static bool validateDocument(int format, uint16_t division,
                                 const std::vector<MidiTrack>& tracks,
                                 bool requireEot,
                                 bool allowOpenSysEx,
                                 ValidationTotals& totals,
                                 std::span<TrackAuthoringState> authoringStates = {}) noexcept
    {
        if (format < 0 || format > 2 || !validDivision(division))
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

    /**
     * Reads one variable-length quantity of at most four significant bytes.
     * The strict form rejects a redundant leading 0x80 byte; the tolerant form
     * skips such bytes and reports them through @p overlong. Templated so the
     * strict per-event path carries no tolerance test.
     */
    template <bool Tolerant>
    static bool readVlq(ByteCursor& cursor, uint32_t& value,
                        bool& overlong) noexcept
    {
        uint8_t byte = 0;
        if (!cursor.readU8(byte))
            return false;
        if (byte == 0x80u)
        {
            if constexpr (Tolerant)
            {
                do
                {
                    overlong = true;
                    if (!cursor.readU8(byte))
                        return false;
                } while (byte == 0x80u);
            }
            else
            {
                return false;
            }
        }
        value = byte & 0x7fu;
        if ((byte & 0x80u) == 0)
            return true;
        for (unsigned count = 1; count < 4; ++count)
        {
            if (!cursor.readU8(byte))
                return false;
            value = static_cast<uint32_t>((value << 7) | (byte & 0x7fu));
            if ((byte & 0x80u) == 0)
                return true;
        }
        return false;
    }

    /** Ok, ran out of chunk bytes inside the event, or malformed. */
    enum class EventParse : uint8_t
    {
        Ok,
        Truncated,
        Invalid
    };

    /**
     * Parses one event. SysEx continuation rules are applied by the caller;
     * running status is updated only by a status byte. Payloads are checked
     * in place and copied only when Store is set; @p endsF7 reports whether a
     * SysEx payload ends with 0xF7.
     */
    template <bool TolerantVlq, bool Store>
    static EventParse parseEvent(ByteCursor& cursor,
                                 bool& overlong, uint8_t& runningStatus,
                                 ValidationTotals& totals, MidiEvent& event,
                                 bool& endsF7)
    {
        const auto vlqFailure = [&cursor]() noexcept {
            return cursor.remaining() == 0 ? EventParse::Truncated
                                           : EventParse::Invalid;
        };
        if (!readVlq<TolerantVlq>(cursor, event.deltaTicks, overlong))
            return vlqFailure();

        uint8_t first = 0;
        if (!cursor.readU8(first))
            return EventParse::Truncated;
        if (first < 0x80)
        {
            if (runningStatus == 0)
                return EventParse::Invalid;
            event.kind = MidiEventKind::Channel;
            event.status = runningStatus;
            event.data1 = first;
            if (channelDataCount(runningStatus) == 2)
            {
                if (!cursor.readU8(event.data2))
                    return EventParse::Truncated;
                if (event.data2 >= 0x80)
                    return EventParse::Invalid;
            }
        }
        else if (first <= 0xef)
        {
            event.kind = MidiEventKind::Channel;
            event.status = first;
            runningStatus = first;
            if (!cursor.readU8(event.data1))
                return EventParse::Truncated;
            if (event.data1 >= 0x80)
                return EventParse::Invalid;
            if (channelDataCount(first) == 2)
            {
                if (!cursor.readU8(event.data2))
                    return EventParse::Truncated;
                if (event.data2 >= 0x80)
                    return EventParse::Invalid;
            }
        }
        else if (first == 0xf0 || first == 0xf7 || first == 0xff)
        {
            runningStatus = 0;
            if (first == 0xff)
            {
                event.kind = MidiEventKind::Meta;
                if (!cursor.readU8(event.metaType))
                    return EventParse::Truncated;
                if (event.metaType >= 0x80)
                    return EventParse::Invalid;
            }
            else
            {
                event.kind = first == 0xf0 ? MidiEventKind::SysExF0
                                           : MidiEventKind::SysExF7;
            }
            uint32_t length = 0;
            if (!readVlq<TolerantVlq>(cursor, length, overlong))
                return vlqFailure();
            if (length > cursor.remaining())
                return EventParse::Truncated;
            const uint8_t* data = nullptr;
            if (totals.payloadBytes > kMaxAggregatePayloadBytes - length
                || !cursor.readSpan(length, data))
                return EventParse::Invalid;
            totals.payloadBytes += length;
            endsF7 = length != 0 && data[length - 1] == 0xf7;
            if (event.kind == MidiEventKind::Meta
                && ((event.metaType == 0x51 && length == 3
                     && data[0] == 0 && data[1] == 0 && data[2] == 0)
                    || (event.metaType == 0x2f && length != 0)))
                return EventParse::Invalid;
            if constexpr (Store)
                event.payload.assign(data, data + length);
        }
        else
        {
            return EventParse::Invalid;
        }
        return validateEventShape(event) ? EventParse::Ok : EventParse::Invalid;
    }

    /** At least 8 bytes at @p pos holding a printable tag and a length that fits. */
    static bool looksLikeChunkHeader(const std::vector<uint8_t>& bytes,
                                     size_t pos, size_t end) noexcept
    {
        if (pos > end || end - pos < 8)
            return false;
        for (size_t i = 0; i < 4; ++i)
            if (bytes[pos + i] < 0x20 || bytes[pos + i] > 0x7e)
                return false;
        const uint32_t length = (static_cast<uint32_t>(bytes[pos + 4]) << 24)
                              | (static_cast<uint32_t>(bytes[pos + 5]) << 16)
                              | (static_cast<uint32_t>(bytes[pos + 6]) << 8)
                              | static_cast<uint32_t>(bytes[pos + 7]);
        return length <= end - pos - 8;
    }

    static bool startsTrackChunk(const std::vector<uint8_t>& bytes,
                                 size_t pos, size_t end) noexcept
    {
        return pos <= end && end - pos >= 8
            && std::memcmp(bytes.data() + pos, "MTrk", 4) == 0;
    }

    enum class TrackParse : uint8_t
    {
        Ok,
        Failed,
        RetryWithoutExtension
    };

    /**
     * Parses one track chunk [begin, declaredEnd). @p resumeAt receives the
     * file position where chunk parsing continues. With TrackLengthMismatch
     * allowed and @p allowExtension set, a chunk that ends without its
     * end-of-track event and is not followed by a chunk header is parsed on
     * past its declared end; RetryWithoutExtension asks the caller to parse
     * it again without that when no end-of-track event is found there either.
     * TolerantVlq selects the VLQ reader at compile time, so the strict event
     * loop carries no tolerance test. Without Store nothing is allocated: the
     * pass only counts in @p count the events a storing pass keeps, with the
     * same checks and outcome, so the storing pass can reserve exactly.
     */
    template <bool TolerantVlq, bool Store>
    static TrackParse parseTrack(const std::vector<uint8_t>& bytes, size_t begin,
                                 size_t declaredEnd, size_t fileEnd,
                                 MidiReadRecovery allowed, bool allowExtension,
                                 MidiReadRecovery& applied,
                                 ValidationTotals& totals, MidiTrack& result,
                                 size_t& count, size_t& resumeAt)
    {
        const bool closeSysExAllowed =
            hasRecovery(allowed, MidiReadRecovery::UnterminatedSysEx);
        ByteCursor cursor(bytes, begin, declaredEnd);
        uint8_t runningStatus = 0;
        bool sysexOpen = false;
        size_t openPacket = 0;
        bool sawEot = false;
        bool overlong = false;
        bool extended = false;
        uint64_t absoluteTick = 0;

        const auto closeOpenSysEx = [&]() {
            if (!closeSysExAllowed
                || totals.payloadBytes >= kMaxAggregatePayloadBytes)
                return false;
            if constexpr (Store)
                result.events[openPacket].payload.push_back(0xf7);
            ++totals.payloadBytes;
            sysexOpen = false;
            applied = applied | MidiReadRecovery::UnterminatedSysEx;
            return true;
        };

        for (;;)
        {
            while (cursor.remaining() > 0)
            {
                if (totals.events >= kMaxEvents)
                    return TrackParse::Failed;
                const size_t eventStart = cursor.position();
                const uint8_t statusBefore = runningStatus;
                MidiEvent event;
                bool endsF7 = false;
                const EventParse parsed = parseEvent<TolerantVlq, Store>(
                    cursor, overlong, runningStatus, totals, event, endsF7);
                if (parsed == EventParse::Invalid)
                    return TrackParse::Failed;
                if (parsed == EventParse::Truncated)
                {
                    cursor.seek(eventStart);
                    runningStatus = statusBefore;
                    break;
                }
                if (absoluteTick > std::numeric_limits<uint64_t>::max() - event.deltaTicks)
                    return TrackParse::Failed;
                absoluteTick += event.deltaTicks;

                if (sysexOpen && event.kind != MidiEventKind::SysExF7
                    && !closeOpenSysEx())
                    return TrackParse::Failed;
                if (event.kind == MidiEventKind::SysExF0
                    || (event.kind == MidiEventKind::SysExF7 && sysexOpen))
                {
                    sysexOpen = !endsF7;
                    openPacket = count;
                }
                const bool endOfTrack = isEot(event);
                ++totals.events;
                ++count;
                if constexpr (Store)
                    result.events.push_back(std::move(event));
                if (endOfTrack)
                {
                    sawEot = true;
                    break;
                }
            }
            if (sawEot || extended || !allowExtension
                || !hasRecovery(allowed, MidiReadRecovery::TrackLengthMismatch)
                || declaredEnd >= fileEnd
                || looksLikeChunkHeader(bytes, declaredEnd, fileEnd))
                break;
            // The chunk ended without its end-of-track event and no chunk
            // header follows: its declared length understated the events.
            cursor.setEnd(std::min(fileEnd, begin + static_cast<size_t>(kMaxTrackChunkBytes)));
            extended = true;
        }

        if (overlong)
            applied = applied | MidiReadRecovery::NoncanonicalVlq;

        if (extended)
        {
            if (!sawEot)
                return TrackParse::RetryWithoutExtension;
            applied = applied | MidiReadRecovery::TrackLengthMismatch;
            resumeAt = cursor.position();
            return TrackParse::Ok;
        }

        resumeAt = declaredEnd;
        if (sawEot)
        {
            if (cursor.remaining() == 0)
                return TrackParse::Ok;
            const size_t after = cursor.position();
            if (hasRecovery(allowed, MidiReadRecovery::TrackLengthMismatch)
                && startsTrackChunk(bytes, after, fileEnd))
            {
                // The next track chunk starts inside this one's declared length.
                applied = applied | MidiReadRecovery::TrackLengthMismatch;
                resumeAt = after;
                return TrackParse::Ok;
            }
            if (!hasRecovery(allowed, MidiReadRecovery::DataAfterEndOfTrack))
                return TrackParse::Failed;
            applied = applied | MidiReadRecovery::DataAfterEndOfTrack;
            return TrackParse::Ok;
        }

        // The track ended, cleanly or inside an incomplete event, without its
        // end-of-track event.
        if (!hasRecovery(allowed, MidiReadRecovery::MissingEndOfTrack)
            || (sysexOpen && !closeOpenSysEx())
            || totals.events >= kMaxEvents)
            return TrackParse::Failed;
        ++totals.events;
        ++count;
        if constexpr (Store)
        {
            MidiEvent endOfTrack;
            endOfTrack.kind = MidiEventKind::Meta;
            endOfTrack.metaType = 0x2f;
            result.events.push_back(std::move(endOfTrack));
        }
        applied = applied | MidiReadRecovery::MissingEndOfTrack;
        return TrackParse::Ok;
    }

    static bool parseFile(const std::vector<uint8_t>& bytes,
                          const MidiReadOptions& options, MidiFile& result)
    {
        if (bytes.size() < 14 || bytes.size() > kMaxInputBytes)
            return false;
        const MidiReadRecovery allowed = options.tolerate & MidiReadRecovery::All;
        const bool tolerantVlq = hasRecovery(allowed, MidiReadRecovery::NoncanonicalVlq);
        MidiReadRecovery applied = MidiReadRecovery::None;
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
        if (format > 2 || !validDivision(division)
            || ((division & 0x8000u) != 0 && !options.acceptSmpteDivision))
            return false;
        if (trackCount == 0 || trackCount > kMaxTracks
            || (format == 0 && trackCount != 1))
        {
            if (!hasRecovery(allowed, MidiReadRecovery::TrackCountMismatch))
                return false;
            applied = applied | MidiReadRecovery::TrackCountMismatch;
        }

        std::vector<MidiTrack> tracks;
        tracks.reserve(std::min<size_t>(trackCount, kMaxTracks));
        ValidationTotals totals;
        while (cursor.remaining() > 0)
        {
            if (cursor.remaining() < 8)
            {
                if (!hasRecovery(allowed, MidiReadRecovery::TrailingBytes))
                    return false;
                applied = applied | MidiReadRecovery::TrailingBytes;
                break;
            }
            uint32_t length = 0;
            if (!cursor.readTag(tag) || !cursor.readBE32(length))
                return false;
            const bool isTrack = sameTag(tag, "MTrk");
            size_t chunkLength = length;
            if (length > cursor.remaining())
            {
                if (isTrack
                    && hasRecovery(allowed, MidiReadRecovery::TrackLengthMismatch))
                {
                    chunkLength = cursor.remaining();
                    applied = applied | MidiReadRecovery::TrackLengthMismatch;
                }
                else if (!isTrack && !sameTag(tag, "MThd")
                         && hasRecovery(allowed, MidiReadRecovery::TrailingBytes))
                {
                    applied = applied | MidiReadRecovery::TrailingBytes;
                    break;
                }
                else
                {
                    return false;
                }
            }
            const size_t begin = cursor.position();
            const size_t end = begin + chunkLength;
            size_t resumeAt = end;

            if (isTrack)
            {
                const bool extra = format == 0 ? !tracks.empty()
                                               : tracks.size() >= trackCount;
                if (extra)
                {
                    if (!hasRecovery(allowed, MidiReadRecovery::TrackCountMismatch))
                        return false;
                    applied = applied | MidiReadRecovery::TrackCountMismatch;
                }
                if (chunkLength > kMaxTrackChunkBytes)
                    return false;
                if (format != 0 || !extra)
                {
                    // A format 0 file keeps its first track chunk only.
                    if (tracks.size() >= kMaxTracks)
                        return false;
                    // A counting pass runs the same checks without
                    // allocating; the storing pass then reserves exactly the
                    // events it keeps. That is faster than growing the vector
                    // and never needs more memory, and a track the counting
                    // pass rejects allocates nothing.
                    MidiTrack track;
                    const auto pass = [&](auto store, bool allowExtension,
                                          ValidationTotals& passTotals,
                                          MidiReadRecovery& passApplied,
                                          size_t& passCount) {
                        constexpr bool storing = decltype(store)::value;
                        passCount = 0;
                        return tolerantVlq
                            ? parseTrack<true, storing>(
                                  bytes, begin, end, bytes.size(), allowed, allowExtension,
                                  passApplied, passTotals, track, passCount, resumeAt)
                            : parseTrack<false, storing>(
                                  bytes, begin, end, bytes.size(), allowed, allowExtension,
                                  passApplied, passTotals, track, passCount, resumeAt);
                    };
                    ValidationTotals countTotals = totals;
                    MidiReadRecovery countApplied = applied;
                    size_t count = 0;
                    bool allowExtension = true;
                    TrackParse counted = pass(std::false_type {}, true, countTotals,
                                              countApplied, count);
                    if (counted == TrackParse::RetryWithoutExtension)
                    {
                        allowExtension = false;
                        countTotals = totals;
                        countApplied = applied;
                        counted = pass(std::false_type {}, false, countTotals,
                                       countApplied, count);
                    }
                    if (counted != TrackParse::Ok)
                        return false;
                    track.events.reserve(count);
                    size_t stored = 0;
                    if (pass(std::true_type {}, allowExtension, totals, applied, stored)
                            != TrackParse::Ok
                        || stored != count)
                        return false;
                    tracks.push_back(std::move(track));
                }
            }
            else
            {
                if (sameTag(tag, "MThd") || chunkLength > kMaxTrackChunkBytes)
                    return false;
            }
            cursor.seek(resumeAt);
        }
        if (tracks.size() != trackCount)
        {
            if (!hasRecovery(allowed, MidiReadRecovery::TrackCountMismatch)
                || tracks.empty())
                return false;
            applied = applied | MidiReadRecovery::TrackCountMismatch;
        }

        ValidationTotals verified;
        std::vector<TrackAuthoringState> authoringStates(tracks.size());
        if (!validateDocument(format, division, tracks, true, false, verified,
                              authoringStates))
            return false;
        result.format_ = format;
        result.division_ = division;
        result.tracks_ = std::move(tracks);
        result.validationTotals_ = verified;
        result.trackAuthoringStates_ = std::move(authoringStates);
        result.recoveries_ = applied;
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
        if (format_ < 0 || format_ > 2 || division_ == 0 || track >= tracks_.size())
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

    /** Adds ticks * tempo / ppqn microseconds to the exact sum whole + remainder / ppqn. */
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

    /**
     * Microseconds per SMPTE tick as numerator / denominator: 10^6 divided by
     * framesPerSecond * ticksPerFrame, with 30000/1001 frames per second for
     * the 29 (30 drop frame) code. Every denominator is at most 3 * 255.
     */
    void smpteTickMicroseconds(uint32_t& numerator, uint16_t& denominator) const noexcept
    {
        const unsigned ticks = division_ & 0xffu;
        switch (smpteFramesPerSecond())
        {
            case 24: numerator = 125000; denominator = static_cast<uint16_t>(3 * ticks); break;
            case 25: numerator = 40000; denominator = static_cast<uint16_t>(ticks); break;
            case 29: numerator = 100100; denominator = static_cast<uint16_t>(3 * ticks); break;
            default: numerator = 100000; denominator = static_cast<uint16_t>(3 * ticks); break;
        }
    }

    bool integrateTicks(uint64_t targetTick, size_t track, uint64_t& whole,
                        uint64_t& remainder, uint16_t& denominator) const noexcept
    {
        const std::optional<size_t> source = tempoTrack(track);
        if (!source)
            return false;
        whole = 0;
        remainder = 0;
        if (isSmpteDivision())
        {
            // SMPTE ticks are fixed fractions of a second; tempo is ignored.
            uint32_t numerator = 0;
            smpteTickMicroseconds(numerator, denominator);
            return addInterval(targetTick, numerator, denominator, whole, remainder);
        }
        denominator = division_;
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
            if (!addInterval(eventTick - integratedTick, tempo, denominator, whole, remainder))
                return false;
            integratedTick = eventTick;
            tempo = nextTempo;
        }
        return addInterval(targetTick - integratedTick, tempo, denominator, whole, remainder);
    }
};

} // namespace dspark
