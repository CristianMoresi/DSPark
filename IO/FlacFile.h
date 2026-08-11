// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file FlacFile.h
 * @brief Native, bounded, decode-only FLAC reader.
 *
 * Implements RFC 9639 without a codec dependency. openRead() eagerly validates
 * the complete native FLAC stream, every frame CRC, and any nonzero STREAMINFO
 * MD5 while building a compact frame index. Compressed bytes are retained, but
 * decoded PCM is not: range reads decode only intersecting indexed frames.
 * Ogg encapsulation and encoding are intentionally unsupported.
 *
 * File parsing, checksum work, and range decoding allocate and block. They are
 * intended for offline or worker threads, never the real-time audio thread.
 * The header is omitted from the umbrella when DSPARK_NO_FILE_IO is defined.
 *
 * Threading: owner-managed. One instance is used by one thread at a time.
 *
 * Dependencies: IO/AudioFile.h (Core/AudioBuffer.h, Core/AudioSpec.h).
 */

#include "AudioFile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class FlacFile
 * @brief Decode-only native FLAC implementation with eager integrity checks.
 */
class FlacFile : public AudioFile
{
public:
    /** @brief Maximum compressed input retained by one instance. */
    static constexpr uint64_t kMaxInputBytes = 256ull * 1024 * 1024;
    /** @brief Maximum number of metadata block headers scanned. */
    static constexpr uint64_t kMaxMetadataBlocks = 65536;
    /** @brief Maximum number of indexed audio frames. */
    static constexpr uint64_t kMaxFrames = 1048576;
    /** @brief Maximum decoded interchannel sample count. */
    static constexpr uint64_t kMaxInterchannelSamples = 1ull << 31;
    /** @brief Maximum logical decoded PCM byte count. */
    static constexpr uint64_t kMaxDecodedPcmBytes = 8ull << 30;
    /** @brief Maximum zero run accepted in one Rice unary code. */
    static constexpr uint64_t kMaxRiceUnaryZeros = 1ull << 20;
    /** @brief Stable explanation returned by documentation for write failure. */
    static constexpr std::string_view kWriteUnsupportedReason =
        "FlacFile is decode-only; FLAC encoding is not supported.";

    ~FlacFile() override { close(); }

    /**
     * @brief Opens and fully validates a native FLAC stream.
     * @return True only after metadata, frames, CRCs, coverage, and MD5 pass.
     */
    [[nodiscard]] bool openRead(const std::filesystem::path& path) override
    {
        close();

        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input.is_open())
            return false;
        const std::streampos end = input.tellg();
        if (end <= std::streampos(0))
            return false;
        const uint64_t fileSize = static_cast<uint64_t>(end);
        if (fileSize > kMaxInputBytes
            || fileSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            return false;

        OpenState candidate;
        candidate.bytes.resize(static_cast<size_t>(fileSize));
        input.seekg(0, std::ios::beg);
        input.read(reinterpret_cast<char*>(candidate.bytes.data()),
                   static_cast<std::streamsize>(candidate.bytes.size()));
        if (input.gcount() != static_cast<std::streamsize>(candidate.bytes.size()))
            return false;
        if (!validateOpen(candidate))
            return false;

        bytes_ = std::move(candidate.bytes);
        frames_ = std::move(candidate.frames);
        stream_ = candidate.stream;
        info_ = candidate.info;
        open_ = true;
        return true;
    }

    /**
     * @brief FLAC encoding is unsupported; closes prior state and returns false.
     *
     * The destination path is never opened or modified.
     */
    [[nodiscard]] bool openWrite(const std::filesystem::path& path,
                                 const AudioFileInfo& info) override
    {
        (void)path;
        (void)info;
        close();
        return false;
    }

    /** @brief Returns current stream metadata, or default metadata when closed. */
    [[nodiscard]] AudioFileInfo getInfo() const override { return info_; }

    /** @brief Reads from sample zero, clamped to the destination dimensions. */
    [[nodiscard]] bool readSamples(AudioBufferView<float> dest) override
    {
        if (!open_) return false;
        return readSamples(dest, 0, info_.numSamples);
    }

    /**
     * @brief Decodes an exact valid file range into the destination capacity.
     *
     * The requested range must be wholly valid. Transfer is clamped to the
     * destination's frames and channels, leaving every excess element untouched.
     */
    [[nodiscard]] bool readSamples(AudioBufferView<float> dest,
                                   int64_t startFrame,
                                   int64_t numFrames) override
    {
        if (!open_ || startFrame < 0 || numFrames <= 0)
            return false;
        if (startFrame > info_.numSamples
            || numFrames > info_.numSamples - startFrame)
            return false;

        const int destChannels = std::max(0, dest.getNumChannels());
        const int destSamples = std::max(0, dest.getNumSamples());
        const int channelsToCopy = std::min(destChannels,
            static_cast<int>(info_.numChannels));
        const int64_t framesToCopy = std::min<int64_t>(numFrames, destSamples);
        if (channelsToCopy == 0 || framesToCopy == 0)
            return true;

        const uint64_t first = static_cast<uint64_t>(startFrame);
        const uint64_t last = first + static_cast<uint64_t>(framesToCopy);
        auto it = std::lower_bound(frames_.begin(), frames_.end(), first,
            [](const FrameIndex& frame, uint64_t sample) {
                return frame.firstSample + frame.blockSize <= sample;
            });

        int64_t written = 0;
        std::vector<int64_t> pcm;
        while (it != frames_.end() && it->firstSample < last)
        {
            const size_t frameNumber = static_cast<size_t>(it - frames_.begin());
            DecodedFrame decoded;
            if (!decodeFrame(bytes_, it->offset, stream_, frameNumber,
                             it->firstSample, pcm, decoded)
                || decoded.endOffset != it->offset + it->size
                || decoded.blockSize != it->blockSize)
            {
                close();
                return false;
            }

            const uint64_t overlapBegin = std::max(first, it->firstSample);
            const uint64_t overlapEnd = std::min(last,
                it->firstSample + static_cast<uint64_t>(it->blockSize));
            const size_t sourceOffset = static_cast<size_t>(overlapBegin - it->firstSample);
            const size_t count = static_cast<size_t>(overlapEnd - overlapBegin);
            const size_t destinationOffset = static_cast<size_t>(overlapBegin - first);

            for (int channel = 0; channel < channelsToCopy; ++channel)
            {
                float* output = dest.getChannel(channel) + destinationOffset;
                const int64_t* source = pcm.data()
                    + static_cast<size_t>(channel) * it->blockSize + sourceOffset;
                for (size_t i = 0; i < count; ++i)
                    output[i] = std::ldexp(static_cast<float>(source[i]),
                                           1 - static_cast<int>(stream_.bitsPerSample));
            }
            written += static_cast<int64_t>(count);
            ++it;
        }
        if (written != framesToCopy)
        {
            close();
            return false;
        }
        return true;
    }

    /** @brief Always returns false because this class is decode-only. */
    [[nodiscard]] bool writeSamples(AudioBufferView<const float> src) override
    {
        (void)src;
        return false;
    }

    /** @brief Releases compressed data, metadata, and the frame index. */
    void close() override
    {
        std::vector<uint8_t>().swap(bytes_);
        std::vector<FrameIndex>().swap(frames_);
        stream_ = {};
        info_ = {};
        open_ = false;
    }

    /** @brief True only after a complete stream has passed eager validation. */
    [[nodiscard]] bool isOpen() const noexcept override { return open_; }

private:
    struct StreamInfo
    {
        uint16_t minBlockSize = 0;
        uint16_t maxBlockSize = 0;
        uint32_t minFrameSize = 0;
        uint32_t maxFrameSize = 0;
        uint32_t sampleRate = 0;
        uint8_t channels = 0;
        uint8_t bitsPerSample = 0;
        uint64_t declaredSamples = 0;
        std::array<uint8_t, 16> md5 {};
    };

    struct FrameIndex
    {
        size_t offset = 0;
        size_t size = 0;
        uint64_t firstSample = 0;
        uint32_t blockSize = 0;
    };

    struct DecodedFrame
    {
        size_t endOffset = 0;
        uint32_t blockSize = 0;
        bool variableBlock = false;
    };

    struct OpenState
    {
        std::vector<uint8_t> bytes;
        std::vector<FrameIndex> frames;
        StreamInfo stream;
        AudioFileInfo info;
    };

    class ByteCursor
    {
    public:
        ByteCursor(const std::vector<uint8_t>& bytes, size_t begin, size_t end)
            : bytes_(bytes), pos_(begin), end_(end)
        {
        }

        [[nodiscard]] size_t position() const noexcept { return pos_; }
        [[nodiscard]] size_t remaining() const noexcept { return end_ - pos_; }

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

        bool readBE24(uint32_t& value) noexcept
        {
            uint8_t a = 0, b = 0, c = 0;
            if (!readU8(a) || !readU8(b) || !readU8(c)) return false;
            value = (static_cast<uint32_t>(a) << 16)
                  | (static_cast<uint32_t>(b) << 8)
                  | static_cast<uint32_t>(c);
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

    class BitCursor
    {
    public:
        BitCursor(const std::vector<uint8_t>& bytes, size_t beginByte,
                  size_t endByte)
            : bytes_(bytes), bitPos_(beginByte * 8), endBit_(endByte * 8)
        {
        }

        [[nodiscard]] size_t bitPosition() const noexcept { return bitPos_; }
        [[nodiscard]] size_t bytePosition() const noexcept { return bitPos_ / 8; }
        [[nodiscard]] uint64_t remainingBits() const noexcept
        {
            return static_cast<uint64_t>(endBit_ - bitPos_);
        }

        bool readBits(unsigned count, uint64_t& value) noexcept
        {
            value = 0;
            if (count > 64 || static_cast<uint64_t>(count) > remainingBits())
                return false;
            for (unsigned i = 0; i < count; ++i)
            {
                const uint8_t byte = bytes_[bitPos_ / 8];
                const unsigned shift = 7u - static_cast<unsigned>(bitPos_ & 7u);
                value = (value << 1) | ((byte >> shift) & 1u);
                ++bitPos_;
            }
            return true;
        }

        bool readSigned(unsigned count, int64_t& value) noexcept
        {
            if (count == 0)
            {
                value = 0;
                return true;
            }
            if (count > 63) return false;
            uint64_t raw = 0;
            if (!readBits(count, raw)) return false;
            const uint64_t sign = uint64_t { 1 } << (count - 1);
            if ((raw & sign) == 0)
            {
                value = static_cast<int64_t>(raw);
                return true;
            }
            const uint64_t magnitude = (uint64_t { 1 } << count) - raw;
            value = -static_cast<int64_t>(magnitude);
            return true;
        }

        bool readUnary(uint64_t limit, uint64_t& zeros) noexcept
        {
            zeros = 0;
            for (;;)
            {
                uint64_t bit = 0;
                if (!readBits(1, bit)) return false;
                if (bit != 0) return true;
                if (zeros >= limit) return false;
                ++zeros;
            }
        }

        bool alignWithZeroPadding() noexcept
        {
            while ((bitPos_ & 7u) != 0)
            {
                uint64_t bit = 0;
                if (!readBits(1, bit) || bit != 0) return false;
            }
            return true;
        }

    private:
        const std::vector<uint8_t>& bytes_;
        size_t bitPos_ = 0;
        size_t endBit_ = 0;
    };

    class Md5
    {
    public:
        Md5() noexcept { reset(); }

        void update(std::span<const uint8_t> data) noexcept
        {
            totalBytes_ += data.size();
            while (!data.empty())
            {
                const size_t take = std::min(data.size(), buffer_.size() - bufferSize_);
                std::memcpy(buffer_.data() + bufferSize_, data.data(), take);
                bufferSize_ += take;
                data = data.subspan(take);
                if (bufferSize_ == buffer_.size())
                {
                    transform(buffer_.data());
                    bufferSize_ = 0;
                }
            }
        }

        [[nodiscard]] std::array<uint8_t, 16> finish() noexcept
        {
            const uint64_t messageBits = totalBytes_ * 8;
            std::array<uint8_t, 120> padding {};
            padding[0] = 0x80;
            const size_t paddingSize = bufferSize_ < 56
                ? 56 - bufferSize_ : 64 + 56 - bufferSize_;
            update(std::span<const uint8_t>(padding.data(), paddingSize));
            uint8_t length[8] = {};
            for (unsigned i = 0; i < 8; ++i)
                length[i] = static_cast<uint8_t>(messageBits >> (8 * i));
            update(length);

            std::array<uint8_t, 16> digest {};
            for (size_t word = 0; word < 4; ++word)
                for (unsigned byte = 0; byte < 4; ++byte)
                    digest[word * 4 + byte] = static_cast<uint8_t>(
                        state_[word] >> (8 * byte));
            return digest;
        }

    private:
        std::array<uint32_t, 4> state_ {};
        std::array<uint8_t, 64> buffer_ {};
        size_t bufferSize_ = 0;
        uint64_t totalBytes_ = 0;

        void reset() noexcept
        {
            state_ = { 0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u };
            bufferSize_ = 0;
            totalBytes_ = 0;
        }

        static uint32_t rotateLeft(uint32_t value, unsigned count) noexcept
        {
            return static_cast<uint32_t>((value << count) | (value >> (32 - count)));
        }

        void transform(const uint8_t* block) noexcept
        {
            static constexpr uint32_t constants[64] = {
                0xd76aa478u,0xe8c7b756u,0x242070dbu,0xc1bdceeeu,
                0xf57c0fafu,0x4787c62au,0xa8304613u,0xfd469501u,
                0x698098d8u,0x8b44f7afu,0xffff5bb1u,0x895cd7beu,
                0x6b901122u,0xfd987193u,0xa679438eu,0x49b40821u,
                0xf61e2562u,0xc040b340u,0x265e5a51u,0xe9b6c7aau,
                0xd62f105du,0x02441453u,0xd8a1e681u,0xe7d3fbc8u,
                0x21e1cde6u,0xc33707d6u,0xf4d50d87u,0x455a14edu,
                0xa9e3e905u,0xfcefa3f8u,0x676f02d9u,0x8d2a4c8au,
                0xfffa3942u,0x8771f681u,0x6d9d6122u,0xfde5380cu,
                0xa4beea44u,0x4bdecfa9u,0xf6bb4b60u,0xbebfbc70u,
                0x289b7ec6u,0xeaa127fau,0xd4ef3085u,0x04881d05u,
                0xd9d4d039u,0xe6db99e5u,0x1fa27cf8u,0xc4ac5665u,
                0xf4292244u,0x432aff97u,0xab9423a7u,0xfc93a039u,
                0x655b59c3u,0x8f0ccc92u,0xffeff47du,0x85845dd1u,
                0x6fa87e4fu,0xfe2ce6e0u,0xa3014314u,0x4e0811a1u,
                0xf7537e82u,0xbd3af235u,0x2ad7d2bbu,0xeb86d391u
            };
            static constexpr unsigned shifts[64] = {
                7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
                5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
                4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
                6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
            };

            uint32_t words[16] = {};
            for (size_t i = 0; i < 16; ++i)
                words[i] = static_cast<uint32_t>(block[i * 4])
                         | (static_cast<uint32_t>(block[i * 4 + 1]) << 8)
                         | (static_cast<uint32_t>(block[i * 4 + 2]) << 16)
                         | (static_cast<uint32_t>(block[i * 4 + 3]) << 24);

            uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
            for (unsigned i = 0; i < 64; ++i)
            {
                uint32_t f = 0;
                unsigned index = 0;
                if (i < 16) { f = (b & c) | (~b & d); index = i; }
                else if (i < 32) { f = (d & b) | (~d & c); index = (5 * i + 1) & 15u; }
                else if (i < 48) { f = b ^ c ^ d; index = (3 * i + 5) & 15u; }
                else { f = c ^ (b | ~d); index = (7 * i) & 15u; }
                const uint32_t next = b + rotateLeft(a + f + constants[i]
                                                      + words[index], shifts[i]);
                a = d; d = c; c = b; b = next;
            }
            state_[0] += a;
            state_[1] += b;
            state_[2] += c;
            state_[3] += d;
        }
    };

    std::vector<uint8_t> bytes_;
    std::vector<FrameIndex> frames_;
    StreamInfo stream_;
    AudioFileInfo info_ {};
    bool open_ = false;

    static uint8_t crc8(std::span<const uint8_t> data) noexcept
    {
        uint8_t crc = 0;
        for (uint8_t byte : data)
        {
            crc ^= byte;
            for (unsigned bit = 0; bit < 8; ++bit)
                crc = static_cast<uint8_t>((crc & 0x80u) != 0
                    ? static_cast<uint8_t>((crc << 1) ^ 0x07u)
                    : static_cast<uint8_t>(crc << 1));
        }
        return crc;
    }

    static uint16_t crc16(std::span<const uint8_t> data) noexcept
    {
        uint16_t crc = 0;
        for (uint8_t byte : data)
        {
            crc ^= static_cast<uint16_t>(byte) << 8;
            for (unsigned bit = 0; bit < 8; ++bit)
                crc = static_cast<uint16_t>((crc & 0x8000u) != 0
                    ? static_cast<uint16_t>((crc << 1) ^ 0x8005u)
                    : static_cast<uint16_t>(crc << 1));
        }
        return crc;
    }

    static bool checkedAdd(int64_t a, int64_t b, int64_t& result) noexcept
    {
        if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b)
            || (b < 0 && a < std::numeric_limits<int64_t>::min() - b))
            return false;
        result = a + b;
        return true;
    }

    static bool checkedSubtract(int64_t a, int64_t b, int64_t& result) noexcept
    {
        if (b == std::numeric_limits<int64_t>::min())
            return false;
        return checkedAdd(a, -b, result);
    }

    static bool checkedMultiply(int64_t a, int64_t b, int64_t& result) noexcept
    {
        if (a == 0 || b == 0) { result = 0; return true; }
        if (a == -1 && b == std::numeric_limits<int64_t>::min()) return false;
        if (b == -1 && a == std::numeric_limits<int64_t>::min()) return false;
        if (a > 0)
        {
            if ((b > 0 && a > std::numeric_limits<int64_t>::max() / b)
                || (b < 0 && b < std::numeric_limits<int64_t>::min() / a))
                return false;
        }
        else
        {
            if ((b > 0 && a < std::numeric_limits<int64_t>::min() / b)
                || (b < 0 && a < std::numeric_limits<int64_t>::max() / b))
                return false;
        }
        result = a * b;
        return true;
    }

    static bool floorDividePowerOfTwo(int64_t value, unsigned shift,
                                      int64_t& result) noexcept
    {
        if (shift > 62) return false;
        if (shift == 0) { result = value; return true; }
        const int64_t divisor = int64_t { 1 } << shift;
        result = value / divisor;
        if (value < 0 && value % divisor != 0)
            --result;
        return true;
    }

    static bool sampleInDepth(int64_t value, unsigned depth) noexcept
    {
        if (depth == 0 || depth > 33) return false;
        const int64_t magnitude = int64_t { 1 } << (depth - 1);
        return value >= -magnitude && value <= magnitude - 1;
    }

    static bool parseStreamInfo(const uint8_t* data, StreamInfo& info) noexcept
    {
        info.minBlockSize = static_cast<uint16_t>(
            (static_cast<uint16_t>(data[0]) << 8) | data[1]);
        info.maxBlockSize = static_cast<uint16_t>(
            (static_cast<uint16_t>(data[2]) << 8) | data[3]);
        info.minFrameSize = (static_cast<uint32_t>(data[4]) << 16)
                          | (static_cast<uint32_t>(data[5]) << 8) | data[6];
        info.maxFrameSize = (static_cast<uint32_t>(data[7]) << 16)
                          | (static_cast<uint32_t>(data[8]) << 8) | data[9];
        const uint64_t packed = (static_cast<uint64_t>(data[10]) << 56)
                              | (static_cast<uint64_t>(data[11]) << 48)
                              | (static_cast<uint64_t>(data[12]) << 40)
                              | (static_cast<uint64_t>(data[13]) << 32)
                              | (static_cast<uint64_t>(data[14]) << 24)
                              | (static_cast<uint64_t>(data[15]) << 16)
                              | (static_cast<uint64_t>(data[16]) << 8)
                              | static_cast<uint64_t>(data[17]);
        info.sampleRate = static_cast<uint32_t>(packed >> 44);
        info.channels = static_cast<uint8_t>(((packed >> 41) & 7u) + 1u);
        info.bitsPerSample = static_cast<uint8_t>(((packed >> 36) & 31u) + 1u);
        info.declaredSamples = packed & 0xfffffffffull;
        std::copy(data + 18, data + 34, info.md5.begin());

        if (info.minBlockSize < 16 || info.maxBlockSize < 16
            || info.minBlockSize > info.maxBlockSize)
            return false;
        if (info.minFrameSize != 0 && info.maxFrameSize != 0
            && info.minFrameSize > info.maxFrameSize)
            return false;
        if (info.sampleRate == 0 || info.sampleRate > 1048575
            || info.channels == 0 || info.channels > 8
            || info.bitsPerSample < 4 || info.bitsPerSample > 32
            || info.declaredSamples > kMaxInterchannelSamples)
            return false;
        return true;
    }

    static bool parseCodedNumber(ByteCursor& cursor, bool variable,
                                 uint64_t& value) noexcept
    {
        uint8_t first = 0;
        if (!cursor.readU8(first)) return false;
        unsigned length = 0;
        uint64_t minimum = 0;
        if (first < 0x80) { length = 1; value = first; }
        else if ((first & 0xe0u) == 0xc0u) { length = 2; value = first & 0x1fu; minimum = 0x80; }
        else if ((first & 0xf0u) == 0xe0u) { length = 3; value = first & 0x0fu; minimum = 0x800; }
        else if ((first & 0xf8u) == 0xf0u) { length = 4; value = first & 0x07u; minimum = 0x10000; }
        else if ((first & 0xfcu) == 0xf8u) { length = 5; value = first & 0x03u; minimum = 0x200000; }
        else if ((first & 0xfeu) == 0xfcu) { length = 6; value = first & 0x01u; minimum = 0x4000000; }
        else if (first == 0xfeu) { length = 7; value = 0; minimum = 0x80000000ull; }
        else return false;

        for (unsigned i = 1; i < length; ++i)
        {
            uint8_t continuation = 0;
            if (!cursor.readU8(continuation) || (continuation & 0xc0u) != 0x80u)
                return false;
            value = (value << 6) | (continuation & 0x3fu);
        }
        if (length > 1 && value < minimum)
            return false;
        if (variable)
            return value <= 0xfffffffffull;
        return length <= 6 && value <= 0x7fffffffu;
    }

    static bool decodeBlockSize(uint8_t code, ByteCursor& cursor,
                                uint32_t& blockSize) noexcept
    {
        if (code == 0) return false;
        if (code == 1) blockSize = 192;
        else if (code >= 2 && code <= 5)
            blockSize = 576u << (code - 2u);
        else if (code == 6)
        {
            uint8_t value = 0;
            if (!cursor.readU8(value)) return false;
            blockSize = static_cast<uint32_t>(value) + 1u;
        }
        else if (code == 7)
        {
            uint16_t value = 0;
            if (!cursor.readBE16(value) || value == 0xffffu) return false;
            blockSize = static_cast<uint32_t>(value) + 1u;
        }
        else
            blockSize = 1u << code;
        return blockSize > 0 && blockSize <= 65535;
    }

    static bool decodeSampleRate(uint8_t code, ByteCursor& cursor,
                                 const StreamInfo& stream,
                                 uint32_t& sampleRate) noexcept
    {
        static constexpr uint32_t rates[12] = {
            0, 88200, 176400, 192000, 8000, 16000,
            22050, 24000, 32000, 44100, 48000, 96000
        };
        if (code <= 11)
            sampleRate = code == 0 ? stream.sampleRate : rates[code];
        else if (code == 12)
        {
            uint8_t value = 0;
            if (!cursor.readU8(value)) return false;
            sampleRate = static_cast<uint32_t>(value) * 1000u;
        }
        else if (code == 13)
        {
            uint16_t value = 0;
            if (!cursor.readBE16(value)) return false;
            sampleRate = value;
        }
        else if (code == 14)
        {
            uint16_t value = 0;
            if (!cursor.readBE16(value)) return false;
            sampleRate = static_cast<uint32_t>(value) * 10u;
        }
        else return false;
        return sampleRate != 0 && sampleRate == stream.sampleRate;
    }

    static bool decodeResidual(BitCursor& bits, uint32_t blockSize,
                               unsigned predictorOrder,
                               std::span<int64_t> output) noexcept
    {
        uint64_t method = 0, partitionOrder = 0;
        if (!bits.readBits(2, method) || method > 1
            || !bits.readBits(4, partitionOrder))
            return false;
        const uint32_t partitions = 1u << static_cast<unsigned>(partitionOrder);
        if (partitions == 0 || blockSize % partitions != 0)
            return false;
        const uint32_t partitionSize = blockSize / partitions;
        if (partitionSize <= predictorOrder)
            return false;
        const unsigned parameterBits = method == 0 ? 4u : 5u;
        const uint64_t escape = (uint64_t { 1 } << parameterBits) - 1u;
        size_t outputIndex = predictorOrder;

        for (uint32_t partition = 0; partition < partitions; ++partition)
        {
            uint64_t parameter = 0;
            if (!bits.readBits(parameterBits, parameter)) return false;
            const uint32_t count = partitionSize
                - (partition == 0 ? static_cast<uint32_t>(predictorOrder) : 0u);
            if (parameter == escape)
            {
                uint64_t width = 0;
                if (!bits.readBits(5, width) || width > 31) return false;
                for (uint32_t i = 0; i < count; ++i)
                {
                    int64_t residual = 0;
                    if (!bits.readSigned(static_cast<unsigned>(width), residual)
                        || residual < -2147483647ll || residual > 2147483647ll)
                        return false;
                    output[outputIndex++] = residual;
                }
            }
            else
            {
                for (uint32_t i = 0; i < count; ++i)
                {
                    uint64_t quotient = 0, remainder = 0;
                    if (!bits.readUnary(kMaxRiceUnaryZeros, quotient)
                        || !bits.readBits(static_cast<unsigned>(parameter), remainder)
                        || quotient > (std::numeric_limits<uint64_t>::max()
                                       >> static_cast<unsigned>(parameter)))
                        return false;
                    const uint64_t folded = (quotient << static_cast<unsigned>(parameter))
                                          | remainder;
                    if (folded > 4294967294ull)
                        return false;
                    const int64_t residual = (folded & 1u) == 0
                        ? static_cast<int64_t>(folded / 2u)
                        : -static_cast<int64_t>(folded / 2u) - 1;
                    if (residual < -2147483647ll || residual > 2147483647ll)
                        return false;
                    output[outputIndex++] = residual;
                }
            }
        }
        return outputIndex == output.size();
    }

    static bool decodeSubframe(BitCursor& bits, uint32_t blockSize,
                               unsigned storedDepth,
                               std::span<int64_t> output) noexcept
    {
        uint64_t zero = 0, type = 0, wastedFlag = 0;
        if (!bits.readBits(1, zero) || zero != 0
            || !bits.readBits(6, type)
            || !bits.readBits(1, wastedFlag))
            return false;

        unsigned wasted = 0;
        if (wastedFlag != 0)
        {
            uint64_t zeros = 0;
            if (!bits.readUnary(32, zeros) || zeros >= storedDepth)
                return false;
            wasted = static_cast<unsigned>(zeros + 1);
        }
        if (storedDepth <= wasted)
            return false;
        const unsigned depth = storedDepth - wasted;

        unsigned predictorOrder = 0;
        if (type == 0)
        {
            int64_t value = 0;
            if (!bits.readSigned(depth, value)) return false;
            std::fill(output.begin(), output.end(), value);
        }
        else if (type == 1)
        {
            for (int64_t& value : output)
                if (!bits.readSigned(depth, value)) return false;
        }
        else if (type >= 8 && type <= 12)
        {
            predictorOrder = static_cast<unsigned>(type - 8);
            if (predictorOrder > blockSize) return false;
            for (unsigned i = 0; i < predictorOrder; ++i)
                if (!bits.readSigned(depth, output[i])) return false;
            if (!decodeResidual(bits, blockSize, predictorOrder, output))
                return false;
            static constexpr int coefficients[5][4] = {
                { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 2, -1, 0, 0 },
                { 3, -3, 1, 0 }, { 4, -6, 4, -1 }
            };
            for (size_t i = predictorOrder; i < output.size(); ++i)
            {
                int64_t prediction = 0;
                for (unsigned j = 0; j < predictorOrder; ++j)
                {
                    int64_t product = 0;
                    if (!checkedMultiply(output[i - j - 1], coefficients[predictorOrder][j], product)
                        || !checkedAdd(prediction, product, prediction))
                        return false;
                }
                if (!checkedAdd(prediction, output[i], output[i])
                    || !sampleInDepth(output[i], depth))
                    return false;
            }
        }
        else if (type >= 32 && type <= 63)
        {
            predictorOrder = static_cast<unsigned>(type - 31);
            if (predictorOrder > blockSize) return false;
            for (unsigned i = 0; i < predictorOrder; ++i)
                if (!bits.readSigned(depth, output[i])) return false;
            uint64_t precisionMinusOne = 0;
            int64_t shift = 0;
            if (!bits.readBits(4, precisionMinusOne) || precisionMinusOne == 15
                || !bits.readSigned(5, shift) || shift < 0)
                return false;
            const unsigned precision = static_cast<unsigned>(precisionMinusOne + 1);
            std::array<int64_t, 32> coefficients {};
            for (unsigned i = 0; i < predictorOrder; ++i)
                if (!bits.readSigned(precision, coefficients[i])) return false;
            if (!decodeResidual(bits, blockSize, predictorOrder, output))
                return false;
            for (size_t i = predictorOrder; i < output.size(); ++i)
            {
                int64_t sum = 0;
                for (unsigned j = 0; j < predictorOrder; ++j)
                {
                    int64_t product = 0;
                    if (!checkedMultiply(coefficients[j], output[i - j - 1], product)
                        || !checkedAdd(sum, product, sum))
                        return false;
                }
                int64_t prediction = 0;
                if (!floorDividePowerOfTwo(sum, static_cast<unsigned>(shift), prediction)
                    || !checkedAdd(prediction, output[i], output[i])
                    || !sampleInDepth(output[i], depth))
                    return false;
            }
        }
        else
            return false;

        if (type <= 1)
            for (int64_t value : output)
                if (!sampleInDepth(value, depth)) return false;

        if (wasted != 0)
        {
            const int64_t factor = int64_t { 1 } << wasted;
            for (int64_t& value : output)
                if (!checkedMultiply(value, factor, value)
                    || !sampleInDepth(value, storedDepth))
                    return false;
        }
        return true;
    }

    static bool restoreChannels(uint8_t channelCode, uint8_t bitsPerSample,
                                uint32_t blockSize,
                                std::vector<int64_t>& pcm) noexcept
    {
        if (channelCode < 8)
        {
            for (int64_t value : pcm)
                if (!sampleInDepth(value, bitsPerSample)) return false;
            return true;
        }

        int64_t* first = pcm.data();
        int64_t* second = pcm.data() + blockSize;
        for (uint32_t i = 0; i < blockSize; ++i)
        {
            int64_t left = 0, right = 0;
            if (channelCode == 8)
            {
                left = first[i];
                if (!checkedSubtract(left, second[i], right)) return false;
            }
            else if (channelCode == 9)
            {
                right = second[i];
                if (!checkedAdd(first[i], right, left)) return false;
            }
            else
            {
                int64_t doubledMid = 0, adjustedMid = 0;
                if (!checkedMultiply(first[i], 2, doubledMid)
                    || !checkedAdd(doubledMid, second[i] % 2 != 0 ? 1 : 0, adjustedMid))
                    return false;
                int64_t leftNumerator = 0, rightNumerator = 0;
                if (!checkedAdd(adjustedMid, second[i], leftNumerator)
                    || !checkedSubtract(adjustedMid, second[i], rightNumerator)
                    || !floorDividePowerOfTwo(leftNumerator, 1, left)
                    || !floorDividePowerOfTwo(rightNumerator, 1, right))
                    return false;
            }
            if (!sampleInDepth(left, bitsPerSample)
                || !sampleInDepth(right, bitsPerSample))
                return false;
            first[i] = left;
            second[i] = right;
        }
        return true;
    }

    static bool decodeFrame(const std::vector<uint8_t>& bytes, size_t offset,
                            const StreamInfo& stream, size_t expectedFrameNumber,
                            uint64_t expectedFirstSample,
                            std::vector<int64_t>& pcm,
                            DecodedFrame& decoded)
    {
        if (offset > bytes.size() || bytes.size() - offset < 8)
            return false;
        ByteCursor cursor(bytes, offset, bytes.size());
        uint8_t first = 0, second = 0, third = 0, fourth = 0;
        if (!cursor.readU8(first) || !cursor.readU8(second)
            || !cursor.readU8(third) || !cursor.readU8(fourth)
            || first != 0xff || (second & 0xfeu) != 0xf8u
            || (fourth & 1u) != 0)
            return false;
        const bool variable = (second & 1u) != 0;
        if (variable == (stream.minBlockSize == stream.maxBlockSize))
            return false;
        const uint8_t blockCode = third >> 4;
        const uint8_t rateCode = third & 0x0fu;
        const uint8_t channelCode = fourth >> 4;
        const uint8_t depthCode = (fourth >> 1) & 7u;
        if (channelCode > 10 || depthCode == 3)
            return false;
        const uint8_t channels = channelCode <= 7
            ? static_cast<uint8_t>(channelCode + 1) : uint8_t { 2 };
        static constexpr uint8_t depths[8] = { 0, 8, 12, 0, 16, 20, 24, 32 };
        const uint8_t bitDepth = depthCode == 0 ? stream.bitsPerSample : depths[depthCode];
        if (channels != stream.channels || bitDepth != stream.bitsPerSample)
            return false;

        uint64_t codedNumber = 0;
        if (!parseCodedNumber(cursor, variable, codedNumber))
            return false;
        if ((variable && codedNumber != expectedFirstSample)
            || (!variable && codedNumber != expectedFrameNumber))
            return false;

        uint32_t blockSize = 0, sampleRate = 0;
        if (!decodeBlockSize(blockCode, cursor, blockSize)
            || !decodeSampleRate(rateCode, cursor, stream, sampleRate)
            || blockSize > stream.maxBlockSize)
            return false;

        const size_t crcPosition = cursor.position();
        uint8_t storedCrc8 = 0;
        if (!cursor.readU8(storedCrc8)
            || crc8(std::span<const uint8_t>(bytes.data() + offset,
                                            crcPosition - offset)) != storedCrc8)
            return false;

        if (static_cast<uint64_t>(blockSize) * channels
            > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            return false;
        pcm.resize(static_cast<size_t>(blockSize) * channels);
        BitCursor bits(bytes, cursor.position(), bytes.size());
        for (uint8_t channel = 0; channel < channels; ++channel)
        {
            unsigned depth = bitDepth;
            if ((channelCode == 8 && channel == 1)
                || (channelCode == 9 && channel == 0)
                || (channelCode == 10 && channel == 1))
                ++depth;
            std::span<int64_t> channelPcm(
                pcm.data() + static_cast<size_t>(channel) * blockSize, blockSize);
            if (!decodeSubframe(bits, blockSize, depth, channelPcm))
                return false;
        }
        if (!bits.alignWithZeroPadding())
            return false;
        const size_t footerPosition = bits.bytePosition();
        if (footerPosition > bytes.size() || bytes.size() - footerPosition < 2)
            return false;
        const uint16_t storedCrc16 = static_cast<uint16_t>(
            (static_cast<uint16_t>(bytes[footerPosition]) << 8)
            | bytes[footerPosition + 1]);
        if (crc16(std::span<const uint8_t>(bytes.data() + offset,
                                          footerPosition - offset)) != storedCrc16)
            return false;
        if (!restoreChannels(channelCode, bitDepth, blockSize, pcm))
            return false;

        decoded.endOffset = footerPosition + 2;
        decoded.blockSize = blockSize;
        decoded.variableBlock = variable;
        return true;
    }

    static void updateMd5(Md5& md5, const std::vector<int64_t>& pcm,
                          uint8_t channels, uint8_t bitsPerSample,
                          uint32_t blockSize) noexcept
    {
        const unsigned bytesPerSample = (bitsPerSample + 7u) / 8u;
        uint8_t encoded[4] = {};
        for (uint32_t sample = 0; sample < blockSize; ++sample)
        {
            for (uint8_t channel = 0; channel < channels; ++channel)
            {
                const int64_t value = pcm[static_cast<size_t>(channel) * blockSize + sample];
                const uint64_t representation = static_cast<uint64_t>(value);
                for (unsigned byte = 0; byte < bytesPerSample; ++byte)
                    encoded[byte] = static_cast<uint8_t>(representation >> (8 * byte));
                md5.update(std::span<const uint8_t>(encoded, bytesPerSample));
            }
        }
    }

    static bool allZero(const std::array<uint8_t, 16>& value) noexcept
    {
        for (uint8_t byte : value) if (byte != 0) return false;
        return true;
    }

    static bool validateOpen(OpenState& state)
    {
        if (state.bytes.size() < 4
            || std::memcmp(state.bytes.data(), "fLaC", 4) != 0)
            return false;
        ByteCursor cursor(state.bytes, 4, state.bytes.size());
        uint64_t metadataBlocks = 0;
        bool sawStreamInfo = false;
        bool lastMetadata = false;

        while (!lastMetadata)
        {
            if (metadataBlocks >= kMaxMetadataBlocks) return false;
            ++metadataBlocks;
            uint8_t header = 0;
            uint32_t length = 0;
            if (!cursor.readU8(header) || !cursor.readBE24(length))
                return false;
            lastMetadata = (header & 0x80u) != 0;
            const uint8_t type = header & 0x7fu;
            if (type == 127 || length > cursor.remaining())
                return false;
            const uint8_t* data = nullptr;
            if (!cursor.readSpan(length, data))
                return false;
            if (!sawStreamInfo)
            {
                if (type != 0 || length != 34
                    || !parseStreamInfo(data, state.stream))
                    return false;
                sawStreamInfo = true;
            }
            else if (type == 0)
                return false;
        }
        if (!sawStreamInfo || cursor.remaining() == 0)
            return false;

        Md5 md5;
        uint64_t coverage = 0;
        bool haveStrategy = false;
        bool strategy = false;
        uint32_t previousBlockSize = 0;
        std::vector<int64_t> pcm;
        while (cursor.remaining() > 0)
        {
            if (state.frames.size() >= kMaxFrames)
                return false;
            if (haveStrategy && previousBlockSize < state.stream.minBlockSize)
                return false;
            if (haveStrategy && !strategy
                && previousBlockSize != state.stream.minBlockSize)
                return false;

            DecodedFrame decoded;
            const size_t frameOffset = cursor.position();
            if (!decodeFrame(state.bytes, frameOffset, state.stream,
                             state.frames.size(), coverage, pcm, decoded)
                || decoded.endOffset <= frameOffset
                || decoded.endOffset > state.bytes.size())
                return false;
            if (haveStrategy && decoded.variableBlock != strategy)
                return false;
            haveStrategy = true;
            strategy = decoded.variableBlock;
            previousBlockSize = decoded.blockSize;
            const size_t frameSize = decoded.endOffset - frameOffset;
            if ((state.stream.minFrameSize != 0
                 && frameSize < state.stream.minFrameSize)
                || (state.stream.maxFrameSize != 0
                    && frameSize > state.stream.maxFrameSize))
                return false;
            if (coverage > kMaxInterchannelSamples - decoded.blockSize)
                return false;
            const uint64_t firstSample = coverage;
            coverage += decoded.blockSize;
            const uint64_t bytesPerSample = (state.stream.bitsPerSample + 7u) / 8u;
            if (coverage > kMaxDecodedPcmBytes
                / (static_cast<uint64_t>(state.stream.channels) * bytesPerSample))
                return false;
            updateMd5(md5, pcm, state.stream.channels,
                      state.stream.bitsPerSample, decoded.blockSize);
            state.frames.push_back({ frameOffset, frameSize, firstSample,
                                     decoded.blockSize });
            if (!cursor.skip(frameSize))
                return false;
        }

        if (state.frames.empty() || coverage == 0
            || (state.stream.declaredSamples != 0
                && state.stream.declaredSamples != coverage))
            return false;
        const std::array<uint8_t, 16> digest = md5.finish();
        if (!allZero(state.stream.md5) && digest != state.stream.md5)
            return false;

        state.info.sampleRate = static_cast<double>(state.stream.sampleRate);
        state.info.numChannels = state.stream.channels;
        state.info.numSamples = static_cast<int64_t>(coverage);
        state.info.bitsPerSample = state.stream.bitsPerSample;
        state.info.isFloatingPoint = false;
        return true;
    }
};

} // namespace dspark
