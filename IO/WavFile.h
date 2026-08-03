// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WavFile.h
 * @brief High-performance, pure C++ WAV file reader/writer.
 *
 * Provides bit-accurate parsing and encoding of standard and Extensible WAV
 * files with cache-friendly channel-major conversion loops.
 *
 * @details
 * Supports the following formats:
 * - PCM integer: 8-bit unsigned, 16-bit, 24-bit, 32-bit signed
 * - IEEE float: 32-bit and 64-bit floating-point
 * - WAVE_FORMAT_EXTENSIBLE wrappers of both, including reduced-precision
 *   containers (e.g. 24 valid bits stored left-justified in a 32-bit
 *   container): samples are decoded by container width, which reproduces
 *   the reduced precision exactly.
 *
 * Floating-point conversions to integer formats apply correct mathematical
 * rounding to avoid DC offsets and harmonic distortion caused by simple
 * truncation. Endianness is handled safely via byte-wise reconstruction to
 * avoid Undefined Behavior (UB) on non-x86 architectures.
 *
 * The writer targets classic RIFF/WAVE, whose 32-bit sizes cap a file at
 * 4 GiB: writeSamples() refuses (returns false) any write that would
 * overflow that limit instead of silently corrupting the header.
 *
 * @warning File operations allocate memory internally (for chunk buffers)
 * and block the thread pending disk I/O. NEVER call open, read, or write
 * methods from the real-time audio thread.
 *
 * Threading: owner-managed. One instance serves one file from one thread at
 * a time; no internal synchronization (see AudioFile.h).
 *
 * Dependencies: IO/AudioFile.h (Core/AudioBuffer.h, Core/AudioSpec.h).
 */

#include "AudioFile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <vector>

namespace dspark {

/**
 * @class WavFile
 * @brief Complete WAV file reader and writer in pure C++20.
 *
 * Inherits from AudioFile to provide a uniform interface. Handles both
 * standard PCM and IEEE float formats, mono through multi-channel (up to
 * 64 channels).
 *
 * @note Files with more than 2 channels or more than 16 bits per sample are
 * written as WAVE_FORMAT_EXTENSIBLE, per the Microsoft specification. Plain
 * PCM/float headers are used otherwise for maximum compatibility.
 */
class WavFile : public AudioFile
{
public:
    ~WavFile() override { close(); }

    // -- AudioFile interface ---------------------------------------------------

    /**
     * @brief Opens a WAV file for reading.
     * @param path Platform-independent path to the WAV file.
     * @return True if the file was found and contains a valid RIFF/WAVE header.
     */
    [[nodiscard]] bool openRead(const std::filesystem::path& path) override
    {
        close();
        file_.open(path, std::ios::binary | std::ios::in);
        if (!file_.is_open()) return false;

        if (!parseHeader())
        {
            close();
            return false;
        }

        mode_ = Mode::Read;
        resizeIoBuffer();
        return true;
    }

    /**
     * @brief Opens a WAV file for writing, overwriting if it exists.
     *
     * The requested format is validated BEFORE the destination is touched:
     * an unsupported format returns false without creating or truncating
     * anything on disk.
     *
     * @param path Platform-independent output path.
     * @param info Configuration for sample rate, bit depth, and channels.
     * @return True if the file was created and the preliminary header written.
     */
    [[nodiscard]] bool openWrite(const std::filesystem::path& path, const AudioFileInfo& info) override
    {
        close();

        // Validate/normalize the requested format first so we never emit a
        // corrupt header AND never destroy an existing file on a doomed
        // request. WAV supports PCM at 8/16/24/32-bit and IEEE float at
        // 32/64-bit; the fmt chunk stores the rate as uint32.
        if (info.numChannels < 1 || info.numChannels > kMaxChannels) return false;
        if (!(info.sampleRate >= 1.0) || info.sampleRate > 4294967295.0) return false;
        if (info.isFloatingPoint)
        {
            if (info.bitsPerSample != 32 && info.bitsPerSample != 64)
                return false;
        }
        else if (info.bitsPerSample != 8 && info.bitsPerSample != 16 &&
                 info.bitsPerSample != 24 && info.bitsPerSample != 32)
        {
            return false;
        }

        file_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!file_.is_open()) return false;

        info_ = info;
        mode_ = Mode::Write;
        totalFramesWritten_ = 0;

        // Write placeholder header (sizes will be patched on close)
        if (!writeHeader())
        {
            close();
            return false;
        }

        resizeIoBuffer();
        return true;
    }

    [[nodiscard]] AudioFileInfo getInfo() const override { return info_; }

    [[nodiscard]] bool readSamples(AudioBufferView<float> dest) override
    {
        // An open-but-empty file (zero-length data chunk) reads zero frames
        // successfully; only unopened handles or I/O errors report failure.
        if (mode_ == Mode::Read && file_.is_open() && info_.numSamples == 0)
            return true;
        return readSamples(dest, 0, info_.numSamples);
    }

    [[nodiscard]] bool readSamples(AudioBufferView<float> dest,
                                   int64_t startFrame, int64_t numFrames) override
    {
        if (mode_ != Mode::Read || !file_.is_open()) return false;
        if (startFrame < 0 || numFrames <= 0) return false;
        if (startFrame + numFrames > info_.numSamples) return false;

        const int frameSize = static_cast<int>(info_.bitsPerSample / 8 * info_.numChannels);

        // Seek to the exact start frame within the 'data' chunk
        auto seekPos = dataChunkOffset_ + static_cast<std::streamoff>(startFrame * frameSize);
        file_.seekg(seekPos, std::ios::beg);
        if (!file_.good()) return false;

        const int nCh = std::min(dest.getNumChannels(), static_cast<int>(info_.numChannels));
        int64_t framesRemaining = std::min(numFrames, static_cast<int64_t>(dest.getNumSamples()));
        int64_t destOffset = 0;

        while (framesRemaining > 0)
        {
            const auto toRead = std::min<int64_t>(framesRemaining, kChunkFrames);
            const auto rawBytes = static_cast<std::streamsize>(toRead * frameSize);

            file_.read(reinterpret_cast<char*>(ioBuffer_.data()), rawBytes);
            if (file_.gcount() != rawBytes) return false;

            deinterleave(ioBuffer_.data(), dest, nCh, static_cast<int>(toRead), static_cast<int>(destOffset));

            destOffset += toRead;
            framesRemaining -= toRead;
        }

        return true;
    }

    [[nodiscard]] bool writeSamples(AudioBufferView<const float> src) override
    {
        if (mode_ != Mode::Write || !file_.is_open()) return false;

        const int nCh = static_cast<int>(info_.numChannels);
        const int srcCh = std::min(src.getNumChannels(), nCh);
        const int nS  = src.getNumSamples();
        const int frameSize = static_cast<int>(info_.bitsPerSample / 8) * nCh;

        // Classic RIFF sizes are 32-bit: refuse a write that would push the
        // data chunk past what the header can describe (header <= 128 bytes).
        const int64_t dataAfter = (totalFramesWritten_ + nS) * static_cast<int64_t>(frameSize);
        if (dataAfter > static_cast<int64_t>(0xFFFFFFFFu) - 128) return false;

        int framesRemaining = nS;
        int srcOffset = 0;

        while (framesRemaining > 0)
        {
            const int toWrite = std::min(framesRemaining, kChunkFrames);

            interleave(src, ioBuffer_.data(), nCh, srcCh, toWrite, srcOffset);

            auto rawBytes = static_cast<std::streamsize>(static_cast<int64_t>(toWrite) * frameSize);
            file_.write(reinterpret_cast<const char*>(ioBuffer_.data()), rawBytes);
            if (!file_.good()) return false;

            srcOffset += toWrite;
            framesRemaining -= toWrite;
        }

        totalFramesWritten_ += nS;
        return true;
    }

    void close() override
    {
        if (!file_.is_open()) { mode_ = Mode::Closed; info_ = {}; return; }

        if (mode_ == Mode::Write)
            finaliseHeader();

        file_.close();
        ioBuffer_.clear();
        ioBuffer_.shrink_to_fit();
        mode_ = Mode::Closed;
        info_ = {}; // getInfo() contract: defaults once no file is open
    }

    [[nodiscard]] bool isOpen() const noexcept override
    {
        return file_.is_open() && mode_ != Mode::Closed;
    }

private:
    // -- RIFF/WAV constants ----------------------------------------------------

    static constexpr uint16_t kFormatPCM         = 1;
    static constexpr uint16_t kFormatIEEEFloat   = 3;
    static constexpr uint16_t kFormatExtensible  = 0xFFFE;
    static constexpr int kChunkFrames            = 8192;
    static constexpr uint32_t kMaxChannels       = 64;

    enum class Mode { Closed, Read, Write };

    std::fstream file_;
    Mode mode_ = Mode::Closed;
    AudioFileInfo info_ {};
    std::vector<uint8_t> ioBuffer_;

    std::streamoff dataChunkOffset_ = 0;
    std::streamoff dataChunkSizeOffset_ = 0;
    uint32_t dataChunkSize_ = 0;
    int64_t totalFramesWritten_ = 0;

    void resizeIoBuffer()
    {
        const size_t bytesNeeded = static_cast<size_t>(kChunkFrames)
                                 * info_.numChannels * (info_.bitsPerSample / 8);
        if (ioBuffer_.size() < bytesNeeded)
            ioBuffer_.resize(bytesNeeded);
    }

    // -- Header parsing (read) -------------------------------------------------

    bool parseHeader()
    {
        char riffId[4];
        if (!readBytes(riffId, 4)) return false;
        if (std::memcmp(riffId, "RIFF", 4) != 0) return false;

        uint32_t riffSize = 0;
        if (!readLE32(riffSize)) return false;
        (void)riffSize; // Deliberately ignored; many DAWs write invalid lengths

        char waveId[4];
        if (!readBytes(waveId, 4)) return false;
        if (std::memcmp(waveId, "WAVE", 4) != 0) return false;

        bool hasFmt = false, hasData = false;

        while (file_.good() && !(hasFmt && hasData))
        {
            char chunkId[4];
            uint32_t chunkSize = 0;
            if (!readBytes(chunkId, 4)) break;
            if (!readLE32(chunkSize)) break;

            // RIFF chunks are word-aligned: an odd-sized chunk is followed by
            // one pad byte that is not part of the declared size. All skips
            // below must account for it (in 64-bit arithmetic, so a corrupt
            // size near UINT32_MAX cannot wrap the seek to zero).
            const std::streamoff pad = (chunkSize & 1);

            if (std::memcmp(chunkId, "fmt ", 4) == 0)
            {
                if (!parseFmtChunk(chunkSize)) return false;
                if (pad) file_.seekg(pad, std::ios::cur);
                hasFmt = true;
            }
            else if (std::memcmp(chunkId, "data", 4) == 0)
            {
                auto currentPos = file_.tellg();
                file_.seekg(0, std::ios::end);
                auto fileEnd = file_.tellg();
                file_.seekg(currentPos);

                auto remaining = fileEnd - currentPos;
                if (static_cast<std::streamoff>(chunkSize) > remaining)
                    chunkSize = static_cast<uint32_t>(remaining);

                dataChunkOffset_ = file_.tellg();
                dataChunkSize_ = chunkSize;
                hasData = true;

                if (!hasFmt)
                {
                    file_.seekg(static_cast<std::streamoff>(chunkSize) + pad, std::ios::cur);
                }
            }
            else
            {
                file_.seekg(static_cast<std::streamoff>(chunkSize) + pad, std::ios::cur);
            }
        }

        // numSamples is derived here rather than inside the 'data' branch: if
        // 'data' precedes 'fmt ' (legal but unusual chunk order), bit depth
        // and channel count were still unknown at that point.
        if (hasFmt && hasData)
        {
            const int bytesPerFrame = static_cast<int>(info_.bitsPerSample / 8 * info_.numChannels);
            if (bytesPerFrame > 0)
                info_.numSamples = static_cast<int64_t>(dataChunkSize_) / bytesPerFrame;
        }

        return hasFmt && hasData;
    }

    bool parseFmtChunk(uint32_t chunkSize)
    {
        if (chunkSize < 16) return false;

        uint16_t audioFormat = 0;
        uint16_t numChannels = 0;
        uint32_t sampleRate = 0;
        uint32_t byteRate = 0;
        uint16_t blockAlign = 0;
        uint16_t bitsPerSample = 0;

        if (!readLE16(audioFormat)) return false;
        if (!readLE16(numChannels)) return false;
        if (!readLE32(sampleRate)) return false;
        if (!readLE32(byteRate)) return false;
        (void)byteRate;
        if (!readLE16(blockAlign)) return false;
        (void)blockAlign;
        if (!readLE16(bitsPerSample)) return false;

        if (audioFormat == kFormatExtensible && chunkSize >= 40)
        {
            uint16_t cbSize = 0;
            readLE16(cbSize);
            (void)cbSize; // Standard layout assumed; trailing bytes skipped below

            // wValidBitsPerSample tells how many bits are significant, but the
            // container width (wBitsPerSample) governs the stored layout: valid
            // bits are left-justified, so decoding by container width yields
            // the exact same values. Do NOT adopt validBits as the frame size
            // (that mis-strides real 24-in-32 files). Read it only for sanity.
            uint16_t validBitsPerSample = 0;
            readLE16(validBitsPerSample);
            if (validBitsPerSample > bitsPerSample) return false;

            uint32_t channelMask = 0;
            readLE32(channelMask);
            (void)channelMask;

            uint16_t subFormat = 0;
            if (!readLE16(subFormat)) return false;
            audioFormat = subFormat;

            file_.seekg(14, std::ios::cur); // rest of the 16-byte subformat GUID

            // Skip any extra bytes after the 40 standard EXTENSIBLE bytes so
            // the chunk walker stays aligned on unusual writers.
            if (chunkSize > 40)
                file_.seekg(static_cast<std::streamoff>(chunkSize - 40), std::ios::cur);
        }
        else if (chunkSize > 16)
        {
            auto skip = static_cast<std::streamoff>(chunkSize - 16);
            file_.seekg(skip, std::ios::cur);
        }

        info_.sampleRate      = static_cast<double>(sampleRate);
        info_.numChannels     = numChannels;
        info_.bitsPerSample   = bitsPerSample;
        info_.isFloatingPoint = (audioFormat == kFormatIEEEFloat);

        if (audioFormat != kFormatPCM && audioFormat != kFormatIEEEFloat) return false;
        if (sampleRate == 0) return false;
        if (numChannels == 0 || numChannels > kMaxChannels) return false;
        if (info_.isFloatingPoint)
        {
            if (bitsPerSample != 32 && bitsPerSample != 64) return false;
        }
        else if (bitsPerSample != 8 && bitsPerSample != 16 &&
                 bitsPerSample != 24 && bitsPerSample != 32)
        {
            // Integer PCM has no 64-bit variant; without this gate a PCM-64
            // header would pass validation and then "read" silence (no
            // decoder branch exists for it).
            return false;
        }

        return true;
    }

    // -- Header writing --------------------------------------------------------

    bool writeHeader()
    {
        const uint16_t formatTag = getWriteFormatTag();
        const bool extensible = (formatTag == kFormatExtensible);
        const uint32_t fmtChunkSize = extensible ? 40u : 16u;

        const int bytesPerSample = static_cast<int>(info_.bitsPerSample / 8);
        const auto blockAlign = static_cast<uint16_t>(info_.numChannels * bytesPerSample);
        const auto rate = static_cast<uint32_t>(std::llround(info_.sampleRate));
        const uint64_t byteRate64 = static_cast<uint64_t>(rate) * blockAlign;
        const auto byteRate = static_cast<uint32_t>(
            std::min<uint64_t>(byteRate64, 0xFFFFFFFFu));

        writeBytes("RIFF", 4);
        writeLE32(0); // Patched on close
        writeBytes("WAVE", 4);

        writeBytes("fmt ", 4);
        writeLE32(fmtChunkSize);
        writeLE16(formatTag);
        writeLE16(static_cast<uint16_t>(info_.numChannels));
        writeLE32(rate);
        writeLE32(byteRate);
        writeLE16(blockAlign);
        writeLE16(static_cast<uint16_t>(info_.bitsPerSample));

        if (extensible)
        {
            writeLE16(22);
            writeLE16(static_cast<uint16_t>(info_.bitsPerSample));
            writeLE32(0);

            uint16_t subFormat = info_.isFloatingPoint ? kFormatIEEEFloat : kFormatPCM;
            writeLE16(subFormat);
            static constexpr uint8_t kGuidSuffix[14] = {
                0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71
            };
            writeBytes(reinterpret_cast<const char*>(kGuidSuffix), 14);
        }

        writeBytes("data", 4);
        dataChunkSizeOffset_ = file_.tellp();
        writeLE32(0); // Patched on close

        dataChunkOffset_ = file_.tellp();
        return file_.good();
    }

    void finaliseHeader()
    {
        if (!file_.is_open()) return;

        // A transient failure (e.g. one failed writeSamples) leaves sticky
        // failbits that would turn the header patch into a silent no-op.
        // Clear them so whatever WAS written gets a consistent header.
        file_.clear();

        const int bytesPerSample = static_cast<int>(info_.bitsPerSample / 8);
        const auto dataSize = static_cast<uint32_t>(
            totalFramesWritten_ * info_.numChannels * bytesPerSample);

        file_.seekp(dataChunkSizeOffset_, std::ios::beg);
        writeLE32(dataSize);

        file_.seekp(0, std::ios::end);
        auto fileSize = static_cast<uint32_t>(file_.tellp());
        uint32_t riffSize = fileSize - 8;
        file_.seekp(4, std::ios::beg);
        writeLE32(riffSize);

        file_.flush();
    }

    [[nodiscard]] uint16_t getWriteFormatTag() const noexcept
    {
        if (info_.numChannels > 2 || info_.bitsPerSample > 16) return kFormatExtensible;
        if (info_.isFloatingPoint) return kFormatIEEEFloat;
        return kFormatPCM;
    }

    // -- Sample conversion (read: raw to float) --------------------------------

    void deinterleave(const uint8_t* raw, AudioBufferView<float>& dest, int nCh, int numFrames, int destOffset) const
    {
        if (info_.isFloatingPoint) {
            if (info_.bitsPerSample == 32) deinterleaveImpl<float, 32>(raw, dest, nCh, numFrames, destOffset);
            else deinterleaveImpl<double, 64>(raw, dest, nCh, numFrames, destOffset);
        } else {
            switch (info_.bitsPerSample) {
                case 8:  deinterleaveImpl<uint8_t, 8>(raw, dest, nCh, numFrames, destOffset); break;
                case 16: deinterleaveImpl<int16_t, 16>(raw, dest, nCh, numFrames, destOffset); break;
                case 24: deinterleaveImpl<int32_t, 24>(raw, dest, nCh, numFrames, destOffset); break;
                case 32: deinterleaveImpl<int32_t, 32>(raw, dest, nCh, numFrames, destOffset); break;
            }
        }
    }

    template <typename T, int Bits>
    void deinterleaveImpl(const uint8_t* raw, AudioBufferView<float>& dest, int nCh, int numFrames, int destOffset) const
    {
        const int bytesPerSample = Bits / 8;
        const int stride = static_cast<int>(info_.numChannels) * bytesPerSample;

        // Channel-major loop: each output channel is written contiguously
        // (cache-friendly); the byte-wise strided source reads are the cost
        // of endian-safe decoding.
        for (int ch = 0; ch < nCh; ++ch)
        {
            float* out = dest.getChannel(ch) + destOffset;
            const uint8_t* channelRaw = raw + (ch * bytesPerSample);

            for (int f = 0; f < numFrames; ++f)
            {
                const uint8_t* ptr = channelRaw + (f * stride);

                if constexpr (Bits == 8) {
                    out[f] = (static_cast<float>(ptr[0]) - 128.0f) * (1.0f / 128.0f);
                }
                else if constexpr (Bits == 16) {
                    auto val = static_cast<int16_t>(ptr[0] | (ptr[1] << 8));
                    out[f] = static_cast<float>(val) * (1.0f / 32768.0f);
                }
                else if constexpr (Bits == 24) {
                    int32_t val = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16);
                    if (val & 0x800000) val |= 0xFF000000;
                    out[f] = static_cast<float>(val) * (1.0f / 8388608.0f);
                }
                else if constexpr (Bits == 32 && std::is_same_v<T, int32_t>) {
                    // Assemble in unsigned to avoid signed overflow UB on the high
                    // byte (ptr[3] << 24 with the top bit set), then reinterpret.
                    int32_t val = static_cast<int32_t>(
                          static_cast<uint32_t>(ptr[0])
                        | (static_cast<uint32_t>(ptr[1]) << 8)
                        | (static_cast<uint32_t>(ptr[2]) << 16)
                        | (static_cast<uint32_t>(ptr[3]) << 24));
                    out[f] = static_cast<float>(val) * (1.0f / 2147483648.0f);
                }
                else if constexpr (Bits == 32 && std::is_same_v<T, float>) {
                    float v; std::memcpy(&v, ptr, 4); out[f] = v;
                }
                else if constexpr (Bits == 64) {
                    double v; std::memcpy(&v, ptr, 8); out[f] = static_cast<float>(v);
                }
            }
        }
    }

    // -- Sample conversion (write: float to raw) --------------------------------

    void interleave(AudioBufferView<const float> src, uint8_t* raw,
                    int fileCh, int srcCh, int numFrames, int srcOffset) const
    {
        if (info_.isFloatingPoint) {
            if (info_.bitsPerSample == 32) interleaveImpl<float, 32>(src, raw, fileCh, srcCh, numFrames, srcOffset);
            else interleaveImpl<double, 64>(src, raw, fileCh, srcCh, numFrames, srcOffset);
        } else {
            switch (info_.bitsPerSample) {
                case 8:  interleaveImpl<uint8_t, 8>(src, raw, fileCh, srcCh, numFrames, srcOffset); break;
                case 16: interleaveImpl<int16_t, 16>(src, raw, fileCh, srcCh, numFrames, srcOffset); break;
                case 24: interleaveImpl<int32_t, 24>(src, raw, fileCh, srcCh, numFrames, srcOffset); break;
                case 32: interleaveImpl<int32_t, 32>(src, raw, fileCh, srcCh, numFrames, srcOffset); break;
            }
        }
    }

    template <typename T, int Bits>
    void interleaveImpl(AudioBufferView<const float> src, uint8_t* raw,
                        int fileCh, int srcCh, int numFrames, int srcOffset) const
    {
        const int bytesPerSample = Bits / 8;

        // Frame-major loop is required here to emit contiguous interleaved
        // data. Channels the source view does not carry are written as
        // silence (same contract as Mp3File) instead of reading out of range.
        for (int f = 0; f < numFrames; ++f)
        {
            for (int ch = 0; ch < fileCh; ++ch)
            {
                const float value = (ch < srcCh) ? src.getChannel(ch)[srcOffset + f] : 0.0f;
                uint8_t* ptr = raw + ((f * fileCh + ch) * bytesPerSample);

                // Precise rounding & strict clipping bounds
                if constexpr (Bits == 8) {
                    // 8-bit WAV is unsigned with 128 as zero, and the reader
                    // scales by 128. Writing with 127 put every sample on a
                    // different grid, so an 8-bit round trip came back 0.78%
                    // quiet - a whole quantisation step of systematic error on
                    // top of the quantisation itself. Only +1.0 now clamps,
                    // which is inherent to the 128-step scale.
                    auto val = static_cast<uint8_t>(std::clamp(std::round(value * 128.0f) + 128.0f, 0.0f, 255.0f));
                    ptr[0] = val;
                }
                else if constexpr (Bits == 16) {
                    auto val = static_cast<int16_t>(std::clamp(std::round(value * 32768.0f), -32768.0f, 32767.0f));
                    ptr[0] = val & 0xFF; ptr[1] = (val >> 8) & 0xFF;
                }
                else if constexpr (Bits == 24) {
                    auto val = static_cast<int32_t>(std::clamp(std::round(value * 8388608.0f), -8388608.0f, 8388607.0f));
                    ptr[0] = val & 0xFF; ptr[1] = (val >> 8) & 0xFF; ptr[2] = (val >> 16) & 0xFF;
                }
                else if constexpr (Bits == 32 && std::is_same_v<T, int32_t>) {
                    auto val = static_cast<int32_t>(std::clamp(std::round(static_cast<double>(value) * 2147483648.0), -2147483648.0, 2147483647.0));
                    ptr[0] = val & 0xFF; ptr[1] = (val >> 8) & 0xFF; ptr[2] = (val >> 16) & 0xFF; ptr[3] = (val >> 24) & 0xFF;
                }
                else if constexpr (Bits == 32 && std::is_same_v<T, float>) {
                    std::memcpy(ptr, &value, 4);
                }
                else if constexpr (Bits == 64) {
                    double d = static_cast<double>(value);
                    std::memcpy(ptr, &d, 8);
                }
            }
        }
    }

    // -- Low-level I/O helpers (little-endian) ---------------------------------

    bool readBytes(char* buf, int n)
    {
        file_.read(buf, n);
        return file_.gcount() == n;
    }

    bool readLE16(uint16_t& val)
    {
        uint8_t b[2];
        file_.read(reinterpret_cast<char*>(b), 2);
        if (file_.gcount() != 2) return false;
        val = static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
        return true;
    }

    bool readLE32(uint32_t& val)
    {
        uint8_t b[4];
        file_.read(reinterpret_cast<char*>(b), 4);
        if (file_.gcount() != 4) return false;
        val = static_cast<uint32_t>(b[0])
            | (static_cast<uint32_t>(b[1]) << 8)
            | (static_cast<uint32_t>(b[2]) << 16)
            | (static_cast<uint32_t>(b[3]) << 24);
        return true;
    }

    void writeBytes(const char* buf, int n)
    {
        file_.write(buf, n);
    }

    void writeLE16(uint16_t val)
    {
        uint8_t b[2] = {
            static_cast<uint8_t>(val & 0xFF),
            static_cast<uint8_t>((val >> 8) & 0xFF)
        };
        file_.write(reinterpret_cast<const char*>(b), 2);
    }

    void writeLE32(uint32_t val)
    {
        uint8_t b[4] = {
            static_cast<uint8_t>(val & 0xFF),
            static_cast<uint8_t>((val >> 8) & 0xFF),
            static_cast<uint8_t>((val >> 16) & 0xFF),
            static_cast<uint8_t>((val >> 24) & 0xFF)
        };
        file_.write(reinterpret_cast<const char*>(b), 4);
    }
};

} // namespace dspark
