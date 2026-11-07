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
 * - PCM whose bit depth is not a whole number of bytes (12-bit, 20-bit and
 *   the like): the container width is nBlockAlign / nChannels, as the
 *   format defines it, and the left-justified samples decode exactly.
 * - Reading only: G.711 A-law (format 6) and mu-law (format 7), also inside
 *   WAVE_FORMAT_EXTENSIBLE, IMA/DVI ADPCM (format 0x11) and Microsoft ADPCM
 *   (format 2). Every coding decodes to the 16-bit linear values its
 *   specification defines: ITU-T G.711 for the companded codes, the IMA
 *   Digital Audio Focus and Technical Working Groups reference algorithm
 *   (1992) for IMA ADPCM, and Microsoft's ADPCM with the coefficient table
 *   stored in the fmt chunk (the prediction divides by 256, truncating toward
 *   zero, the common decoder convention). ADPCM is read in mono and stereo.
 *   Blocks are nBlockAlign bytes with wSamplesPerBlock frames; a short last
 *   block decodes the frames it holds, and a fact chunk that ends inside the
 *   last block trims its padding. Writing produces PCM or IEEE float only.
 *
 * Floating-point conversions to integer formats apply correct mathematical
 * rounding to avoid DC offsets and harmonic distortion caused by simple
 * truncation. Endianness is handled safely via byte-wise reconstruction to
 * avoid Undefined Behavior (UB) on non-x86 architectures.
 *
 * RF64 and BW64 files (EBU Tech 3306, ITU-R BS.2088: a ds64 chunk carries
 * the 64-bit sizes) are read like classic RIFF/WAVE files.
 *
 * The writer produces classic RIFF/WAVE by default, whose 32-bit sizes cap
 * a file at 4 GiB: writeSamples() then refuses (returns false) any write that
 * would overflow that limit instead of silently corrupting the header.
 * setRf64Mode() selects RF64 output instead: Rf64Mode::Auto reserves the
 * space for a ds64 chunk (as a JUNK chunk) and promotes the file to RF64 on
 * close() only when it outgrows classic RIFF; Rf64Mode::Always writes RF64
 * regardless of size. An odd-sized data chunk is followed by the RIFF pad
 * byte.
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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

namespace dspark {

/**
 * @class WavFile
 * @brief Complete WAV file reader and writer in pure C++20.
 *
 * Inherits from AudioFile to provide a uniform interface. Handles both
 * standard PCM and IEEE float formats, mono through multi-channel (up to
 * 64 channels), and reads G.711 and IMA/Microsoft ADPCM files.
 *
 * @note An ADPCM block whose header is out of range (an IMA step index above
 * 88, a Microsoft predictor beyond the coefficient table) makes a read that
 * touches it return false. Reads of other blocks are unaffected.
 *
 * @note Files with more than 2 channels or more than 16 bits per sample are
 * written as WAVE_FORMAT_EXTENSIBLE, per the Microsoft specification. Plain
 * PCM/float headers are used otherwise for maximum compatibility.
 */
class WavFile : public AudioFile
{
public:
    /** @brief How the samples of the file open for reading are coded. */
    enum class Encoding
    {
        Pcm,       ///< Linear integer PCM (including EXTENSIBLE and 12/20-bit).
        Float,     ///< IEEE 754 float.
        ALaw,      ///< ITU-T G.711 A-law.
        MuLaw,     ///< ITU-T G.711 mu-law.
        ImaAdpcm,  ///< IMA/DVI ADPCM, 4 bits per sample.
        MsAdpcm    ///< Microsoft ADPCM, 4 bits per sample.
    };

    /** @brief Container used by the next openWrite(). */
    enum class Rf64Mode
    {
        Never,  ///< Classic RIFF/WAVE, at most 4 GiB (the default).
        Auto,   ///< RIFF with a reserved ds64 slot, promoted to RF64 beyond 4 GiB.
        Always  ///< RF64 whatever the size.
    };

    ~WavFile() override { close(); }

    /** @brief Chooses the container of files opened for writing from now on. */
    void setRf64Mode(Rf64Mode mode) noexcept { rf64Mode_ = mode; }

    /** @brief Container chosen for files opened for writing. */
    [[nodiscard]] Rf64Mode getRf64Mode() const noexcept { return rf64Mode_; }

    // -- AudioFile interface ---------------------------------------------------

    /**
     * @brief Opens a WAV file for reading.
     * @param path Platform-independent path to the WAV file.
     * @return True if the file was found and contains a valid RIFF/WAVE header.
     */
    [[nodiscard]] bool openRead(const std::filesystem::path& path) override
    {
        close();
        writeError_ = false;
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

        writeError_ = false;
        file_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!file_.is_open()) return false;

        info_ = info;
        mode_ = Mode::Write;
        totalFramesWritten_ = 0;
        writeContainer_ = rf64Mode_;

        // Write placeholder header (sizes will be patched on close)
        if (!writeHeader())
        {
            writeError_ = true;
            close();
            return false;
        }

        resizeIoBuffer();
        return true;
    }

    /**
     * @brief Format of the open file.
     *
     * bitsPerSample is the fmt chunk's wBitsPerSample for PCM and float: the
     * container width for WAVE_FORMAT_EXTENSIBLE (32 for 24-in-32) and the
     * declared depth for a plain header (12 for 12-bit in a 16-bit
     * container). The companded and ADPCM codings report 16, the linear
     * resolution they decode to, so that writing the decoded audio at that
     * depth is lossless; getEncoding() tells them apart from 16-bit PCM.
     */
    [[nodiscard]] AudioFileInfo getInfo() const override { return info_; }

    /** @brief Sample coding of the file open for reading (Pcm when none is). */
    [[nodiscard]] Encoding getEncoding() const noexcept { return encoding_; }

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
        // Subtraction form: startFrame + numFrames can overflow int64, and
        // the wrapped sum used to pass this check and seek into the header.
        if (startFrame > info_.numSamples || numFrames > info_.numSamples - startFrame)
            return false;

        const int nCh = std::min(dest.getNumChannels(), static_cast<int>(info_.numChannels));
        int64_t framesRemaining = std::min(numFrames, static_cast<int64_t>(dest.getNumSamples()));

        if (encoding_ == Encoding::ImaAdpcm || encoding_ == Encoding::MsAdpcm)
            return readAdpcm(dest, nCh, startFrame, framesRemaining);

        const int frameSize = containerBytes_ * static_cast<int>(info_.numChannels);

        // Seek to the exact start frame within the 'data' chunk
        auto seekPos = dataChunkOffset_ + static_cast<std::streamoff>(startFrame * frameSize);
        file_.seekg(seekPos, std::ios::beg);
        if (!file_.good()) return false;

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
        // data chunk past what the header can describe (header <= 128 bytes,
        // plus the pad byte). RF64 sizes are 64-bit.
        const int64_t dataAfter = (totalFramesWritten_ + nS) * static_cast<int64_t>(frameSize);
        const int64_t limit = (writeContainer_ == Rf64Mode::Never)
                            ? static_cast<int64_t>(0xFFFFFFFFu) - 128
                            : (int64_t { 1 } << 62);
        if (dataAfter > limit) return false;

        int framesRemaining = nS;
        int srcOffset = 0;

        while (framesRemaining > 0)
        {
            const int toWrite = std::min(framesRemaining, kChunkFrames);

            interleave(src, ioBuffer_.data(), nCh, srcCh, toWrite, srcOffset);

            auto rawBytes = static_cast<std::streamsize>(static_cast<int64_t>(toWrite) * frameSize);
            file_.write(reinterpret_cast<const char*>(ioBuffer_.data()), rawBytes);
            if (!file_.good())
            {
                writeError_ = true;
                return false;
            }

            srcOffset += toWrite;
            framesRemaining -= toWrite;
        }

        totalFramesWritten_ += nS;
        return true;
    }

    void close() override
    {
        resetReadFormat();
        if (!file_.is_open()) { mode_ = Mode::Closed; info_ = {}; return; }

        if (mode_ == Mode::Write)
        {
            finaliseHeader();
            if (!file_.good()) writeError_ = true;
        }

        file_.close();
        if (file_.fail()) writeError_ = true;
        ioBuffer_.clear();
        ioBuffer_.shrink_to_fit();
        blockPcm_.clear();
        blockPcm_.shrink_to_fit();
        mode_ = Mode::Closed;
        info_ = {}; // getInfo() contract: defaults once no file is open
    }

    [[nodiscard]] bool isOpen() const noexcept override
    {
        return file_.is_open() && mode_ != Mode::Closed;
    }

    /**
     * @brief True when a write or the header finalisation of the current or
     *        most recent writer failed.
     *
     * Remains readable after close() until the next open call.
     */
    [[nodiscard]] bool hasWriteError() const noexcept override { return writeError_; }

private:
    // -- RIFF/WAV constants ----------------------------------------------------

    static constexpr uint16_t kFormatPCM         = 1;
    static constexpr uint16_t kFormatMsAdpcm     = 2;
    static constexpr uint16_t kFormatIEEEFloat   = 3;
    static constexpr uint16_t kFormatALaw        = 6;
    static constexpr uint16_t kFormatMuLaw       = 7;
    static constexpr uint16_t kFormatImaAdpcm    = 0x11;
    static constexpr uint16_t kFormatExtensible  = 0xFFFE;
    static constexpr int kChunkFrames            = 8192;
    static constexpr uint32_t kMaxChannels       = 64;
    static constexpr uint32_t kMaxMsCoefficients = 256; // indexed by a byte
    static constexpr uint32_t kMaxAdpcmChannels  = 2;

    enum class Mode { Closed, Read, Write };

    std::fstream file_;
    Mode mode_ = Mode::Closed;
    AudioFileInfo info_ {};
    std::vector<uint8_t> ioBuffer_;

    std::streamoff dataChunkOffset_ = 0;
    std::streamoff dataChunkSizeOffset_ = 0;
    uint64_t dataChunkSize_ = 0;
    int64_t totalFramesWritten_ = 0;
    Rf64Mode rf64Mode_ = Rf64Mode::Never;       // for the next openWrite()
    Rf64Mode writeContainer_ = Rf64Mode::Never; // of the file being written
    bool writeError_ = false;

    // Read-side format, set by parseFmtChunk(). PCM, float and G.711 read
    // containerBytes_ per sample; ADPCM reads blockAlign_-byte blocks of
    // samplesPerBlock_ frames into blockPcm_ (interleaved 16-bit).
    Encoding encoding_ = Encoding::Pcm;
    int containerBytes_ = 0;
    uint32_t blockAlign_ = 0;
    uint32_t samplesPerBlock_ = 0;
    uint32_t factFrames_ = 0;
    bool hasFact_ = false;
    std::vector<int16_t> msCoefficients_;       // pairs: coef1, coef2
    std::vector<int16_t> blockPcm_;

    static constexpr uint32_t kRf64Unknown = 0xFFFFFFFFu;
    static constexpr std::streamoff kDs64Offset = 12;   // right after "WAVE"
    static constexpr uint32_t kDs64Size = 28;           // without a chunk table

    [[nodiscard]] bool isAdpcm() const noexcept
    {
        return encoding_ == Encoding::ImaAdpcm || encoding_ == Encoding::MsAdpcm;
    }

    void resetReadFormat() noexcept
    {
        encoding_ = Encoding::Pcm;
        containerBytes_ = 0;
        blockAlign_ = 0;
        samplesPerBlock_ = 0;
        factFrames_ = 0;
        hasFact_ = false;
    }

    void resizeIoBuffer()
    {
        size_t bytesNeeded = 0;
        if (mode_ == Mode::Read && isAdpcm())
        {
            // Both are bounded by the 16-bit nBlockAlign: at most 64 KiB of
            // block bytes and about 2 * nBlockAlign + nChannels samples.
            bytesNeeded = blockAlign_;
            blockPcm_.assign(static_cast<size_t>(samplesPerBlock_) * info_.numChannels, 0);
        }
        else
        {
            const size_t sampleBytes = (mode_ == Mode::Read)
                ? static_cast<size_t>(containerBytes_)
                : static_cast<size_t>(info_.bitsPerSample / 8);
            bytesNeeded = static_cast<size_t>(kChunkFrames) * info_.numChannels * sampleBytes;
        }
        if (ioBuffer_.size() < bytesNeeded)
            ioBuffer_.resize(bytesNeeded);
    }

    // -- Header parsing (read) -------------------------------------------------

    bool parseHeader()
    {
        char riffId[4];
        if (!readBytes(riffId, 4)) return false;
        const bool rf64 = std::memcmp(riffId, "RF64", 4) == 0
                       || std::memcmp(riffId, "BW64", 4) == 0;
        if (!rf64 && std::memcmp(riffId, "RIFF", 4) != 0) return false;

        uint32_t riffSize = 0;
        if (!readLE32(riffSize)) return false;
        (void)riffSize; // Deliberately ignored; many DAWs write invalid lengths

        char waveId[4];
        if (!readBytes(waveId, 4)) return false;
        if (std::memcmp(waveId, "WAVE", 4) != 0) return false;

        // RF64/BW64: the ds64 chunk comes first and holds the 64-bit sizes
        // that the 32-bit fields mark with 0xFFFFFFFF.
        uint64_t ds64DataSize = 0;
        if (rf64)
        {
            char ds64Id[4];
            uint32_t ds64Size = 0;
            uint64_t riffSize64 = 0;
            if (!readBytes(ds64Id, 4) || std::memcmp(ds64Id, "ds64", 4) != 0) return false;
            if (!readLE32(ds64Size) || ds64Size < 24) return false;
            if (!readLE64(riffSize64) || !readLE64(ds64DataSize)) return false;
            (void)riffSize64;
            file_.seekg(static_cast<std::streamoff>(ds64Size) - 16 + (ds64Size & 1), std::ios::cur);
            if (!file_.good()) return false;
        }

        bool hasFmt = false, hasData = false;

        while (file_.good() && !(hasFmt && hasData))
        {
            char chunkId[4];
            uint32_t chunkSize32 = 0;
            if (!readBytes(chunkId, 4)) break;
            if (!readLE32(chunkSize32)) break;
            uint64_t chunkSize = chunkSize32;
            if (rf64 && chunkSize32 == kRf64Unknown && std::memcmp(chunkId, "data", 4) == 0)
                chunkSize = ds64DataSize;

            // RIFF chunks are word-aligned: an odd-sized chunk is followed by
            // one pad byte that is not part of the declared size. All skips
            // below must account for it (in 64-bit arithmetic, so a corrupt
            // size near UINT32_MAX cannot wrap the seek to zero).
            const std::streamoff pad = static_cast<std::streamoff>(chunkSize & 1);

            if (std::memcmp(chunkId, "fmt ", 4) == 0)
            {
                if (chunkSize > 0xFFFFu || !parseFmtChunk(static_cast<uint32_t>(chunkSize))) return false;
                if (pad) file_.seekg(pad, std::ios::cur);
                hasFmt = true;
            }
            else if (std::memcmp(chunkId, "data", 4) == 0)
            {
                auto currentPos = file_.tellg();
                file_.seekg(0, std::ios::end);
                auto fileEnd = file_.tellg();
                file_.seekg(currentPos);

                const auto remaining = static_cast<std::streamoff>(fileEnd - currentPos);
                if (remaining < 0) return false;
                if (chunkSize > static_cast<uint64_t>(remaining))
                    chunkSize = static_cast<uint64_t>(remaining);

                dataChunkOffset_ = file_.tellg();
                dataChunkSize_ = chunkSize;
                hasData = true;

                if (!hasFmt)
                {
                    file_.seekg(static_cast<std::streamoff>(chunkSize) + pad, std::ios::cur);
                }
            }
            else if (std::memcmp(chunkId, "fact", 4) == 0 && chunkSize >= 4)
            {
                // Frame count of a compressed stream; only used to trim the
                // padding of the last ADPCM block (see adpcmFrameCount()).
                if (!readLE32(factFrames_)) break;
                hasFact_ = true;
                file_.seekg(static_cast<std::streamoff>(chunkSize) - 4 + pad, std::ios::cur);
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
            if (isAdpcm())
            {
                info_.numSamples = adpcmFrameCount();
            }
            else
            {
                const uint64_t bytesPerFrame = static_cast<uint64_t>(containerBytes_) * info_.numChannels;
                if (bytesPerFrame > 0)
                    info_.numSamples = static_cast<int64_t>(dataChunkSize_ / bytesPerFrame);
            }
        }

        return hasFmt && hasData;
    }

    // Frames held by an ADPCM block of `bytes` bytes (a full block or the
    // short last one), or 0 when the bytes cannot hold the block header.
    [[nodiscard]] uint32_t adpcmFramesInBlock(uint64_t bytes) const noexcept
    {
        const uint64_t nCh = info_.numChannels;
        uint64_t frames = 0;
        if (encoding_ == Encoding::ImaAdpcm)
        {
            // Header: one 4-byte word per channel holding the first sample;
            // then 4-byte words per channel, 8 samples each, interleaved.
            if (bytes >= 4 * nCh) frames = 1 + ((bytes - 4 * nCh) / (4 * nCh)) * 8;
        }
        else
        {
            // Header: 7 bytes per channel holding the first two samples;
            // then one nibble per sample, channel-interleaved.
            if (bytes >= 7 * nCh) frames = 2 + ((bytes - 7 * nCh) * 2) / nCh;
        }
        return static_cast<uint32_t>(std::min<uint64_t>(frames, samplesPerBlock_));
    }

    [[nodiscard]] int64_t adpcmFrameCount() const noexcept
    {
        const uint64_t blocks = dataChunkSize_ / blockAlign_;
        const uint64_t tail = dataChunkSize_ % blockAlign_;
        const uint64_t lastBlockFrames = adpcmFramesInBlock(tail);
        const uint64_t frames = blocks * samplesPerBlock_ + lastBlockFrames;

        // Encoders pad the last block to nBlockAlign and record the true
        // length in the fact chunk. Trust it only when it ends inside the
        // last block: a stale or foreign count must not cut whole blocks.
        const uint64_t lastBlock = (lastBlockFrames > 0) ? lastBlockFrames : samplesPerBlock_;
        if (hasFact_ && factFrames_ <= frames && factFrames_ + lastBlock > frames)
            return static_cast<int64_t>(factFrames_);
        return static_cast<int64_t>(frames);
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
        if (!readLE16(bitsPerSample)) return false;

        // ADPCM extension (WAVEFORMATEX cbSize + format-specific fields).
        uint32_t fmtSamplesPerBlock = 0;
        bool hasMsCoefficients = false;

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

            // The ADPCM parameters live where EXTENSIBLE keeps its own
            // fields, so an ADPCM subformat has no samples-per-block or
            // coefficient table to decode with.
            if (audioFormat == kFormatImaAdpcm || audioFormat == kFormatMsAdpcm) return false;
        }
        else if ((audioFormat == kFormatImaAdpcm || audioFormat == kFormatMsAdpcm) && chunkSize >= 18)
        {
            uint16_t cbSize = 0;
            if (!readLE16(cbSize)) return false;
            const uint32_t extBytes = std::min<uint32_t>(cbSize, chunkSize - 18);
            uint32_t consumed = 18;

            if (extBytes >= 2)
            {
                uint16_t spb = 0;
                if (!readLE16(spb)) return false;
                fmtSamplesPerBlock = spb;
                consumed += 2;
            }
            if (audioFormat == kFormatMsAdpcm && extBytes >= 4)
            {
                uint16_t numCoef = 0;
                if (!readLE16(numCoef)) return false;
                consumed += 2;
                // The block header indexes the table with one byte; a table
                // that is empty, longer than that, or not all present in the
                // chunk is malformed.
                if (numCoef == 0 || numCoef > kMaxMsCoefficients) return false;
                if (extBytes < 4 + 4u * numCoef) return false;
                msCoefficients_.resize(2u * numCoef);
                for (auto& c : msCoefficients_)
                {
                    uint16_t v = 0;
                    if (!readLE16(v)) return false;
                    c = static_cast<int16_t>(v);
                }
                consumed += 4u * numCoef;
                hasMsCoefficients = true;
            }
            if (chunkSize > consumed)
                file_.seekg(static_cast<std::streamoff>(chunkSize - consumed), std::ios::cur);
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

        if (sampleRate == 0) return false;
        if (numChannels == 0 || numChannels > kMaxChannels) return false;

        switch (audioFormat)
        {
            case kFormatPCM:
            {
                encoding_ = Encoding::Pcm;
                if (bitsPerSample == 0 || bitsPerSample > 32) return false;
                // Integer PCM has no 64-bit variant; without the gate above a
                // PCM-64 header would pass validation and then "read" silence.
                if ((bitsPerSample & 7) == 0)
                {
                    containerBytes_ = bitsPerSample / 8;
                }
                else
                {
                    // A depth that is not a whole number of bytes (12, 20...)
                    // is stored left-justified in a container the block
                    // alignment defines; the low padding bits are zero, so
                    // decoding the whole container is exact.
                    if (blockAlign == 0 || blockAlign % numChannels != 0) return false;
                    containerBytes_ = blockAlign / numChannels;
                    if (containerBytes_ < (bitsPerSample + 7) / 8 || containerBytes_ > 4) return false;
                }
                return true;
            }
            case kFormatIEEEFloat:
                encoding_ = Encoding::Float;
                if (bitsPerSample != 32 && bitsPerSample != 64) return false;
                containerBytes_ = bitsPerSample / 8;
                return true;
            case kFormatALaw:
            case kFormatMuLaw:
                encoding_ = (audioFormat == kFormatALaw) ? Encoding::ALaw : Encoding::MuLaw;
                if (bitsPerSample != 8) return false;
                containerBytes_ = 1;
                info_.bitsPerSample = 16;
                return true;
            case kFormatImaAdpcm:
            case kFormatMsAdpcm:
            {
                encoding_ = (audioFormat == kFormatImaAdpcm) ? Encoding::ImaAdpcm : Encoding::MsAdpcm;
                if (bitsPerSample != 4) return false;
                // Mono and stereo only: no layout for more channels is
                // agreed between encoders and decoders.
                if (numChannels > kMaxAdpcmChannels) return false;
                const uint32_t headerBytes = (encoding_ == Encoding::ImaAdpcm ? 4u : 7u) * numChannels;
                if (blockAlign < headerBytes) return false;
                // IMA data comes in one 4-byte word per channel at a time.
                if (encoding_ == Encoding::ImaAdpcm && blockAlign % headerBytes != 0) return false;
                blockAlign_ = blockAlign;
                samplesPerBlock_ = std::numeric_limits<uint32_t>::max();
                const uint32_t maxFrames = adpcmFramesInBlock(blockAlign);
                // wSamplesPerBlock may be smaller than the block can hold
                // (the decoder then stops there), never larger, and never
                // smaller than the samples the block header itself carries.
                const uint32_t minFrames = (encoding_ == Encoding::ImaAdpcm) ? 1u : 2u;
                if (fmtSamplesPerBlock != 0
                    && (fmtSamplesPerBlock < minFrames || fmtSamplesPerBlock > maxFrames))
                    return false;
                samplesPerBlock_ = (fmtSamplesPerBlock != 0) ? fmtSamplesPerBlock : maxFrames;
                if (encoding_ == Encoding::MsAdpcm && !hasMsCoefficients)
                {
                    // No table in the fmt chunk: the seven standard pairs.
                    static constexpr int16_t kStandard[14] = {
                        256, 0, 512, -256, 0, 0, 192, 64, 240, 0, 460, -208, 392, -232
                    };
                    msCoefficients_.assign(std::begin(kStandard), std::end(kStandard));
                }
                info_.bitsPerSample = 16;
                return true;
            }
            default:
                return false;
        }
    }

    // -- Compressed sample decoding (read only) --------------------------------

    // ITU-T G.711 expansion to the 16-bit grid (A-law: 13-bit magnitudes x 8,
    // mu-law: 14-bit magnitudes x 4).
    [[nodiscard]] static constexpr int16_t aLawToLinear(uint8_t code) noexcept
    {
        const int v = code ^ 0x55;
        const int mantissa = v & 0x0F;
        const int segment = (v >> 4) & 0x07;
        int magnitude = (mantissa << 4) + 8;
        if (segment > 0) magnitude = ((mantissa << 4) + 0x108) << (segment - 1);
        return static_cast<int16_t>((v & 0x80) ? magnitude : -magnitude);
    }

    [[nodiscard]] static constexpr int16_t muLawToLinear(uint8_t code) noexcept
    {
        const int v = ~code & 0xFF;
        const int magnitude = (((v & 0x0F) << 3) + 0x84) << ((v >> 4) & 0x07);
        return static_cast<int16_t>((v & 0x80) ? (0x84 - magnitude) : (magnitude - 0x84));
    }

    template <bool ALaw>
    [[nodiscard]] static constexpr std::array<float, 256> makeG711Table() noexcept
    {
        std::array<float, 256> t {};
        for (int i = 0; i < 256; ++i)
        {
            const int16_t s = ALaw ? aLawToLinear(static_cast<uint8_t>(i))
                                   : muLawToLinear(static_cast<uint8_t>(i));
            t[static_cast<size_t>(i)] = static_cast<float>(s) * (1.0f / 32768.0f);
        }
        return t;
    }

    // IMA ADPCM step size per step index (the reference table).
    static constexpr int16_t kImaStep[89] = {
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
        253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
        1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
        3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
        12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
    };

    // Per (step index, 3-bit magnitude): the difference the reference's
    // "series of shifts" adds (not (2n + 1) * step / 8, which rounds
    // differently) and the next step index. The lookup shortens the serial
    // dependency between consecutive samples.
    struct ImaTables { std::array<int32_t, 89 * 8> diff {}; std::array<uint8_t, 89 * 8> next {}; };

    [[nodiscard]] static constexpr ImaTables makeImaTables() noexcept
    {
        constexpr int indexStep[8] = { -1, -1, -1, -1, 2, 4, 6, 8 };
        ImaTables t;
        for (int i = 0; i < 89; ++i)
            for (int m = 0; m < 8; ++m)
            {
                const int step = kImaStep[i];
                int diff = step >> 3;
                if (m & 1) diff += step >> 2;
                if (m & 2) diff += step >> 1;
                if (m & 4) diff += step;
                t.diff[static_cast<size_t>(i * 8 + m)] = diff;   // up to 61436
                t.next[static_cast<size_t>(i * 8 + m)] =
                    static_cast<uint8_t>(std::clamp(i + indexStep[m], 0, 88));
            }
        return t;
    }

    // Decodes the IMA ADPCM block of `bytes` bytes in ioBuffer_ into
    // blockPcm_; returns the frames decoded, or 0 when a header is invalid.
    [[nodiscard]] uint32_t decodeImaBlock(uint64_t bytes) noexcept
    {
        static constexpr ImaTables kTables = makeImaTables();

        const uint32_t nCh = info_.numChannels;
        const uint32_t frames = adpcmFramesInBlock(bytes);
        if (frames == 0) return 0;
        const uint8_t* b = ioBuffer_.data();
        int16_t* out = blockPcm_.data();

        for (uint32_t ch = 0; ch < nCh; ++ch)
        {
            const uint8_t* h = b + 4 * ch;
            int predictor = static_cast<int16_t>(static_cast<uint16_t>(h[0] | (h[1] << 8)));
            uint32_t index = h[2];             // h[3] is reserved
            if (index > 88) return 0;
            out[ch] = static_cast<int16_t>(predictor);

            // Words of 8 samples, low nibble first, channel words interleaved.
            int16_t* o = out + nCh + ch;
            const uint8_t* w = b + 4 * nCh + 4 * ch;
            for (uint32_t done = 1; done < frames; done += 8, w += 4 * nCh)
            {
                const uint32_t count = std::min<uint32_t>(8, frames - done);
                for (uint32_t k = 0; k < count; ++k, o += nCh)
                {
                    const uint32_t nibble = (w[k >> 1] >> ((k & 1) * 4)) & 0x0F;
                    const uint32_t slot = index * 8 + (nibble & 7);
                    const int diff = kTables.diff[slot];
                    predictor = std::clamp(predictor + ((nibble & 8) ? -diff : diff), -32768, 32767);
                    index = kTables.next[slot];
                    *o = static_cast<int16_t>(predictor);
                }
            }
        }
        return frames;
    }

    // Microsoft ADPCM counterpart of decodeImaBlock().
    [[nodiscard]] uint32_t decodeMsBlock(uint64_t bytes) noexcept
    {
        static constexpr int16_t kAdaptation[16] = {
            230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230
        };
        struct State { int64_t coef1, coef2, delta, s1, s2; };
        std::array<State, kMaxAdpcmChannels> st {};

        const uint32_t nCh = info_.numChannels;
        const uint32_t frames = adpcmFramesInBlock(bytes);
        if (frames == 0) return 0;
        const uint8_t* b = ioBuffer_.data();
        int16_t* out = blockPcm_.data();
        const auto le16 = [](const uint8_t* p) noexcept
        {
            return static_cast<int64_t>(static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8))));
        };

        const size_t numCoef = msCoefficients_.size() / 2;
        for (uint32_t ch = 0; ch < nCh; ++ch)
        {
            const uint8_t predictor = b[ch];
            if (predictor >= numCoef) return 0;
            State& s = st[ch];
            s.coef1 = msCoefficients_[2u * predictor];
            s.coef2 = msCoefficients_[2u * predictor + 1];
            s.delta = le16(b + nCh + 2 * ch);
            s.s1    = le16(b + 3 * nCh + 2 * ch);   // the newer sample
            s.s2    = le16(b + 5 * nCh + 2 * ch);   // the older, output first
            out[ch]       = static_cast<int16_t>(s.s2);
            out[nCh + ch] = static_cast<int16_t>(s.s1);
        }

        // One nibble per sample, high nibble first, channel-interleaved.
        const uint8_t* data = b + 7 * nCh;
        uint32_t i = 0;
        for (uint32_t frame = 2; frame < frames; ++frame)
        {
            for (uint32_t ch = 0; ch < nCh; ++ch, ++i)
            {
                const int nibble = (data[i >> 1] >> ((i & 1) ? 0 : 4)) & 0x0F;
                State& s = st[ch];
                // Division by 256 truncates toward zero, the common decoder
                // convention; an arithmetic shift would floor and drift by a
                // few LSB. The step size is held to 32 bits.
                const int64_t prediction = (s.s1 * s.coef1 + s.s2 * s.coef2) / 256;
                const int64_t sample = std::clamp<int64_t>(
                    prediction + (nibble >= 8 ? nibble - 16 : nibble) * s.delta, -32768, 32767);
                s.s2 = s.s1;
                s.s1 = sample;
                s.delta = std::clamp<int64_t>((kAdaptation[nibble] * s.delta) / 256, 16, 0x7FFFFFFF);
                out[static_cast<size_t>(frame) * nCh + ch] = static_cast<int16_t>(sample);
            }
        }
        return frames;
    }

    bool readAdpcm(AudioBufferView<float>& dest, int nCh, int64_t startFrame, int64_t frames)
    {
        const uint32_t fileCh = info_.numChannels;
        uint64_t block = static_cast<uint64_t>(startFrame) / samplesPerBlock_;
        uint32_t offset = static_cast<uint32_t>(static_cast<uint64_t>(startFrame) % samplesPerBlock_);
        int64_t destOffset = 0;

        while (frames > 0)
        {
            const uint64_t blockStart = block * blockAlign_;
            if (blockStart >= dataChunkSize_) return false;
            const uint64_t bytes = std::min<uint64_t>(blockAlign_, dataChunkSize_ - blockStart);

            file_.seekg(dataChunkOffset_ + static_cast<std::streamoff>(blockStart), std::ios::beg);
            if (!file_.good()) return false;
            file_.read(reinterpret_cast<char*>(ioBuffer_.data()), static_cast<std::streamsize>(bytes));
            if (file_.gcount() != static_cast<std::streamsize>(bytes)) return false;

            const uint32_t decoded = (encoding_ == Encoding::ImaAdpcm) ? decodeImaBlock(bytes)
                                                                       : decodeMsBlock(bytes);
            if (decoded <= offset) return false;   // corrupt block header

            const int64_t take = std::min<int64_t>(frames, decoded - offset);
            for (int ch = 0; ch < nCh; ++ch)
            {
                float* out = dest.getChannel(ch) + destOffset;
                const int16_t* in = blockPcm_.data() + static_cast<size_t>(offset) * fileCh
                                  + static_cast<size_t>(ch);
                for (int64_t f = 0; f < take; ++f)
                    out[f] = static_cast<float>(in[static_cast<size_t>(f) * fileCh]) * (1.0f / 32768.0f);
            }

            destOffset += take;
            frames -= take;
            offset = 0;
            ++block;
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

        writeBytes(writeContainer_ == Rf64Mode::Always ? "RF64" : "RIFF", 4);
        writeLE32(0); // Patched on close
        writeBytes("WAVE", 4);

        if (writeContainer_ != Rf64Mode::Never)
        {
            // ds64 slot (EBU Tech 3306): a JUNK chunk of the same size that
            // close() turns into ds64 when the file becomes RF64.
            writeBytes(writeContainer_ == Rf64Mode::Always ? "ds64" : "JUNK", 4);
            writeLE32(kDs64Size);
            for (uint32_t i = 0; i < kDs64Size; ++i) file_.put('\0');
        }

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
        const uint64_t dataSize = static_cast<uint64_t>(totalFramesWritten_)
                                * info_.numChannels * static_cast<uint64_t>(bytesPerSample);

        // RIFF chunks are word-aligned: an odd data chunk (8-bit or 24-bit
        // with an odd frame count) is followed by one pad byte, counted in
        // the RIFF size but not in the data size.
        file_.seekp(0, std::ios::end);
        if ((dataSize & 1u) != 0) file_.put('\0');
        const auto fileSize = static_cast<uint64_t>(static_cast<std::streamoff>(file_.tellp()));
        const uint64_t riffSize = fileSize - 8;

        const bool asRf64 = writeContainer_ == Rf64Mode::Always
            || (writeContainer_ == Rf64Mode::Auto && riffSize > 0xFFFFFFFFull);
        if (asRf64)
        {
            const int frameBytes = bytesPerSample * static_cast<int>(info_.numChannels);
            file_.seekp(0, std::ios::beg);
            writeBytes("RF64", 4);
            writeLE32(kRf64Unknown);
            file_.seekp(kDs64Offset, std::ios::beg);
            writeBytes("ds64", 4);
            writeLE32(kDs64Size);
            writeLE64(riffSize);
            writeLE64(dataSize);
            writeLE64(frameBytes > 0 ? dataSize / static_cast<uint64_t>(frameBytes) : 0);
            writeLE32(0);                       // no chunk size table
            file_.seekp(dataChunkSizeOffset_, std::ios::beg);
            writeLE32(kRf64Unknown);
        }
        else
        {
            file_.seekp(dataChunkSizeOffset_, std::ios::beg);
            writeLE32(static_cast<uint32_t>(dataSize));
            file_.seekp(4, std::ios::beg);
            writeLE32(static_cast<uint32_t>(riffSize));
        }

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
        if (encoding_ == Encoding::ALaw || encoding_ == Encoding::MuLaw)
        {
            static constexpr std::array<float, 256> kALaw  = makeG711Table<true>();
            static constexpr std::array<float, 256> kMuLaw = makeG711Table<false>();
            const float* table = (encoding_ == Encoding::ALaw) ? kALaw.data() : kMuLaw.data();
            const int stride = static_cast<int>(info_.numChannels);
            for (int ch = 0; ch < nCh; ++ch)
            {
                float* out = dest.getChannel(ch) + destOffset;
                const uint8_t* in = raw + ch;
                for (int f = 0; f < numFrames; ++f)
                    out[f] = table[in[f * stride]];
            }
            return;
        }

        // The container width, not the declared depth, sets the layout
        // (12-bit samples sit in 16-bit containers, 20-bit in 24-bit ones).
        if (encoding_ == Encoding::Float) {
            if (containerBytes_ == 4) deinterleaveImpl<float, 32>(raw, dest, nCh, numFrames, destOffset);
            else deinterleaveImpl<double, 64>(raw, dest, nCh, numFrames, destOffset);
        } else {
            switch (containerBytes_) {
                case 1: deinterleaveImpl<uint8_t, 8>(raw, dest, nCh, numFrames, destOffset); break;
                case 2: deinterleaveImpl<int16_t, 16>(raw, dest, nCh, numFrames, destOffset); break;
                case 3: deinterleaveImpl<int32_t, 24>(raw, dest, nCh, numFrames, destOffset); break;
                case 4: deinterleaveImpl<int32_t, 32>(raw, dest, nCh, numFrames, destOffset); break;
                default: break;
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
                float value = (ch < srcCh) ? src.getChannel(ch)[srcOffset + f] : 0.0f;
                uint8_t* ptr = raw + ((f * fileCh + ch) * bytesPerSample);

                // A NaN has no integer code: std::clamp passes it through and
                // the float-to-int conversion is then undefined (x86 wrote the
                // most negative code, a full-scale click). Integer formats
                // write it as silence; +-Inf clamp to full scale below.
                if constexpr (std::is_integral_v<T>)
                    if (value != value) value = 0.0f;

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

    bool readLE64(uint64_t& val)
    {
        uint32_t lo = 0, hi = 0;
        if (!readLE32(lo) || !readLE32(hi)) return false;
        val = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
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

    void writeLE64(uint64_t val)
    {
        writeLE32(static_cast<uint32_t>(val));
        writeLE32(static_cast<uint32_t>(val >> 32));
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
