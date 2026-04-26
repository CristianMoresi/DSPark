// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Mp3File.h
 * @brief Pure C++20 MPEG-1 Layer III (MP3) codec — decoder and encoder.
 *
 * Full MP3 codec supporting both reading and writing:
 *
 * **Decoder**: Decodes MPEG-1 Layer III to normalised float [-1, 1].
 * - Sample rates: 32000, 44100, 48000 Hz
 * - Mono and stereo (including joint stereo with M/S and intensity)
 * - CBR and VBR, ID3v2 tag skipping
 *
 * **Encoder**: Encodes float audio to MPEG-1 Layer III CBR.
 * - Bitrates: 32–320 kbps (CBR)
 * - Mono and stereo at 32000, 44100, 48000 Hz
 * - Analysis polyphase filterbank + MDCT + Huffman coding
 * - Produces standard-compliant MP3 playable by any decoder
 *
 * The full MP3 decoding pipeline is implemented:
 *   1. Frame sync & header parsing
 *   2. Side information parsing
 *   3. Huffman decoding of spectral data (tables 0-31)
 *   4. Requantization with scalefactors
 *   5. Stereo processing (mid/side, intensity)
 *   6. Reordering (short blocks)
 *   7. Alias reduction (butterfly)
 *   8. IMDCT (36-point long, 12-point short) with windowing
 *   9. Frequency inversion
 *  10. Synthesis polyphase filterbank (32-subband, 512-tap)
 *
 * Dependencies: AudioFile.h (base class), standard C++ library only.
 *
 * @code
 *   // Decode:
 *   dspark::Mp3File mp3;
 *   if (mp3.openRead("song.mp3"))
 *   {
 *       auto info = mp3.getInfo();
 *       dspark::AudioBuffer<float> buffer;
 *       buffer.resize(info.numChannels, static_cast<int>(info.numSamples));
 *       mp3.readSamples(buffer.toView());
 *       mp3.close();
 *   }
 *
 *   // Encode:
 *   dspark::AudioFileInfo encInfo;
 *   encInfo.sampleRate = 44100; encInfo.numChannels = 2; encInfo.bitsPerSample = 128;
 *   dspark::Mp3File enc;
 *   if (enc.openWrite("output.mp3", encInfo))
 *   {
 *       enc.writeSamples(buffer.toView());
 *       enc.close();
 *   }
 * @endcode
 */

#include "AudioFile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>   // [DEBUG] for crash log, remove later
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

namespace dspark {

// ============================================================================
// Mp3File
// ============================================================================

class Mp3File : public AudioFile
{
public:
    ~Mp3File() override { close(); }

    // -- AudioFile interface ---------------------------------------------------

    bool openRead(const char* path) override
    {
        // [DEBUG] crash-log instrumentation, remove later
        auto LOG = [](const char* fmt, auto... args) {
            if (std::FILE* f = std::fopen("dsparklab_crash.log", "a")) {
                std::fprintf(f, fmt, args...);
                std::fputc('\n', f);
                std::fclose(f);
            }
        };
        LOG("[Mp3File] openRead enter");

        close();
        LOG("[Mp3File] after close()");

        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in.is_open()) { LOG("[Mp3File] ifstream open FAILED"); return false; }
        LOG("[Mp3File] ifstream open OK");

        auto fileSize = static_cast<size_t>(in.tellg());
        LOG("[Mp3File] fileSize=%zu", fileSize);
        if (fileSize < 10) { LOG("[Mp3File] fileSize too small"); return false; }
        constexpr size_t kMaxMp3FileSize = 256 * 1024 * 1024;  // 256 MB
        if (fileSize > kMaxMp3FileSize) { LOG("[Mp3File] fileSize too big"); return false; }
        in.seekg(0, std::ios::beg);

        fileData_.resize(fileSize);
        LOG("[Mp3File] fileData_ resized");
        in.read(reinterpret_cast<char*>(fileData_.data()),
                static_cast<std::streamsize>(fileSize));
        LOG("[Mp3File] read() done, gcount=%lld", (long long)in.gcount());
        if (static_cast<size_t>(in.gcount()) != fileSize)
        {
            LOG("[Mp3File] gcount != fileSize -> bail");
            fileData_.clear();
            return false;
        }
        in.close();

        // Skip ID3v2 tag if present
        filePos_ = 0;
        skipID3v2();
        LOG("[Mp3File] skipID3v2 done, filePos_=%zu", filePos_);

        // First pass: count frames to determine total samples
        if (!scanFrames())
        {
            LOG("[Mp3File] scanFrames FAILED");
            fileData_.clear();
            return false;
        }
        LOG("[Mp3File] scanFrames OK: frames=%zu sr=%.0f ch=%d numSamples=%lld",
            frameOffsets_.size(), info_.sampleRate, info_.numChannels,
            (long long)info_.numSamples);

        // Pre-decode all frames into the output buffer
        if (!decodeAll())
        {
            LOG("[Mp3File] decodeAll FAILED");
            fileData_.clear();
            decodedSamples_.clear();
            return false;
        }
        LOG("[Mp3File] decodeAll OK");

        fileData_.clear(); // Free raw data after decode
        isOpen_ = true;
        LOG("[Mp3File] openRead done");
        return true;
    }

    bool openWrite(const char* path, const AudioFileInfo& info) override
    {
        close();
        outFile_.open(path, std::ios::binary | std::ios::trunc);
        if (!outFile_.is_open()) return false;

        info_ = info;
        if (info_.numChannels < 1) info_.numChannels = 1;
        if (info_.numChannels > 2) info_.numChannels = 2;

        // bitsPerSample is used to specify target bitrate (kbps) for MP3
        encBitrate_ = info_.bitsPerSample;
        if (encBitrate_ < 32) encBitrate_ = 128;
        // Snap to valid MPEG-1 Layer III bitrate
        static constexpr int validBr[] = {32,40,48,56,64,80,96,112,128,160,192,224,256,320};
        int best = 128, bestDist = 999;
        for (int br : validBr) {
            int d = std::abs(br - encBitrate_);
            if (d < bestDist) { bestDist = d; best = br; }
        }
        encBitrate_ = best;

        // Snap sample rate
        double sr = info_.sampleRate;
        if (sr <= 36050) info_.sampleRate = 32000;
        else if (sr <= 46050) info_.sampleRate = 44100;
        else info_.sampleRate = 48000;

        // Init encoder state
        for (int ch = 0; ch < kChannelsMax; ++ch) encState_[ch] = {};
        encFrameBuf_.clear();
        encInputPos_ = 0;
        isWriting_ = true;
        isOpen_ = true;
        return true;
    }

    [[nodiscard]] AudioFileInfo getInfo() const override { return info_; }

    bool readSamples(AudioBufferView<float> dest) override
    {
        return readSamples(dest, 0, info_.numSamples);
    }

    bool readSamples(AudioBufferView<float> dest,
                     int64_t startFrame, int64_t numFrames) override
    {
        if (!isOpen_) return false;
        if (startFrame < 0 || numFrames <= 0) return false;
        if (startFrame + numFrames > info_.numSamples) return false;

        const int nCh = std::min(dest.getNumChannels(), info_.numChannels);
        const int64_t toCopy = std::min(numFrames,
                                        static_cast<int64_t>(dest.getNumSamples()));

        for (int ch = 0; ch < nCh; ++ch)
        {
            float* dst = dest.getChannel(ch);
            const float* src = decodedSamples_[ch].data() + startFrame;
            std::memcpy(dst, src, static_cast<size_t>(toCopy) * sizeof(float));
        }

        return true;
    }

    bool writeSamples(AudioBufferView<const float> src) override
    {
        if (!isWriting_ || !outFile_.is_open()) return false;

        const int nCh = std::min(src.getNumChannels(), info_.numChannels);
        const int nS = src.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            for (int ch = 0; ch < info_.numChannels; ++ch)
            {
                float val = (ch < nCh) ? src.getChannel(ch)[i] : 0.0f;
                encInput_[ch][encInputPos_] = static_cast<double>(val);
            }
            ++encInputPos_;

            if (encInputPos_ >= kSamplesPerFrame)
            {
                encEncodeFrame();
                encInputPos_ = 0;
            }
        }
        return true;
    }

    void close() override
    {
        // Flush remaining encoder samples (zero-pad to complete frame)
        if (isWriting_ && outFile_.is_open())
        {
            if (encInputPos_ > 0)
            {
                for (int ch = 0; ch < info_.numChannels; ++ch)
                    for (int i = encInputPos_; i < kSamplesPerFrame; ++i)
                        encInput_[ch][i] = 0.0;
                encEncodeFrame();
            }
            outFile_.close();
        }
        isWriting_ = false;

        fileData_.clear();
        decodedSamples_.clear();
        frameOffsets_.clear();
        info_ = {};
        isOpen_ = false;
        filePos_ = 0;
    }

    [[nodiscard]] bool isOpen() const noexcept override { return isOpen_; }

private:
    // ========================================================================
    // Constants
    // ========================================================================

    static constexpr int kGranules     = 2;
    static constexpr int kChannelsMax  = 2;
    static constexpr int kSamplesPerGranule = 576;
    static constexpr int kSamplesPerFrame   = 1152; // MPEG-1 Layer III
    static constexpr int kSubbands     = 32;
    static constexpr int kSynthSlots   = 16;

    static constexpr int kMaxReservoir = 8192; // Bit reservoir maximum bytes

    // ========================================================================
    // Frame header
    // ========================================================================

    struct FrameHeader
    {
        int version     = 0; // 3 = MPEG1
        int layer       = 0; // 1 = Layer III
        bool crcProtect = false;
        int bitrateIdx  = 0;
        int srateIdx    = 0;
        bool padding    = false;
        int channelMode = 0; // 0=stereo, 1=joint, 2=dual, 3=mono
        int modeExt     = 0;
        bool copyright  = false;
        bool original   = false;
        int emphasis     = 0;

        int bitrate     = 0;
        int sampleRate  = 0;
        int channels    = 0;
        int frameSize   = 0;
        int sideInfoSize = 0;
    };

    // ========================================================================
    // Side information
    // ========================================================================

    struct GranuleChannel
    {
        int part2_3_length  = 0;
        int big_values      = 0;
        int global_gain     = 0;
        int scalefac_compress = 0;
        bool window_switching = false;
        int block_type      = 0;
        int mixed_block     = 0;
        int table_select[3] = {};
        int subblock_gain[3] = {};
        int region0_count   = 0;
        int region1_count   = 0;
        int preflag         = 0;
        int scalefac_scale  = 0;
        int count1table_select = 0;
    };

    struct SideInfo
    {
        int main_data_begin = 0;
        int scfsi[2] = {};           // scalefactor selection info per channel (4 bits each)
        GranuleChannel gr[2][2] = {};
    };

    // ========================================================================
    // Bitstream reader
    // ========================================================================

    class BitReader
    {
    public:
        BitReader() = default;

        void init(const uint8_t* data, size_t sizeBytes)
        {
            data_ = data;
            size_ = sizeBytes * 8;
            pos_  = 0;
        }

        uint32_t readBits(int n)
        {
            if (n == 0) return 0;
            uint32_t val = 0;
            for (int i = 0; i < n; ++i)
            {
                val <<= 1;
                if (pos_ < size_)
                {
                    size_t byteIdx = pos_ >> 3;
                    int    bitIdx  = 7 - static_cast<int>(pos_ & 7);
                    val |= (data_[byteIdx] >> bitIdx) & 1u;
                }
                ++pos_;
            }
            return val;
        }

        int readBit()
        {
            if (pos_ >= size_) { ++pos_; return 0; }
            size_t byteIdx = pos_ >> 3;
            int    bitIdx  = 7 - static_cast<int>(pos_ & 7);
            int val = (data_[byteIdx] >> bitIdx) & 1;
            ++pos_;
            return val;
        }

        [[nodiscard]] size_t getPos() const { return pos_; }
        void setPos(size_t p) { pos_ = p; }
        [[nodiscard]] size_t remaining() const { return (pos_ < size_) ? (size_ - pos_) : 0; }

    private:
        const uint8_t* data_ = nullptr;
        size_t size_ = 0; // In bits
        size_t pos_  = 0; // In bits
    };

    // ========================================================================
    // Bitrate and sample rate tables (MPEG-1)
    // ========================================================================

    static constexpr int kBitrateTable[16] = {
        0, 32, 40, 48, 56, 64, 80, 96,
        112, 128, 160, 192, 224, 256, 320, 0
    };

    static constexpr int kSampleRateTable[4] = {
        44100, 48000, 32000, 0
    };

    // ========================================================================
    // Scalefactor band tables (MPEG-1 Layer III)
    // ========================================================================

    struct BandTable
    {
        int longBands[23]  = {};
        int shortBands[14] = {};
        int longCount       = 0;
        int shortCount      = 0;
    };

    static BandTable getBandTable(int sampleRate)
    {
        BandTable t {};
        if (sampleRate == 44100)
        {
            static constexpr int lb[] = {0,4,8,12,16,20,24,30,36,44,52,62,74,90,110,134,162,196,238,288,342,418,576};
            static constexpr int sb[] = {0,4,8,12,16,22,30,40,52,66,84,106,136,192};
            std::memcpy(t.longBands, lb, sizeof(lb));
            std::memcpy(t.shortBands, sb, sizeof(sb));
            t.longCount = 22;
            t.shortCount = 13;
        }
        else if (sampleRate == 48000)
        {
            static constexpr int lb[] = {0,4,8,12,16,20,24,30,36,42,50,60,72,88,106,128,156,190,230,276,330,384,576};
            static constexpr int sb[] = {0,4,8,12,16,22,28,38,50,64,80,100,126,192};
            std::memcpy(t.longBands, lb, sizeof(lb));
            std::memcpy(t.shortBands, sb, sizeof(sb));
            t.longCount = 22;
            t.shortCount = 13;
        }
        else // 32000
        {
            static constexpr int lb[] = {0,4,8,12,16,20,24,30,36,44,54,66,82,102,126,156,194,240,296,364,448,550,576};
            static constexpr int sb[] = {0,4,8,12,16,22,30,42,58,78,104,138,180,192};
            std::memcpy(t.longBands, lb, sizeof(lb));
            std::memcpy(t.shortBands, sb, sizeof(sb));
            t.longCount = 22;
            t.shortCount = 13;
        }
        return t;
    }

    // ========================================================================
    // Scalefactor band length tables for scalefac_compress
    // ========================================================================

    // slen1 and slen2 indexed by scalefac_compress (0-15)
    static constexpr int kSlen1[16] = {0,0,0,0,3,1,1,1,2,2,2,3,3,3,4,4};
    static constexpr int kSlen2[16] = {0,1,2,3,0,1,2,3,1,2,3,1,2,3,2,3};

    // Pretab for requantization (used when preflag is set)
    static constexpr int kPretab[22] = {
        0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,3,2,0
    };

    // ========================================================================
    // Huffman tables (ISO 11172-3, tables 0-31)
    // ========================================================================

    // Each Huffman table entry: packed (x, y, bits) for the most compact representation.
    // We decode using a tree-walk approach with a flat array per table.
    // For tables 0-15 (pair tables with linbits=0): values are (x, y) each 0-15.
    // For tables 16-31 (pair tables with linbits>0): base values decoded, then linbits added.
    // Tables 32-33 are count1 tables (quadruples).

    // linbits for tables 16-31
    static constexpr int kHuffLinbits[32] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,2,3,4,6,8,10,13,4,5,6,7,8,9,11,13
    };

    // Huffman tree node: for each bit read, go left (0) or right (1).
    // Leaf nodes store the decoded (x, y) value.
    struct HuffNode
    {
        int16_t children[2]; // Index of child for bit 0/1. -1 = leaf.
        uint8_t x, y;        // Decoded values (only valid at leaves).
    };

    // We build Huffman trees from the standard table definitions.
    // The trees are stored in a flat array, root at index 0.
    // To keep this header-only and not excessively large, we define the tables
    // using a compact representation: for each codeword, {length, bits, x, y}.

    struct HuffCode
    {
        uint8_t len;
        uint16_t code;
        uint8_t x, y;
    };

    // We store all table data inline. Each table is a span of HuffCode entries.
    // The actual ISO 11172-3 Huffman tables follow.

    struct HuffTable
    {
        const HuffCode* codes;
        int count;
        int linbits;
        int xmax; // max x/y value in base table
    };

    // -- Table 0: empty (all zero)
    // No codes needed, big_values in region with table 0 means all zeros.

    // -- Table 1
    static constexpr HuffCode kHuff01[] = {
        {1, 0b1, 0, 0},
        {3, 0b001, 0, 1},
        {3, 0b010, 1, 0},
        {3, 0b011, 1, 1},
    };

    // -- Table 2
    static constexpr HuffCode kHuff02[] = {
        {1, 0b1, 0, 0},
        {3, 0b010, 0, 1},
        {3, 0b011, 0, 2},
        {3, 0b001, 1, 0},
        {4, 0b0001, 1, 1},
        {4, 0b0000, 1, 2},
        {3, 0b100, 2, 0},
        {4, 0b0010, 2, 1},
        {4, 0b0011, 2, 2},
    };

    // -- Table 3
    static constexpr HuffCode kHuff03[] = {
        {2, 0b11, 0, 0},
        {2, 0b01, 0, 1},
        {3, 0b001, 1, 0},
        {3, 0b000, 1, 1},
        {2, 0b10, 0, 2},
        {3, 0b010, 2, 0},
        {3, 0b011, 2, 1},
        {4, 0b0001, 2, 2},
    };

    // -- Table 5
    static constexpr HuffCode kHuff05[] = {
        {1, 0b1, 0, 0},
        {3, 0b010, 0, 1},
        {4, 0b0011, 0, 2},
        {5, 0b00011, 0, 3},
        {3, 0b011, 1, 0},
        {4, 0b0010, 1, 1},
        {5, 0b00010, 1, 2},
        {6, 0b000010, 1, 3},
        {4, 0b0001, 2, 0},
        {5, 0b00001, 2, 1},
        {6, 0b000011, 2, 2},
        {7, 0b0000010, 2, 3},
        {5, 0b00000, 3, 0},
        {6, 0b000001, 3, 1},
        {7, 0b0000011, 3, 2},
        {7, 0b0000001, 3, 3},
    };

    // -- Table 6
    static constexpr HuffCode kHuff06[] = {
        {3, 0b111, 0, 0},
        {3, 0b011, 0, 1},
        {5, 0b00101, 0, 2},
        {7, 0b0000001, 0, 3},
        {3, 0b110, 1, 0},
        {3, 0b010, 1, 1},
        {4, 0b0100, 1, 2},
        {5, 0b00100, 1, 3},
        {4, 0b0101, 2, 0},
        {4, 0b0001, 2, 1},
        {5, 0b00110, 2, 2},
        {6, 0b000010, 2, 3},
        {5, 0b00111, 3, 0},
        {5, 0b00011, 3, 1},
        {6, 0b000011, 3, 2},
        {6, 0b000001, 3, 3},
    };

    // -- Table 7
    static constexpr HuffCode kHuff07[] = {
        {1, 0b1, 0, 0},
        {3, 0b010, 0, 1},
        {5, 0b00010, 0, 2},
        {7, 0b0000011, 0, 3},
        {8, 0b00000011, 0, 4},
        {9, 0b000000010, 0, 5},
        {3, 0b011, 1, 0},
        {4, 0b0011, 1, 1},
        {5, 0b00011, 1, 2},
        {7, 0b0000010, 1, 3},
        {8, 0b00000010, 1, 4},
        {8, 0b00000101, 1, 5},
        {5, 0b00001, 2, 0},
        {5, 0b00000, 2, 1},
        {6, 0b000010, 2, 2},
        {7, 0b0000001, 2, 3},
        {8, 0b00000001, 2, 4},
        {8, 0b00000100, 2, 5},
        {6, 0b000011, 3, 0},
        {6, 0b000001, 3, 1},
        {7, 0b0000100, 3, 2},
        {8, 0b00000110, 3, 3},
        {9, 0b000000011, 3, 4},
        {9, 0b000000001, 3, 5},
        {7, 0b0000101, 4, 0},
        {7, 0b0000110, 4, 1},
        {8, 0b00000111, 4, 2},
        {9, 0b000000100, 4, 3},
        {9, 0b000000101, 4, 4},
        {9, 0b000000110, 4, 5},
        {8, 0b00001000, 5, 0},
        {8, 0b00001001, 5, 1},
        {8, 0b00001010, 5, 2},
        {9, 0b000000111, 5, 3},
        {10, 0b0000000010, 5, 4},
        {10, 0b0000000011, 5, 5},
    };

    // -- Table 8
    static constexpr HuffCode kHuff08[] = {
        {2, 0b11, 0, 0},
        {3, 0b010, 0, 1},
        {5, 0b00010, 0, 2},
        {7, 0b0000010, 0, 3},
        {8, 0b00000010, 0, 4},
        {9, 0b000000010, 0, 5},
        {3, 0b100, 1, 0},
        {3, 0b011, 1, 1},
        {4, 0b0011, 1, 2},
        {6, 0b000010, 1, 3},
        {7, 0b0000011, 1, 4},
        {8, 0b00000011, 1, 5},
        {5, 0b00011, 2, 0},
        {5, 0b00001, 2, 1},
        {5, 0b00000, 2, 2},
        {6, 0b000011, 2, 3},
        {7, 0b0000001, 2, 4},
        {8, 0b00000100, 2, 5},
        {6, 0b000001, 3, 0},
        {6, 0b000100, 3, 1},
        {7, 0b0000100, 3, 2},
        {7, 0b0000101, 3, 3},
        {8, 0b00000101, 3, 4},
        {8, 0b00000110, 3, 5},
        {7, 0b0000110, 4, 0},
        {7, 0b0000111, 4, 1},
        {8, 0b00000111, 4, 2},
        {8, 0b00001000, 4, 3},
        {9, 0b000000011, 4, 4},
        {9, 0b000000001, 4, 5},
        {8, 0b00001001, 5, 0},
        {8, 0b00001010, 5, 1},
        {8, 0b00001011, 5, 2},
        {9, 0b000000100, 5, 3},
        {9, 0b000000101, 5, 4},
        {10, 0b0000000010, 5, 5},
    };

    // -- Table 9
    static constexpr HuffCode kHuff09[] = {
        {3, 0b111, 0, 0},
        {3, 0b101, 0, 1},
        {5, 0b01001, 0, 2},
        {6, 0b001110, 0, 3},
        {8, 0b00010000, 0, 4},
        {9, 0b000001010, 0, 5},
        {3, 0b110, 1, 0},
        {3, 0b100, 1, 1},
        {4, 0b0101, 1, 2},
        {5, 0b01000, 1, 3},
        {7, 0b0001100, 1, 4},
        {8, 0b00010001, 1, 5},
        {5, 0b01010, 2, 0},
        {4, 0b0100, 2, 1},
        {5, 0b00111, 2, 2},
        {6, 0b001111, 2, 3},
        {7, 0b0001101, 2, 4},
        {8, 0b00001010, 2, 5},
        {6, 0b001100, 3, 0},
        {5, 0b01011, 3, 1},
        {6, 0b001101, 3, 2},
        {7, 0b0001001, 3, 3},
        {7, 0b0001110, 3, 4},
        {8, 0b00001011, 3, 5},
        {7, 0b0001111, 4, 0},
        {6, 0b001000, 4, 1},
        {7, 0b0001010, 4, 2},
        {7, 0b0001000, 4, 3},
        {8, 0b00010010, 4, 4},
        {9, 0b000001011, 4, 5},
        {8, 0b00001000, 5, 0},
        {7, 0b0001011, 5, 1},
        {8, 0b00001001, 5, 2},
        {8, 0b00010011, 5, 3},
        {9, 0b000001000, 5, 4},
        {9, 0b000001001, 5, 5},
    };

    // -- Table 10
    static constexpr HuffCode kHuff10[] = {
        {1, 0b1, 0, 0},
        {3, 0b010, 0, 1},
        {5, 0b00010, 0, 2},
        {7, 0b0000010, 0, 3},
        {8, 0b00000100, 0, 4},
        {9, 0b000000100, 0, 5},
        {10, 0b0000000010, 0, 6},
        {10, 0b0000000100, 0, 7},
        {3, 0b011, 1, 0},
        {4, 0b0011, 1, 1},
        {5, 0b00011, 1, 2},
        {6, 0b000010, 1, 3},
        {8, 0b00000101, 1, 4},
        {8, 0b00000110, 1, 5},
        {9, 0b000000101, 1, 6},
        {10, 0b0000000101, 1, 7},
        {5, 0b00001, 2, 0},
        {5, 0b00000, 2, 1},
        {6, 0b000011, 2, 2},
        {7, 0b0000011, 2, 3},
        {8, 0b00000011, 2, 4},
        {8, 0b00000111, 2, 5},
        {9, 0b000000011, 2, 6},
        {9, 0b000000110, 2, 7},
        {6, 0b000001, 3, 0},
        {6, 0b000100, 3, 1},
        {7, 0b0000100, 3, 2},
        {7, 0b0000101, 3, 3},
        {8, 0b00001000, 3, 4},
        {8, 0b00001001, 3, 5},
        {9, 0b000000111, 3, 6},
        {10, 0b0000000110, 3, 7},
        {7, 0b0000110, 4, 0},
        {7, 0b0000111, 4, 1},
        {8, 0b00001010, 4, 2},
        {8, 0b00001011, 4, 3},
        {9, 0b000001000, 4, 4},
        {9, 0b000001001, 4, 5},
        {10, 0b0000000111, 4, 6},
        {10, 0b0000001000, 4, 7},
        {8, 0b00001100, 5, 0},
        {8, 0b00001101, 5, 1},
        {8, 0b00001110, 5, 2},
        {9, 0b000001010, 5, 3},
        {9, 0b000001011, 5, 4},
        {10, 0b0000001001, 5, 5},
        {10, 0b0000001010, 5, 6},
        {11, 0b00000000010, 5, 7},
        {9, 0b000001100, 6, 0},
        {9, 0b000001101, 6, 1},
        {9, 0b000001110, 6, 2},
        {10, 0b0000001011, 6, 3},
        {10, 0b0000001100, 6, 4},
        {10, 0b0000001101, 6, 5},
        {11, 0b00000000011, 6, 6},
        {11, 0b00000000100, 6, 7},
        {10, 0b0000000011, 7, 0},
        {9, 0b000001111, 7, 1},
        {10, 0b0000001110, 7, 2},
        {10, 0b0000001111, 7, 3},
        {11, 0b00000000101, 7, 4},
        {11, 0b00000000110, 7, 5},
        {11, 0b00000000111, 7, 6},
        {11, 0b00000000001, 7, 7},
    };

    // -- Table 11
    static constexpr HuffCode kHuff11[] = {
        {2, 0b11, 0, 0},
        {3, 0b010, 0, 1},
        {5, 0b00010, 0, 2},
        {7, 0b0000100, 0, 3},
        {8, 0b00000100, 0, 4},
        {9, 0b000000100, 0, 5},
        {10, 0b0000000100, 0, 6},
        {10, 0b0000000110, 0, 7},
        {3, 0b100, 1, 0},
        {3, 0b011, 1, 1},
        {4, 0b0011, 1, 2},
        {6, 0b000011, 1, 3},
        {7, 0b0000101, 1, 4},
        {8, 0b00000101, 1, 5},
        {9, 0b000000101, 1, 6},
        {9, 0b000000111, 1, 7},
        {5, 0b00011, 2, 0},
        {4, 0b0010, 2, 1},
        {5, 0b00001, 2, 2},
        {6, 0b000010, 2, 3},
        {7, 0b0000011, 2, 4},
        {8, 0b00000011, 2, 5},
        {8, 0b00000110, 2, 6},
        {9, 0b000000110, 2, 7},
        {6, 0b000001, 3, 0},
        {5, 0b00000, 3, 1},
        {6, 0b000100, 3, 2},
        {7, 0b0000110, 3, 3},
        {7, 0b0000111, 3, 4},
        {8, 0b00000111, 3, 5},
        {8, 0b00001000, 3, 6},
        {9, 0b000001000, 3, 7},
        {7, 0b0001000, 4, 0},
        {6, 0b000101, 4, 1},
        {7, 0b0001001, 4, 2},
        {8, 0b00001001, 4, 3},
        {8, 0b00001010, 4, 4},
        {8, 0b00001011, 4, 5},
        {9, 0b000001001, 4, 6},
        {9, 0b000001010, 4, 7},
        {8, 0b00001100, 5, 0},
        {7, 0b0001010, 5, 1},
        {8, 0b00001101, 5, 2},
        {8, 0b00001110, 5, 3},
        {8, 0b00001111, 5, 4},
        {9, 0b000001011, 5, 5},
        {9, 0b000001100, 5, 6},
        {10, 0b0000000101, 5, 7},
        {9, 0b000001101, 6, 0},
        {8, 0b00010000, 6, 1},
        {8, 0b00010001, 6, 2},
        {9, 0b000001110, 6, 3},
        {9, 0b000001111, 6, 4},
        {9, 0b000010000, 6, 5},
        {10, 0b0000000111, 6, 6},
        {10, 0b0000001000, 6, 7},
        {9, 0b000010001, 7, 0},
        {9, 0b000010010, 7, 1},
        {9, 0b000010011, 7, 2},
        {9, 0b000010100, 7, 3},
        {10, 0b0000001001, 7, 4},
        {10, 0b0000001010, 7, 5},
        {10, 0b0000001011, 7, 6},
        {10, 0b0000001100, 7, 7},
    };

    // -- Table 12
    static constexpr HuffCode kHuff12[] = {
        {4, 0b1001, 0, 0},
        {3, 0b110, 0, 1},
        {5, 0b01010, 0, 2},
        {7, 0b0011000, 0, 3},
        {8, 0b00100000, 0, 4},
        {9, 0b000011100, 0, 5},
        {10, 0b0000100010, 0, 6},
        {10, 0b0000100100, 0, 7},
        {3, 0b111, 1, 0},
        {3, 0b101, 1, 1},
        {4, 0b0100, 1, 2},
        {5, 0b01011, 1, 3},
        {7, 0b0011001, 1, 4},
        {8, 0b00100001, 1, 5},
        {8, 0b00100100, 1, 6},
        {9, 0b000011101, 1, 7},
        {4, 0b1000, 2, 0},
        {4, 0b0101, 2, 1},
        {5, 0b00110, 2, 2},
        {6, 0b010010, 2, 3},
        {7, 0b0011010, 2, 4},
        {7, 0b0011100, 2, 5},
        {8, 0b00100010, 2, 6},
        {8, 0b00100101, 2, 7},
        {6, 0b010011, 3, 0},
        {5, 0b01000, 3, 1},
        {5, 0b00111, 3, 2},
        {6, 0b010001, 3, 3},
        {7, 0b0011011, 3, 4},
        {7, 0b0011101, 3, 5},
        {8, 0b00100011, 3, 6},
        {8, 0b00100110, 3, 7},
        {7, 0b0011110, 4, 0},
        {6, 0b010100, 4, 1},
        {6, 0b010101, 4, 2},
        {7, 0b0011111, 4, 3},
        {7, 0b0100000, 4, 4},
        {8, 0b00100111, 4, 5},
        {8, 0b00101000, 4, 6},
        {9, 0b000011110, 4, 7},
        {8, 0b00101001, 5, 0},
        {7, 0b0100001, 5, 1},
        {7, 0b0100010, 5, 2},
        {7, 0b0100011, 5, 3},
        {8, 0b00101010, 5, 4},
        {8, 0b00101011, 5, 5},
        {8, 0b00101100, 5, 6},
        {9, 0b000011111, 5, 7},
        {9, 0b000100000, 6, 0},
        {8, 0b00101101, 6, 1},
        {8, 0b00101110, 6, 2},
        {8, 0b00101111, 6, 3},
        {8, 0b00110000, 6, 4},
        {9, 0b000100001, 6, 5},
        {9, 0b000100010, 6, 6},
        {10, 0b0000100011, 6, 7},
        {9, 0b000100011, 7, 0},
        {8, 0b00110001, 7, 1},
        {9, 0b000100100, 7, 2},
        {9, 0b000100101, 7, 3},
        {9, 0b000100110, 7, 4},
        {9, 0b000100111, 7, 5},
        {10, 0b0000100101, 7, 6},
        {10, 0b0000100110, 7, 7},
    };

    // -- Table 13
    static constexpr HuffCode kHuff13[] = {
        {1, 0b1, 0, 0},
        {4, 0b0101, 0, 1},
        {6, 0b001110, 0, 2},
        {7, 0b0010100, 0, 3},
        {8, 0b00100010, 0, 4},
        {9, 0b000110000, 0, 5},
        {9, 0b000110110, 0, 6},
        {10, 0b0001010010, 0, 7},
        {9, 0b000111110, 0, 8},
        {10, 0b0001100010, 0, 9},
        {11, 0b00010100100, 0, 10},
        {11, 0b00010110010, 0, 11},
        {12, 0b000011010010, 0, 12},
        {12, 0b000011100010, 0, 13},
        {13, 0b0000101000010, 0, 14},
        {13, 0b0000101100010, 0, 15},
        {3, 0b011, 1, 0},
        {4, 0b0100, 1, 1},
        {5, 0b00110, 1, 2},
        {7, 0b0010101, 1, 3},
        {7, 0b0010010, 1, 4},
        {8, 0b00100011, 1, 5},
        {9, 0b000110001, 1, 6},
        {9, 0b000110111, 1, 7},
        {9, 0b000111111, 1, 8},
        {9, 0b000111100, 1, 9},
        {10, 0b0001010011, 1, 10},
        {11, 0b00010100101, 1, 11},
        {11, 0b00010110011, 1, 12},
        {11, 0b00011000100, 1, 13},
        {12, 0b000011010011, 1, 14},
        {12, 0b000011100011, 1, 15},
        {5, 0b00111, 2, 0},
        {5, 0b00100, 2, 1},
        {6, 0b001111, 2, 2},
        {7, 0b0010110, 2, 3},
        {7, 0b0010011, 2, 4},
        {8, 0b00100100, 2, 5},
        {8, 0b00101010, 2, 6},
        {9, 0b000111000, 2, 7},
        {9, 0b001000000, 2, 8},
        {10, 0b0001010100, 2, 9},
        {10, 0b0001100011, 2, 10},
        {11, 0b00010100110, 2, 11},
        {11, 0b00010110100, 2, 12},
        {11, 0b00011000101, 2, 13},
        {12, 0b000011010100, 2, 14},
        {12, 0b000011100100, 2, 15},
        {6, 0b010000, 3, 0},
        {6, 0b001100, 3, 1},
        {6, 0b001101, 3, 2},
        {7, 0b0010111, 3, 3},
        {7, 0b0011000, 3, 4},
        {8, 0b00100101, 3, 5},
        {8, 0b00101011, 3, 6},
        {9, 0b000111001, 3, 7},
        {9, 0b001000001, 3, 8},
        {10, 0b0001010101, 3, 9},
        {10, 0b0001100100, 3, 10},
        {10, 0b0001110010, 3, 11},
        {11, 0b00010110101, 3, 12},
        {11, 0b00011000110, 3, 13},
        {12, 0b000011010101, 3, 14},
        {12, 0b000011100101, 3, 15},
        {7, 0b0011001, 4, 0},
        {7, 0b0010000, 4, 1},
        {7, 0b0010001, 4, 2},
        {7, 0b0011010, 4, 3},
        {8, 0b00100110, 4, 4},
        {8, 0b00101100, 4, 5},
        {8, 0b00110010, 4, 6},
        {9, 0b000111010, 4, 7},
        {9, 0b001000010, 4, 8},
        {10, 0b0001010110, 4, 9},
        {10, 0b0001100101, 4, 10},
        {10, 0b0001110011, 4, 11},
        {11, 0b00010110110, 4, 12},
        {11, 0b00011000111, 4, 13},
        {11, 0b00011100100, 4, 14},
        {12, 0b000011100110, 4, 15},
        {8, 0b00100111, 5, 0},
        {7, 0b0011011, 5, 1},
        {7, 0b0011100, 5, 2},
        {8, 0b00100000, 5, 3},
        {8, 0b00101000, 5, 4},
        {8, 0b00101101, 5, 5},
        {8, 0b00110011, 5, 6},
        {9, 0b000111011, 5, 7},
        {9, 0b001000011, 5, 8},
        {10, 0b0001010111, 5, 9},
        {10, 0b0001100110, 5, 10},
        {10, 0b0001110100, 5, 11},
        {11, 0b00010110111, 5, 12},
        {11, 0b00011001000, 5, 13},
        {11, 0b00011100101, 5, 14},
        {12, 0b000011100111, 5, 15},
        {8, 0b00101001, 6, 0},
        {8, 0b00100001, 6, 1},
        {8, 0b00100010, 6, 2},
        {8, 0b00101001, 6, 3},
        {8, 0b00110000, 6, 4},
        {9, 0b000110010, 6, 5},
        {9, 0b000111100, 6, 6},
        {9, 0b001000100, 6, 7},
        {9, 0b001001000, 6, 8},
        {10, 0b0001011000, 6, 9},
        {10, 0b0001100111, 6, 10},
        {10, 0b0001110101, 6, 11},
        {11, 0b00010111000, 6, 12},
        {11, 0b00011001001, 6, 13},
        {11, 0b00011100110, 6, 14},
        {12, 0b000011101000, 6, 15},
        {9, 0b000110011, 7, 0},
        {8, 0b00101010, 7, 1},
        {8, 0b00101011, 7, 2},
        {9, 0b000110100, 7, 3},
        {9, 0b000111101, 7, 4},
        {9, 0b001000101, 7, 5},
        {9, 0b001001001, 7, 6},
        {10, 0b0001011001, 7, 7},
        {10, 0b0001101000, 7, 8},
        {10, 0b0001110110, 7, 9},
        {11, 0b00010111001, 7, 10},
        {11, 0b00011001010, 7, 11},
        {11, 0b00011100111, 7, 12},
        {12, 0b000011101001, 7, 13},
        {12, 0b000011110000, 7, 14},
        {12, 0b000100000000, 7, 15},
        {9, 0b000110101, 8, 0},
        {8, 0b00110001, 8, 1},
        {9, 0b000110110, 8, 2},
        {9, 0b000111010, 8, 3},
        {9, 0b001000000, 8, 4},
        {9, 0b001000110, 8, 5},
        {9, 0b001001010, 8, 6},
        {10, 0b0001011010, 8, 7},
        {10, 0b0001101001, 8, 8},
        {10, 0b0001110111, 8, 9},
        {11, 0b00010111010, 8, 10},
        {11, 0b00011001011, 8, 11},
        {11, 0b00011101000, 8, 12},
        {12, 0b000011101010, 8, 13},
        {12, 0b000011110001, 8, 14},
        {12, 0b000100000001, 8, 15},
        {9, 0b000111110, 9, 0},
        {9, 0b000111000, 9, 1},
        {9, 0b000111001, 9, 2},
        {9, 0b001000001, 9, 3},
        {9, 0b001000111, 9, 4},
        {10, 0b0001011011, 9, 5},
        {10, 0b0001011100, 9, 6},
        {10, 0b0001101010, 9, 7},
        {10, 0b0001101011, 9, 8},
        {10, 0b0001111000, 9, 9},
        {11, 0b00010111011, 9, 10},
        {11, 0b00011001100, 9, 11},
        {11, 0b00011101001, 9, 12},
        {12, 0b000011101011, 9, 13},
        {12, 0b000011110010, 9, 14},
        {12, 0b000100000010, 9, 15},
        {10, 0b0001100000, 10, 0},
        {9, 0b000111011, 10, 1},
        {10, 0b0001100001, 10, 2},
        {10, 0b0001011101, 10, 3},
        {10, 0b0001011110, 10, 4},
        {10, 0b0001011111, 10, 5},
        {10, 0b0001101100, 10, 6},
        {10, 0b0001101101, 10, 7},
        {11, 0b00010111100, 10, 8},
        {11, 0b00010111101, 10, 9},
        {11, 0b00011001101, 10, 10},
        {11, 0b00011101010, 10, 11},
        {12, 0b000011101100, 10, 12},
        {12, 0b000011110011, 10, 13},
        {12, 0b000100000011, 10, 14},
        {13, 0b0000101000011, 10, 15},
        {10, 0b0001100010, 11, 0},
        {10, 0b0001000100, 11, 1},
        {10, 0b0001100011, 11, 2},
        {10, 0b0001100100, 11, 3},
        {10, 0b0001100101, 11, 4},
        {10, 0b0001101110, 11, 5},
        {10, 0b0001101111, 11, 6},
        {11, 0b00010111110, 11, 7},
        {11, 0b00010111111, 11, 8},
        {11, 0b00011001110, 11, 9},
        {11, 0b00011101011, 11, 10},
        {12, 0b000011101101, 11, 11},
        {12, 0b000011110100, 11, 12},
        {12, 0b000100000100, 11, 13},
        {13, 0b0000101000100, 11, 14},
        {13, 0b0000101100011, 11, 15},
        {11, 0b00011000000, 12, 0},
        {10, 0b0001110000, 12, 1},
        {10, 0b0001110001, 12, 2},
        {11, 0b00011000001, 12, 3},
        {11, 0b00011000010, 12, 4},
        {11, 0b00011000011, 12, 5},
        {11, 0b00011010000, 12, 6},
        {11, 0b00011010001, 12, 7},
        {11, 0b00011101100, 12, 8},
        {12, 0b000011101110, 12, 9},
        {12, 0b000011110101, 12, 10},
        {12, 0b000100000101, 12, 11},
        {12, 0b000100001000, 12, 12},
        {13, 0b0000101000101, 12, 13},
        {13, 0b0000101100100, 12, 14},
        {13, 0b0000110000010, 12, 15},
        {11, 0b00011010010, 13, 0},
        {11, 0b00011000100, 13, 1},
        {11, 0b00011000101, 13, 2},
        {11, 0b00011010011, 13, 3},
        {11, 0b00011010100, 13, 4},
        {11, 0b00011010101, 13, 5},
        {11, 0b00011010110, 13, 6},
        {12, 0b000011101111, 13, 7},
        {12, 0b000011110110, 13, 8},
        {12, 0b000100000110, 13, 9},
        {12, 0b000100001001, 13, 10},
        {12, 0b000100001100, 13, 11},
        {13, 0b0000101000110, 13, 12},
        {13, 0b0000101100101, 13, 13},
        {13, 0b0000110000011, 13, 14},
        {14, 0b00001010001000, 13, 15},
        {12, 0b000011010000, 14, 0},
        {11, 0b00011010111, 14, 1},
        {11, 0b00011011000, 14, 2},
        {11, 0b00011101101, 14, 3},
        {12, 0b000011110111, 14, 4},
        {12, 0b000100000111, 14, 5},
        {12, 0b000100001010, 14, 6},
        {12, 0b000100001101, 14, 7},
        {12, 0b000100010000, 14, 8},
        {12, 0b000100010010, 14, 9},
        {13, 0b0000101000111, 14, 10},
        {13, 0b0000101100110, 14, 11},
        {13, 0b0000110000100, 14, 12},
        {13, 0b0000110010000, 14, 13},
        {14, 0b00001010001001, 14, 14},
        {14, 0b00001011000000, 14, 15},
        {12, 0b000011010001, 15, 0},
        {11, 0b00011011001, 15, 1},
        {12, 0b000011110000, 15, 2},
        {12, 0b000100001000, 15, 3},
        {12, 0b000100001011, 15, 4},
        {12, 0b000100001110, 15, 5},
        {12, 0b000100010001, 15, 6},
        {12, 0b000100010011, 15, 7},
        {12, 0b000100010100, 15, 8},
        {13, 0b0000101001000, 15, 9},
        {13, 0b0000101100111, 15, 10},
        {13, 0b0000110000101, 15, 11},
        {13, 0b0000110010001, 15, 12},
        {14, 0b00001010001010, 15, 13},
        {14, 0b00001011000001, 15, 14},
        {14, 0b00001100000000, 15, 15},
    };

    // -- Table 15
    static constexpr HuffCode kHuff15[] = {
        {3, 0b111, 0, 0},
        {4, 0b1100, 0, 1},
        {5, 0b01110, 0, 2},
        {7, 0b0100100, 0, 3},
        {7, 0b0100110, 0, 4},
        {8, 0b01001010, 0, 5},
        {9, 0b010011010, 0, 6},
        {9, 0b010100010, 0, 7},
        {9, 0b010100110, 0, 8},
        {10, 0b0101010100, 0, 9},
        {10, 0b0101100010, 0, 10},
        {11, 0b01011010100, 0, 11},
        {11, 0b01100000100, 0, 12},
        {11, 0b01100100100, 0, 13},
        {12, 0b011001100100, 0, 14},
        {12, 0b011010000100, 0, 15},
        {3, 0b110, 1, 0},
        {4, 0b1010, 1, 1},
        {5, 0b01100, 1, 2},
        {6, 0b010000, 1, 3},
        {7, 0b0100101, 1, 4},
        {7, 0b0100111, 1, 5},
        {8, 0b01001011, 1, 6},
        {8, 0b01010000, 1, 7},
        {9, 0b010100011, 1, 8},
        {9, 0b010100111, 1, 9},
        {9, 0b010110000, 1, 10},
        {10, 0b0101100011, 1, 11},
        {10, 0b0101101010, 1, 12},
        {11, 0b01100000101, 1, 13},
        {11, 0b01100100101, 1, 14},
        {11, 0b01100110100, 1, 15},
        {4, 0b1011, 2, 0},
        {4, 0b1001, 2, 1},
        {5, 0b01101, 2, 2},
        {6, 0b010001, 2, 3},
        {6, 0b010010, 2, 4},
        {7, 0b0101000, 2, 5},
        {7, 0b0101001, 2, 6},
        {8, 0b01010001, 2, 7},
        {8, 0b01010100, 2, 8},
        {9, 0b010101000, 2, 9},
        {9, 0b010110001, 2, 10},
        {10, 0b0101100100, 2, 11},
        {10, 0b0101101011, 2, 12},
        {10, 0b0110000010, 2, 13},
        {11, 0b01100100110, 2, 14},
        {11, 0b01100110101, 2, 15},
        {5, 0b01111, 3, 0},
        {5, 0b01000, 3, 1},
        {6, 0b010011, 3, 2},
        {6, 0b010100, 3, 3},
        {7, 0b0101010, 3, 4},
        {7, 0b0101011, 3, 5},
        {7, 0b0101100, 3, 6},
        {8, 0b01010101, 3, 7},
        {8, 0b01011000, 3, 8},
        {9, 0b010110010, 3, 9},
        {9, 0b010110100, 3, 10},
        {10, 0b0101101100, 3, 11},
        {10, 0b0110000011, 3, 12},
        {10, 0b0110010010, 3, 13},
        {10, 0b0110011010, 3, 14},
        {11, 0b01101000010, 3, 15},
        {7, 0b0110000, 4, 0},
        {6, 0b010101, 4, 1},
        {6, 0b010110, 4, 2},
        {7, 0b0101101, 4, 3},
        {7, 0b0101110, 4, 4},
        {7, 0b0101111, 4, 5},
        {8, 0b01011001, 4, 6},
        {8, 0b01011010, 4, 7},
        {8, 0b01100000, 4, 8},
        {9, 0b010110101, 4, 9},
        {9, 0b011000010, 4, 10},
        {10, 0b0110001100, 4, 11},
        {10, 0b0110010011, 4, 12},
        {10, 0b0110011011, 4, 13},
        {10, 0b0110100010, 4, 14},
        {11, 0b01101000011, 4, 15},
        {7, 0b0110001, 5, 0},
        {7, 0b0101100, 5, 1},
        {7, 0b0110010, 5, 2},
        {7, 0b0110011, 5, 3},
        {7, 0b0110100, 5, 4},
        {8, 0b01011011, 5, 5},
        {8, 0b01100001, 5, 6},
        {8, 0b01100010, 5, 7},
        {8, 0b01100100, 5, 8},
        {9, 0b010111000, 5, 9},
        {9, 0b011001000, 5, 10},
        {10, 0b0110001101, 5, 11},
        {10, 0b0110100011, 5, 12},
        {10, 0b0110100100, 5, 13},
        {10, 0b0110110010, 5, 14},
        {11, 0b01101100110, 5, 15},
        {8, 0b01100011, 6, 0},
        {7, 0b0110101, 6, 1},
        {7, 0b0110110, 6, 2},
        {7, 0b0110111, 6, 3},
        {8, 0b01011100, 6, 4},
        {8, 0b01100011, 6, 5},
        {8, 0b01100101, 6, 6},
        {8, 0b01100110, 6, 7},
        {9, 0b011000110, 6, 8},
        {9, 0b011001001, 6, 9},
        {9, 0b011010010, 6, 10},
        {10, 0b0110011100, 6, 11},
        {10, 0b0110100101, 6, 12},
        {10, 0b0110110011, 6, 13},
        {10, 0b0110110100, 6, 14},
        {11, 0b01101100111, 6, 15},
        {8, 0b01101000, 7, 0},
        {8, 0b01011101, 7, 1},
        {7, 0b0111000, 7, 2},
        {8, 0b01011110, 7, 3},
        {8, 0b01011111, 7, 4},
        {8, 0b01101001, 7, 5},
        {8, 0b01101010, 7, 6},
        {8, 0b01101011, 7, 7},
        {9, 0b011001010, 7, 8},
        {9, 0b011010011, 7, 9},
        {9, 0b011011000, 7, 10},
        {10, 0b0110110101, 7, 11},
        {10, 0b0110111010, 7, 12},
        {10, 0b0111000010, 7, 13},
        {11, 0b01110001010, 7, 14},
        {11, 0b01110010010, 7, 15},
        {9, 0b011001011, 8, 0},
        {8, 0b01101100, 8, 1},
        {8, 0b01101101, 8, 2},
        {8, 0b01101110, 8, 3},
        {8, 0b01101111, 8, 4},
        {8, 0b01110000, 8, 5},
        {9, 0b011001100, 8, 6},
        {9, 0b011010100, 8, 7},
        {9, 0b011011001, 8, 8},
        {9, 0b011011010, 8, 9},
        {10, 0b0110111011, 8, 10},
        {10, 0b0111000011, 8, 11},
        {10, 0b0111001000, 8, 12},
        {10, 0b0111010010, 8, 13},
        {11, 0b01110010011, 8, 14},
        {11, 0b01110100110, 8, 15},
        {9, 0b011001101, 9, 0},
        {8, 0b01110001, 9, 1},
        {9, 0b011010101, 9, 2},
        {9, 0b011010110, 9, 3},
        {9, 0b011010111, 9, 4},
        {9, 0b011011011, 9, 5},
        {9, 0b011011100, 9, 6},
        {9, 0b011011101, 9, 7},
        {9, 0b011100010, 9, 8},
        {10, 0b0111000100, 9, 9},
        {10, 0b0111001001, 9, 10},
        {10, 0b0111010011, 9, 11},
        {10, 0b0111010100, 9, 12},
        {11, 0b01110100111, 9, 13},
        {11, 0b01110110010, 9, 14},
        {11, 0b01111000010, 9, 15},
        {10, 0b0110111100, 10, 0},
        {9, 0b011011110, 10, 1},
        {9, 0b011011111, 10, 2},
        {9, 0b011100011, 10, 3},
        {9, 0b011100100, 10, 4},
        {9, 0b011100101, 10, 5},
        {9, 0b011100110, 10, 6},
        {10, 0b0111001010, 10, 7},
        {10, 0b0111010000, 10, 8},
        {10, 0b0111010101, 10, 9},
        {10, 0b0111011000, 10, 10},
        {10, 0b0111100000, 10, 11},
        {11, 0b01110110011, 10, 12},
        {11, 0b01111000011, 10, 13},
        {11, 0b01111010010, 10, 14},
        {11, 0b01111100010, 10, 15},
        {10, 0b0110111101, 11, 0},
        {9, 0b011100111, 11, 1},
        {10, 0b0111001011, 11, 2},
        {10, 0b0111010001, 11, 3},
        {10, 0b0111010110, 11, 4},
        {10, 0b0111011001, 11, 5},
        {10, 0b0111100001, 11, 6},
        {10, 0b0111100010, 11, 7},
        {10, 0b0111100100, 11, 8},
        {10, 0b0111101000, 11, 9},
        {11, 0b01111000100, 11, 10},
        {11, 0b01111010011, 11, 11},
        {11, 0b01111100011, 11, 12},
        {11, 0b01111110010, 11, 13},
        {12, 0b011111100110, 11, 14},
        {12, 0b011111110010, 11, 15},
        {11, 0b01110110100, 12, 0},
        {10, 0b0111001100, 12, 1},
        {10, 0b0111010111, 12, 2},
        {10, 0b0111011010, 12, 3},
        {10, 0b0111100011, 12, 4},
        {10, 0b0111100101, 12, 5},
        {10, 0b0111101001, 12, 6},
        {10, 0b0111101010, 12, 7},
        {10, 0b0111110000, 12, 8},
        {11, 0b01111010100, 12, 9},
        {11, 0b01111100100, 12, 10},
        {11, 0b01111110011, 12, 11},
        {11, 0b01111111000, 12, 12},
        {12, 0b011111100111, 12, 13},
        {12, 0b011111110011, 12, 14},
        {12, 0b100000000010, 12, 15},
        {11, 0b01110110101, 13, 0},
        {10, 0b0111011011, 13, 1},
        {10, 0b0111011100, 13, 2},
        {10, 0b0111100110, 13, 3},
        {10, 0b0111101011, 13, 4},
        {10, 0b0111101100, 13, 5},
        {10, 0b0111110001, 13, 6},
        {11, 0b01111010101, 13, 7},
        {11, 0b01111100101, 13, 8},
        {11, 0b01111110100, 13, 9},
        {11, 0b01111111001, 13, 10},
        {12, 0b011111110100, 13, 11},
        {12, 0b100000000011, 13, 12},
        {12, 0b100000010010, 13, 13},
        {12, 0b100000100010, 13, 14},
        {12, 0b100000110010, 13, 15},
        {11, 0b01111000000, 14, 0},
        {10, 0b0111101101, 14, 1},
        {10, 0b0111101110, 14, 2},
        {10, 0b0111110010, 14, 3},
        {10, 0b0111110011, 14, 4},
        {11, 0b01111010110, 14, 5},
        {11, 0b01111100110, 14, 6},
        {11, 0b01111110101, 14, 7},
        {11, 0b01111111010, 14, 8},
        {11, 0b01111111100, 14, 9},
        {12, 0b011111110101, 14, 10},
        {12, 0b100000000100, 14, 11},
        {12, 0b100000010011, 14, 12},
        {12, 0b100000100011, 14, 13},
        {12, 0b100000110011, 14, 14},
        {13, 0b1000001000100, 14, 15},
        {11, 0b01111000001, 15, 0},
        {10, 0b0111110100, 15, 1},
        {11, 0b01111010111, 15, 2},
        {11, 0b01111100111, 15, 3},
        {11, 0b01111110110, 15, 4},
        {11, 0b01111111011, 15, 5},
        {11, 0b01111111101, 15, 6},
        {11, 0b01111111110, 15, 7},
        {12, 0b011111111110, 15, 8},
        {12, 0b100000000101, 15, 9},
        {12, 0b100000010100, 15, 10},
        {12, 0b100000100100, 15, 11},
        {12, 0b100000110100, 15, 12},
        {13, 0b1000001000101, 15, 13},
        {13, 0b1000001100100, 15, 14},
        {13, 0b1000010000100, 15, 15},
    };

    // For tables 16-23 and 24-31, they share base tables with different linbits.
    // Tables 16-23 share structure (max=15, varying linbits 1-13)
    // Tables 24-31 share structure (max=15, varying linbits 4-13)
    // We reuse tables 15 and 13 as base decoders for these.

    // -- Count1 tables (A and B) for the count1 region
    // Table A (table_select=0): used with count1table_select=0
    // Decodes 4 values (v,w,x,y) each 0 or 1
    struct Count1Code
    {
        uint8_t len;
        uint8_t code;
        uint8_t v, w, x, y;
    };

    // Table A (Huffman table for count1, table 32)
    static constexpr Count1Code kCount1A[] = {
        {1,  0b1,      0, 0, 0, 0},
        {4,  0b0101,   0, 0, 0, 1},
        {4,  0b0100,   0, 0, 1, 0},
        {5,  0b00101,  0, 0, 1, 1},
        {4,  0b0110,   0, 1, 0, 0},
        {5,  0b00110,  0, 1, 0, 1},
        {5,  0b00100,  0, 1, 1, 0},
        {6,  0b001110, 0, 1, 1, 1},
        {4,  0b0111,   1, 0, 0, 0},
        {5,  0b00111,  1, 0, 0, 1},
        {5,  0b00011,  1, 0, 1, 0},
        {6,  0b001111, 1, 0, 1, 1},
        {5,  0b00010,  1, 1, 0, 0},
        {6,  0b000010, 1, 1, 0, 1},
        {6,  0b000011, 1, 1, 1, 0},
        {6,  0b000001, 1, 1, 1, 1},
    };

    // Table B (table 33): all codewords are 4 bits, direct mapping
    static constexpr Count1Code kCount1B[] = {
        {4, 0b1111, 0, 0, 0, 0},
        {4, 0b1110, 0, 0, 0, 1},
        {4, 0b1101, 0, 0, 1, 0},
        {4, 0b1100, 0, 0, 1, 1},
        {4, 0b1011, 0, 1, 0, 0},
        {4, 0b1010, 0, 1, 0, 1},
        {4, 0b1001, 0, 1, 1, 0},
        {4, 0b1000, 0, 1, 1, 1},
        {4, 0b0111, 1, 0, 0, 0},
        {4, 0b0110, 1, 0, 0, 1},
        {4, 0b0101, 1, 0, 1, 0},
        {4, 0b0100, 1, 0, 1, 1},
        {4, 0b0011, 1, 1, 0, 0},
        {4, 0b0010, 1, 1, 0, 1},
        {4, 0b0001, 1, 1, 1, 0},
        {4, 0b0000, 1, 1, 1, 1},
    };

    // ========================================================================
    // Synthesis window coefficients (D[i], 512 values)
    // Derived from the standard (ISO 11172-3 Table B.3)
    // ========================================================================

    static constexpr double kSynthWindow[512] = {
         0.000000000, -0.000015259, -0.000015259, -0.000015259,
        -0.000015259, -0.000015259, -0.000015259, -0.000030518,
        -0.000030518, -0.000030518, -0.000030518, -0.000045776,
        -0.000045776, -0.000061035, -0.000061035, -0.000076294,
        -0.000076294, -0.000091553, -0.000106812, -0.000106812,
        -0.000122070, -0.000137329, -0.000152588, -0.000167847,
        -0.000198364, -0.000213623, -0.000244141, -0.000259399,
        -0.000289917, -0.000320435, -0.000366211, -0.000396729,
        -0.000442505, -0.000473022, -0.000534058, -0.000579834,
        -0.000625610, -0.000686646, -0.000747681, -0.000808716,
        -0.000885010, -0.000961304, -0.001037598, -0.001113892,
        -0.001205444, -0.001296997, -0.001388550, -0.001480103,
        -0.001586914, -0.001693726, -0.001785278, -0.001907349,
        -0.002014160, -0.002120972, -0.002243042, -0.002349854,
        -0.002456665, -0.002578735, -0.002685547, -0.002792358,
        -0.002899170, -0.002990723, -0.003082275, -0.003173828,
         0.003250122,  0.003326416,  0.003387451,  0.003433228,
         0.003463745,  0.003479004,  0.003479004,  0.003463745,
         0.003417969,  0.003372192,  0.003280640,  0.003173828,
         0.003051758,  0.002883911,  0.002700806,  0.002487183,
         0.002227783,  0.001937866,  0.001617432,  0.001266479,
         0.000869751,  0.000442505, -0.000030518, -0.000549316,
        -0.001098633, -0.001693726, -0.002334595, -0.003005981,
        -0.003723145, -0.004486084, -0.005294800, -0.006118774,
        -0.007003784, -0.007919312, -0.008865356, -0.009841919,
        -0.010848999, -0.011886597, -0.012939453, -0.014022827,
        -0.015121460, -0.016235352, -0.017349243, -0.018463135,
        -0.019577026, -0.020690918, -0.021789551, -0.022857666,
        -0.023910522, -0.024932861, -0.025909424, -0.026840210,
        -0.027725220, -0.028533936, -0.029281616, -0.029937744,
        -0.030532837, -0.031005859, -0.031387329, -0.031661987,
        -0.031814575, -0.031845093, -0.031738281, -0.031478882,
         0.031082153,  0.030517578,  0.029785156,  0.028884888,
         0.027801514,  0.026535034,  0.025085449,  0.023422241,
         0.021575928,  0.019531250,  0.017257690,  0.014801025,
         0.012115479,  0.009231567,  0.006134033,  0.002822876,
        -0.000686646, -0.004394531, -0.008316040, -0.012420654,
        -0.016708374, -0.021179199, -0.025817871, -0.030609131,
        -0.035552979, -0.040634155, -0.045837402, -0.051132202,
        -0.056533813, -0.061996460, -0.067520142, -0.073059082,
        -0.078628540, -0.084182739, -0.089706421, -0.095169067,
        -0.100540161, -0.105819702, -0.110946655, -0.115921021,
        -0.120697021, -0.125259399, -0.129562378, -0.133590698,
        -0.137298584, -0.140670776, -0.143676758, -0.146255493,
        -0.148422241, -0.150115967, -0.151306152, -0.151962280,
        -0.152069092, -0.151596069, -0.150497437, -0.148773193,
        -0.146362305, -0.143264771, -0.139450073, -0.134887695,
         0.129577637,  0.123474121,  0.116577148,  0.108856201,
         0.100311279,  0.090927124,  0.080688477,  0.069595337,
         0.057617187,  0.044784546,  0.031082153,  0.016510010,
         0.001068115, -0.015228271, -0.032379150, -0.050354004,
        -0.069168091, -0.088775635, -0.109161377, -0.130310059,
        -0.152206421, -0.174789429, -0.198059082, -0.221984863,
        -0.246505737, -0.271591187, -0.297210693, -0.323318481,
        -0.349868774, -0.376800537, -0.404083252, -0.431655884,
        -0.459472656, -0.487472534, -0.515609741, -0.543823242,
        -0.572036743, -0.600219727, -0.628295898, -0.656219482,
        -0.683914185, -0.711318970, -0.738372803, -0.765029907,
        -0.791213989, -0.816864014, -0.841949463, -0.866363525,
        -0.890090942, -0.913055420, -0.935195923, -0.956481934,
        -0.976852417, -0.996246338, -1.014617920, -1.031936646,
        -1.048156738, -1.063217163, -1.077117920, -1.089782715,
        -1.101211548, -1.111373901, -1.120223999, -1.127746582,
        -1.133926392, -1.138763428, -1.142211914, -1.144287109,
         1.144989014,  1.144287109,  1.142211914,  1.138763428,
         1.133926392,  1.127746582,  1.120223999,  1.111373901,
         1.101211548,  1.089782715,  1.077117920,  1.063217163,
         1.048156738,  1.031936646,  1.014617920,  0.996246338,
         0.976852417,  0.956481934,  0.935195923,  0.913055420,
         0.890090942,  0.866363525,  0.841949463,  0.816864014,
         0.791213989,  0.765029907,  0.738372803,  0.711318970,
         0.683914185,  0.656219482,  0.628295898,  0.600219727,
         0.572036743,  0.543823242,  0.515609741,  0.487472534,
         0.459472656,  0.431655884,  0.404083252,  0.376800537,
         0.349868774,  0.323318481,  0.297210693,  0.271591187,
         0.246505737,  0.221984863,  0.198059082,  0.174789429,
         0.152206421,  0.130310059,  0.109161377,  0.088775635,
         0.069168091,  0.050354004,  0.032379150,  0.015228271,
        -0.001068115, -0.016510010, -0.031082153, -0.044784546,
        -0.057617187, -0.069595337, -0.080688477, -0.090927124,
        -0.100311279, -0.108856201, -0.116577148, -0.123474121,
        -0.129577637, -0.134887695, -0.139450073, -0.143264771,
        -0.146362305, -0.148773193, -0.150497437, -0.151596069,
        -0.152069092, -0.151962280, -0.151306152, -0.150115967,
        -0.148422241, -0.146255493, -0.143676758, -0.140670776,
        -0.137298584, -0.133590698, -0.129562378, -0.125259399,
        -0.120697021, -0.115921021, -0.110946655, -0.105819702,
        -0.100540161, -0.095169067, -0.089706421, -0.084182739,
        -0.078628540, -0.073059082, -0.067520142, -0.061996460,
        -0.056533813, -0.051132202, -0.045837402, -0.040634155,
        -0.035552979, -0.030609131, -0.025817871, -0.021179199,
        -0.016708374, -0.012420654, -0.008316040, -0.004394531,
         0.000686646, -0.002822876, -0.006134033, -0.009231567,
        -0.012115479, -0.014801025, -0.017257690, -0.019531250,
        -0.021575928, -0.023422241, -0.025085449, -0.026535034,
        -0.027801514, -0.028884888, -0.029785156, -0.030517578,
        -0.031082153, -0.031478882, -0.031738281, -0.031845093,
        -0.031814575, -0.031661987, -0.031387329, -0.031005859,
        -0.030532837, -0.029937744, -0.029281616, -0.028533936,
        -0.027725220, -0.026840210, -0.025909424, -0.024932861,
        -0.023910522, -0.022857666, -0.021789551, -0.020690918,
        -0.019577026, -0.018463135, -0.017349243, -0.016235352,
        -0.015121460, -0.014022827, -0.012939453, -0.011886597,
        -0.010848999, -0.009841919, -0.008865356, -0.007919312,
        -0.007003784, -0.006118774, -0.005294800, -0.004486084,
        -0.003723145, -0.003005981, -0.002334595, -0.001693726,
        -0.001098633, -0.000549316,  0.000030518,  0.000442505,
         0.000869751,  0.001266479,  0.001617432,  0.001937866,
         0.002227783,  0.002487183,  0.002700806,  0.002883911,
         0.003051758,  0.003173828,  0.003280640,  0.003372192,
         0.003417969,  0.003463745,  0.003479004,  0.003479004,
         0.003463745,  0.003433228,  0.003387451,  0.003326416,
         0.003250122,  0.003173828,  0.003082275,  0.002990723,
         0.002899170,  0.002792358,  0.002685547,  0.002578735,
         0.002456665,  0.002349854,  0.002243042,  0.002120972,
         0.002014160,  0.001907349,  0.001785278,  0.001693726,
         0.001586914,  0.001480103,  0.001388550,  0.001296997,
         0.001205444,  0.001113892,  0.001037598,  0.000961304,
         0.000885010,  0.000808716,  0.000747681,  0.000686646,
         0.000625610,  0.000579834,  0.000534058,  0.000473022,
         0.000442505,  0.000396729,  0.000366211,  0.000320435,
         0.000289917,  0.000259399,  0.000244141,  0.000213623,
         0.000198364,  0.000167847,  0.000152588,  0.000137329,
         0.000122070,  0.000106812,  0.000106812,  0.000091553,
         0.000076294,  0.000076294,  0.000061035,  0.000061035,
         0.000045776,  0.000045776,  0.000030518,  0.000030518,
         0.000030518,  0.000030518,  0.000015259,  0.000015259,
         0.000015259,  0.000015259,  0.000015259,  0.000015259,
    };

    // ========================================================================
    // IMDCT window coefficients
    // ========================================================================

    // Normal (type 0) window for 36-point IMDCT
    static constexpr double kNormalWindow[36] = {
        0.043619387, 0.130526192, 0.216439614, 0.300705800, 0.382683432, 0.461748613,
        0.537299608, 0.608761429, 0.675590208, 0.737277337, 0.793353340, 0.843391446,
        0.887010833, 0.923879533, 0.953716951, 0.976296007, 0.991444861, 0.999048222,
        0.999048222, 0.991444861, 0.976296007, 0.953716951, 0.923879533, 0.887010833,
        0.843391446, 0.793353340, 0.737277337, 0.675590208, 0.608761429, 0.537299608,
        0.461748613, 0.382683432, 0.300705800, 0.216439614, 0.130526192, 0.043619387
    };

    // Start block (type 1) window
    static constexpr double kStartWindow[36] = {
        0.043619387, 0.130526192, 0.216439614, 0.300705800, 0.382683432, 0.461748613,
        0.537299608, 0.608761429, 0.675590208, 0.737277337, 0.793353340, 0.843391446,
        0.887010833, 0.923879533, 0.953716951, 0.976296007, 0.991444861, 0.999048222,
        1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000,
        0.991444861, 0.923879533, 0.793353340, 0.608761429, 0.382683432, 0.130526192,
        0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000
    };

    // Stop block (type 3) window
    static constexpr double kStopWindow[36] = {
        0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000,
        0.130526192, 0.382683432, 0.608761429, 0.793353340, 0.923879533, 0.991444861,
        1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000,
        0.999048222, 0.991444861, 0.976296007, 0.953716951, 0.923879533, 0.887010833,
        0.843391446, 0.793353340, 0.737277337, 0.675590208, 0.608761429, 0.537299608,
        0.461748613, 0.382683432, 0.300705800, 0.216439614, 0.130526192, 0.043619387
    };

    // Short block window (12-point)
    static constexpr double kShortWindow[12] = {
        0.130526192, 0.382683432, 0.608761429, 0.793353340, 0.923879533, 0.991444861,
        0.991444861, 0.923879533, 0.793353340, 0.608761429, 0.382683432, 0.130526192
    };

    // ========================================================================
    // Alias reduction butterfly coefficients
    // ========================================================================

    struct AliasCoeffs
    {
        double cs[8];
        double ca[8];
    };

    static AliasCoeffs getAliasCoeffs()
    {
        // ci values from the standard
        static constexpr double ci[8] = {
            -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037
        };
        AliasCoeffs c {};
        for (int i = 0; i < 8; ++i)
        {
            c.cs[i] = 1.0 / std::sqrt(1.0 + ci[i] * ci[i]);
            c.ca[i] = ci[i] / std::sqrt(1.0 + ci[i] * ci[i]);
        }
        return c;
    }

    // ========================================================================
    // Decoder state (per-channel)
    // ========================================================================

    struct ChannelState
    {
        double prevBlock[576]    = {};  // Previous IMDCT output for overlap-add
        double synthBuf[1024]    = {};  // Synthesis filterbank circular buffer
        int    synthOffset       = 0;   // Current position in synthesis buffer
    };

    // ========================================================================
    // Main decode state
    // ========================================================================

    std::vector<uint8_t> fileData_;
    size_t filePos_ = 0;

    AudioFileInfo info_ {};
    bool isOpen_ = false;

    std::vector<size_t> frameOffsets_; // Byte offset of each frame in fileData_
    std::vector<std::vector<float>> decodedSamples_; // [channel][sample]

    // Bit reservoir
    std::vector<uint8_t> reservoir_;
    size_t reservoirSize_ = 0;

    // Per-channel persistent state
    ChannelState channelState_[kChannelsMax] = {};

    // ========================================================================
    // Encoder state
    // ========================================================================

    /// @note Encoder limitation: The MP3 encoder does not implement a bit reservoir
    /// (main_data_begin is always 0). This means CBR mode allocates the same number
    /// of bits to every frame regardless of complexity. Quality is acceptable for
    /// most use cases but not equivalent to LAME or other production encoders.
    /// This is a deliberate trade-off for zero-dependency, header-only operation.

    struct EncChannelState
    {
        double analysisBuf[512] = {};
        double mdctOverlap[32][18] = {};   // Previous 18 subband samples per subband
    };

    std::ofstream outFile_;
    bool isWriting_ = false;
    int encBitrate_ = 128;
    int encPaddingAccum_ = 0;       // Running accumulator for padding distribution
    EncChannelState encState_[kChannelsMax] = {};
    double encInput_[kChannelsMax][kSamplesPerFrame] = {};
    int encInputPos_ = 0;
    std::vector<uint8_t> encFrameBuf_;

    // ========================================================================
    // ID3v2 tag skipping
    // ========================================================================

    void skipID3v2()
    {
        if (filePos_ + 10 > fileData_.size()) return;
        if (fileData_[filePos_]     == 'I' &&
            fileData_[filePos_ + 1] == 'D' &&
            fileData_[filePos_ + 2] == '3')
        {
            // ID3v2 tag size is stored as synchsafe integer in bytes 6-9
            uint32_t size = (static_cast<uint32_t>(fileData_[filePos_ + 6]) << 21)
                          | (static_cast<uint32_t>(fileData_[filePos_ + 7]) << 14)
                          | (static_cast<uint32_t>(fileData_[filePos_ + 8]) << 7)
                          |  static_cast<uint32_t>(fileData_[filePos_ + 9]);
            filePos_ += 10 + size;
        }
    }

    // ========================================================================
    // Frame scanning (first pass)
    // ========================================================================

    bool scanFrames()
    {
        frameOffsets_.clear();
        size_t pos = filePos_;
        int sampleRate = 0;
        int channels = 0;

        while (pos + 4 <= fileData_.size())
        {
            // Find sync word
            if ((fileData_[pos] & 0xFF) != 0xFF ||
                (fileData_[pos + 1] & 0xE0) != 0xE0)
            {
                ++pos;
                continue;
            }

            FrameHeader hdr {};
            if (!parseFrameHeader(pos, hdr))
            {
                ++pos;
                continue;
            }

            // Only support MPEG-1 Layer III
            if (hdr.version != 3 || hdr.layer != 1)
            {
                ++pos;
                continue;
            }

            if (frameOffsets_.empty())
            {
                sampleRate = hdr.sampleRate;
                channels = hdr.channels;
            }

            frameOffsets_.push_back(pos);
            pos += static_cast<size_t>(hdr.frameSize);
        }

        if (frameOffsets_.empty()) return false;

        info_.sampleRate = static_cast<double>(sampleRate);
        info_.numChannels = channels;
        info_.numSamples = static_cast<int64_t>(frameOffsets_.size()) * kSamplesPerFrame;
        info_.bitsPerSample = 16; // Nominal (decoded to float)
        info_.isFloatingPoint = true;

        return true;
    }

    // ========================================================================
    // Frame header parsing
    // ========================================================================

    bool parseFrameHeader(size_t pos, FrameHeader& hdr) const
    {
        if (pos + 4 > fileData_.size()) return false;

        uint32_t header = (static_cast<uint32_t>(fileData_[pos])     << 24)
                        | (static_cast<uint32_t>(fileData_[pos + 1]) << 16)
                        | (static_cast<uint32_t>(fileData_[pos + 2]) << 8)
                        |  static_cast<uint32_t>(fileData_[pos + 3]);

        // Sync word check
        if ((header & 0xFFE00000u) != 0xFFE00000u) return false;

        hdr.version    = static_cast<int>((header >> 19) & 3);
        hdr.layer      = static_cast<int>((header >> 17) & 3);
        hdr.crcProtect = !((header >> 16) & 1);
        hdr.bitrateIdx = static_cast<int>((header >> 12) & 0xF);
        hdr.srateIdx   = static_cast<int>((header >> 10) & 3);
        hdr.padding    = ((header >> 9) & 1) != 0;
        hdr.channelMode = static_cast<int>((header >> 6) & 3);
        hdr.modeExt    = static_cast<int>((header >> 4) & 3);
        hdr.copyright  = ((header >> 3) & 1) != 0;
        hdr.original   = ((header >> 2) & 1) != 0;
        hdr.emphasis   = static_cast<int>(header & 3);

        // Validate MPEG-1, Layer III
        if (hdr.version != 3) return false;       // Only MPEG-1 (version=3)
        if (hdr.layer != 1) return false;          // Layer III (layer=1 in header)
        if (hdr.bitrateIdx == 0 || hdr.bitrateIdx == 15) return false;
        if (hdr.srateIdx >= 3) return false;

        hdr.bitrate    = kBitrateTable[hdr.bitrateIdx] * 1000;
        hdr.sampleRate = kSampleRateTable[hdr.srateIdx];
        hdr.channels   = (hdr.channelMode == 3) ? 1 : 2;
        hdr.sideInfoSize = (hdr.channels == 1) ? 17 : 32;

        // Frame size in bytes: 144 * bitrate / sampleRate + padding
        hdr.frameSize = (144 * hdr.bitrate) / hdr.sampleRate + (hdr.padding ? 1 : 0);

        if (hdr.frameSize < 4 + (hdr.crcProtect ? 2 : 0) + hdr.sideInfoSize)
            return false;

        return true;
    }

    // ========================================================================
    // Side information parsing
    // ========================================================================

    bool parseSideInfo(BitReader& br, const FrameHeader& hdr, SideInfo& si) const
    {
        int nch = hdr.channels;

        si.main_data_begin = static_cast<int>(br.readBits(9));

        // Private bits
        if (nch == 1)
            br.readBits(5);
        else
            br.readBits(3);

        // SCFSI bands (4 bits per channel)
        for (int ch = 0; ch < nch; ++ch)
        {
            si.scfsi[ch] = static_cast<int>(br.readBits(4));
        }

        // Granule info
        for (int gr = 0; gr < kGranules; ++gr)
        {
            for (int ch = 0; ch < nch; ++ch)
            {
                auto& g = si.gr[gr][ch];
                g.part2_3_length     = static_cast<int>(br.readBits(12));
                g.big_values         = static_cast<int>(br.readBits(9));
                g.global_gain        = static_cast<int>(br.readBits(8));
                g.scalefac_compress  = static_cast<int>(br.readBits(4));
                g.window_switching   = br.readBits(1) != 0;

                if (g.window_switching)
                {
                    g.block_type      = static_cast<int>(br.readBits(2));
                    g.mixed_block     = static_cast<int>(br.readBits(1));
                    g.table_select[0] = static_cast<int>(br.readBits(5));
                    g.table_select[1] = static_cast<int>(br.readBits(5));
                    g.table_select[2] = 0;
                    g.subblock_gain[0] = static_cast<int>(br.readBits(3));
                    g.subblock_gain[1] = static_cast<int>(br.readBits(3));
                    g.subblock_gain[2] = static_cast<int>(br.readBits(3));

                    // region bounds for short/mixed blocks
                    if (g.block_type == 2 && g.mixed_block == 0)
                        g.region0_count = 8;
                    else
                        g.region0_count = 7;
                    g.region1_count = 20 - g.region0_count;
                }
                else
                {
                    g.block_type      = 0;
                    g.mixed_block     = 0;
                    g.table_select[0] = static_cast<int>(br.readBits(5));
                    g.table_select[1] = static_cast<int>(br.readBits(5));
                    g.table_select[2] = static_cast<int>(br.readBits(5));
                    g.subblock_gain[0] = 0;
                    g.subblock_gain[1] = 0;
                    g.subblock_gain[2] = 0;
                    g.region0_count   = static_cast<int>(br.readBits(4));
                    g.region1_count   = static_cast<int>(br.readBits(3));
                }

                g.preflag          = static_cast<int>(br.readBits(1));
                g.scalefac_scale   = static_cast<int>(br.readBits(1));
                g.count1table_select = static_cast<int>(br.readBits(1));
            }
        }

        return true;
    }

    // ========================================================================
    // Huffman decoding
    // ========================================================================

    // Decode a pair (x, y) from the big_values region using the given table index.
    bool decodePair(BitReader& br, int tableIdx, int& x, int& y) const
    {
        if (tableIdx == 0) { x = 0; y = 0; return true; }
        if (tableIdx == 4) // Table 4 not used in practice, treat like table 0
        { x = 0; y = 0; return true; }
        if (tableIdx == 14) // Table 14 not defined, treat like table 0
        { x = 0; y = 0; return true; }

        int linbits = kHuffLinbits[tableIdx];

        // Select which base table to use for decoding
        // Tables 16-23 share structure with table 16 (like table 13 base)
        // Tables 24-31 share structure with table 24 (like table 15 base)
        const HuffCode* codes = nullptr;
        int codeCount = 0;

        switch (tableIdx)
        {
            case 1:  codes = kHuff01; codeCount = sizeof(kHuff01)/sizeof(HuffCode); break;
            case 2:  codes = kHuff02; codeCount = sizeof(kHuff02)/sizeof(HuffCode); break;
            case 3:  codes = kHuff03; codeCount = sizeof(kHuff03)/sizeof(HuffCode); break;
            case 5:  codes = kHuff05; codeCount = sizeof(kHuff05)/sizeof(HuffCode); break;
            case 6:  codes = kHuff06; codeCount = sizeof(kHuff06)/sizeof(HuffCode); break;
            case 7:  codes = kHuff07; codeCount = sizeof(kHuff07)/sizeof(HuffCode); break;
            case 8:  codes = kHuff08; codeCount = sizeof(kHuff08)/sizeof(HuffCode); break;
            case 9:  codes = kHuff09; codeCount = sizeof(kHuff09)/sizeof(HuffCode); break;
            case 10: codes = kHuff10; codeCount = sizeof(kHuff10)/sizeof(HuffCode); break;
            case 11: codes = kHuff11; codeCount = sizeof(kHuff11)/sizeof(HuffCode); break;
            case 12: codes = kHuff12; codeCount = sizeof(kHuff12)/sizeof(HuffCode); break;
            case 13: codes = kHuff13; codeCount = sizeof(kHuff13)/sizeof(HuffCode); break;
            case 15: codes = kHuff15; codeCount = sizeof(kHuff15)/sizeof(HuffCode); break;
            case 16: case 17: case 18: case 19: case 20: case 21: case 22: case 23:
                codes = kHuff13; codeCount = sizeof(kHuff13)/sizeof(HuffCode); break;
            case 24: case 25: case 26: case 27: case 28: case 29: case 30: case 31:
                codes = kHuff15; codeCount = sizeof(kHuff15)/sizeof(HuffCode); break;
            default: x = 0; y = 0; return true;
        }

        // Decode by matching bits against the table
        if (!decodeHuffSymbol(br, codes, codeCount, x, y))
            return false;

        // Add linbits for large values
        if (linbits > 0)
        {
            if (x == 15)
                x += static_cast<int>(br.readBits(linbits));
            if (y == 15)
                y += static_cast<int>(br.readBits(linbits));
        }

        // Sign bits
        if (x != 0)
        {
            if (br.readBit()) x = -x;
        }
        if (y != 0)
        {
            if (br.readBit()) y = -y;
        }

        return true;
    }

    // Match bits from the stream against a Huffman code table
    bool decodeHuffSymbol(BitReader& br, const HuffCode* codes, int count,
                          int& xOut, int& yOut) const
    {
        // We accumulate bits and check against codes at each length.
        // This is a straightforward sequential decode (adequate for correctness).
        uint32_t acc = 0;
        int maxLen = 0;
        for (int i = 0; i < count; ++i)
            if (codes[i].len > maxLen) maxLen = codes[i].len;

        size_t startPos = br.getPos();

        for (int len = 1; len <= maxLen; ++len)
        {
            acc = (acc << 1) | static_cast<uint32_t>(br.readBit());

            for (int i = 0; i < count; ++i)
            {
                if (codes[i].len == len && codes[i].code == acc)
                {
                    xOut = codes[i].x;
                    yOut = codes[i].y;
                    return true;
                }
            }
        }

        // Not found: reset and return zeros
        br.setPos(startPos);
        xOut = 0;
        yOut = 0;
        return false;
    }

    // Decode count1 region (quadruples v,w,x,y each 0 or 1)
    bool decodeCount1(BitReader& br, int tableSelect, int& v, int& w, int& x, int& y) const
    {
        const Count1Code* codes = (tableSelect == 0) ? kCount1A : kCount1B;
        int count = 16;

        uint32_t acc = 0;
        int maxLen = (tableSelect == 0) ? 6 : 4;

        for (int len = 1; len <= maxLen; ++len)
        {
            acc = (acc << 1) | static_cast<uint32_t>(br.readBit());

            for (int i = 0; i < count; ++i)
            {
                if (codes[i].len == len && codes[i].code == acc)
                {
                    v = codes[i].v;
                    w = codes[i].w;
                    x = codes[i].x;
                    y = codes[i].y;

                    // Sign bits
                    if (v) { if (br.readBit()) v = -1; }
                    if (w) { if (br.readBit()) w = -1; }
                    if (x) { if (br.readBit()) x = -1; }
                    if (y) { if (br.readBit()) y = -1; }
                    return true;
                }
            }
        }

        v = w = x = y = 0;
        return false;
    }

    // Full Huffman decode for one granule/channel
    void huffmanDecode(BitReader& br, const GranuleChannel& gc, const FrameHeader& hdr,
                       int is[576]) const
    {
        std::memset(is, 0, 576 * sizeof(int));

        BandTable bands = getBandTable(hdr.sampleRate);

        // Determine region boundaries
        int region1Start, region2Start;
        if (gc.window_switching && gc.block_type == 2)
        {
            region1Start = 36; // 12 short bands * 3
            region2Start = kSamplesPerGranule;
        }
        else
        {
            int r0 = gc.region0_count + 1;
            int r1 = gc.region0_count + 1 + gc.region1_count + 1;
            if (r0 > bands.longCount) r0 = bands.longCount;
            if (r1 > bands.longCount) r1 = bands.longCount;
            region1Start = bands.longBands[r0];
            region2Start = bands.longBands[r1];
        }

        int bigValuesEnd = gc.big_values * 2;
        if (bigValuesEnd > 576) bigValuesEnd = 576;

        size_t part2_3_end = br.getPos() + static_cast<size_t>(gc.part2_3_length);

        // Big values region
        int idx = 0;
        for (; idx < bigValuesEnd && br.getPos() < part2_3_end; idx += 2)
        {
            int tableIdx;
            if (idx < region1Start)
                tableIdx = gc.table_select[0];
            else if (idx < region2Start)
                tableIdx = gc.table_select[1];
            else
                tableIdx = gc.table_select[2];

            int x = 0, y = 0;
            decodePair(br, tableIdx, x, y);
            if (idx < 576)     is[idx] = x;
            if (idx + 1 < 576) is[idx + 1] = y;
        }

        // Count1 region (quadruples, until part2_3 boundary)
        while (idx + 3 < 576 && br.getPos() < part2_3_end)
        {
            int v = 0, w = 0, x = 0, y = 0;
            if (!decodeCount1(br, gc.count1table_select, v, w, x, y))
                break;

            // Check we haven't overread
            if (br.getPos() > part2_3_end + 2)
            {
                // Went past the boundary, zero out
                break;
            }

            is[idx]     = v;
            is[idx + 1] = w;
            is[idx + 2] = x;
            is[idx + 3] = y;
            idx += 4;
        }

        // Ensure we're at the right position
        br.setPos(part2_3_end);
    }

    // ========================================================================
    // Scalefactor decoding
    // ========================================================================

    void decodeScalefactors(BitReader& br, const GranuleChannel& gc, int gr, int /*ch*/,
                            int scfsi, int scalefac[39], size_t& bitsRead) const
    {
        std::memset(scalefac, 0, 39 * sizeof(int));

        int slen1 = kSlen1[gc.scalefac_compress];
        int slen2 = kSlen2[gc.scalefac_compress];

        size_t startBits = br.getPos();

        if (gc.window_switching && gc.block_type == 2)
        {
            if (gc.mixed_block)
            {
                // Mixed: 8 long bands with slen1, then 3*3 short bands
                for (int sfb = 0; sfb < 8; ++sfb)
                    scalefac[sfb] = static_cast<int>(br.readBits(slen1));
                for (int sfb = 3; sfb < 6; ++sfb)
                    for (int win = 0; win < 3; ++win)
                        scalefac[sfb * 3 + win - 1] = static_cast<int>(br.readBits(slen1));
                for (int sfb = 6; sfb < 12; ++sfb)
                    for (int win = 0; win < 3; ++win)
                        scalefac[sfb * 3 + win - 1] = static_cast<int>(br.readBits(slen2));
            }
            else
            {
                // Pure short blocks: 12 bands * 3 windows
                for (int sfb = 0; sfb < 6; ++sfb)
                    for (int win = 0; win < 3; ++win)
                        scalefac[sfb * 3 + win] = static_cast<int>(br.readBits(slen1));
                for (int sfb = 6; sfb < 12; ++sfb)
                    for (int win = 0; win < 3; ++win)
                        scalefac[sfb * 3 + win] = static_cast<int>(br.readBits(slen2));
            }
        }
        else
        {
            // Long blocks
            // scfsi controls whether to reuse scalefactors from granule 0
            // Band groups: [0-5], [6-10], [11-15], [16-20]
            static constexpr int bandStart[4] = {0, 6, 11, 16};
            static constexpr int bandEnd[4]   = {6, 11, 16, 21};
            static constexpr int lens[4]      = {0, 0, 1, 1}; // 0=slen1, 1=slen2

            for (int group = 0; group < 4; ++group)
            {
                int slen = (lens[group] == 0) ? slen1 : slen2;
                if (gr == 1 && (scfsi & (8 >> group)))
                {
                    // Reuse from granule 0 (scalefactors already set externally)
                    // Skip reading, keep existing values
                }
                else
                {
                    for (int sfb = bandStart[group]; sfb < bandEnd[group]; ++sfb)
                        scalefac[sfb] = static_cast<int>(br.readBits(slen));
                }
            }
        }

        bitsRead = br.getPos() - startBits;
    }

    // ========================================================================
    // Requantization
    // ========================================================================

    void requantize(const int is[576], const int scalefac[39],
                    const GranuleChannel& gc, const FrameHeader& hdr,
                    double xr[576]) const
    {
        // PERFORMANCE: pre-compute pow(i, 4/3) for i in [0, 8191] once.
        // Huffman-decoded `is` values for MPEG-1 Layer III stay in this range
        // (after pretab/range expansion). Avoids ~2300 std::pow calls per frame.
        static const std::array<double, 8192> kPow43 = [] {
            std::array<double, 8192> t{};
            for (int i = 0; i < 8192; ++i)
                t[i] = std::pow(static_cast<double>(i), 4.0 / 3.0);
            return t;
        }();

        // Helper: signed pow43 with safety clamp.
        // kPow43 has static storage duration — referenced directly, not
        // captured (MSVC C3495: "lambda capture must have automatic storage").
        auto pow43 = [](int v) -> double {
            int a = (v < 0) ? -v : v;
            if (a >= 8192) a = 8191;
            return (v < 0) ? -kPow43[a] : kPow43[a];
        };

        BandTable bands = getBandTable(hdr.sampleRate);

        // exp2() instead of pow(2.0, x) — ~3× faster on MSVC.
        double globalGainFactor = std::exp2((static_cast<double>(gc.global_gain) - 210.0) / 4.0);
        double scalefacMult = (gc.scalefac_scale) ? 1.0 : 0.5;

        if (gc.window_switching && gc.block_type == 2)
        {
            if (gc.mixed_block)
            {
                // Long bands first (bands 0..7)
                int longEnd = bands.longBands[8];
                for (int sfb = 0; sfb < 8; ++sfb)
                {
                    int start = bands.longBands[sfb];
                    int end   = bands.longBands[sfb + 1];
                    double sfPow = std::exp2(-scalefacMult *
                        (scalefac[sfb] + gc.preflag * kPretab[sfb]));
                    double mult = globalGainFactor * sfPow;
                    for (int i = start; i < end && i < 576; ++i)
                        xr[i] = mult * pow43(is[i]);
                }

                // Short bands
                for (int sfb = 3; sfb < bands.shortCount; ++sfb)
                {
                    int width = bands.shortBands[sfb + 1] - bands.shortBands[sfb];
                    for (int win = 0; win < 3; ++win)
                    {
                        double sbGain = std::exp2(-2.0 * gc.subblock_gain[win]);
                        int sfIdx = sfb * 3 + win - 1;
                        if (sfIdx < 0) sfIdx = 0;
                        double sfPow = std::exp2(-scalefacMult * scalefac[sfIdx]);
                        double mult = globalGainFactor * sbGain * sfPow;

                        for (int k = 0; k < width; ++k)
                        {
                            int i = longEnd + (sfb - 3) * 3 * width + win * width + k;
                            if (i >= 576) break;
                            xr[i] = mult * pow43(is[i]);
                        }
                    }
                }
            }
            else
            {
                // Pure short blocks
                for (int sfb = 0; sfb < bands.shortCount; ++sfb)
                {
                    int width = bands.shortBands[sfb + 1] - bands.shortBands[sfb];
                    for (int win = 0; win < 3; ++win)
                    {
                        double sbGain = std::exp2(-2.0 * gc.subblock_gain[win]);
                        double sfPow = std::exp2(-scalefacMult * scalefac[sfb * 3 + win]);
                        double mult = globalGainFactor * sbGain * sfPow;

                        for (int k = 0; k < width; ++k)
                        {
                            int i = sfb * width * 3 + win * width + k;
                            if (i >= 576) break;
                            xr[i] = mult * pow43(is[i]);
                        }
                    }
                }
            }
        }
        else
        {
            // Long blocks
            for (int sfb = 0; sfb < bands.longCount; ++sfb)
            {
                int start = bands.longBands[sfb];
                int end   = bands.longBands[sfb + 1];
                double sfPow = std::exp2(-scalefacMult *
                    (scalefac[sfb] + gc.preflag * kPretab[sfb]));
                double mult = globalGainFactor * sfPow;
                for (int i = start; i < end && i < 576; ++i)
                    xr[i] = mult * pow43(is[i]);
            }
        }
    }

    // ========================================================================
    // Stereo processing
    // ========================================================================

    void stereoProcess(double xr[2][576], const GranuleChannel gc[2],
                       const FrameHeader& hdr, const int scalefac[2][39]) const
    {
        if (hdr.channels < 2) return;

        bool msStereo = (hdr.channelMode == 1) && ((hdr.modeExt & 2) != 0);
        bool iStereo  = (hdr.channelMode == 1) && ((hdr.modeExt & 1) != 0);

        if (msStereo)
        {
            // M/S stereo: M = (L+R)/sqrt(2), S = (L-R)/sqrt(2)
            // Recover: L = (M+S)/sqrt(2), R = (M-S)/sqrt(2)
            static constexpr double kInvSqrt2 = 0.7071067811865476;

            int isEnd = 576; // Apply to all unless intensity stereo limits it

            if (iStereo)
            {
                // Find the last nonzero sample in channel 1 to determine the IS bound
                isEnd = 0;
                for (int i = 575; i >= 0; --i)
                {
                    if (xr[1][i] != 0.0)
                    {
                        isEnd = i + 1;
                        break;
                    }
                }
            }

            for (int i = 0; i < isEnd; ++i)
            {
                double m = xr[0][i];
                double s = xr[1][i];
                xr[0][i] = (m + s) * kInvSqrt2;
                xr[1][i] = (m - s) * kInvSqrt2;
            }
        }

        if (iStereo)
        {
            // Intensity stereo: where channel 1 is zero, distribute channel 0
            // using the scalefactor of channel 1 as the stereo position.
            BandTable bands = getBandTable(hdr.sampleRate);

            // Simple intensity stereo (MPEG-1):
            // For each scalefactor band where channel 1 is zero:
            //   ratio = tan(scalefac[1][sfb] * PI/12)
            //   left  = xr[0] * k_l, right = xr[0] * k_r
            //   k_l = ratio / (1 + ratio), k_r = 1 / (1 + ratio)
            // When scalefac = 7, both channels get equal share.

            static constexpr double kISRatio[7] = {
                0.000000000, 0.267949192, 0.577350269, 1.000000000,
                1.732050808, 3.732050808, 1e10  // tan(6*PI/12) -> infinity
            };

            if (gc[1].window_switching && gc[1].block_type == 2)
            {
                // Short blocks intensity stereo
                for (int sfb = 0; sfb < bands.shortCount; ++sfb)
                {
                    int width = bands.shortBands[sfb + 1] - bands.shortBands[sfb];
                    for (int win = 0; win < 3; ++win)
                    {
                        int sfIdx = sfb * 3 + win;
                        if (sfIdx >= 36) break;
                        int sf = scalefac[1][sfIdx];
                        if (sf >= 7) continue;

                        // Check if all values in this band/window are zero in ch1
                        bool allZero = true;
                        int baseIdx = sfb * width * 3 + win * width;
                        for (int k = 0; k < width; ++k)
                        {
                            if (baseIdx + k < 576 && xr[1][baseIdx + k] != 0.0)
                            { allZero = false; break; }
                        }

                        if (allZero)
                        {
                            double ratio = kISRatio[sf];
                            double kl = ratio / (1.0 + ratio);
                            double kr = 1.0 / (1.0 + ratio);

                            for (int k = 0; k < width; ++k)
                            {
                                int i = baseIdx + k;
                                if (i >= 576) break;
                                xr[1][i] = xr[0][i] * kr;
                                xr[0][i] = xr[0][i] * kl;
                            }
                        }
                    }
                }
            }
            else
            {
                // Long blocks intensity stereo
                for (int sfb = 0; sfb < bands.longCount; ++sfb)
                {
                    int start = bands.longBands[sfb];
                    int end   = bands.longBands[sfb + 1];
                    int sf = scalefac[1][sfb];
                    if (sf >= 7) continue;

                    bool allZero = true;
                    for (int i = start; i < end; ++i)
                    {
                        if (xr[1][i] != 0.0) { allZero = false; break; }
                    }

                    if (allZero)
                    {
                        double ratio = kISRatio[sf];
                        double kl = ratio / (1.0 + ratio);
                        double kr = 1.0 / (1.0 + ratio);

                        for (int i = start; i < end; ++i)
                        {
                            xr[1][i] = xr[0][i] * kr;
                            xr[0][i] = xr[0][i] * kl;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Reordering (short blocks)
    // ========================================================================

    void reorder(double xr[576], const GranuleChannel& gc, const FrameHeader& hdr) const
    {
        if (!gc.window_switching || gc.block_type != 2) return;

        BandTable bands = getBandTable(hdr.sampleRate);
        double tmp[576];
        std::memset(tmp, 0, sizeof(tmp));

        int startBand = gc.mixed_block ? 3 : 0;
        int startIdx = gc.mixed_block ? bands.longBands[8] : 0;

        // Copy long-block portion unchanged
        if (gc.mixed_block)
        {
            for (int i = 0; i < startIdx; ++i)
                tmp[i] = xr[i];
        }

        // Reorder short blocks: from (sfb, win, freq) to (sfb, freq, win)
        for (int sfb = startBand; sfb < bands.shortCount; ++sfb)
        {
            int width = bands.shortBands[sfb + 1] - bands.shortBands[sfb];
            for (int win = 0; win < 3; ++win)
            {
                for (int k = 0; k < width; ++k)
                {
                    int srcIdx = startIdx + (sfb - startBand) * width * 3 + win * width + k;
                    int dstIdx = startIdx + (sfb - startBand) * width * 3 + k * 3 + win;
                    if (srcIdx < 576 && dstIdx < 576)
                        tmp[dstIdx] = xr[srcIdx];
                }
            }
        }

        std::memcpy(xr, tmp, sizeof(tmp));
    }

    // ========================================================================
    // Alias reduction
    // ========================================================================

    void aliasReduction(double xr[576], const GranuleChannel& gc) const
    {
        if (gc.window_switching && gc.block_type == 2 && !gc.mixed_block)
            return; // No alias reduction for pure short blocks

        static const AliasCoeffs ac = getAliasCoeffs();

        int sbLimit = gc.window_switching ? 1 : 31; // Only first subband for mixed

        for (int sb = 0; sb < sbLimit; ++sb)
        {
            for (int i = 0; i < 8; ++i)
            {
                int idx1 = (sb + 1) * 18 - 1 - i;
                int idx2 = (sb + 1) * 18 + i;
                if (idx1 < 0 || idx2 >= 576) break;

                double a = xr[idx1];
                double b = xr[idx2];
                xr[idx1] = a * ac.cs[i] - b * ac.ca[i];
                xr[idx2] = b * ac.cs[i] + a * ac.ca[i];
            }
        }
    }

    // ========================================================================
    // IMDCT
    // ========================================================================

    void imdct(const double xr[576], const GranuleChannel& gc, int ch,
               double output[576])
    {
        ChannelState& state = channelState_[ch];

        if (gc.window_switching && gc.block_type == 2)
        {
            if (gc.mixed_block)
            {
                // First 2 subbands: 36-point IMDCT with normal window
                for (int sb = 0; sb < 2; ++sb)
                {
                    double in[18];
                    double out36[36];
                    for (int i = 0; i < 18; ++i)
                        in[i] = xr[sb * 18 + i];

                    imdct36(in, out36);
                    applyWindow(out36, 0); // normal

                    for (int i = 0; i < 18; ++i)
                    {
                        output[sb * 18 + i] = out36[i] + state.prevBlock[sb * 18 + i];
                        state.prevBlock[sb * 18 + i] = out36[i + 18];
                    }
                }

                // Remaining subbands: 12-point IMDCT * 3 windows
                for (int sb = 2; sb < 32; ++sb)
                {
                    double tmp[36];
                    std::memset(tmp, 0, sizeof(tmp));

                    for (int win = 0; win < 3; ++win)
                    {
                        double in[6], out12[12];
                        for (int i = 0; i < 6; ++i)
                            in[i] = xr[sb * 18 + win * 6 + i];

                        imdct12(in, out12);

                        for (int i = 0; i < 12; ++i)
                            tmp[6 * win + i] += out12[i] * kShortWindow[i];
                    }

                    for (int i = 0; i < 18; ++i)
                    {
                        output[sb * 18 + i] = tmp[i] + state.prevBlock[sb * 18 + i];
                        state.prevBlock[sb * 18 + i] = tmp[i + 18];
                    }
                }
            }
            else
            {
                // Pure short blocks: 12-point IMDCT * 3 windows per subband
                for (int sb = 0; sb < 32; ++sb)
                {
                    double tmp[36];
                    std::memset(tmp, 0, sizeof(tmp));

                    for (int win = 0; win < 3; ++win)
                    {
                        double in[6], out12[12];
                        for (int i = 0; i < 6; ++i)
                            in[i] = xr[sb * 18 + win * 6 + i];

                        imdct12(in, out12);

                        for (int i = 0; i < 12; ++i)
                            tmp[6 * win + i] += out12[i] * kShortWindow[i];
                    }

                    for (int i = 0; i < 18; ++i)
                    {
                        output[sb * 18 + i] = tmp[i] + state.prevBlock[sb * 18 + i];
                        state.prevBlock[sb * 18 + i] = tmp[i + 18];
                    }
                }
            }
        }
        else
        {
            // Long blocks: 36-point IMDCT per subband
            int windowType = gc.block_type; // 0=normal, 1=start, 3=stop

            for (int sb = 0; sb < 32; ++sb)
            {
                double in[18];
                double out36[36];
                for (int i = 0; i < 18; ++i)
                    in[i] = xr[sb * 18 + i];

                imdct36(in, out36);
                applyWindow(out36, windowType);

                for (int i = 0; i < 18; ++i)
                {
                    output[sb * 18 + i] = out36[i] + state.prevBlock[sb * 18 + i];
                    state.prevBlock[sb * 18 + i] = out36[i + 18];
                }
            }
        }
    }

    // 36-point IMDCT
    // PERFORMANCE: cosines are CONSTANTS that depend only on (k,n), not on input
    // audio. Pre-compute once in a LUT instead of recomputing 648 cosines per call.
    static void imdct36(const double in[18], double out[36])
    {
        static const auto kCos = [] {
            std::array<std::array<double, 18>, 36> t{};
            constexpr double kPi = 3.14159265358979323846;
            for (int k = 0; k < 36; ++k)
                for (int n = 0; n < 18; ++n)
                    t[k][n] = std::cos(kPi / 72.0 *
                                       (2.0 * k + 1.0 + 18.0) *
                                       (2.0 * n + 1.0));
            return t;
        }();

        for (int k = 0; k < 36; ++k)
        {
            double sum = 0.0;
            for (int n = 0; n < 18; ++n)
                sum += in[n] * kCos[k][n];
            out[k] = sum;
        }
    }

    // 12-point IMDCT (same LUT optimisation)
    static void imdct12(const double in[6], double out[12])
    {
        static const auto kCos = [] {
            std::array<std::array<double, 6>, 12> t{};
            constexpr double kPi = 3.14159265358979323846;
            for (int k = 0; k < 12; ++k)
                for (int n = 0; n < 6; ++n)
                    t[k][n] = std::cos(kPi / 24.0 *
                                       (2.0 * k + 1.0 + 6.0) *
                                       (2.0 * n + 1.0));
            return t;
        }();

        for (int k = 0; k < 12; ++k)
        {
            double sum = 0.0;
            for (int n = 0; n < 6; ++n)
                sum += in[n] * kCos[k][n];
            out[k] = sum;
        }
    }

    // Apply window to IMDCT output
    static void applyWindow(double out[36], int blockType)
    {
        const double* win;
        switch (blockType)
        {
            case 1:  win = kStartWindow; break;
            case 3:  win = kStopWindow;  break;
            default: win = kNormalWindow; break;
        }
        for (int i = 0; i < 36; ++i)
            out[i] *= win[i];
    }

    // ========================================================================
    // Frequency inversion
    // ========================================================================

    static void frequencyInversion(double output[576])
    {
        // Negate every other sample in every other subband
        for (int sb = 1; sb < 32; sb += 2)
        {
            for (int i = 1; i < 18; i += 2)
            {
                output[sb * 18 + i] = -output[sb * 18 + i];
            }
        }
    }

    // ========================================================================
    // Synthesis polyphase filterbank
    // ========================================================================

    void synthesize(const double input[576], int ch, float pcm[576])
    {
        // PERFORMANCE: pre-compute the 64×32 matrixing cosines ONCE.
        // Previous version called std::cos() 2048 times per time-slot × 18 slots
        // × 2 channels × ~13k frames ~= 1 billion cosines per 5-min file.
        static const auto kMatrixCos = [] {
            std::array<std::array<double, 32>, 64> t{};
            constexpr double kPi = 3.14159265358979323846;
            for (int i = 0; i < 64; ++i)
                for (int k = 0; k < 32; ++k)
                    t[i][k] = std::cos(kPi / 64.0 *
                                       (16.0 + static_cast<double>(i)) *
                                       (2.0 * static_cast<double>(k) + 1.0));
            return t;
        }();

        ChannelState& state = channelState_[ch];

        // Process 18 samples per subband (= 18 blocks of 32 subbands)
        for (int ss = 0; ss < 18; ++ss)
        {
            // Extract 32 subband samples for this time slot
            double S[32];
            for (int sb = 0; sb < 32; ++sb)
                S[sb] = input[sb * 18 + ss];

            // Matrixing: 64 values from 32 subbands via DCT (LUT)
            double V[64];
            for (int i = 0; i < 64; ++i)
            {
                double sum = 0.0;
                const auto& row = kMatrixCos[i];
                for (int k = 0; k < 32; ++k)
                    sum += S[k] * row[k];
                V[i] = sum;
            }

            // Shift synthesis buffer (1024 = 2^10 → use bitmask, not %)
            state.synthOffset = (state.synthOffset - 64) & 1023;

            // Store V into synthesis buffer (bitmask)
            for (int i = 0; i < 64; ++i)
                state.synthBuf[(state.synthOffset + i) & 1023] = V[i];

            // Build 512 U values and window (bitmask)
            double U[512];
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 32; ++j)
                {
                    U[i * 64 + j]      = state.synthBuf[(state.synthOffset + i * 128 + j) & 1023];
                    U[i * 64 + 32 + j] = state.synthBuf[(state.synthOffset + i * 128 + 96 + j) & 1023];
                }
            }

            // Window and sum
            for (int j = 0; j < 32; ++j)
            {
                double sum = 0.0;
                for (int i = 0; i < 16; ++i)
                {
                    sum += U[i * 32 + j] * kSynthWindow[i * 32 + j];
                }
                // Clamp to [-1.0, 1.0]
                if (sum > 1.0) sum = 1.0;
                if (sum < -1.0) sum = -1.0;
                pcm[ss * 32 + j] = static_cast<float>(sum);
            }
        }
    }

    // ========================================================================
    // Encoder: Bit writer
    // ========================================================================

    class BitWriter
    {
    public:
        BitWriter() = default;
        void init(std::vector<uint8_t>& buf) { buf_ = &buf; bitPos_ = 0; }

        void writeBits(uint32_t val, int n)
        {
            for (int i = n - 1; i >= 0; --i)
            {
                size_t byteIdx = bitPos_ >> 3;
                while (byteIdx >= buf_->size()) buf_->push_back(0);
                int bitIdx = 7 - static_cast<int>(bitPos_ & 7);
                if ((val >> static_cast<unsigned>(i)) & 1u)
                    (*buf_)[byteIdx] |= static_cast<uint8_t>(1u << bitIdx);
                ++bitPos_;
            }
        }

        [[nodiscard]] size_t getBitPos() const { return bitPos_; }
        void padToByte() { while (bitPos_ & 7) writeBits(0, 1); }

    private:
        std::vector<uint8_t>* buf_ = nullptr;
        size_t bitPos_ = 0;
    };

    // ========================================================================
    // Encoder: Analysis polyphase filterbank
    // ========================================================================

    void encAnalysis(const double* pcm32, int ch, double S[32])
    {
        auto& st = encState_[ch];

        // Shift buffer and insert 32 new samples
        std::memmove(st.analysisBuf + 32, st.analysisBuf, 480 * sizeof(double));
        for (int i = 0; i < 32; ++i)
            st.analysisBuf[i] = pcm32[31 - i];

        // Window and partial sum into 64 values
        double Y[64] = {};
        for (int i = 0; i < 64; ++i)
            for (int j = 0; j < 8; ++j)
                Y[i] += st.analysisBuf[i + j * 64] * kSynthWindow[i + j * 64];

        // Matrixing (DCT) → 32 subband samples
        static constexpr double kPi = 3.14159265358979323846;
        for (int k = 0; k < 32; ++k)
        {
            double sum = 0.0;
            for (int i = 0; i < 64; ++i)
                sum += Y[i] * std::cos(kPi / 64.0 * (2.0 * k + 1.0) * (i - 16.0));
            S[k] = sum;
        }
    }

    // ========================================================================
    // Encoder: Forward MDCT-36 with windowing
    // ========================================================================

    static void encMdct36(const double z[36], double X[18])
    {
        static constexpr double kPi = 3.14159265358979323846;
        // Apply normal window
        double zw[36];
        for (int n = 0; n < 36; ++n)
            zw[n] = z[n] * kNormalWindow[n];

        for (int k = 0; k < 18; ++k)
        {
            double sum = 0.0;
            for (int n = 0; n < 36; ++n)
                sum += zw[n] * std::cos(kPi / 36.0 * (2.0 * n + 19.0) * (2.0 * k + 1.0));
            X[k] = sum;
        }
    }

    // ========================================================================
    // Encoder: Huffman table selection and bit counting
    // ========================================================================

    // Find best Huffman table for a region of pairs, return bit count
    static int encSelectTable(const int* ix, int count, int& tableOut)
    {
        if (count <= 0) { tableOut = 0; return 0; }

        int maxVal = 0;
        for (int i = 0; i < count; ++i)
        {
            int v = ix[i] < 0 ? -ix[i] : ix[i];
            if (v > maxVal) maxVal = v;
        }

        if (maxVal == 0) { tableOut = 0; return 0; }

        // Candidate tables by max value
        struct TableCandidate { int table; int xmax; };
        static constexpr TableCandidate candidates[] = {
            {1, 1}, {2, 2}, {5, 3}, {7, 5}, {10, 7}, {13, 15},
            {16, 16}, {17, 18}, {18, 22}, {19, 30}, {20, 46},
            {21, 78}, {22, 142}, {23, 8206},
            {24, 30}, {25, 46}, {26, 78}, {27, 142}, {28, 270},
            {29, 526}, {30, 2062}, {31, 8206}
        };

        int bestBits = 999999;
        int bestTable = 0;
        for (auto& c : candidates)
        {
            if (c.xmax < maxVal) continue;
            int bits = encCountTableBits(ix, count, c.table);
            if (bits < bestBits) { bestBits = bits; bestTable = c.table; }
        }
        tableOut = bestTable;
        return bestBits;
    }

    // Count bits needed to encode pairs with a specific table
    static int encCountTableBits(const int* ix, int count, int tableIdx)
    {
        int linbits = kHuffLinbits[tableIdx];
        const HuffCode* codes = nullptr;
        int codeCount = 0;
        encGetTable(tableIdx, codes, codeCount);
        if (tableIdx != 0 && codes == nullptr) return 999999;

        int totalBits = 0;
        for (int i = 0; i < count; i += 2)
        {
            int x = ix[i] < 0 ? -ix[i] : ix[i];
            int y = (i + 1 < count) ? (ix[i+1] < 0 ? -ix[i+1] : ix[i+1]) : 0;

            if (tableIdx == 0) { if (x != 0 || y != 0) return 999999; continue; }

            int xBase = (linbits > 0 && x > 14) ? 15 : x;
            int yBase = (linbits > 0 && y > 14) ? 15 : y;

            // Find codeword length
            int cLen = 0;
            bool found = false;
            for (int c = 0; c < codeCount; ++c)
            {
                if (codes[c].x == xBase && codes[c].y == yBase)
                {
                    cLen = codes[c].len;
                    found = true;
                    break;
                }
            }
            if (!found) return 999999;

            totalBits += cLen;
            if (linbits > 0 && x > 14) totalBits += linbits;
            if (linbits > 0 && y > 14) totalBits += linbits;
            if (x != 0) totalBits += 1; // sign bit
            if (y != 0) totalBits += 1;
        }
        return totalBits;
    }

    // Get code table pointer for a given table index
    static void encGetTable(int tableIdx, const HuffCode*& codes, int& count)
    {
        codes = nullptr; count = 0;
        switch (tableIdx)
        {
            case 1:  codes = kHuff01; count = sizeof(kHuff01)/sizeof(HuffCode); break;
            case 2:  codes = kHuff02; count = sizeof(kHuff02)/sizeof(HuffCode); break;
            case 3:  codes = kHuff03; count = sizeof(kHuff03)/sizeof(HuffCode); break;
            case 5:  codes = kHuff05; count = sizeof(kHuff05)/sizeof(HuffCode); break;
            case 6:  codes = kHuff06; count = sizeof(kHuff06)/sizeof(HuffCode); break;
            case 7:  codes = kHuff07; count = sizeof(kHuff07)/sizeof(HuffCode); break;
            case 8:  codes = kHuff08; count = sizeof(kHuff08)/sizeof(HuffCode); break;
            case 9:  codes = kHuff09; count = sizeof(kHuff09)/sizeof(HuffCode); break;
            case 10: codes = kHuff10; count = sizeof(kHuff10)/sizeof(HuffCode); break;
            case 11: codes = kHuff11; count = sizeof(kHuff11)/sizeof(HuffCode); break;
            case 12: codes = kHuff12; count = sizeof(kHuff12)/sizeof(HuffCode); break;
            case 13: codes = kHuff13; count = sizeof(kHuff13)/sizeof(HuffCode); break;
            case 15: codes = kHuff15; count = sizeof(kHuff15)/sizeof(HuffCode); break;
            case 16: case 17: case 18: case 19: case 20: case 21: case 22: case 23:
                codes = kHuff13; count = sizeof(kHuff13)/sizeof(HuffCode); break;
            case 24: case 25: case 26: case 27: case 28: case 29: case 30: case 31:
                codes = kHuff15; count = sizeof(kHuff15)/sizeof(HuffCode); break;
            default: break;
        }
    }

    // ========================================================================
    // Encoder: Quantization (inner loop — find global_gain)
    // ========================================================================

    static int encQuantize(const double xr[576], int ix[576], int globalGain)
    {
        double step = std::pow(2.0, 0.25 * (globalGain - 210));
        if (step < 1e-30) step = 1e-30;

        int maxIx = 0;
        for (int i = 0; i < 576; ++i)
        {
            double val = std::abs(xr[i]) / step;
            // The MP3 quantization: nint(val^0.75)
            int q = static_cast<int>(std::pow(val, 0.75) + 0.4054);
            if (q > 8191) q = 8191;
            ix[i] = (xr[i] >= 0.0) ? q : -q;
            if (q > maxIx) maxIx = q;
        }
        return maxIx;
    }

    // Count total Huffman bits for a quantized granule
    static int encCountGranuleBits(const int ix[576], GranuleChannel& gc,
                                    const BandTable& bands)
    {
        // Find last non-zero
        int rzero = 576;
        while (rzero > 0 && ix[rzero - 1] == 0) --rzero;
        if (rzero == 0) { gc.big_values = 0; return 0; }

        // Count1 region: from end, find quads of {0,1}
        int count1End = rzero;
        int count1Start = count1End;
        while (count1Start >= 4)
        {
            bool allSmall = true;
            for (int j = count1Start - 4; j < count1Start; ++j)
            {
                int v = ix[j] < 0 ? -ix[j] : ix[j];
                if (v > 1) { allSmall = false; break; }
            }
            if (!allSmall) break;
            count1Start -= 4;
        }

        gc.big_values = count1Start / 2;
        int bigEnd = gc.big_values * 2;

        // Choose region boundaries
        int r0 = 0, r1 = 0;
        if (bigEnd > 0)
        {
            // Find best split using scalefactor bands
            int bestR0 = 1, bestR1 = 2;
            for (int t = 1; t < bands.longCount && bands.longBands[t] < bigEnd; ++t)
                bestR1 = t;
            bestR0 = (bestR1 > 1) ? bestR1 / 2 : 1;
            if (bestR0 >= bestR1) bestR0 = bestR1 - 1;
            if (bestR0 < 1) bestR0 = 1;
            r0 = bestR0;
            r1 = bestR1;
        }

        gc.region0_count = r0 - 1;
        gc.region1_count = r1 - r0 - 1;
        if (gc.region0_count < 0) gc.region0_count = 0;
        if (gc.region1_count < 0) gc.region1_count = 0;

        int reg0End = (r0 < bands.longCount) ? std::min(bands.longBands[r0], bigEnd) : bigEnd;
        int reg1End = (r1 < bands.longCount) ? std::min(bands.longBands[r1], bigEnd) : bigEnd;

        // Select tables for each region
        int bits = 0;
        bits += encSelectTable(ix, reg0End, gc.table_select[0]);
        bits += encSelectTable(ix + reg0End, reg1End - reg0End, gc.table_select[1]);
        bits += encSelectTable(ix + reg1End, bigEnd - reg1End, gc.table_select[2]);

        // Count1 region
        int count1Bits_A = 0, count1Bits_B = 0;
        for (int i = count1Start; i < count1End; i += 4)
        {
            int v = ix[i] != 0 ? 1 : 0;
            int w = (i+1 < 576 && ix[i+1] != 0) ? 1 : 0;
            int x = (i+2 < 576 && ix[i+2] != 0) ? 1 : 0;
            int y = (i+3 < 576 && ix[i+3] != 0) ? 1 : 0;
            int signBits = v + w + x + y;

            // Table A
            for (int c = 0; c < 16; ++c)
                if (kCount1A[c].v == v && kCount1A[c].w == w &&
                    kCount1A[c].x == x && kCount1A[c].y == y)
                { count1Bits_A += kCount1A[c].len + signBits; break; }

            // Table B: always 4 bits
            count1Bits_B += 4 + signBits;
        }

        if (count1Bits_A <= count1Bits_B)
        {
            gc.count1table_select = 0;
            bits += count1Bits_A;
        }
        else
        {
            gc.count1table_select = 1;
            bits += count1Bits_B;
        }

        return bits;
    }

    // ========================================================================
    // Encoder: Huffman bitstream writing
    // ========================================================================

    static void encHuffWrite(BitWriter& bw, const int* ix, int count, int tableIdx)
    {
        if (tableIdx == 0 || count <= 0) return;
        int linbits = kHuffLinbits[tableIdx];
        const HuffCode* codes = nullptr;
        int codeCount = 0;
        encGetTable(tableIdx, codes, codeCount);

        for (int i = 0; i < count; i += 2)
        {
            int x = ix[i] < 0 ? -ix[i] : ix[i];
            int y = (i + 1 < count) ? (ix[i+1] < 0 ? -ix[i+1] : ix[i+1]) : 0;
            int xBase = (linbits > 0 && x > 14) ? 15 : x;
            int yBase = (linbits > 0 && y > 14) ? 15 : y;

            // Find and write codeword
            for (int c = 0; c < codeCount; ++c)
            {
                if (codes[c].x == xBase && codes[c].y == yBase)
                {
                    bw.writeBits(codes[c].code, codes[c].len);
                    break;
                }
            }

            if (linbits > 0 && x > 14)
                bw.writeBits(static_cast<uint32_t>(x - 15), linbits);
            if (x != 0) bw.writeBits(ix[i] < 0 ? 1u : 0u, 1);

            if (linbits > 0 && y > 14)
                bw.writeBits(static_cast<uint32_t>(y - 15), linbits);
            if (y != 0) bw.writeBits((i+1 < count && ix[i+1] < 0) ? 1u : 0u, 1);
        }
    }

    static void encCount1Write(BitWriter& bw, const int* ix, int start, int end, int tableSelect)
    {
        const Count1Code* codes = (tableSelect == 0) ? kCount1A : kCount1B;
        for (int i = start; i < end; i += 4)
        {
            int v = (ix[i] != 0) ? 1 : 0;
            int w = (i+1 < 576 && ix[i+1] != 0) ? 1 : 0;
            int x = (i+2 < 576 && ix[i+2] != 0) ? 1 : 0;
            int y = (i+3 < 576 && ix[i+3] != 0) ? 1 : 0;

            for (int c = 0; c < 16; ++c)
            {
                if (codes[c].v == v && codes[c].w == w &&
                    codes[c].x == x && codes[c].y == y)
                {
                    bw.writeBits(codes[c].code, codes[c].len);
                    break;
                }
            }

            if (v) bw.writeBits(ix[i] < 0 ? 1u : 0u, 1);
            if (w) bw.writeBits((i+1<576 && ix[i+1]<0) ? 1u : 0u, 1);
            if (x) bw.writeBits((i+2<576 && ix[i+2]<0) ? 1u : 0u, 1);
            if (y) bw.writeBits((i+3<576 && ix[i+3]<0) ? 1u : 0u, 1);
        }
    }

    // ========================================================================
    // Encoder: Full frame encoding pipeline
    // ========================================================================

    void encEncodeFrame()
    {
        const int nch = info_.numChannels;
        const int sr = static_cast<int>(info_.sampleRate);
        BandTable bands = getBandTable(sr);

        // Compute frame size
        int srIdx = (sr == 44100) ? 0 : (sr == 48000) ? 1 : 2;
        int brIdx = 0;
        for (int i = 1; i < 15; ++i)
            if (kBitrateTable[i] == encBitrate_) { brIdx = i; break; }

        int baseFrameSize = 144 * encBitrate_ * 1000 / sr;
        // Distribute padding across frames using a running accumulator
        // to maintain correct average bitrate (not padding every frame)
        int remainder = (144 * encBitrate_ * 1000) % sr;
        encPaddingAccum_ += remainder;
        bool padding = false;
        if (encPaddingAccum_ >= sr) {
            padding = true;
            encPaddingAccum_ -= sr;
        }
        int frameSize = baseFrameSize + (padding ? 1 : 0);

        int sideInfoSize = (nch == 1) ? 17 : 32;
        int headerSize = 4;
        int availBytes = frameSize - headerSize - sideInfoSize;
        int availBits = availBytes * 8;

        // 1. Analysis filterbank: PCM → subbands
        double subbands[2][32][36] = {}; // [ch][subband][timeSlot]
        for (int ch = 0; ch < nch; ++ch)
        {
            for (int ts = 0; ts < 36; ++ts)
            {
                double S[32];
                encAnalysis(&encInput_[ch][ts * 32], ch, S);
                for (int sb = 0; sb < 32; ++sb)
                    subbands[ch][sb][ts] = S[sb];
            }
        }

        // 2. MDCT for each granule
        SideInfo si {};
        int ix[2][2][576] = {};    // [gr][ch]
        double xr[2][2][576] = {}; // [gr][ch]

        for (int gr = 0; gr < 2; ++gr)
        {
            for (int ch = 0; ch < nch; ++ch)
            {
                // For each subband: combine overlap + new → MDCT
                for (int sb = 0; sb < 32; ++sb)
                {
                    double mdctIn[36];
                    // First 18: from overlap
                    for (int n = 0; n < 18; ++n)
                        mdctIn[n] = encState_[ch].mdctOverlap[sb][n];
                    // Next 18: new subband samples
                    for (int n = 0; n < 18; ++n)
                        mdctIn[18 + n] = subbands[ch][sb][gr * 18 + n];
                    // Save overlap for next
                    for (int n = 0; n < 18; ++n)
                        encState_[ch].mdctOverlap[sb][n] = subbands[ch][sb][gr * 18 + n];

                    double mdctOut[18];
                    encMdct36(mdctIn, mdctOut);
                    for (int k = 0; k < 18; ++k)
                        xr[gr][ch][sb * 18 + k] = mdctOut[k];
                }
            }
        }

        // 3. Quantization — find global_gain via binary search
        int bitsPerGranule = availBits / 2;
        for (int gr = 0; gr < 2; ++gr)
        {
            for (int ch = 0; ch < nch; ++ch)
            {
                auto& gc = si.gr[gr][ch];
                gc.block_type = 0;
                gc.window_switching = false;
                gc.mixed_block = 0;
                gc.preflag = 0;
                gc.scalefac_scale = 0;
                gc.scalefac_compress = 0;

                int targetBits = bitsPerGranule / nch;

                // Binary search for global_gain
                int lo = 0, hi = 255, bestGain = 210;
                int bestBits = 999999;
                while (lo <= hi)
                {
                    int mid = (lo + hi) / 2;
                    int tmpIx[576];
                    encQuantize(xr[gr][ch], tmpIx, mid);

                    GranuleChannel tmpGc = gc;
                    int bits = encCountGranuleBits(tmpIx, tmpGc, bands);

                    if (bits <= targetBits)
                    {
                        if (bits <= bestBits || mid < bestGain)
                        {
                            bestBits = bits;
                            bestGain = mid;
                            gc = tmpGc;
                            std::memcpy(ix[gr][ch], tmpIx, sizeof(tmpIx));
                        }
                        hi = mid - 1; // Try lower gain (more bits, better quality)
                    }
                    else
                    {
                        lo = mid + 1; // Increase gain (fewer bits)
                    }
                }
                gc.global_gain = bestGain;
                gc.part2_3_length = bestBits;
            }
        }

        // 4. Write frame to buffer
        encFrameBuf_.clear();
        encFrameBuf_.resize(static_cast<size_t>(frameSize), 0);
        BitWriter bw;
        bw.init(encFrameBuf_);

        // Frame header (32 bits)
        bw.writeBits(0xFFF, 12);       // Sync word
        bw.writeBits(1, 1);            // MPEG-1
        bw.writeBits(0b01, 2);         // Layer III
        bw.writeBits(1, 1);            // No CRC
        bw.writeBits(static_cast<uint32_t>(brIdx), 4);
        bw.writeBits(static_cast<uint32_t>(srIdx), 2);
        bw.writeBits(padding ? 1u : 0u, 1);
        bw.writeBits(0, 1);            // Private bit
        bw.writeBits(nch == 1 ? 3u : 0u, 2); // Channel mode (3=mono, 0=stereo)
        bw.writeBits(0, 2);            // Mode extension
        bw.writeBits(0, 1);            // Copyright
        bw.writeBits(1, 1);            // Original
        bw.writeBits(0, 2);            // Emphasis

        // Side information
        bw.writeBits(0, 9);            // main_data_begin = 0 (no reservoir)
        bw.writeBits(0, nch == 1 ? 5 : 3); // Private bits

        // SCFSI
        for (int ch = 0; ch < nch; ++ch)
            bw.writeBits(0, 4);         // No SCFSI

        // Granule channel side info
        for (int gr = 0; gr < 2; ++gr)
        {
            for (int ch = 0; ch < nch; ++ch)
            {
                auto& gc = si.gr[gr][ch];
                bw.writeBits(static_cast<uint32_t>(gc.part2_3_length), 12);
                bw.writeBits(static_cast<uint32_t>(gc.big_values), 9);
                bw.writeBits(static_cast<uint32_t>(gc.global_gain), 8);
                bw.writeBits(static_cast<uint32_t>(gc.scalefac_compress), 4);
                bw.writeBits(0, 1);      // window_switching_flag = 0
                bw.writeBits(static_cast<uint32_t>(gc.table_select[0]), 5);
                bw.writeBits(static_cast<uint32_t>(gc.table_select[1]), 5);
                bw.writeBits(static_cast<uint32_t>(gc.table_select[2]), 5);
                bw.writeBits(static_cast<uint32_t>(gc.region0_count), 4);
                bw.writeBits(static_cast<uint32_t>(gc.region1_count), 3);
                bw.writeBits(static_cast<uint32_t>(gc.preflag), 1);
                bw.writeBits(static_cast<uint32_t>(gc.scalefac_scale), 1);
                bw.writeBits(static_cast<uint32_t>(gc.count1table_select), 1);
            }
        }

        // Main data: scalefactors (all zero, so slen1=slen2=0 → 0 bits) + Huffman
        for (int gr = 0; gr < 2; ++gr)
        {
            for (int ch = 0; ch < nch; ++ch)
            {
                auto& gc = si.gr[gr][ch];
                int bigEnd = gc.big_values * 2;

                // Region boundaries
                int r0 = gc.region0_count + 1;
                int r1 = gc.region0_count + gc.region1_count + 2;
                if (r0 > bands.longCount) r0 = bands.longCount;
                if (r1 > bands.longCount) r1 = bands.longCount;
                int reg0End = std::min(bands.longBands[r0], bigEnd);
                int reg1End = std::min(bands.longBands[r1], bigEnd);

                encHuffWrite(bw, ix[gr][ch], reg0End, gc.table_select[0]);
                encHuffWrite(bw, ix[gr][ch] + reg0End, reg1End - reg0End, gc.table_select[1]);
                encHuffWrite(bw, ix[gr][ch] + reg1End, bigEnd - reg1End, gc.table_select[2]);

                // Count1 region
                int rzero = 576;
                while (rzero > bigEnd && ix[gr][ch][rzero - 1] == 0) --rzero;
                encCount1Write(bw, ix[gr][ch], bigEnd, rzero, gc.count1table_select);
            }
        }

        // Pad remaining bits with zeros
        bw.padToByte();
        encFrameBuf_.resize(static_cast<size_t>(frameSize), 0);

        // Write frame to file
        outFile_.write(reinterpret_cast<const char*>(encFrameBuf_.data()),
                       static_cast<std::streamsize>(frameSize));
    }

    // ========================================================================
    // Full decode pipeline
    // ========================================================================

    bool decodeAll()
    {
        // [DEBUG] crash-log instrumentation, remove later
        auto LOG = [](const char* fmt, auto... args) {
            if (std::FILE* f = std::fopen("dsparklab_crash.log", "a")) {
                std::fprintf(f, fmt, args...);
                std::fputc('\n', f);
                std::fclose(f);
            }
        };

        int nch = info_.numChannels;
        int64_t totalSamples = info_.numSamples;
        LOG("[Mp3File] decodeAll enter: nch=%d totalSamples=%lld",
            nch, (long long)totalSamples);

        decodedSamples_.resize(nch);
        for (int ch = 0; ch < nch; ++ch)
            decodedSamples_[ch].resize(static_cast<size_t>(totalSamples), 0.0f);
        LOG("[Mp3File] decodeAll: output buffers allocated");

        // Reset channel state
        for (int ch = 0; ch < kChannelsMax; ++ch)
            channelState_[ch] = {};

        // Initialize reservoir
        reservoir_.resize(kMaxReservoir, 0);
        reservoirSize_ = 0;

        // Persistent scalefactors for SCFSI
        int prevScalefac[2][39] = {};

        int64_t sampleIdx = 0;

        const size_t totalFrames = frameOffsets_.size();
        for (size_t frameIdx = 0; frameIdx < totalFrames; ++frameIdx)
        {
            // [DEBUG] log progress every 500 frames
            if ((frameIdx % 500) == 0)
                LOG("[Mp3File] decodeAll progress: frame %zu/%zu",
                    frameIdx, totalFrames);

            size_t frameStart = frameOffsets_[frameIdx];

            FrameHeader hdr {};
            if (!parseFrameHeader(frameStart, hdr))
                continue;

            size_t headerSize = 4 + (hdr.crcProtect ? 2 : 0);

            // Parse side information
            BitReader siBr;
            siBr.init(fileData_.data() + frameStart + headerSize,
                      static_cast<size_t>(hdr.sideInfoSize));

            SideInfo si {};
            if (!parseSideInfo(siBr, hdr, si))
                continue;

            // Main data starts after header + side info
            size_t mainDataStart = frameStart + headerSize + static_cast<size_t>(hdr.sideInfoSize);
            size_t mainDataSize  = static_cast<size_t>(hdr.frameSize) - headerSize -
                                   static_cast<size_t>(hdr.sideInfoSize);

            // Build main_data buffer from reservoir + current frame data
            // Copy current frame's main data into reservoir
            size_t mainDataBegin = static_cast<size_t>(si.main_data_begin);

            // The main_data_begin tells us how many bytes before the current
            // frame's side info the main data starts (in the reservoir).
            std::vector<uint8_t> mainData;

            if (mainDataBegin > 0)
            {
                if (mainDataBegin > reservoirSize_)
                    mainDataBegin = reservoirSize_;

                // Get data from reservoir
                size_t resStart = reservoirSize_ - mainDataBegin;
                mainData.insert(mainData.end(),
                    reservoir_.begin() + static_cast<ptrdiff_t>(resStart),
                    reservoir_.begin() + static_cast<ptrdiff_t>(reservoirSize_));
            }

            // Append current frame's main data
            if (mainDataStart + mainDataSize <= fileData_.size())
            {
                mainData.insert(mainData.end(),
                    fileData_.begin() + static_cast<ptrdiff_t>(mainDataStart),
                    fileData_.begin() + static_cast<ptrdiff_t>(mainDataStart + mainDataSize));
            }

            // Update reservoir: shift and append current frame's data
            if (mainDataSize > 0 && mainDataStart + mainDataSize <= fileData_.size())
            {
                size_t newTotal = reservoirSize_ + mainDataSize;
                if (newTotal > kMaxReservoir)
                {
                    size_t shift = newTotal - kMaxReservoir;
                    if (shift < reservoirSize_)
                    {
                        std::memmove(reservoir_.data(),
                                     reservoir_.data() + shift,
                                     reservoirSize_ - shift);
                        reservoirSize_ -= shift;
                    }
                    else
                    {
                        reservoirSize_ = 0;
                    }
                }
                size_t canCopy = std::min(mainDataSize, kMaxReservoir - reservoirSize_);
                std::memcpy(reservoir_.data() + reservoirSize_,
                            fileData_.data() + mainDataStart, canCopy);
                reservoirSize_ += canCopy;
            }

            // Now decode the two granules
            BitReader mainBr;
            mainBr.init(mainData.data(), mainData.size());

            int scalefac[2][39] = {};

            for (int gr = 0; gr < kGranules; ++gr)
            {
                double xr[2][576] = {};

                for (int ch = 0; ch < nch; ++ch)
                {
                    const auto& gc = si.gr[gr][ch];

                    // Decode scalefactors
                    if (gr == 1)
                    {
                        // Copy previous granule's scalefactors for SCFSI bands
                        for (int i = 0; i < 39; ++i)
                            scalefac[ch][i] = prevScalefac[ch][i];
                    }

                    size_t sfBits = 0;
                    size_t posBeforeSf = mainBr.getPos();
                    decodeScalefactors(mainBr, gc, gr, ch, si.scfsi[ch],
                                       scalefac[ch], sfBits);

                    // Huffman decode spectral values
                    int is[576] = {};

                    huffmanDecode(mainBr, gc, hdr, is);

                    // Ensure we've consumed exactly part2_3_length bits from the start
                    mainBr.setPos(posBeforeSf + static_cast<size_t>(gc.part2_3_length));

                    // Requantize
                    requantize(is, scalefac[ch], gc, hdr, xr[ch]);

                    // Save scalefactors for next granule's SCFSI
                    if (gr == 0)
                    {
                        for (int i = 0; i < 39; ++i)
                            prevScalefac[ch][i] = scalefac[ch][i];
                    }
                }

                // Stereo processing
                if (nch == 2)
                    stereoProcess(xr, si.gr[gr], hdr, scalefac);

                // Per-channel post-processing
                for (int ch = 0; ch < nch; ++ch)
                {
                    // Reorder (short blocks)
                    reorder(xr[ch], si.gr[gr][ch], hdr);

                    // Alias reduction
                    aliasReduction(xr[ch], si.gr[gr][ch]);

                    // IMDCT + overlap-add
                    double imdctOut[576];
                    imdct(xr[ch], si.gr[gr][ch], ch, imdctOut);

                    // Frequency inversion
                    frequencyInversion(imdctOut);

                    // Synthesis filterbank -> PCM
                    float pcm[576];
                    synthesize(imdctOut, ch, pcm);

                    // Store output samples
                    for (int i = 0; i < kSamplesPerGranule; ++i)
                    {
                        int64_t outIdx = sampleIdx + i;
                        if (outIdx < totalSamples)
                            decodedSamples_[ch][static_cast<size_t>(outIdx)] = pcm[i];
                    }
                }

                sampleIdx += kSamplesPerGranule;
            }
        }

        LOG("[Mp3File] decodeAll: loop done, sampleIdx=%lld", (long long)sampleIdx);
        return true;
    }
};

} // namespace dspark
