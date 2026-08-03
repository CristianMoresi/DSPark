// DSPark Tests - IO
// WavFile and Mp3File round-trip tests

#include "dspark_test.h"
#include "TestSignals.h"

#include "../IO/WavFile.h"
#include "../IO/Mp3File.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// Helper: create a temp file path
static const char* tempWav16()  { return "dspark_test_16.wav"; }
static const char* tempWav24()  { return "dspark_test_24.wav"; }
static const char* tempWavF32() { return "dspark_test_f32.wav"; }
static const char* tempWavStereo() { return "dspark_test_stereo.wav"; }
static const char* tempMp3()    { return "dspark_test.mp3"; }

// Cleanup helper
struct FileCleanup
{
    const char* path;
    ~FileCleanup() { std::remove(path); }
};

// ============================================================================
// WavFile - 16-bit PCM
// ============================================================================

DSPARK_TEST(WavFile_16bit_roundtrip)
{
    FileCleanup cleanup { tempWav16() };

    constexpr int N = 4096;
    constexpr int CH = 1;

    // Write
    {
        WavFile writer;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = CH;
        info.bitsPerSample = 16;
        info.isFloatingPoint = false;
        info.numSamples = N;

        EXPECT_TRUE(writer.openWrite(tempWav16(), info));

        AudioBuffer<float> buf;
        buf.resize(CH, N);
        generateSine(buf.getChannel(0), N, 440.0f, 44100.0f, 0.9f);

        EXPECT_TRUE(writer.writeSamples(std::as_const(buf).toView()));
        writer.close();
    }

    // Read back
    {
        WavFile reader;
        EXPECT_TRUE(reader.openRead(tempWav16()));

        auto info = reader.getInfo();
        EXPECT_EQ(info.numChannels, CH);
        EXPECT_NEAR(info.sampleRate, 44100.0, 1.0);
        EXPECT_EQ(info.bitsPerSample, 16);

        AudioBuffer<float> buf;
        buf.resize(info.numChannels, static_cast<int>(info.numSamples));
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        // Generate expected
        std::vector<float> expected(N);
        generateSine(expected.data(), N, 440.0f, 44100.0f, 0.9f);

        // 16-bit quantization error: 1 LSB = 1/32768 ~ 3e-5
        for (int i = 0; i < N; ++i)
            EXPECT_NEAR(buf.getChannel(0)[i], expected[i], 0.001f);
    }
}

// ============================================================================
// WavFile - 24-bit PCM
// ============================================================================

DSPARK_TEST(WavFile_24bit_roundtrip)
{
    FileCleanup cleanup { tempWav24() };

    constexpr int N = 2048;

    {
        WavFile writer;
        AudioFileInfo info;
        info.sampleRate = 48000.0;
        info.numChannels = 1;
        info.bitsPerSample = 24;
        info.isFloatingPoint = false;
        info.numSamples = N;

        EXPECT_TRUE(writer.openWrite(tempWav24(), info));

        AudioBuffer<float> buf;
        buf.resize(1, N);
        generateSine(buf.getChannel(0), N, 1000.0f, 48000.0f, 0.8f);

        EXPECT_TRUE(writer.writeSamples(std::as_const(buf).toView()));
        writer.close();
    }

    {
        WavFile reader;
        EXPECT_TRUE(reader.openRead(tempWav24()));

        auto info = reader.getInfo();
        EXPECT_EQ(info.bitsPerSample, 24);

        AudioBuffer<float> buf;
        buf.resize(1, static_cast<int>(info.numSamples));
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        std::vector<float> expected(N);
        generateSine(expected.data(), N, 1000.0f, 48000.0f, 0.8f);

        // 24-bit: 1 LSB = 1/8388608 ~ 1.2e-7
        for (int i = 0; i < N; ++i)
            EXPECT_NEAR(buf.getChannel(0)[i], expected[i], 0.0001f);
    }
}

// ============================================================================
// WavFile - 32-bit float
// ============================================================================

DSPARK_TEST(WavFile_float32_exact_roundtrip)
{
    FileCleanup cleanup { tempWavF32() };

    constexpr int N = 1024;

    AudioBuffer<float> original;
    original.resize(1, N);
    generateSine(original.getChannel(0), N, 440.0f, 44100.0f);

    {
        WavFile writer;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 1;
        info.bitsPerSample = 32;
        info.isFloatingPoint = true;
        info.numSamples = N;

        EXPECT_TRUE(writer.openWrite(tempWavF32(), info));
        EXPECT_TRUE(writer.writeSamples(std::as_const(original).toView()));
        writer.close();
    }

    {
        WavFile reader;
        EXPECT_TRUE(reader.openRead(tempWavF32()));

        AudioBuffer<float> buf;
        buf.resize(1, N);
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        // 32-bit float should be bit-perfect
        for (int i = 0; i < N; ++i)
            EXPECT_NEAR(buf.getChannel(0)[i], original.getChannel(0)[i], 1e-7f);
    }
}

// ============================================================================
// WavFile - Stereo
// ============================================================================

DSPARK_TEST(WavFile_stereo_channels_preserved)
{
    FileCleanup cleanup { tempWavStereo() };

    constexpr int N = 2048;

    {
        WavFile writer;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 2;
        info.bitsPerSample = 16;
        info.isFloatingPoint = false;
        info.numSamples = N;

        EXPECT_TRUE(writer.openWrite(tempWavStereo(), info));

        AudioBuffer<float> buf;
        buf.resize(2, N);
        generateSine(buf.getChannel(0), N, 440.0f, 44100.0f);
        generateSine(buf.getChannel(1), N, 880.0f, 44100.0f);

        EXPECT_TRUE(writer.writeSamples(std::as_const(buf).toView()));
        writer.close();
    }

    {
        WavFile reader;
        EXPECT_TRUE(reader.openRead(tempWavStereo()));

        auto info = reader.getInfo();
        EXPECT_EQ(info.numChannels, 2);

        AudioBuffer<float> buf;
        buf.resize(2, static_cast<int>(info.numSamples));
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        // L should be 440 Hz, R should be 880 Hz
        float magL440 = measureFrequencyMagnitude(buf.getChannel(0), N, 440.0f, 44100.0f);
        float magL880 = measureFrequencyMagnitude(buf.getChannel(0), N, 880.0f, 44100.0f);
        float magR440 = measureFrequencyMagnitude(buf.getChannel(1), N, 440.0f, 44100.0f);
        float magR880 = measureFrequencyMagnitude(buf.getChannel(1), N, 880.0f, 44100.0f);

        EXPECT_GT(magL440, magL880 * 5.0f); // L is 440 Hz dominant
        EXPECT_GT(magR880, magR440 * 5.0f); // R is 880 Hz dominant
    }
}

// ============================================================================
// Mp3File - Encode/Decode round-trip
// ============================================================================

DSPARK_TEST(Mp3File_roundtrip_similarity)
{
    FileCleanup cleanup { tempMp3() };

    constexpr int N = 44100; // 1 second
    constexpr int CH = 2;

    // Generate original
    std::vector<float> origL(N), origR(N);
    generateSine(origL.data(), N, 440.0f, 44100.0f, 0.8f);
    generateSine(origR.data(), N, 880.0f, 44100.0f, 0.6f);

    // Encode
    {
        Mp3File writer;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = CH;
        info.bitsPerSample = 192; // kbps for MP3
        info.numSamples = N;

        EXPECT_TRUE(writer.openWrite(tempMp3(), info));

        AudioBuffer<float> buf;
        buf.resize(CH, N);
        std::copy(origL.begin(), origL.end(), buf.getChannel(0));
        std::copy(origR.begin(), origR.end(), buf.getChannel(1));

        EXPECT_TRUE(writer.writeSamples(std::as_const(buf).toView()));
        writer.close();
    }

    // Decode
    {
        Mp3File reader;
        EXPECT_TRUE(reader.openRead(tempMp3()));

        auto info = reader.getInfo();
        EXPECT_EQ(info.numChannels, CH);
        EXPECT_NEAR(info.sampleRate, 44100.0, 1.0);

        int readLen = static_cast<int>(info.numSamples);
        AudioBuffer<float> buf;
        buf.resize(CH, readLen);
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        // MP3 is lossy and adds encoder delay (~576-2304 samples).
        // Verify basic signal integrity rather than exact alignment.
        if (readLen > 4096)
        {
            // Check the decoded signal has non-trivial energy
            float rms = measureRMS(buf.getChannel(0), readLen);
            EXPECT_GT(rms, 0.05f);

            // Check output is bounded and valid
            EXPECT_NO_NAN(buf.getChannel(0), readLen);
            EXPECT_BOUNDED(buf.getChannel(0), readLen, -1.5f, 1.5f);
        }
    }
}

DSPARK_TEST(Mp3File_metadata_preserved)
{
    FileCleanup cleanup { tempMp3() };

    {
        Mp3File writer;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 1;
        info.bitsPerSample = 128; // kbps
        info.numSamples = 44100;

        EXPECT_TRUE(writer.openWrite(tempMp3(), info));

        AudioBuffer<float> buf;
        buf.resize(1, 44100);
        generateSine(buf.getChannel(0), 44100, 440.0f, 44100.0f);

        EXPECT_TRUE(writer.writeSamples(std::as_const(buf).toView()));
        writer.close();
    }

    {
        Mp3File reader;
        EXPECT_TRUE(reader.openRead(tempMp3()));
        auto info = reader.getInfo();

        EXPECT_NEAR(info.sampleRate, 44100.0, 1.0);
        // Channel count from decoded stream
        EXPECT_TRUE(info.numChannels >= 1);
        EXPECT_TRUE(info.numSamples > 0);

        reader.close();
    }
}

// ============================================================================
// AudioFileInfo::toSpec - dimension mapping and degenerate inputs
// ============================================================================

DSPARK_TEST(AudioFileInfo_toSpec_maps_and_sanitizes)
{
    // Normal file, explicit block size
    AudioFileInfo info;
    info.sampleRate = 48000.0;
    info.numChannels = 2;
    info.numSamples = 100000;

    AudioSpec spec = info.toSpec(512);
    EXPECT_TRUE(spec.isValid());
    EXPECT_NEAR(spec.sampleRate, 48000.0, 1e-9);
    EXPECT_EQ(spec.maxBlockSize, 512);
    EXPECT_EQ(spec.numChannels, 2);

    // Offline mode (no block size) uses the full file length
    AudioSpec offline = info.toSpec();
    EXPECT_TRUE(offline.isValid());
    EXPECT_EQ(offline.maxBlockSize, 100000);

    // Negative block size behaves like "unspecified" (full length)
    EXPECT_EQ(info.toSpec(-5).maxBlockSize, 100000);

    // Frame counts beyond INT_MAX clamp instead of overflowing the cast
    AudioFileInfo huge = info;
    huge.numSamples = int64_t(1) << 40;
    EXPECT_EQ(huge.toSpec().maxBlockSize, std::numeric_limits<int>::max());

    // Corrupt negative frame count clamps to 0 -> invalid spec, never a
    // negative block size masquerading as usable
    AudioFileInfo corrupt = info;
    corrupt.numSamples = -1234;
    AudioSpec bad = corrupt.toSpec();
    EXPECT_EQ(bad.maxBlockSize, 0);
    EXPECT_TRUE(!bad.isValid());

    // Default-constructed info (unopened file) yields an invalid spec that
    // framework processors reject as a no-op
    AudioFileInfo unopened;
    EXPECT_TRUE(!unopened.toSpec().isValid());
}

// ============================================================================
// WavFile - robustness against unusual and hostile files
// ============================================================================

namespace {

void wavLE16(std::ofstream& f, uint16_t v)
{
    char b[2] = { char(v & 0xFF), char((v >> 8) & 0xFF) };
    f.write(b, 2);
}

void wavLE32(std::ofstream& f, uint32_t v)
{
    char b[4] = { char(v & 0xFF), char((v >> 8) & 0xFF),
                  char((v >> 16) & 0xFF), char((v >> 24) & 0xFF) };
    f.write(b, 4);
}

} // namespace

DSPARK_TEST(WavFile_extensible_24in32_reads_exact)
{
    // WAVE_FORMAT_EXTENSIBLE, 32-bit container, 24 valid bits: the classic
    // "24-in-32" layout (valid bits left-justified). The container width
    // governs the frame stride; decoding by container reproduces the values
    // exactly. The old reader adopted validBits as the frame size and
    // mis-strided the whole file into garbage.
    FileCleanup cleanup { "dspark_test_24in32.wav" };

    const int32_t src[4] = {
        int32_t(0x40000000),  // +0.5
        int32_t(0xC0000000),  // -0.5
        int32_t(0x7FFFFF00),  // ~ +1.0 at 24-bit precision
        0
    };
    {
        std::ofstream f("dspark_test_24in32.wav", std::ios::binary | std::ios::trunc);
        const uint32_t dataSize = 4 * 4;
        f.write("RIFF", 4); wavLE32(f, 4 + 8 + 40 + 8 + dataSize); f.write("WAVE", 4);
        f.write("fmt ", 4); wavLE32(f, 40);
        wavLE16(f, 0xFFFE); wavLE16(f, 1); wavLE32(f, 48000); wavLE32(f, 48000 * 4);
        wavLE16(f, 4); wavLE16(f, 32);
        wavLE16(f, 22);          // cbSize
        wavLE16(f, 24);          // validBitsPerSample
        wavLE32(f, 0);           // channelMask
        wavLE16(f, 1);           // subformat PCM
        static const uint8_t guid[14] = { 0,0,0,0, 0x10,0,0x80,0, 0,0xAA,0,0x38,0x9B,0x71 };
        f.write(reinterpret_cast<const char*>(guid), 14);
        f.write("data", 4); wavLE32(f, dataSize);
        for (int32_t s : src) wavLE32(f, uint32_t(s));
    }

    WavFile r;
    EXPECT_TRUE(r.openRead("dspark_test_24in32.wav"));
    auto info = r.getInfo();
    EXPECT_EQ(info.bitsPerSample, 32);     // container width, not valid bits
    EXPECT_EQ(int(info.numSamples), 4);    // 16 bytes / 4-byte frames

    AudioBuffer<float> buf;
    buf.resize(1, 4);
    EXPECT_TRUE(r.readSamples(buf.toView()));
    r.close();

    EXPECT_NEAR(buf.getChannel(0)[0],  0.5f, 1e-6f);
    EXPECT_NEAR(buf.getChannel(0)[1], -0.5f, 1e-6f);
    EXPECT_NEAR(buf.getChannel(0)[2],  1.0f, 1e-4f);
    EXPECT_NEAR(buf.getChannel(0)[3],  0.0f, 1e-9f);
}

DSPARK_TEST(WavFile_invalid_openwrite_preserves_target)
{
    // openWrite must validate the requested format BEFORE touching the
    // destination: the old code opened with ios::trunc first, so a rejected
    // format destroyed the existing file.
    FileCleanup cleanup { "dspark_test_precious.wav" };

    {
        WavFile w;
        AudioFileInfo gi;
        gi.sampleRate = 44100.0; gi.numChannels = 1; gi.bitsPerSample = 16;
        EXPECT_TRUE(w.openWrite("dspark_test_precious.wav", gi));
        AudioBuffer<float> b; b.resize(1, 100);
        for (int i = 0; i < 100; ++i) b.getChannel(0)[i] = 0.25f;
        EXPECT_TRUE(w.writeSamples(std::as_const(b).toView()));
        w.close();
    }

    auto sizeOf = [](const char* p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        return static_cast<long long>(f.tellg());
    };
    const long long before = sizeOf("dspark_test_precious.wav");
    EXPECT_TRUE(before > 0);

    WavFile w2;
    AudioFileInfo bad;
    bad.sampleRate = 44100.0; bad.numChannels = 1; bad.bitsPerSample = 13;
    EXPECT_TRUE(!w2.openWrite("dspark_test_precious.wav", bad));
    AudioFileInfo nanRate;
    nanRate.sampleRate = std::numeric_limits<double>::quiet_NaN();
    nanRate.numChannels = 1; nanRate.bitsPerSample = 16;
    EXPECT_TRUE(!w2.openWrite("dspark_test_precious.wav", nanRate));
    AudioFileInfo tooWide;
    tooWide.sampleRate = 44100.0; tooWide.numChannels = 1000; tooWide.bitsPerSample = 16;
    EXPECT_TRUE(!w2.openWrite("dspark_test_precious.wav", tooWide));

    EXPECT_EQ(sizeOf("dspark_test_precious.wav"), before);

    // A valid file still reads back fine afterwards
    WavFile r;
    EXPECT_TRUE(r.openRead("dspark_test_precious.wav"));
    EXPECT_EQ(int(r.getInfo().numSamples), 100);
    r.close();
}

DSPARK_TEST(WavFile_unusual_headers_handled)
{
    FileCleanup c1 { "dspark_test_pcm64.wav" };
    FileCleanup c2 { "dspark_test_oddorder.wav" };
    FileCleanup c3 { "dspark_test_junk.bin" };

    // (a) Integer PCM claiming 64 bits: no such WAV format exists and there
    // is no decoder branch for it. The old reader accepted the header and
    // then "read" nothing while reporting success.
    {
        std::ofstream f("dspark_test_pcm64.wav", std::ios::binary | std::ios::trunc);
        const uint32_t dataSize = 8 * 8;
        f.write("RIFF", 4); wavLE32(f, 4 + 8 + 16 + 8 + dataSize); f.write("WAVE", 4);
        f.write("fmt ", 4); wavLE32(f, 16);
        wavLE16(f, 1); wavLE16(f, 1); wavLE32(f, 48000); wavLE32(f, 48000 * 8);
        wavLE16(f, 8); wavLE16(f, 64);
        f.write("data", 4); wavLE32(f, dataSize);
        for (int i = 0; i < 16; ++i) wavLE32(f, 0x11223344u);
    }
    {
        WavFile r;
        EXPECT_TRUE(!r.openRead("dspark_test_pcm64.wav"));
    }

    // (b) Odd-sized data chunk BEFORE fmt (legal but unusual): RIFF requires
    // a pad byte after odd chunks; without skipping it the walker lands one
    // byte off and never finds fmt, rejecting a legal file.
    {
        std::ofstream f("dspark_test_oddorder.wav", std::ios::binary | std::ios::trunc);
        const uint32_t dataSize = 3;
        f.write("RIFF", 4); wavLE32(f, 4 + 8 + dataSize + 1 + 8 + 16); f.write("WAVE", 4);
        f.write("data", 4); wavLE32(f, dataSize);
        const uint8_t smp[3] = { 128, 255, 0 };
        f.write(reinterpret_cast<const char*>(smp), 3);
        f.put(0); // RIFF pad byte
        f.write("fmt ", 4); wavLE32(f, 16);
        wavLE16(f, 1); wavLE16(f, 1); wavLE32(f, 48000); wavLE32(f, 48000);
        wavLE16(f, 1); wavLE16(f, 8);
    }
    {
        WavFile r;
        EXPECT_TRUE(r.openRead("dspark_test_oddorder.wav"));
        EXPECT_EQ(int(r.getInfo().numSamples), 3);
        EXPECT_EQ(r.getInfo().bitsPerSample, 8);
        AudioBuffer<float> b; b.resize(1, 3);
        EXPECT_TRUE(r.readSamples(b.toView()));
        EXPECT_NEAR(b.getChannel(0)[0], 0.0f, 1e-6f);            // 128 -> 0
        EXPECT_NEAR(b.getChannel(0)[1], 127.0f / 128.0f, 1e-6f); // 255 -> ~+1
        EXPECT_NEAR(b.getChannel(0)[2], -1.0f, 1e-6f);           // 0 -> -1
        r.close();
    }

    // (c) Truncated header (no data chunk): open fails AND getInfo() honours
    // the interface contract ("default values if no file is open") instead
    // of leaking the partial parse.
    {
        std::ofstream f("dspark_test_junk.bin", std::ios::binary | std::ios::trunc);
        f.write("RIFF", 4); wavLE32(f, 100); f.write("WAVE", 4);
        f.write("fmt ", 4); wavLE32(f, 16);
        wavLE16(f, 1); wavLE16(f, 7); wavLE32(f, 12345); wavLE32(f, 0);
        wavLE16(f, 0); wavLE16(f, 16);
    }
    {
        WavFile r;
        EXPECT_TRUE(!r.openRead("dspark_test_junk.bin"));
        auto info = r.getInfo();
        EXPECT_EQ(info.numChannels, 0u);
        EXPECT_NEAR(info.sampleRate, 44100.0, 1e-9);
    }
}

DSPARK_TEST(WavFile_narrow_view_write_pads_silence)
{
    // Writing a mono view into a stereo file must fill the missing channel
    // with silence (Mp3File contract). The old writer read src channel 1
    // out of range: assert in debug, out-of-bounds read (measured segfault)
    // in release.
    FileCleanup cleanup { "dspark_test_narrow.wav" };

    {
        WavFile w;
        AudioFileInfo si;
        si.sampleRate = 44100.0; si.numChannels = 2; si.bitsPerSample = 16;
        EXPECT_TRUE(w.openWrite("dspark_test_narrow.wav", si));
        AudioBuffer<float> mono; mono.resize(1, 64);
        for (int i = 0; i < 64; ++i) mono.getChannel(0)[i] = 0.5f;
        EXPECT_TRUE(w.writeSamples(std::as_const(mono).toView()));
        w.close();
    }

    WavFile r;
    EXPECT_TRUE(r.openRead("dspark_test_narrow.wav"));
    AudioBuffer<float> b; b.resize(2, 64);
    EXPECT_TRUE(r.readSamples(b.toView()));
    r.close();
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_NEAR(b.getChannel(0)[i], 0.5f, 1e-3f);
        EXPECT_NEAR(b.getChannel(1)[i], 0.0f, 1e-9f);
    }
}

DSPARK_TEST(WavFile_8bit_and_int32_roundtrip)
{
    // Coverage for the two integer formats the suite never exercised.
    FileCleanup c1 { "dspark_test_8.wav" };
    FileCleanup c2 { "dspark_test_i32.wav" };

    constexpr int N = 1024;
    std::vector<float> expected(N);
    generateSine(expected.data(), N, 997.0f, 44100.0f, 0.8f);

    // 8-bit unsigned PCM (write scale 127, read scale 1/128 -> ~1% worst case)
    {
        WavFile w;
        AudioFileInfo info;
        info.sampleRate = 44100.0; info.numChannels = 1; info.bitsPerSample = 8;
        EXPECT_TRUE(w.openWrite("dspark_test_8.wav", info));
        AudioBuffer<float> buf; buf.resize(1, N);
        std::copy(expected.begin(), expected.end(), buf.getChannel(0));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();

        WavFile r;
        EXPECT_TRUE(r.openRead("dspark_test_8.wav"));
        EXPECT_EQ(r.getInfo().bitsPerSample, 8);
        AudioBuffer<float> back; back.resize(1, N);
        EXPECT_TRUE(r.readSamples(back.toView()));
        r.close();
        for (int i = 0; i < N; ++i)
            EXPECT_NEAR(back.getChannel(0)[i], expected[i], 0.02f);
    }

    // 32-bit integer PCM (float mantissa limits accuracy to ~2^-24)
    {
        WavFile w;
        AudioFileInfo info;
        info.sampleRate = 44100.0; info.numChannels = 1; info.bitsPerSample = 32;
        EXPECT_TRUE(w.openWrite("dspark_test_i32.wav", info));
        AudioBuffer<float> buf; buf.resize(1, N);
        std::copy(expected.begin(), expected.end(), buf.getChannel(0));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();

        WavFile r;
        EXPECT_TRUE(r.openRead("dspark_test_i32.wav"));
        EXPECT_EQ(r.getInfo().bitsPerSample, 32);
        EXPECT_TRUE(!r.getInfo().isFloatingPoint);
        AudioBuffer<float> back; back.resize(1, N);
        EXPECT_TRUE(r.readSamples(back.toView()));
        r.close();
        for (int i = 0; i < N; ++i)
            EXPECT_NEAR(back.getChannel(0)[i], expected[i], 2e-7f);
    }
}

// ============================================================================
// Mp3File - part2_3_length budget (ISO 11172-3, 2.4.1.7)
// ============================================================================

namespace {

// Minimal MSB-first bit packer for hand-crafting an MP3 frame.
struct Mp3TestBits
{
    std::vector<uint8_t> bytes;
    size_t bitPos = 0;
    void put(uint32_t val, int n)
    {
        for (int i = n - 1; i >= 0; --i)
        {
            size_t byteIdx = bitPos >> 3;
            while (byteIdx >= bytes.size()) bytes.push_back(0);
            int bitIdx = 7 - int(bitPos & 7);
            if ((val >> unsigned(i)) & 1u) bytes[byteIdx] |= uint8_t(1u << bitIdx);
            ++bitPos;
        }
    }
};

} // namespace

DSPARK_TEST(Mp3File_part23_budget_includes_scalefactor_bits)
{
    // part2_3_length counts scalefactor bits AND Huffman bits. This frame
    // carries scalefac_compress=15 (74 scalefactor bits), big_values=0 and
    // part2_3_length=74: the granule holds NO Huffman data at all, so a
    // correct decoder produces exact silence. The filler bits after the
    // scalefactors are valid count1 quads on purpose - a decoder that
    // budgets part2_3_length AFTER the scalefactors (the old bug) reads 74
    // bits of them as spurious spectral values and produces audible energy
    // (measured 2568 total |pcm| across 4 frames before the fix).
    FileCleanup cleanup { "dspark_test_crafted.mp3" };

    Mp3TestBits b;
    // Header: MPEG-1 Layer III, no CRC, 64 kbps, 48 kHz, mono
    b.put(0x7FF, 11); b.put(3, 2); b.put(1, 2); b.put(1, 1);
    b.put(5, 4); b.put(1, 2); b.put(0, 1); b.put(0, 1);
    b.put(3, 2); b.put(0, 2); b.put(0, 1); b.put(1, 1); b.put(0, 2);
    // Side info (mono, 17 bytes)
    b.put(0, 9); b.put(0, 5); b.put(0, 4);
    // granule 0: 74 scalefactor bits only
    b.put(74, 12); b.put(0, 9); b.put(210, 8); b.put(15, 4); b.put(0, 1);
    b.put(1, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
    b.put(0, 1); b.put(0, 1); b.put(0, 1);
    // granule 1: fully silent
    b.put(0, 12); b.put(0, 9); b.put(210, 8); b.put(0, 4); b.put(0, 1);
    b.put(0, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
    b.put(0, 1); b.put(0, 1); b.put(0, 1);
    // Main data: 74 zero scalefactor bits, then count1-table-A quads
    // (code 0111 + sign 0) that must never be read.
    b.put(0, 32); b.put(0, 32); b.put(0, 10);
    for (int i = 0; i < 40; ++i) b.put(0b01110, 5);

    std::vector<uint8_t> frame = b.bytes;
    frame.resize(192, 0); // 144 * 64000 / 48000

    {
        std::ofstream f("dspark_test_crafted.mp3", std::ios::binary | std::ios::trunc);
        for (int i = 0; i < 4; ++i)
            f.write(reinterpret_cast<const char*>(frame.data()),
                    static_cast<std::streamsize>(frame.size()));
    }

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_crafted.mp3"));
    auto info = r.getInfo();
    EXPECT_NEAR(info.sampleRate, 48000.0, 1e-9);
    EXPECT_EQ(info.numChannels, 1u);
    EXPECT_EQ(info.bitsPerSample, 32); // delivery format, not "16"
    EXPECT_TRUE(info.isFloatingPoint);

    AudioBuffer<float> buf;
    buf.resize(1, static_cast<int>(info.numSamples));
    EXPECT_TRUE(r.readSamples(buf.toView()));
    r.close();

    double energy = 0.0;
    for (int i = 0; i < static_cast<int>(info.numSamples); ++i)
        energy += std::fabs(double(buf.getChannel(0)[i]));
    EXPECT_NEAR(energy, 0.0, 1e-12); // exact silence; the old decoder measured 2568
}

// ============================================================================
// Mp3File - hostile-input and bitstream-conformance pins (file I/O hardening)
// ============================================================================

// Builds one MPEG-1 Layer III frame: no CRC, 64 kbps, 48 kHz, mono, 192 bytes.
// `writeGranule(gr, bits)` contributes that granule's side-info fields and
// main data; the caller supplies both through the two callbacks.
template <typename SideInfoFn, typename MainDataFn>
static std::vector<uint8_t> mp3CraftFrame(SideInfoFn sideInfo, MainDataFn mainData,
                                          uint32_t scfsi = 0)
{
    Mp3TestBits b;
    b.put(0x7FF, 11); b.put(3, 2); b.put(1, 2); b.put(1, 1);   // sync, MPEG-1, layer III, no CRC
    b.put(5, 4); b.put(1, 2); b.put(0, 1); b.put(0, 1);        // 64 kbps, 48 kHz, no padding
    b.put(3, 2); b.put(0, 2); b.put(0, 1); b.put(1, 1); b.put(0, 2); // mono
    b.put(0, 9); b.put(0, 5); b.put(scfsi, 4);                 // main_data_begin, private, scfsi
    sideInfo(b);
    while (b.bitPos < 21 * 8) b.put(0, 1);                     // pad to the end of the side info
    mainData(b);
    std::vector<uint8_t> frame = b.bytes;
    frame.resize(192, 0);
    return frame;
}

static void mp3WriteFile(const char* path, const std::vector<uint8_t>& frame, int repeats)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    for (int i = 0; i < repeats; ++i)
        f.write(reinterpret_cast<const char*>(frame.data()),
                static_cast<std::streamsize>(frame.size()));
}

static double mp3DecodeEnergy(const char* path)
{
    Mp3File r;
    if (!r.openRead(path)) return -1.0;
    auto info = r.getInfo();
    AudioBuffer<float> buf;
    buf.resize(static_cast<int>(info.numChannels), static_cast<int>(info.numSamples));
    if (!r.readSamples(buf.toView())) return -1.0;
    double e = 0.0;
    for (int i = 0; i < static_cast<int>(info.numSamples); ++i)
    {
        const double v = double(buf.getChannel(0)[i]);
        e += v * v;
    }
    r.close();
    return e;
}

// A stream truncated inside a frame: the last frame contributes only its four
// sync bytes, so its 17-byte side info is not in the file at all. The frame
// scan only proves the sync word is present, so the decoder used to hand
// parseSideInfo a reader over 17 bytes past the end of the buffer -- a heap
// over-read on untrusted input, reproduced under AddressSanitizer as
// "heap-buffer-overflow ... READ of size 1 ... in BitReader::readBits" and
// minimised to this shape. Decoding must simply skip the incomplete frame.
DSPARK_TEST(Mp3File_frame_truncated_before_side_info_is_not_overread)
{
    FileCleanup cleanup { "dspark_test_trunc.mp3" };

    auto silentSide = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            b.put(0, 12); b.put(0, 9); b.put(210, 8); b.put(0, 4); b.put(0, 1);
            b.put(0, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
            b.put(0, 1); b.put(0, 1); b.put(0, 1);
        }
    };
    std::vector<uint8_t> frame = mp3CraftFrame(silentSide, [](Mp3TestBits&) {});

    {
        std::ofstream f("dspark_test_trunc.mp3", std::ios::binary | std::ios::trunc);
        for (int i = 0; i < 2; ++i)
            f.write(reinterpret_cast<const char*>(frame.data()),
                    static_cast<std::streamsize>(frame.size()));
        f.write(reinterpret_cast<const char*>(frame.data()), 4); // header only
    }

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_trunc.mp3"));
    auto info = r.getInfo();
    EXPECT_EQ(info.numChannels, 1u);
    EXPECT_TRUE(info.numSamples > 0);

    AudioBuffer<float> buf;
    buf.resize(1, static_cast<int>(info.numSamples));
    EXPECT_TRUE(r.readSamples(buf.toView()));
    EXPECT_NO_NAN(buf.getChannel(0), static_cast<int>(info.numSamples));
    r.close();
}

// ISO 11172-3 2.4.1.7 orders a big_values pair as hcod, linbits(x), sign(x),
// linbits(y), sign(y). Reading both escapes before both signs consumes the same
// number of bits, so nothing looks wrong locally, but it assigns them to the
// wrong fields whenever y escapes while x is non-zero -- and from there the
// rest of the granule is garbage.
//
// Two frames differ only in the 10-bit linbits field of the single coded pair
// (x=1, y=15+linbits, Huffman table 22). Because both carry the same
// global_gain and an all-zero scalefactor set, the ratio of their decoded
// energies is fixed by the ISO requantisation formula alone and is independent
// of any filterbank scaling: (1038/15)^(8/3) = 8.05e4. Reading the escape
// before the sign yields y=526 instead of 1038 and a ratio of 1.3e4.
DSPARK_TEST(Mp3File_escape_pair_reads_linbits_then_sign)
{
    FileCleanup c1 { "dspark_test_esc_hi.mp3" };
    FileCleanup c2 { "dspark_test_esc_lo.mp3" };

    auto side = [](Mp3TestBits& b) {
        // granule 0 carries the pair: part2_3_length = 8 + 1 + 10 + 1 = 20 bits
        b.put(20, 12); b.put(1, 9); b.put(150, 8); b.put(0, 4); b.put(0, 1);
        b.put(22, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
        b.put(0, 1); b.put(0, 1); b.put(0, 1);
        // granule 1 silent
        b.put(0, 12); b.put(0, 9); b.put(210, 8); b.put(0, 4); b.put(0, 1);
        b.put(0, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
        b.put(0, 1); b.put(0, 1); b.put(0, 1);
    };
    // kHuff16 entry for (x=1, y=15) is length 8, code 0x09.
    auto payload = [](uint32_t linbits) {
        return [linbits](Mp3TestBits& b) {
            b.put(0x09, 8);       // hcod (1, 15)
            b.put(0, 1);          // sign(x) -> x = +1
            b.put(linbits, 10);   // linbits(y)
            b.put(0, 1);          // sign(y) -> y = +(15 + linbits)
        };
    };

    mp3WriteFile("dspark_test_esc_hi.mp3", mp3CraftFrame(side, payload(1023)), 6);
    mp3WriteFile("dspark_test_esc_lo.mp3", mp3CraftFrame(side, payload(0)), 6);

    const double eHi = mp3DecodeEnergy("dspark_test_esc_hi.mp3");
    const double eLo = mp3DecodeEnergy("dspark_test_esc_lo.mp3");
    EXPECT_GT(eHi, 0.0);
    EXPECT_GT(eLo, 0.0);

    const double ratio = eHi / eLo;
    EXPECT_GT(ratio, 4.0e4);   // correct: 8.05e4, escape-before-sign: 1.3e4
    EXPECT_LT(ratio, 1.6e5);
}

// scfsi (ISO 11172-3 2.4.2.7): a set scfsi band makes granule 1 REUSE granule
// 0's scalefactors for that band group instead of re-sending them. Clearing the
// scalefactor array at the top of the granule threw the inherited values away
// and requantised those bands with no attenuation at all.
//
// The two files below must decode identically: one inherits scalefactor band
// group 0 through scfsi, the other re-sends the same values explicitly. With
// scalefac[0]=7 and scalefac_scale=0 the inherited attenuation is 2^-3.5, so
// losing it makes granule 1 about 128x louder in energy.
DSPARK_TEST(Mp3File_scfsi_granule1_inherits_granule0_scalefactors)
{
    FileCleanup c1 { "dspark_test_scfsi_on.mp3" };
    FileCleanup c2 { "dspark_test_scfsi_off.mp3" };

    // scalefac_compress = 4 -> slen1 = 3, slen2 = 0: 11 x 3 = 33 scalefactor bits.
    // Huffman: table 1, pair (1,1) = code 0b000 (3 bits) + two sign bits.
    auto granuleSide = [](Mp3TestBits& b, int part23) {
        b.put(uint32_t(part23), 12); b.put(1, 9); b.put(200, 8); b.put(4, 4); b.put(0, 1);
        b.put(1, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
        b.put(0, 1); b.put(0, 1); b.put(0, 1);
    };
    auto scalefactorsFull = [](Mp3TestBits& b) {
        b.put(7, 3);                                  // sfb 0
        for (int i = 1; i < 11; ++i) b.put(0, 3);     // sfb 1..10
    };
    auto huff = [](Mp3TestBits& b) { b.put(0b000, 3); b.put(0, 1); b.put(0, 1); };

    // scfsi = 0b1000: band group 0 (sfb 0..5) inherited by granule 1.
    mp3WriteFile("dspark_test_scfsi_on.mp3", mp3CraftFrame(
        [&](Mp3TestBits& b) { granuleSide(b, 38); granuleSide(b, 20); },
        [&](Mp3TestBits& b) {
            scalefactorsFull(b);                            // granule 0
            huff(b);
            for (int i = 0; i < 5; ++i) b.put(0, 3);        // granule 1: sfb 6..10 only
            huff(b);
        }, 0b1000), 6);

    // Control: no inheritance, granule 1 re-sends the identical scalefactors.
    mp3WriteFile("dspark_test_scfsi_off.mp3", mp3CraftFrame(
        [&](Mp3TestBits& b) { granuleSide(b, 38); granuleSide(b, 38); },
        [&](Mp3TestBits& b) {
            scalefactorsFull(b); huff(b);
            scalefactorsFull(b); huff(b);
        }, 0), 6);

    const double eOn  = mp3DecodeEnergy("dspark_test_scfsi_on.mp3");
    const double eOff = mp3DecodeEnergy("dspark_test_scfsi_off.mp3");
    EXPECT_GT(eOn, 0.0);
    EXPECT_GT(eOff, 0.0);
    EXPECT_NEAR(eOn / eOff, 1.0, 0.02);   // pre-fix the inheriting file measured ~65x
}

// The encoder has to produce the transform the decoder inverts. It did not:
// the forward MDCT ran at twice the correct argument scale, the analysis window
// was the synthesis window (32x hot), and the frequency inversion and the alias
// butterfly the decoder undoes were never applied. The result was a legal MP3
// carrying noise -- a full-scale, 98%-clipped decode whose correlation with the
// input was 0.024, confirmed independently by a third-party decoder.
DSPARK_TEST(Mp3File_encoder_output_decodes_back_to_its_input)
{
    FileCleanup cleanup { "dspark_test_rt.mp3" };

    constexpr int N = 44100;
    std::vector<float> orig(N);
    for (int i = 0; i < N; ++i)
        orig[static_cast<size_t>(i)] =
            0.5f * std::sin(2.0f * pi<float> * 1000.0f * float(i) / 44100.0f);

    {
        Mp3File w;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 1;
        info.bitsPerSample = 320;   // kbps
        info.numSamples = N;
        EXPECT_TRUE(w.openWrite("dspark_test_rt.mp3", info));
        AudioBuffer<float> buf;
        buf.resize(1, N);
        std::copy(orig.begin(), orig.end(), buf.getChannel(0));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
    }

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_rt.mp3"));
    auto info = r.getInfo();
    const int n = static_cast<int>(info.numSamples);
    EXPECT_TRUE(n > N);
    AudioBuffer<float> dec;
    dec.resize(1, n);
    EXPECT_TRUE(r.readSamples(dec.toView()));
    r.close();
    EXPECT_NO_NAN(dec.getChannel(0), n);

    // Best-lag normalised correlation over the codec's delay range.
    const float* d = dec.getChannel(0);
    double best = -1.0;
    for (int lag = 0; lag <= 4096; ++lag)
    {
        const int use = std::min(N - 4096, n - lag - 4096);
        if (use < 8192) break;
        double sa = 0.0, sb = 0.0, sab = 0.0;
        for (int i = 0; i < use; ++i)
        {
            const double a = orig[static_cast<size_t>(i)];
            const double b = d[i + lag];
            sa += a * a; sb += b * b; sab += a * b;
        }
        const double den = std::sqrt(sa * sb);
        if (den > 0.0 && sab / den > best) best = sab / den;
    }
    EXPECT_GT(best, 0.70);   // pre-fix this measured 0.008 - 0.024
}

// 8-bit WAV is unsigned with 128 as zero and the reader scales by 128, but the
// writer scaled by 127. Every 8-bit round trip therefore came back 0.78% quiet
// - a whole quantisation step of SYSTEMATIC error, on top of quantisation. With
// both sides on the same grid, every value that the 128-step scale can express
// round-trips to within half a step and -1.0 is exact; only +1.0 clamps, which
// the scale itself forces.
DSPARK_TEST(WavFile_8bit_roundtrip_has_no_systematic_gain_error)
{
    FileCleanup cleanup { "dspark_test_8bit_gain.wav" };

    const float probes[] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f };
    constexpr int N = 8;

    {
        WavFile w;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 1;
        info.bitsPerSample = 8;
        info.isFloatingPoint = false;
        info.numSamples = N;
        EXPECT_TRUE(w.openWrite("dspark_test_8bit_gain.wav", info));
        AudioBuffer<float> b;
        b.resize(1, N);
        for (int i = 0; i < N; ++i) b.getChannel(0)[i] = probes[i];
        EXPECT_TRUE(w.writeSamples(std::as_const(b).toView()));
        w.close();
    }

    WavFile r;
    EXPECT_TRUE(r.openRead("dspark_test_8bit_gain.wav"));
    AudioBuffer<float> b;
    b.resize(1, N);
    EXPECT_TRUE(r.readSamples(b.toView()));
    r.close();

    // Every probe sits exactly on the 128-step grid, so the round trip is exact.
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(b.getChannel(0)[i], probes[i], 1e-6f);
    EXPECT_BOUNDED(b.getChannel(0), N, -1.0f, 1.0f);
}
