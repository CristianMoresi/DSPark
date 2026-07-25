// DSPark Tests - IO
// WavFile and Mp3File round-trip tests

#include "dspark_test.h"
#include "TestSignals.h"

#include "../IO/WavFile.h"
#include "../IO/Mp3File.h"

#include <algorithm>
#include <cstdio>
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
