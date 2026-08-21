// DSPark Tests - IO
// WavFile and Mp3File round-trip tests

#include "dspark_test.h"
#include "TestSignals.h"

#include "../IO/WavFile.h"
#include "../IO/Mp3File.h"
#include "../IO/MidiFile.h"
#include "../IO/FlacFile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// One atomically claimed root per suite process keeps every generated I/O file
// away from both the launch directory and every concurrent suite process.  The
// current directory is changed once so the older relative-path cases below are
// covered as well; immutable fixtures use their configured absolute path.
class TestIOProcessRoot
{
public:
    TestIOProcessRoot()
        : launchDirectory_(std::filesystem::current_path())
    {
#if defined(DSPARK_TESTIO_FIXED_NAME_MUTANT)
        // Targeted concurrency control: retaining the caller's directory
        // restores the historical fixed-name collision without changing any
        // individual test expectation.
        path_ = launchDirectory_;
#else
        const std::filesystem::path base = std::filesystem::temp_directory_path();
        std::random_device entropy;
        const auto stamp = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const uint64_t randomPart = (static_cast<uint64_t>(entropy()) << 32)
                                  ^ static_cast<uint64_t>(entropy());
        for (uint64_t attempt = 0; attempt < 256; ++attempt)
        {
            path_ = base / ("dspark-testio-" + std::to_string(stamp) + "-"
                            + std::to_string(randomPart) + "-"
                            + std::to_string(attempt));
            std::error_code ec;
            if (std::filesystem::create_directory(path_, ec))
            {
                ownsPath_ = true;
                std::filesystem::current_path(path_);
                return;
            }
        }
        throw std::runtime_error("could not create process-unique TestIO directory");
#endif
    }

    TestIOProcessRoot(const TestIOProcessRoot&) = delete;
    TestIOProcessRoot& operator=(const TestIOProcessRoot&) = delete;

    ~TestIOProcessRoot()
    {
        if (!ownsPath_) return;
        std::error_code ignored;
        std::filesystem::current_path(launchDirectory_, ignored);
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path launchDirectory_;
    std::filesystem::path path_;
    bool ownsPath_ = false;
};

static TestIOProcessRoot testIOProcessRoot;

// Helpers retained as names because several round-trip tests share them, but
// their relative paths now resolve only inside testIOProcessRoot.
static const char* tempWav16()  { return "dspark_test_16.wav"; }
static const char* tempWav24()  { return "dspark_test_24.wav"; }
static const char* tempWavF32() { return "dspark_test_f32.wav"; }
static const char* tempWavStereo() { return "dspark_test_stereo.wav"; }
static const char* tempMp3()    { return "dspark_test.mp3"; }

#if defined(DSPARK_TESTIO_FIXED_NAME_MUTANT)
namespace {

constexpr const char* testIOSyncRootVariable =
    "DSPARK_TESTIO_COLLISION_SYNC_ROOT";
constexpr const char* testIOSyncIdentityVariable =
    "DSPARK_TESTIO_COLLISION_SYNC_IDENTITY";
constexpr const char* testIOSyncParticipantsVariable =
    "DSPARK_TESTIO_COLLISION_SYNC_PARTICIPANTS";
constexpr const char* testIOSyncTimeoutVariable =
    "DSPARK_TESTIO_COLLISION_SYNC_TIMEOUT_MS";
constexpr const char* testIOSyncMutantVariable =
    "DSPARK_TESTIO_COLLISION_SYNC_MUTANT";

struct TestIOCollisionSynchronization
{
    std::filesystem::path root;
    uint64_t identity = 0;
    uint64_t participants = 0;
    std::chrono::milliseconds timeout { 0 };
    std::string mutant;
};

uint64_t parseTestIOSyncUnsigned(const char* value, const char* variable)
{
    if (value == nullptr || *value == '\0')
        throw std::runtime_error(std::string("DSPARK_TESTIO_SYNC_ERROR: missing ")
                                 + variable);
    uint64_t parsed = 0;
    for (const char* cursor = value; *cursor != '\0'; ++cursor)
    {
        if (*cursor < '0' || *cursor > '9')
            throw std::runtime_error(std::string("DSPARK_TESTIO_SYNC_ERROR: invalid ")
                                     + variable);
        const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            throw std::runtime_error(std::string("DSPARK_TESTIO_SYNC_ERROR: overflow ")
                                     + variable);
        parsed = parsed * 10 + digit;
    }
    return parsed;
}

TestIOCollisionSynchronization testIOCollisionSynchronization()
{
    TestIOCollisionSynchronization sync;
    const char* root = std::getenv(testIOSyncRootVariable);
    if (root == nullptr || *root == '\0')
        throw std::runtime_error(std::string("DSPARK_TESTIO_SYNC_ERROR: missing ")
                                 + testIOSyncRootVariable);
    sync.root = std::filesystem::path(root);
    sync.identity = parseTestIOSyncUnsigned(
        std::getenv(testIOSyncIdentityVariable), testIOSyncIdentityVariable);
    sync.participants = parseTestIOSyncUnsigned(
        std::getenv(testIOSyncParticipantsVariable),
        testIOSyncParticipantsVariable);
    const uint64_t timeoutMilliseconds = parseTestIOSyncUnsigned(
        std::getenv(testIOSyncTimeoutVariable), testIOSyncTimeoutVariable);
    if (sync.participants < 2 || sync.participants > 64
        || sync.identity >= sync.participants
        || timeoutMilliseconds == 0 || timeoutMilliseconds > 60000)
        throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: invalid synchronization bounds");
    sync.timeout = std::chrono::milliseconds(timeoutMilliseconds);
    if (const char* mutant = std::getenv(testIOSyncMutantVariable))
        sync.mutant = mutant;
    if (!sync.mutant.empty() && sync.mutant != "missing-readiness"
        && sync.mutant != "missing-attempted")
        throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: unknown synchronization mutant");
    std::error_code ec;
    if (!std::filesystem::is_directory(sync.root, ec) || ec)
        throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: invalid synchronization root");
    return sync;
}

void publishTestIOSyncDirectory(const std::filesystem::path& path,
                                const std::string& duplicateDiagnostic)
{
    std::error_code ec;
    if (std::filesystem::create_directory(path, ec)) return;
    if (!ec)
        throw std::runtime_error(duplicateDiagnostic);
    throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: publication failed: "
                             + ec.message());
}

bool testIOSyncDirectoryExists(const std::filesystem::path& path,
                               const char* phase)
{
    std::error_code ec;
    const std::filesystem::file_status status = std::filesystem::status(path, ec);
    if (!ec) return std::filesystem::is_directory(status);
    if (ec == std::errc::no_such_file_or_directory) return false;
    throw std::runtime_error(std::string("DSPARK_TESTIO_SYNC_ERROR: ") + phase
                             + " scan failed: " + ec.message());
}

size_t countTestIOReady(const TestIOCollisionSynchronization& sync)
{
    size_t observed = 0;
    for (uint64_t id = 0; id < sync.participants; ++id)
    {
        const bool exists = testIOSyncDirectoryExists(
            sync.root / ("ready-" + std::to_string(id)), "readiness");
        if (exists) ++observed;
    }
    return observed;
}

size_t countTestIOAttempted(const TestIOCollisionSynchronization& sync)
{
    size_t observed = 0;
    for (uint64_t id = 0; id < sync.participants; ++id)
    {
        const bool winner = testIOSyncDirectoryExists(
            sync.root / ("attempted-" + std::to_string(id) + "-winner"),
            "attempted");
        const bool loser = testIOSyncDirectoryExists(
            sync.root / ("attempted-" + std::to_string(id) + "-loser"),
            "attempted");
        if (winner && loser)
            throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: ambiguous attempted outcome");
        if (winner || loser) ++observed;
    }
    return observed;
}

template <typename Counter>
size_t waitForTestIOSyncPhase(const TestIOCollisionSynchronization& sync,
                              Counter counter)
{
    const auto deadline = std::chrono::steady_clock::now() + sync.timeout;
    for (;;)
    {
        const size_t observed = counter(sync);
        if (observed == sync.participants) return observed;
        if (std::chrono::steady_clock::now() >= deadline) return observed;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

} // namespace
#endif

// Cleanup helper
struct FileCleanup
{
    const char* path;
    ~FileCleanup() { std::remove(path); }
};

DSPARK_TEST(TestIO_process_temporary_root_is_exclusive)
{
    // create_directory is the portable atomic claim.  Three concurrent suite
    // processes each succeed inside their own root; the fixed-root mutant
    // makes two of them collide while a two-phase runner-owned barrier keeps
    // the winner's claim live until every participant has attempted it.
#if defined(DSPARK_TESTIO_FIXED_NAME_MUTANT)
    const TestIOCollisionSynchronization sync = testIOCollisionSynchronization();
    if (sync.mutant != "missing-readiness")
        publishTestIOSyncDirectory(
            sync.root / ("ready-" + std::to_string(sync.identity)),
            "DSPARK_TESTIO_SYNC_ERROR: duplicate process identity "
                + std::to_string(sync.identity));
    const size_t ready = waitForTestIOSyncPhase(sync, countTestIOReady);
    if (ready != sync.participants)
        throw std::runtime_error(
            "DSPARK_TESTIO_SYNC_TIMEOUT: readiness expected="
            + std::to_string(sync.participants) + " observed="
            + std::to_string(ready));
#endif

    const std::filesystem::path claim = "dspark_process_exclusive_claim";
    std::error_code ec;
    const bool acquired = std::filesystem::create_directory(claim, ec);
    if (ec)
        throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: atomic claim failed: "
                                 + ec.message());

#if defined(DSPARK_TESTIO_FIXED_NAME_MUTANT)
    if (sync.mutant != "missing-attempted")
        publishTestIOSyncDirectory(
            sync.root / ("attempted-" + std::to_string(sync.identity)
                         + (acquired ? "-winner" : "-loser")),
            "DSPARK_TESTIO_SYNC_ERROR: duplicate attempted publication");
    const size_t attempted = waitForTestIOSyncPhase(sync, countTestIOAttempted);
    if (acquired)
    {
        std::filesystem::remove(claim, ec);
        if (ec)
            throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: claim cleanup failed: "
                                     + ec.message());
    }
    if (attempted != sync.participants)
        throw std::runtime_error(
            "DSPARK_TESTIO_SYNC_TIMEOUT: attempted expected="
            + std::to_string(sync.participants) + " observed="
            + std::to_string(attempted));
#else
    if (acquired) std::filesystem::remove(claim, ec);
#endif
    EXPECT_TRUE(acquired);
}

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

// ============================================================================
// Mp3File - polyphase synthesis window and the short-block (block_type 2) path
// ============================================================================

// Builds one 48 kHz / 64 kbps / mono frame whose granules carry a SHORT block
// (block_type 2, not mixed) with a single nonzero spectral value at bitstream
// index `2*pairs`. scalefac_compress 0 means slen1 = slen2 = 0, so the granule
// spends no bits at all on scalefactors and part2_3_length is exactly the
// Huffman length: `pairs` copies of the table-1 code for (0,0) (one bit each)
// followed by the code for (1,0) plus its sign bit.
static std::vector<uint8_t> mp3CraftShortBlockFrame(int pairs)
{
    auto side = [pairs](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            b.put(uint32_t(pairs + 3), 12);        // part2_3_length
            b.put(uint32_t(pairs + 1), 9);         // big_values
            b.put(190, 8);                         // global_gain
            b.put(0, 4);                           // scalefac_compress -> no sf bits
            b.put(1, 1);                           // window_switching
            b.put(2, 2); b.put(0, 1);              // block_type 2, not mixed
            b.put(1, 5); b.put(1, 5);              // table_select 1, 1
            b.put(0, 3); b.put(0, 3); b.put(0, 3); // subblock_gain
            b.put(0, 1); b.put(0, 1); b.put(0, 1); // preflag, scalefac_scale, count1
        }
    };
    auto main_ = [pairs](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            for (int i = 0; i < pairs; ++i) b.put(0b1, 1);   // (0,0)
            b.put(0b01, 2); b.put(0, 1);                     // (1,0), sign +
        }
    };
    return mp3CraftFrame(side, main_);
}

// Sum of |pcm| in each of the 18 subband slots of a granule, folded over all
// granules of the file: a 18-bin picture of WHEN inside a granule the energy is.
static void mp3GranuleSlotProfile(const char* path, double slot[18], int& granules)
{
    for (int i = 0; i < 18; ++i) slot[i] = 0.0;
    granules = 0;

    Mp3File r;
    if (!r.openRead(path)) return;
    auto info = r.getInfo();
    AudioBuffer<float> buf;
    buf.resize(1, static_cast<int>(info.numSamples));
    if (!r.readSamples(buf.toView())) return;
    r.close();

    const size_t n = static_cast<size_t>(info.numSamples);
    for (size_t g = 2; (g + 2) * 576 <= n; ++g)
    {
        ++granules;
        for (int s = 0; s < 18; ++s)
            for (int j = 0; j < 32; ++j)
                slot[s] += std::fabs(double(buf.getChannel(0)[g * 576 + size_t(s * 32 + j)]));
    }
}

// The 512 synthesis-window coefficients are the polyphase prototype of the
// cosine-modulated bank the decoder ends in: the filter for subband k is
// kSynthWindow[n] * (-1)^floor(n/64) * cos(pi/64*(2k+1)*(n+16)). 115 of the 512
// carried the wrong SIGN, which left that prototype with a -25 dB stopband
// instead of about -100 dB, so every decode came back with images of itself at
// +-m*fs/32. This frame excites subband 0 and NOTHING else - one spectral value
// per granule, at xr[0] - so all it may contain is subband 0's own band,
// 0..fs/64. Everything at or above 3*fs/64 is an image the prototype failed to
// reject: -36.4 dB with the broken signs, -108.9 dB with them corrected.
DSPARK_TEST(Mp3File_synthesis_window_rejects_out_of_band_images)
{
    FileCleanup cleanup { "dspark_test_subband0.mp3" };

    auto side = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            b.put(3, 12); b.put(1, 9); b.put(190, 8); b.put(0, 4); b.put(0, 1);
            b.put(1, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
            b.put(0, 1); b.put(0, 1); b.put(0, 1);
        }
    };
    auto main_ = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr) { b.put(0b01, 2); b.put(0, 1); }
    };
    mp3WriteFile("dspark_test_subband0.mp3", mp3CraftFrame(side, main_), 8);

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_subband0.mp3"));
    auto info = r.getInfo();
    AudioBuffer<float> buf;
    buf.resize(1, static_cast<int>(info.numSamples));
    EXPECT_TRUE(r.readSamples(buf.toView()));
    r.close();
    EXPECT_TRUE(info.numSamples >= 3072);
    EXPECT_NO_NAN(buf.getChannel(0), static_cast<int>(info.numSamples));

    constexpr int kN = 2048;
    constexpr int kOff = 1024;
    const double fs = 48000.0;
    const float* x = buf.getChannel(0);

    // Naive DFT at the bins that matter: subband 0 spans 0..fs/64, and the
    // first image of a 32-band bank lands at or above 3*fs/64.
    auto binMag = [&](int k) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < kN; ++n)
        {
            const double a = 2.0 * pi<double> * double(k) * double(n) / double(kN);
            re += double(x[kOff + n]) * std::cos(a);
            im -= double(x[kOff + n]) * std::sin(a);
        }
        return std::sqrt(re * re + im * im);
    };

    const int kPassEnd = static_cast<int>((fs / 64.0) * kN / fs);        // 32
    const int kStopStart = static_cast<int>((3.0 * fs / 64.0) * kN / fs); // 96
    double passPeak = 0.0, stopPeak = 0.0;
    for (int k = 0; k <= kPassEnd; ++k) passPeak = std::max(passPeak, binMag(k));
    for (int k = kStopStart; k <= kN / 2; ++k) stopPeak = std::max(stopPeak, binMag(k));

    EXPECT_GT(passPeak, 1.0);
    const double rejectionDb = 20.0 * std::log10(stopPeak / passPeak);
    EXPECT_LT(rejectionDb, -80.0);   // broken signs: -36.4 dB; corrected: -108.9 dB
}

// A short block stores three windows per scalefactor band, so band sfb starts
// at the CUMULATIVE offset 3*shortBands[sfb]. Scaling the band index by the
// CURRENT band's width instead was only right while the widths stayed equal:
// at 48 kHz the short widths are 4,4,4,4,6,6,10,12,14,16,20,26,66, so sfb 10
// computed a base of 600 and the top three bands fell outside the granule,
// where an `index < 576` guard silently discarded them. This frame codes ONE
// value in sfb 10 (bitstream index 240) and nothing else: pre-fix the decode is
// digital silence, because both requantize() and reorder() threw that band away.
DSPARK_TEST(Mp3File_short_block_decodes_its_top_scalefactor_bands)
{
    FileCleanup cleanup { "dspark_test_shorthi.mp3" };
    mp3WriteFile("dspark_test_shorthi.mp3", mp3CraftShortBlockFrame(120), 8);

    const double energy = mp3DecodeEnergy("dspark_test_shorthi.mp3");
    EXPECT_GT(energy, 0.5);   // pre-fix this was exactly 0.0
}

// reorder() interleaves the three short windows (index 3*i + win), and the
// 12-point IMDCT must read them back the same way, xr[sb*18 + win + 3*i] (ISO
// 11172-3 2.4.3.4.9 reads in[win + 3*m]). Reading each window's six values
// contiguously fed every IMDCT a mixture of all three. The two files below
// differ ONLY in which short window carries the value, inside sfb 1 - a band
// whose base offset is the same before and after the base-offset fix, so this
// case isolates the IMDCT read order. A correct decoder emits window 2 exactly
// two short windows (12 subband slots) after window 0 and with the same shape.
DSPARK_TEST(Mp3File_short_block_windows_are_read_interleaved)
{
    FileCleanup c1 { "dspark_test_shortw0.mp3" };
    FileCleanup c2 { "dspark_test_shortw2.mp3" };

    // sfb 1 at 48 kHz: base 12, width 4 -> window w, first value = index 12+4w.
    mp3WriteFile("dspark_test_shortw0.mp3", mp3CraftShortBlockFrame(6), 8);
    mp3WriteFile("dspark_test_shortw2.mp3", mp3CraftShortBlockFrame(10), 8);

    double p0[18], p2[18];
    int g0 = 0, g2 = 0;
    mp3GranuleSlotProfile("dspark_test_shortw0.mp3", p0, g0);
    mp3GranuleSlotProfile("dspark_test_shortw2.mp3", p2, g2);
    EXPECT_GT(g0, 4);
    EXPECT_EQ(g0, g2);

    double t0 = 0.0, t2 = 0.0;
    for (int s = 0; s < 18; ++s) { t0 += p0[s]; t2 += p2[s]; }
    EXPECT_GT(t0, 0.0);
    EXPECT_NEAR(t2 / t0, 1.0, 0.02);   // the same value, only in another window

    // Same shape, delayed by two short windows: p0[s] == p2[(s+12) % 18].
    for (int s = 0; s < 18; ++s)
        EXPECT_NEAR(p2[(s + 12) % 18] / t2, p0[s] / t0, 0.01);

    // ... and the two are NOT the same profile, which is what the pre-fix
    // decoder produced: it could not tell the three windows apart.
    double maxDiff = 0.0;
    for (int s = 0; s < 18; ++s)
        maxDiff = std::max(maxDiff, std::fabs(p2[s] / t2 - p0[s] / t0));
    EXPECT_GT(maxDiff, 0.05);
}

// The three short windows occupy samples 6..29 of the 36-sample block, so the
// IMDCT must accumulate at tmp[6*win + i + 6]. At tmp[6*win + i] the block came
// out six samples early and its overlap with the neighbouring start/stop blocks
// never cancelled. Here ONLY granule 0 is coded, in window 2, and window 2 lives
// entirely in samples 18..29 - the overlap tail. Its energy therefore belongs to
// granule 1 in full, and granule 0 must be exactly silent. Six samples early it
// is not: the pre-fix decoder leaked 1.6e-2 into granule 0.
DSPARK_TEST(Mp3File_short_block_starts_six_samples_into_the_window)
{
    FileCleanup cleanup { "dspark_test_shortgran.mp3" };

    auto side = [](Mp3TestBits& b) {
        b.put(13, 12); b.put(11, 9); b.put(190, 8); b.put(0, 4); b.put(1, 1);
        b.put(2, 2); b.put(0, 1); b.put(1, 5); b.put(1, 5);
        b.put(0, 3); b.put(0, 3); b.put(0, 3);
        b.put(0, 1); b.put(0, 1); b.put(0, 1);
        b.put(0, 12); b.put(0, 9); b.put(190, 8); b.put(0, 4); b.put(0, 1);
        b.put(0, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
        b.put(0, 1); b.put(0, 1); b.put(0, 1);
    };
    auto main_ = [](Mp3TestBits& b) {
        for (int i = 0; i < 10; ++i) b.put(0b1, 1);
        b.put(0b01, 2); b.put(0, 1);
    };
    auto silent = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            b.put(0, 12); b.put(0, 9); b.put(190, 8); b.put(0, 4); b.put(0, 1);
            b.put(0, 5); b.put(0, 5); b.put(0, 5); b.put(0, 4); b.put(0, 3);
            b.put(0, 1); b.put(0, 1); b.put(0, 1);
        }
    };

    const std::vector<uint8_t> coded = mp3CraftFrame(side, main_);
    const std::vector<uint8_t> quiet = mp3CraftFrame(silent, [](Mp3TestBits&) {});
    {
        std::ofstream f("dspark_test_shortgran.mp3", std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(coded.data()),
                static_cast<std::streamsize>(coded.size()));
        for (int i = 0; i < 7; ++i)
            f.write(reinterpret_cast<const char*>(quiet.data()),
                    static_cast<std::streamsize>(quiet.size()));
    }

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_shortgran.mp3"));
    auto info = r.getInfo();
    EXPECT_TRUE(info.numSamples >= 1152);
    AudioBuffer<float> buf;
    buf.resize(1, static_cast<int>(info.numSamples));
    EXPECT_TRUE(r.readSamples(buf.toView()));
    r.close();

    double g[2] = { 0.0, 0.0 };
    for (int gi = 0; gi < 2; ++gi)
        for (int n = 0; n < 576; ++n)
            g[gi] += std::fabs(double(buf.getChannel(0)[gi * 576 + n]));

    EXPECT_GT(g[1], 1.0);       // window 2 lands in the next granule, in full
    EXPECT_LT(g[0], 1e-9);      // ... and none of it here; pre-fix: 1.6e-2
}

// ============================================================================
// Mp3File - the last long scalefactor band, and the encoder's region fields
// ============================================================================

// Decodes a whole file and returns its samples for channel 0.
static std::vector<double> mp3DecodeChannel0(const char* path)
{
    std::vector<double> out;
    Mp3File r;
    if (!r.openRead(path)) return out;
    auto info = r.getInfo();
    AudioBuffer<float> buf;
    buf.resize(static_cast<int>(info.numChannels), static_cast<int>(info.numSamples));
    if (!r.readSamples(buf.toView())) return out;
    r.close();
    out.resize(static_cast<size_t>(info.numSamples));
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = double(buf.getChannel(0)[i]);
    return out;
}

// ISO 11172-3 2.4.2.7 sends long scalefactors for bands 0..20 only: band 21 has
// none and is implicitly 0. The decoder seeds granule 1's scalefactor array with
// granule 0's so that scfsi inheritance works, but scfsi only ever covers bands
// 0..20 - so when granule 0 is a SHORT block, whose 12 bands x 3 windows fill
// entries 0..35, entry 21 still held one of ITS per-window scalefactors and
// granule 1 requantised long band 21 with it.
//
// Band 21 is the top of the spectrum (coefficients 384..575 at 48 kHz), and the
// granule that follows a short block is the stop block of a transient, so this
// darkened the top octave of every transient by up to 2^-7.5. These two files
// differ ONLY in a short-block scalefactor of granule 0, at the entry that band
// 21 aliased onto; granule 1 carries one coefficient at index 384. A conformant
// decoder must return byte-identical audio for both.
DSPARK_TEST(Mp3File_long_band_21_ignores_the_previous_granule_short_scalefactors)
{
    FileCleanup cleanupA { "dspark_test_band21_a.mp3" };
    FileCleanup cleanupB { "dspark_test_band21_b.mp3" };

    // granule 0: short block, silent, 54 bits of scalefactors (scalefac_compress
    //            3 -> slen1 0, slen2 3, so only bands 6..11 spend bits);
    // granule 1: stop block carrying (1,0) at coefficients 384,385 - long band 21.
    auto build = [](uint32_t shortSf21) {
        auto side = [](Mp3TestBits& b) {
            b.put(54, 12); b.put(0, 9); b.put(190, 8); b.put(3, 4); b.put(1, 1);
            b.put(2, 2); b.put(0, 1); b.put(0, 5); b.put(0, 5);
            b.put(0, 3); b.put(0, 3); b.put(0, 3);
            b.put(0, 1); b.put(0, 1); b.put(0, 1);
            b.put(195, 12); b.put(193, 9); b.put(190, 8); b.put(0, 4); b.put(1, 1);
            b.put(3, 2); b.put(0, 1); b.put(1, 5); b.put(1, 5);
            b.put(0, 3); b.put(0, 3); b.put(0, 3);
            b.put(0, 1); b.put(0, 1); b.put(0, 1);
        };
        auto main_ = [shortSf21](Mp3TestBits& b) {
            // granule 0: 18 short scalefactors of 3 bits (bands 6..11 x 3 windows).
            // Entry 21 is the fourth of them - the one long band 21 aliases onto.
            for (int i = 0; i < 18; ++i) b.put(i == 3 ? shortSf21 : 0u, 3);
            // granule 1: 192 (0,0) pairs, then (1,0) at coefficients 384,385.
            for (int i = 0; i < 192; ++i) b.put(0b1, 1);
            b.put(0b01, 2); b.put(0, 1);
        };
        return mp3CraftFrame(side, main_);
    };

    mp3WriteFile("dspark_test_band21_a.mp3", build(0), 8);
    mp3WriteFile("dspark_test_band21_b.mp3", build(7), 8);

    const std::vector<double> a = mp3DecodeChannel0("dspark_test_band21_a.mp3");
    const std::vector<double> b = mp3DecodeChannel0("dspark_test_band21_b.mp3");
    EXPECT_EQ(a.size(), b.size());
    EXPECT_TRUE(a.size() >= 4608);

    double energy = 0.0, worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        energy += std::fabs(a[i]);
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    EXPECT_GT(energy, 1.0);      // the band-21 coefficient really is audible
    EXPECT_LT(worst, 1e-12);     // pre-fix the two decodes differed by 2^-3.5
}

// region0_count is a 4-bit side-info field and region1_count a 3-bit one, so the
// encoder's region split must satisfy r0 <= 16 and r1 - r0 - 1 <= 7. Splitting
// the band range in half did not: a granule whose coefficients reach past
// scalefactor band 17 gives r1 >= 17 and a region1_count of 8..10, which the
// 3-bit field truncated. The writer then emitted the Huffman regions using the
// untruncated value while the bitstream carried the truncated one, so the
// decoder placed the region-1/region-2 boundary up to nine scalefactor bands too
// low and read the rest of the granule with the wrong table and the wrong
// linbits count.
//
// The first frequency at which a steady tone pushes the granule past band 17 is
// the boundary of subband 10, 6890.6 Hz at 44.1 kHz: below it the round trip was
// exact and above it collapsed - 7 kHz came back at correlation 0.026 with its
// amplitude deleted, 7.5 kHz at 0.0014 with 141% of the source amplitude at the
// wrong frequency.
DSPARK_TEST(Mp3File_encoder_codes_tones_above_the_region_boundary)
{
    FileCleanup cleanup { "dspark_test_hf.mp3" };

    constexpr int N = 22050;
    for (double freq : { 6500.0, 7000.0, 7500.0, 11000.0 })
    {
        std::vector<float> orig(N);
        for (int i = 0; i < N; ++i)
            orig[static_cast<size_t>(i)] =
                0.5f * std::sin(2.0f * pi<float> * float(freq) * float(i) / 44100.0f);

        {
            Mp3File w;
            AudioFileInfo info;
            info.sampleRate = 44100.0;
            info.numChannels = 1;
            info.bitsPerSample = 320;   // kbps
            info.numSamples = N;
            EXPECT_TRUE(w.openWrite("dspark_test_hf.mp3", info));
            AudioBuffer<float> buf;
            buf.resize(1, N);
            std::copy(orig.begin(), orig.end(), buf.getChannel(0));
            EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
            w.close();
        }

        Mp3File r;
        EXPECT_TRUE(r.openRead("dspark_test_hf.mp3"));
        auto info = r.getInfo();
        const int n = static_cast<int>(info.numSamples);
        AudioBuffer<float> dec;
        dec.resize(1, n);
        EXPECT_TRUE(r.readSamples(dec.toView()));
        r.close();
        EXPECT_NO_NAN(dec.getChannel(0), n);

        const float* d = dec.getChannel(0);
        double best = -1.0;
        for (int lag = 0; lag <= 4096; ++lag)
        {
            const int use = std::min(N - 4096, n - lag - 4096);
            if (use < 8192) break;
            double sa = 0.0, sb = 0.0, sab = 0.0;
            for (int i = 0; i < use; ++i)
            {
                const double x = orig[static_cast<size_t>(i)];
                const double y = d[i + lag];
                sa += x * x; sb += y * y; sab += x * y;
            }
            const double den = std::sqrt(sa * sb);
            if (den > 0.0 && sab / den > best) best = sab / den;
        }

        double ms = 0.0;
        int used = 0;
        for (int i = 4096; i < n - 4096; ++i) { ms += double(d[i]) * double(d[i]); ++used; }
        const double rms = std::sqrt(ms / std::max(1, used));

        EXPECT_GT(best, 0.99);        // 7 kHz measured 0.026 before the fix
        EXPECT_GT(rms, 0.336);        // source RMS is 0.3536; 7 kHz returned 0.0007
        EXPECT_LT(rms, 0.372);        // 7.5 kHz returned 0.498, 141% of the source
    }
}

// The encoder's analysis filterbank and the decoder's synthesis filterbank share
// one prototype: ISO 11172-3 gives the analysis window as C[i] = D[i]/32. Nothing
// pinned that /32 or the cascade it belongs to, and the two are only checkable
// together - windowing the encoder's input with D itself makes the subband
// samples 32x hot, which the decoder then reproduces faithfully. At 320 kbps the
// quantiser is transparent enough that the round-trip gain of a broadband signal
// is the cascade's own gain: it must be 1, not 32 and not 1/32.
DSPARK_TEST(Mp3File_encoder_decoder_cascade_has_unity_gain)
{
    FileCleanup cleanup { "dspark_test_cascade.mp3" };

    constexpr int N = 22050;
    std::vector<float> orig(N);
    for (int i = 0; i < N; ++i)
    {
        const double t = double(i) / 44100.0;
        double v = 0.0;
        for (double f : { 220.0, 1370.0, 3100.0, 5300.0, 9100.0 })
            v += std::sin(2.0 * pi<double> * f * t);
        orig[static_cast<size_t>(i)] = float(0.14 * v);
    }

    {
        Mp3File w;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 1;
        info.bitsPerSample = 320;
        info.numSamples = N;
        EXPECT_TRUE(w.openWrite("dspark_test_cascade.mp3", info));
        AudioBuffer<float> buf;
        buf.resize(1, N);
        std::copy(orig.begin(), orig.end(), buf.getChannel(0));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
    }

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_cascade.mp3"));
    auto info = r.getInfo();
    const int n = static_cast<int>(info.numSamples);
    AudioBuffer<float> dec;
    dec.resize(1, n);
    EXPECT_TRUE(r.readSamples(dec.toView()));
    r.close();
    EXPECT_NO_NAN(dec.getChannel(0), n);

    const float* d = dec.getChannel(0);
    double best = -1.0;
    int bestLag = 0;
    for (int lag = 0; lag <= 4096; ++lag)
    {
        const int use = std::min(N - 4096, n - lag - 4096);
        if (use < 8192) break;
        double sa = 0.0, sb = 0.0, sab = 0.0;
        for (int i = 0; i < use; ++i)
        {
            const double x = orig[static_cast<size_t>(i)];
            const double y = d[i + lag];
            sa += x * x; sb += y * y; sab += x * y;
        }
        const double den = std::sqrt(sa * sb);
        if (den > 0.0 && sab / den > best) { best = sab / den; bestLag = lag; }
    }

    const int use = std::min(N - 4096, n - bestLag - 4096);
    double sxy = 0.0, syy = 0.0, sxx = 0.0;
    for (int i = 0; i < use; ++i)
    {
        const double x = orig[static_cast<size_t>(i)];
        const double y = d[i + bestLag];
        sxy += x * y; syy += y * y; sxx += x * x;
    }
    const double gain = sxy / syy;      // the scale the decode needs to match the source
    double err = 0.0;
    for (int i = 0; i < use; ++i)
    {
        const double e = orig[static_cast<size_t>(i)] - gain * double(d[i + bestLag]);
        err += e * e;
    }
    const double residualDb = 10.0 * std::log10(err / sxx);

    EXPECT_GT(best, 0.99);
    EXPECT_NEAR(gain, 1.0, 0.01);       // 32 or 1/32 if the /32 is dropped
    EXPECT_LT(residualDb, -40.0);
}

// A MIXED block (block_type 2 with mixed_block_flag 1) keeps subbands 0 and 1
// long and switches the rest to three short windows. Its short region therefore
// begins where the long region ends, at longBands[8] == 3*shortBands[3] == 36,
// and band sfb sits at the CUMULATIVE offset 36 + 3*(shortBands[sfb] -
// shortBands[3]) - the same cumulative rule the non-mixed path uses, offset by
// the long region. LAME never emits mixed blocks, so no recorded fixture reaches
// this path and it has to be crafted. This frame puts one value at bitstream
// index 240, which is band 10 window 0 line 0 at 48 kHz: frequency line 80, i.e.
// subband 13, 9750..10500 Hz. Scaling the band index by the current band's width
// instead put it outside the granule, where a bounds guard discarded it.
DSPARK_TEST(Mp3File_mixed_block_short_region_uses_cumulative_band_offsets)
{
    FileCleanup cleanup { "dspark_test_mixed.mp3" };

    auto side = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            b.put(123, 12); b.put(121, 9); b.put(190, 8); b.put(0, 4); b.put(1, 1);
            b.put(2, 2); b.put(1, 1);              // block_type 2, MIXED
            b.put(1, 5); b.put(1, 5);
            b.put(0, 3); b.put(0, 3); b.put(0, 3);
            b.put(0, 1); b.put(0, 1); b.put(0, 1);
        }
    };
    auto main_ = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr)
        {
            for (int i = 0; i < 120; ++i) b.put(0b1, 1);   // (0,0) up to index 239
            b.put(0b01, 2); b.put(0, 1);                   // (1,0) at 240, 241
        }
    };
    mp3WriteFile("dspark_test_mixed.mp3", mp3CraftFrame(side, main_), 8);

    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_mixed.mp3"));
    auto info = r.getInfo();
    EXPECT_TRUE(info.numSamples >= 3072);
    AudioBuffer<float> buf;
    buf.resize(1, static_cast<int>(info.numSamples));
    EXPECT_TRUE(r.readSamples(buf.toView()));
    r.close();
    EXPECT_NO_NAN(buf.getChannel(0), static_cast<int>(info.numSamples));

    constexpr int kN = 2048;
    constexpr int kOff = 1152;
    const double fs = 48000.0;
    const float* x = buf.getChannel(0);
    auto binMag = [&](int k) {
        double re = 0.0, im = 0.0;
        for (int nn = 0; nn < kN; ++nn)
        {
            const double a = 2.0 * pi<double> * double(k) * double(nn) / double(kN);
            re += double(x[kOff + nn]) * std::cos(a);
            im -= double(x[kOff + nn]) * std::sin(a);
        }
        return std::sqrt(re * re + im * im);
    };

    // Subband 13 spans 13*fs/64 .. 14*fs/64 == 9750..10500 Hz.
    const int lo = int(9750.0 * kN / fs);
    const int hi = int(10500.0 * kN / fs);
    double inBand = 0.0, outBand = 0.0;
    for (int k = 1; k <= kN / 2; ++k)
    {
        const double m = binMag(k);
        if (k >= lo && k <= hi) inBand = std::max(inBand, m);
        else                    outBand = std::max(outBand, m);
    }
    EXPECT_GT(inBand, 1.0);                                   // pre-fix: digital silence
    EXPECT_LT(20.0 * std::log10(outBand / inBand), -20.0);    // and it is where it belongs
}

// ============================================================================
// MidiFile - semantic SMF, hostile input, and deterministic timing
// ============================================================================

namespace midi_test {

class TempDirectory
{
public:
    explicit TempDirectory(const std::string& label)
    {
        static std::atomic<uint64_t> serial { 0 };
        const auto stamp = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const std::filesystem::path base = testIOProcessRoot.path();
        for (uint64_t attempt = 0; attempt < 128; ++attempt)
        {
            const uint64_t id = serial.fetch_add(1, std::memory_order_relaxed);
            path_ = base / ("dspark-midi-" + label + "-"
                            + std::to_string(stamp) + "-" + std::to_string(id));
            std::error_code ec;
            if (std::filesystem::create_directory(path_, ec))
                return;
        }
        throw std::runtime_error("could not create unique MIDI test directory");
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    ~TempDirectory()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] std::filesystem::path file(const std::string& name) const
    {
        return path_ / name;
    }

private:
    std::filesystem::path path_;
};

void be16(std::vector<uint8_t>& out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

void be32(std::vector<uint8_t>& out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

void tag(std::vector<uint8_t>& out, const char* value)
{
    out.insert(out.end(), value, value + 4);
}

std::vector<uint8_t> withEot(std::vector<uint8_t> events)
{
    const uint8_t eot[] = { 0x00, 0xff, 0x2f, 0x00 };
    events.insert(events.end(), std::begin(eot), std::end(eot));
    return events;
}

std::vector<uint8_t> smf(uint16_t format, uint16_t division,
                         const std::vector<std::vector<uint8_t>>& tracks,
                         const std::vector<uint8_t>& headerExtension = {},
                         const std::vector<uint8_t>& alien = {})
{
    std::vector<uint8_t> out;
    tag(out, "MThd");
    be32(out, static_cast<uint32_t>(6 + headerExtension.size()));
    be16(out, format);
    be16(out, static_cast<uint16_t>(tracks.size()));
    be16(out, division);
    out.insert(out.end(), headerExtension.begin(), headerExtension.end());
    if (!alien.empty())
    {
        tag(out, "TEST");
        be32(out, static_cast<uint32_t>(alien.size()));
        out.insert(out.end(), alien.begin(), alien.end());
    }
    for (const auto& trackData : tracks)
    {
        tag(out, "MTrk");
        be32(out, static_cast<uint32_t>(trackData.size()));
        out.insert(out.end(), trackData.begin(), trackData.end());
    }
    return out;
}

bool writeBytes(const std::filesystem::path& path, std::span<const uint8_t> bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

std::vector<uint8_t> readBytes(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return {};
    const std::streampos end = in.tellg();
    if (end <= std::streampos(0)) return {};
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes.size())) return {};
    return bytes;
}

bool reject(const std::vector<uint8_t>& bytes, const std::string& label)
{
    TempDirectory temp(label);
    const auto path = temp.file("input.mid");
    if (!writeBytes(path, bytes)) return false;
    MidiFile file;
    if (!file.create(1, 96, 1)) return false;
    if (file.read(path)) return false;
    return file.format() == -1 && file.ticksPerQuarter() == 0
        && file.tracks().empty();
}

bool record(bool condition, const std::string& name)
{
    if (!condition)
        std::cerr << "    MIDI subcase failed: " << name << "\n";
    return condition;
}

uint64_t fnv1a(uint64_t hash, std::span<const uint8_t> bytes)
{
    for (uint8_t byte : bytes)
    {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

} // namespace midi_test

DSPARK_TEST(MidiFile_RP001_primary_byte_oracles)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(DSPARK_TEST_FIXTURE_DIR) / "midi";
    const auto format0Bytes = midi_test::readBytes(root / "rp001-format0.mid");
    const auto format1Bytes = midi_test::readBytes(root / "rp001-format1.mid");
    EXPECT_EQ(format0Bytes.size(), size_t(81));
    EXPECT_EQ(format1Bytes.size(), size_t(118));

    MidiFile f0;
    EXPECT_TRUE(f0.read(root / "rp001-format0.mid"));
    EXPECT_EQ(f0.format(), 0);
    EXPECT_EQ(f0.ticksPerQuarter(), uint16_t(96));
    EXPECT_EQ(f0.tracks().size(), size_t(1));
    EXPECT_EQ(f0.tracks()[0].events.size(), size_t(14));
    EXPECT_EQ(f0.tracks()[0].events[6].status, uint8_t(0x92));
    EXPECT_EQ(f0.tracks()[0].events[6].data1, uint8_t(0x3c));
    EXPECT_EQ(f0.tracks()[0].events[9].deltaTicks, uint32_t(192));
    EXPECT_EQ(f0.tracks()[0].events.back().metaType, uint8_t(0x2f));

    MidiFile f1;
    EXPECT_TRUE(f1.read(root / "rp001-format1.mid"));
    EXPECT_EQ(f1.format(), 1);
    EXPECT_EQ(f1.tracks().size(), size_t(4));
    size_t eventCount = 0;
    for (const MidiTrack& track : f1.tracks()) eventCount += track.events.size();
    EXPECT_EQ(eventCount, size_t(17));
    EXPECT_EQ(f1.tracks()[1].events[2].status, uint8_t(0x90));
    EXPECT_EQ(f1.tracks()[1].events[2].data2, uint8_t(0x00));

    midi_test::TempDirectory temp("rp001-roundtrip");
    const auto rewritten = temp.file("rewritten.mid");
    EXPECT_TRUE(f1.write(rewritten));
    MidiFile reread;
    EXPECT_TRUE(reread.read(rewritten));
    EXPECT_TRUE(reread.tracks() == f1.tracks());
}

DSPARK_TEST(MidiFile_semantic_authoring_and_roundtrip)
{
    MidiFile midi;
    EXPECT_TRUE(midi.create(1, 960, 2));

    const uint8_t tempo[] = { 0x07, 0xa1, 0x20 };
    const uint8_t signature[] = { 4, 2, 24, 8 };
    EXPECT_TRUE(midi.addMetaEvent(0, 0, 0x51, tempo));
    EXPECT_TRUE(midi.addMetaEvent(0, 0, 0x58, signature));
    EXPECT_TRUE(midi.addChannelEvent(0, 0, 0x80, 60, 64));
    EXPECT_TRUE(midi.addChannelEvent(0, 1, 0x91, 61, 0));
    EXPECT_TRUE(midi.addChannelEvent(0, 2, 0xa2, 62, 3));
    EXPECT_TRUE(midi.addChannelEvent(0, 3, 0xb3, 7, 100));
    EXPECT_TRUE(midi.addChannelEvent(0, 4, 0xc4, 10));
    EXPECT_TRUE(midi.addChannelEvent(0, 5, 0xd5, 11));
    EXPECT_TRUE(midi.addChannelEvent(0, 6, 0xe6, 0, 64));

    const uint8_t complete[] = { 0x43, 0x12, 0xf7 };
    const uint8_t escape[] = { 0xf8, 0x01 };
    const uint8_t split0[] = { 0x43, 0x12 };
    const uint8_t split1[] = { 0x00, 0x43 };
    const uint8_t split2[] = { 0x12, 0xf7 };
    EXPECT_TRUE(midi.addSysExEvent(0, 7, MidiEventKind::SysExF0, complete));
    EXPECT_TRUE(midi.addSysExEvent(0, 8, MidiEventKind::SysExF7, escape));
    EXPECT_TRUE(midi.addSysExEvent(0, 9, MidiEventKind::SysExF0, split0));
    EXPECT_TRUE(midi.addSysExEvent(0, 10, MidiEventKind::SysExF7, split1));
    EXPECT_TRUE(midi.addSysExEvent(0, 11, MidiEventKind::SysExF7, split2));

    std::vector<uint8_t> opaque(130);
    for (size_t i = 0; i < opaque.size(); ++i)
        opaque[i] = static_cast<uint8_t>((i * 17) & 0xffu);
    EXPECT_TRUE(midi.addMetaEvent(0, 12, 0x7f, opaque));
    EXPECT_TRUE(midi.addMetaEvent(0, 13, 0x2f));
    const uint8_t ignoredTempo[] = { 0x03, 0x0d, 0x40 };
    EXPECT_TRUE(midi.addMetaEvent(1, 0, 0x51, ignoredTempo));
    EXPECT_TRUE(midi.addMetaEvent(1, 0, 0x2f));

    const MidiTrack beforeFailure = midi.tracks()[0];
    const uint8_t zeroTempo[] = { 0, 0, 0 };
    EXPECT_TRUE(!midi.addMetaEvent(0, 0, 0x51, zeroTempo));
    EXPECT_TRUE(!midi.addChannelEvent(0, 0, 0xf0, 0, 0));
    EXPECT_TRUE(midi.tracks()[0] == beforeFailure);
    EXPECT_TRUE(!midi.create(0, 0, 1));
    EXPECT_EQ(midi.format(), 1);

    midi_test::TempDirectory temp("semantic");
    const auto path = temp.file("semantic.mid");
    EXPECT_TRUE(midi.write(path));
    MidiFile decoded;
    EXPECT_TRUE(decoded.read(path));
    EXPECT_TRUE(decoded.tracks() == midi.tracks());

    bool all = true;
    const auto& events = decoded.tracks()[0].events;
    const char* names[] = {
        "format-1-write-read", "channel-80", "channel-90-velocity-zero",
        "channel-a0", "channel-b0", "channel-c0-one-byte",
        "channel-d0-one-byte", "channel-e0", "multiple-channels",
        "f0-complete", "f7-escape", "f0-split", "f7-continuation",
        "f7-terminal", "tempo-meta", "time-signature-meta",
        "unknown-oversized-meta", "final-eot", "transactional-mutator",
        "tempo-outside-track-zero-retained"
    };
    const bool checks[] = {
        decoded.format() == 1, events[2].status == 0x80,
        events[3].status == 0x91 && events[3].data2 == 0,
        events[4].status == 0xa2, events[5].status == 0xb3,
        events[6].status == 0xc4 && events[6].data2 == 0,
        events[7].status == 0xd5 && events[7].data2 == 0,
        events[8].status == 0xe6, events[3].status != events[4].status,
        events[9].kind == MidiEventKind::SysExF0 && events[9].payload.back() == 0xf7,
        events[10].kind == MidiEventKind::SysExF7,
        events[11].kind == MidiEventKind::SysExF0 && events[11].payload.back() != 0xf7,
        events[12].kind == MidiEventKind::SysExF7 && events[12].payload.back() != 0xf7,
        events[13].kind == MidiEventKind::SysExF7 && events[13].payload.back() == 0xf7,
        events[0].metaType == 0x51, events[1].metaType == 0x58,
        events[14].metaType == 0x7f && events[14].payload.size() == 130,
        events.back().metaType == 0x2f, midi.tracks()[0] == beforeFailure,
        decoded.tracks()[1].events[0].metaType == 0x51
    };
    static_assert(std::size(names) == std::size(checks));
    for (size_t i = 0; i < std::size(names); ++i)
        all = midi_test::record(checks[i], names[i]) && all;
    EXPECT_TRUE(all);
}

DSPARK_TEST(MidiFile_authoring_cache_state_stays_synchronized)
{
    MidiFile midi;
    EXPECT_TRUE(midi.create(1, 480, 2));

    const uint8_t openPacket[] = { 0x43, 0x12 };
    const uint8_t continuation[] = { 0x01, 0x02 };
    const uint8_t terminal[] = { 0x03, 0xf7 };
    const uint8_t escape[] = { 0x7d, 0x01 };
    const uint8_t text[] = { 'c', 'a', 'c', 'h', 'e' };

    EXPECT_TRUE(midi.addChannelEvent(0, 1, 0x90, 60, 100));
    EXPECT_TRUE(midi.addMetaEvent(1, 2, 0x01, text));
    EXPECT_TRUE(midi.addSysExEvent(0, 3, MidiEventKind::SysExF0,
                                  openPacket));
    EXPECT_TRUE(!midi.addChannelEvent(0, 4, 0x80, 60, 0));
    EXPECT_TRUE(!midi.addMetaEvent(0, 4, 0x01, text));
    EXPECT_TRUE(!midi.addSysExEvent(0, 4, MidiEventKind::SysExF0,
                                   openPacket));
    EXPECT_TRUE(midi.addSysExEvent(0, 4, MidiEventKind::SysExF7,
                                  continuation));
    EXPECT_TRUE(!midi.addChannelEvent(0, 5, 0x80, 60, 0));
    EXPECT_TRUE(midi.addSysExEvent(0, 5, MidiEventKind::SysExF7, terminal));
    EXPECT_TRUE(midi.addMetaEvent(0, 6, 0x01, text));
    EXPECT_TRUE(midi.addChannelEvent(1, 7, 0xc1, 11));
    EXPECT_TRUE(midi.addSysExEvent(1, 8, MidiEventKind::SysExF7, escape));
    EXPECT_TRUE(midi.addMetaEvent(0, 9, 0x2f));
    EXPECT_TRUE(midi.addMetaEvent(1, 10, 0x2f));

    const std::vector<MidiTrack> beforeFailures = midi.tracks();
    EXPECT_TRUE(!midi.addChannelEvent(0, 0, 0x90, 1, 1));
    EXPECT_TRUE(!midi.addMetaEvent(1, 0, 0x01, text));
    EXPECT_TRUE(!midi.addSysExEvent(0, 0, MidiEventKind::SysExF7, escape));
    EXPECT_TRUE(!midi.create(0, 0, 1));
    EXPECT_TRUE(midi.tracks() == beforeFailures);

    const auto added = midi.addTrack();
    EXPECT_TRUE(added.has_value());
    EXPECT_EQ(*added, size_t(2));
    EXPECT_TRUE(midi.addChannelEvent(*added, 11, 0xe2, 0, 64));
    EXPECT_TRUE(midi.addSysExEvent(*added, 12, MidiEventKind::SysExF0,
                                  openPacket));
    EXPECT_TRUE(midi.addSysExEvent(*added, 13, MidiEventKind::SysExF7,
                                  terminal));
    EXPECT_TRUE(midi.addMetaEvent(*added, 14, 0x2f));

    midi_test::TempDirectory temp("authoring-cache");
    const auto firstPath = temp.file("first.mid");
    EXPECT_TRUE(midi.write(firstPath));

    MidiFile decoded;
    EXPECT_TRUE(decoded.read(firstPath));
    EXPECT_TRUE(decoded.tracks() == midi.tracks());
    EXPECT_TRUE(!decoded.addChannelEvent(0, 0, 0x90, 1, 1));
    EXPECT_TRUE(!decoded.addMetaEvent(1, 0, 0x01, text));
    EXPECT_TRUE(!decoded.addSysExEvent(2, 0, MidiEventKind::SysExF7, escape));

    MidiFile moved = std::move(decoded);
    EXPECT_TRUE(moved.tracks() == midi.tracks());
    EXPECT_TRUE(!moved.addChannelEvent(2, 0, 0x90, 1, 1));
    MidiFile moveAssigned;
    EXPECT_TRUE(moveAssigned.create(0, 96));
    moveAssigned = std::move(moved);
    EXPECT_TRUE(moveAssigned.tracks() == midi.tracks());
    EXPECT_TRUE(!moveAssigned.addMetaEvent(0, 0, 0x01, text));

    const auto secondPath = temp.file("second.mid");
    EXPECT_TRUE(moveAssigned.write(secondPath));
    MidiFile roundTripped;
    EXPECT_TRUE(roundTripped.read(secondPath));
    EXPECT_TRUE(roundTripped.tracks() == midi.tracks());

    moveAssigned.clear();
    EXPECT_EQ(moveAssigned.format(), -1);
    EXPECT_TRUE(moveAssigned.tracks().empty());
    EXPECT_TRUE(moveAssigned.create(0, 96));
    EXPECT_TRUE(moveAssigned.addChannelEvent(0, 1, 0x90, 64, 127));
    const MidiTrack beforeInvalidAddTrack = moveAssigned.tracks()[0];
    EXPECT_TRUE(!moveAssigned.addTrack().has_value());
    EXPECT_TRUE(moveAssigned.tracks()[0] == beforeInvalidAddTrack);
    EXPECT_TRUE(moveAssigned.addMetaEvent(0, 2, 0x2f));
    EXPECT_EQ(moveAssigned.tracks()[0].events.size(), size_t(2));
}

DSPARK_TEST(MidiFile_format2_tempo_map_and_exact_rational_time)
{
    const std::vector<uint8_t> track0 = midi_test::withEot({
        0x00, 0xff, 0x51, 0x03, 0x09, 0x27, 0xc0,
        0x00, 0xff, 0x51, 0x03, 0x04, 0x93, 0xe0,
        0x03, 0xff, 0x51, 0x03, 0x06, 0x1a, 0x80
    });
    const std::vector<uint8_t> track1 = midi_test::withEot({
        0x00, 0xff, 0x51, 0x03, 0x0f, 0x42, 0x40
    });
    const auto bytes = midi_test::smf(2, 3, { track0, track1 });
    midi_test::TempDirectory temp("format2");
    const auto path = temp.file("patterns.mid");
    EXPECT_TRUE(midi_test::writeBytes(path, bytes));

    MidiFile midi;
    EXPECT_TRUE(midi.read(path));
    EXPECT_EQ(midi.format(), 2);
    const auto map0 = midi.tempoMap(0);
    const auto map1 = midi.tempoMap(1);
    EXPECT_TRUE(map0.has_value());
    EXPECT_TRUE(map1.has_value());
    EXPECT_EQ(map0->size(), size_t(2));
    EXPECT_EQ((*map0)[0].tick, uint64_t(0));
    EXPECT_EQ((*map0)[0].microsecondsPerQuarter, uint32_t(300000));
    EXPECT_EQ((*map0)[1].tick, uint64_t(3));
    EXPECT_EQ((*map0)[1].microsecondsPerQuarter, uint32_t(400000));
    EXPECT_EQ((*map1)[0].microsecondsPerQuarter, uint32_t(1000000));
    EXPECT_TRUE(!midi.tempoMap(2).has_value());
    EXPECT_EQ(*midi.tickToMicroseconds(2, 0), uint64_t(200000));
    EXPECT_EQ(*midi.tickToMicroseconds(4, 0), uint64_t(433333));
    EXPECT_NEAR(*midi.tickToSeconds(4, 0), 13.0 / 30.0, 1e-15);
    EXPECT_EQ(*midi.tickToMicroseconds(3, 1), uint64_t(1000000));
    EXPECT_TRUE(!midi.write(temp.file("format2-write.mid")));
}

DSPARK_TEST(MidiFile_VLQ_PPQN_and_chunk_boundaries)
{
    const uint32_t deltas[] = {
        0u, 0x7fu, 0x80u, 0x3fffu, 0x4000u, 0x1fffffu,
        0x200000u, 0x0fffffffu
    };
    midi_test::TempDirectory temp("boundaries");
    bool all = true;
    for (size_t i = 0; i < std::size(deltas); ++i)
    {
        MidiFile source;
        bool ok = source.create(0, 96)
            && source.addChannelEvent(0, deltas[i], 0x90, 60, 64)
            && source.addMetaEvent(0, 0, 0x2f);
        const auto path = temp.file("vlq-" + std::to_string(i) + ".mid");
        ok = ok && source.write(path);
        MidiFile decoded;
        ok = ok && decoded.read(path)
            && decoded.tracks()[0].events[0].deltaTicks == deltas[i];
        all = midi_test::record(ok, "VLQ-" + std::to_string(deltas[i])) && all;
    }

    MidiFile ppqn1, ppqnMax;
    all = midi_test::record(ppqn1.create(0, 1), "PPQN-1") && all;
    all = midi_test::record(ppqnMax.create(0, 0x7fff), "PPQN-32767") && all;

    const auto exact = midi_test::smf(0, 96,
        { midi_test::withEot({ 0x00, 0x90, 60, 64 }) });
    const auto extended = midi_test::smf(0, 96,
        { midi_test::withEot({}) }, { 0xaa, 0x55 }, { 1, 2, 3, 4, 5 });
    const auto exactPath = temp.file("exact.mid");
    const auto extendedPath = temp.file("extended.mid");
    MidiFile parser;
    all = midi_test::record(midi_test::writeBytes(exactPath, exact)
                            && parser.read(exactPath), "exact-track-chunk-end") && all;
    all = midi_test::record(midi_test::writeBytes(extendedPath, extended)
                            && parser.read(extendedPath), "header-extension") && all;
    all = midi_test::record(parser.tracks().size() == 1,
                            "bounded-alien-chunk") && all;
    EXPECT_TRUE(all);
}

DSPARK_TEST(MidiFile_public_resource_policy_boundaries)
{
    static_assert(MidiFile::kMaxInputBytes == 256ull * 1024 * 1024);
    static_assert(MidiFile::kMaxTracks == uint32_t(4096));
    static_assert(MidiFile::kMaxEvents == uint64_t(2000000));
    static_assert(MidiFile::kMaxAggregatePayloadBytes == 128ull * 1024 * 1024);
    static_assert(MidiFile::kMaxTrackChunkBytes == 128ull * 1024 * 1024);

    MidiFile exactTracks;
    EXPECT_TRUE(exactTracks.create(1, 96, MidiFile::kMaxTracks));
    EXPECT_EQ(exactTracks.tracks().size(), size_t(MidiFile::kMaxTracks));
    EXPECT_TRUE(!exactTracks.addTrack().has_value());
    EXPECT_TRUE(!exactTracks.create(1, 96,
                                    static_cast<size_t>(MidiFile::kMaxTracks) + 1));
    EXPECT_EQ(exactTracks.tracks().size(), size_t(MidiFile::kMaxTracks));

    MidiFile delta;
    EXPECT_TRUE(delta.create(0, 96));
    EXPECT_TRUE(delta.addChannelEvent(0, 0x0fffffff, 0x90, 60, 64));
    EXPECT_TRUE(!delta.addChannelEvent(0, 0x10000000, 0x90, 60, 64));
    EXPECT_EQ(delta.tracks()[0].events.size(), size_t(1));

    midi_test::TempDirectory temp("midi-resource");
    const auto inputOver = temp.file("input-over.mid");
    {
        std::ofstream output(inputOver, std::ios::binary | std::ios::trunc);
        output.seekp(static_cast<std::streamoff>(MidiFile::kMaxInputBytes));
        output.put(0);
    }
    MidiFile parser;
    EXPECT_TRUE(!parser.read(inputOver));
    EXPECT_EQ(parser.format(), -1);

    auto lengthOver = midi_test::smf(0, 96, { midi_test::withEot({}) });
    lengthOver[18] = 0x08;
    lengthOver[19] = 0x00;
    lengthOver[20] = 0x00;
    lengthOver[21] = 0x01;
    EXPECT_TRUE(midi_test::reject(lengthOver, "track-chunk-cap-over"));

    auto alienOver = midi_test::smf(0, 96, { midi_test::withEot({}) }, {}, { 0 });
    alienOver[18] = 0x08;
    alienOver[19] = 0x00;
    alienOver[20] = 0x00;
    alienOver[21] = 0x01;
    EXPECT_TRUE(midi_test::reject(alienOver, "alien-chunk-cap-over"));
}

DSPARK_TEST(MidiFile_malformed_corpus_58_named_cases)
{
    using midi_test::smf;
    using midi_test::withEot;
    const auto valid = smf(0, 96, { withEot({}) });
    std::vector<std::pair<std::string, std::vector<uint8_t>>> cases;

    auto changed = [&](const std::string& name, size_t offset, uint8_t value) {
        auto bytes = valid; bytes[offset] = value; cases.push_back({ name, std::move(bytes) });
    };
    changed("bad-magic", 0, 'X');
    auto shortHeader = valid; shortHeader.resize(13); cases.push_back({ "header-truncated", shortHeader });
    changed("header-length-five", 7, 5);
    auto headerOverrun = valid; headerOverrun[7] = 100; cases.push_back({ "header-extension-overrun", headerOverrun });
    changed("unknown-format", 9, 3);
    changed("zero-track-count", 11, 0);
    auto format0Two = valid; format0Two[11] = 2; cases.push_back({ "format0-two-tracks", format0Two });
    changed("zero-ppqn", 13, 0);
    changed("smpte-division", 12, 0xe2);
    auto shortChunkHeader = valid; shortChunkHeader.resize(20); cases.push_back({ "track-header-truncated", shortChunkHeader });
    auto chunkOverrun = valid; chunkOverrun[21] = 5; cases.push_back({ "track-chunk-overrun", chunkOverrun });
    auto extraTrack = valid; extraTrack.insert(extraTrack.end(), valid.begin() + 14, valid.end()); cases.push_back({ "extra-mtrk", extraTrack });
    auto trailing = valid; trailing.push_back(0); cases.push_back({ "trailing-junk", trailing });
    auto duplicateHeader = valid; duplicateHeader.insert(duplicateHeader.end(), { 'M','T','h','d',0,0,0,0 }); cases.push_back({ "duplicate-mthd", duplicateHeader });
    cases.push_back({ "missing-eot", smf(0, 96, { { 0x00, 0x90, 60, 64 } }) });
    cases.push_back({ "duplicate-eot", smf(0, 96, { { 0x00,0xff,0x2f,0, 0x00,0xff,0x2f,0 } }) });
    cases.push_back({ "nonfinal-eot", smf(0, 96, { { 0x00,0xff,0x2f,0, 0x00,0x90,60,64 } }) });
    cases.push_back({ "nonempty-eot", smf(0, 96, { { 0x00,0xff,0x2f,1,0, 0x00,0xff,0x2f,0 } }) });
    cases.push_back({ "eot-overlong-length", smf(0, 96, { { 0x00,0xff,0x2f,0x80,0x00 } }) });

    for (uint8_t status : { uint8_t(0xf1), uint8_t(0xf2), uint8_t(0xf3),
                            uint8_t(0xf4), uint8_t(0xf5), uint8_t(0xf6),
                            uint8_t(0xf8), uint8_t(0xf9), uint8_t(0xfa),
                            uint8_t(0xfb), uint8_t(0xfc), uint8_t(0xfd),
                            uint8_t(0xfe) })
    {
        cases.push_back({ "illegal-system-status-" + std::to_string(status),
                          smf(0, 96, { withEot({ 0x00, status }) }) });
    }

    cases.push_back({ "running-status-at-start", smf(0, 96, { withEot({ 0x00,60,64 }) }) });
    cases.push_back({ "running-after-meta", smf(0, 96, { withEot({ 0x00,0x90,60,64, 0x00,0xff,1,0, 0x00,61,64 }) }) });
    cases.push_back({ "running-after-f0", smf(0, 96, { withEot({ 0x00,0x90,60,64, 0x00,0xf0,1,0xf7, 0x00,61,64 }) }) });
    cases.push_back({ "running-after-f7", smf(0, 96, { withEot({ 0x00,0x90,60,64, 0x00,0xf7,1,1, 0x00,61,64 }) }) });
    cases.push_back({ "open-sysex-at-end", smf(0, 96, { { 0x00,0xf0,1,1 } }) });
    cases.push_back({ "channel-inside-sysex", smf(0, 96, { withEot({ 0x00,0xf0,1,1, 0x00,0x90,60,64 }) }) });
    cases.push_back({ "meta-inside-sysex", smf(0, 96, { withEot({ 0x00,0xf0,1,1, 0x00,0xff,1,0 }) }) });
    cases.push_back({ "f0-inside-sysex", smf(0, 96, { withEot({ 0x00,0xf0,1,1, 0x00,0xf0,1,0xf7 }) }) });
    cases.push_back({ "channel-data1-high", smf(0, 96, { withEot({ 0x00,0x90,0x80,0 }) }) });
    cases.push_back({ "channel-data2-high", smf(0, 96, { withEot({ 0x00,0x90,60,0x80 }) }) });
    cases.push_back({ "channel-one-byte-incomplete", smf(0, 96, { { 0x00,0xc0 } }) });
    cases.push_back({ "channel-two-byte-incomplete", smf(0, 96, { { 0x00,0x90,60 } }) });
    cases.push_back({ "delta-vlq-overlong", smf(0, 96, { { 0x80,0x00,0xff,0x2f,0 } }) });
    cases.push_back({ "delta-vlq-five-byte", smf(0, 96, { { 0x81,0x80,0x80,0x80,0x00,0xff,0x2f,0 } }) });
    cases.push_back({ "delta-vlq-unterminated", smf(0, 96, { { 0x81 } }) });
    cases.push_back({ "meta-length-overlong", smf(0, 96, { { 0,0xff,1,0x80,0, 0,0xff,0x2f,0 } }) });
    cases.push_back({ "meta-length-unterminated", smf(0, 96, { { 0,0xff,1,0x81 } }) });
    cases.push_back({ "meta-length-overrun", smf(0, 96, { { 0,0xff,1,2,0 } }) });
    cases.push_back({ "sysex-length-overlong", smf(0, 96, { { 0,0xf7,0x80,0, 0,0xff,0x2f,0 } }) });
    cases.push_back({ "sysex-length-unterminated", smf(0, 96, { { 0,0xf7,0x81 } }) });
    cases.push_back({ "sysex-length-overrun", smf(0, 96, { { 0,0xf7,2,0 } }) });
    cases.push_back({ "sysex-length-missing", smf(0, 96, { { 0,0xf0 } }) });
    cases.push_back({ "zero-tempo", smf(0, 96, { withEot({ 0,0xff,0x51,3,0,0,0 }) }) });
    cases.push_back({ "meta-type-high", smf(0, 96, { withEot({ 0,0xff,0x80,0 }) }) });
    cases.push_back({ "empty-track-chunk", smf(0, 96, { {} }) });
    auto hugeTrack = valid; hugeTrack[18] = 0x08; hugeTrack[19] = 0x00;
    hugeTrack[20] = 0x00; hugeTrack[21] = 0x01;
    cases.push_back({ "track-cap-over", hugeTrack });
    auto hugeAlien = smf(0, 96, { withEot({}) }, {}, { 0 });
    hugeAlien[18] = 0x08; hugeAlien[19] = 0x00; hugeAlien[20] = 0x00; hugeAlien[21] = 0x01;
    cases.push_back({ "alien-cap-over", hugeAlien });

    EXPECT_TRUE(cases.size() >= 58);
    bool all = true;
    for (const auto& test : cases)
        all = midi_test::record(midi_test::reject(test.second, test.first), test.first) && all;
    EXPECT_TRUE(all);
}

DSPARK_TEST(MidiFile_running_status_is_cancelled_by_meta)
{
    const auto bytes = midi_test::smf(0, 96, { midi_test::withEot({
        0,0x90,60,64, 0,0xff,1,0, 0,61,64
    }) });
    EXPECT_TRUE(midi_test::reject(bytes, "mut-running-meta"));
}

DSPARK_TEST(MidiFile_rejects_noncanonical_VLQ)
{
    const auto bytes = midi_test::smf(0, 96,
        { { 0x80,0x00,0xff,0x2f,0x00 } });
    EXPECT_TRUE(midi_test::reject(bytes, "mut-vlq"));
}

DSPARK_TEST(MidiFile_requires_one_final_EOT)
{
    const auto bytes = midi_test::smf(0, 96,
        { { 0x00,0x90,60,64 } });
    EXPECT_TRUE(midi_test::reject(bytes, "mut-eot"));
}

DSPARK_TEST(MidiFile_enforces_track_chunk_remaining_bytes)
{
    const auto bytes = midi_test::smf(0, 96,
        { { 0x00,0xff,0x01,0x04,0x41, 0x00,0xff,0x2f,0x00 } });
    EXPECT_TRUE(midi_test::reject(bytes, "mut-track-remaining"));
}

DSPARK_TEST(MidiFile_rejects_SMPTE_division)
{
    const auto bytes = midi_test::smf(0, 0xe250,
        { midi_test::withEot({}) });
    EXPECT_TRUE(midi_test::reject(bytes, "mut-smpte"));
}

DSPARK_TEST(MidiFile_hostile_mutation_campaign_20000)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(DSPARK_TEST_FIXTURE_DIR) / "midi";
    const std::vector<uint8_t> corpus[] = {
        midi_test::readBytes(root / "rp001-format0.mid"),
        midi_test::readBytes(root / "rp001-format1.mid")
    };
    EXPECT_TRUE(!corpus[0].empty() && !corpus[1].empty());

    uint64_t state = 0x4d4944495f323031ull;
    uint64_t mutationHash = 1469598103934665603ull;
    uint64_t accepted = 0, rejected = 0;
    int64_t worstMicros = 0;
    const auto campaignStart = std::chrono::steady_clock::now();
    for (size_t caseIndex = 0; caseIndex < 20000; ++caseIndex)
    {
        auto next = [&]() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        };
        std::vector<uint8_t> bytes = corpus[caseIndex & 1u];
        const unsigned edits = 1u + static_cast<unsigned>(next() % 4u);
        for (unsigned edit = 0; edit < edits; ++edit)
        {
            const size_t pos = static_cast<size_t>(next() % bytes.size());
            bytes[pos] ^= static_cast<uint8_t>(1u << (next() & 7u));
        }
        mutationHash = midi_test::fnv1a(mutationHash, bytes);

        midi_test::TempDirectory temp("mutation-" + std::to_string(caseIndex));
        const auto path = temp.file("mutated.mid");
        EXPECT_TRUE(midi_test::writeBytes(path, bytes));
        MidiFile file;
        const auto start = std::chrono::steady_clock::now();
        const bool ok = file.read(path);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        worstMicros = std::max(worstMicros, elapsed);
        EXPECT_TRUE(elapsed < 2000000);
        if (ok) ++accepted;
        else
        {
            ++rejected;
            EXPECT_EQ(file.format(), -1);
            EXPECT_TRUE(file.tracks().empty());
        }
    }
    const auto campaignSeconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - campaignStart).count();
    EXPECT_TRUE(campaignSeconds < 180);
    EXPECT_EQ(accepted + rejected, uint64_t(20000));
    std::cout << "MIDI mutation seed=0x4d4944495f323031 cases=20000 accepted="
              << accepted << " rejected=" << rejected << " corpus-fnv64="
              << mutationHash << " worst-us=" << worstMicros << "\n";
}

// ============================================================================
// FlacFile - RFC 9639, independent fixtures, hardening and range semantics
// ============================================================================

namespace flac_test {

struct Fixture
{
    const char* name;
    uint32_t sampleRate;
    uint32_t channels;
    uint32_t bits;
    int64_t frames;
};

constexpr std::array fixtures {
    Fixture { "rfc9639-d1", 44100, 2, 16, 1 },
    Fixture { "rfc9639-d2", 44100, 2, 16, 19 },
    Fixture { "rfc9639-d3", 32000, 1, 8, 24 },
    Fixture { "libflac-1ch-8bit", 48000, 1, 8, 64 },
    Fixture { "libflac-1ch-16bit", 48000, 1, 16, 64 },
    Fixture { "libflac-1ch-24bit", 48000, 1, 24, 64 },
    Fixture { "libflac-1ch-32bit", 48000, 1, 32, 64 },
    Fixture { "libflac-2ch-8bit", 48000, 2, 8, 64 },
    Fixture { "libflac-2ch-16bit", 48000, 2, 16, 64 },
    Fixture { "libflac-2ch-24bit", 48000, 2, 24, 64 },
    Fixture { "libflac-2ch-32bit", 48000, 2, 32, 64 },
    Fixture { "decl-constant", 48000, 1, 16, 32 },
    Fixture { "decl-verbatim-wasted", 48000, 1, 16, 32 },
    Fixture { "decl-fixed0-rice4", 48000, 1, 16, 32 },
    Fixture { "decl-fixed1-rice5", 48000, 1, 16, 32 },
    Fixture { "decl-fixed2-escape0", 48000, 1, 16, 32 },
    Fixture { "decl-fixed3-escape31", 48000, 1, 24, 32 },
    Fixture { "decl-fixed4-partitioned", 48000, 1, 24, 32 },
    Fixture { "decl-lpc1", 48000, 1, 16, 32 },
    Fixture { "decl-lpc12", 48000, 1, 24, 32 },
    Fixture { "decl-lpc32-mid-side", 96000, 2, 32, 33 },
    Fixture { "decl-left-side", 44100, 2, 16, 32 },
    Fixture { "decl-side-right", 44100, 2, 16, 32 },
    Fixture { "decl-mid-side-negative-odd", 44100, 2, 16, 32 },
    Fixture { "decl-variable-uncommon-rate", 12345, 1, 16, 33 },
    Fixture { "decl-rice-unary-cap", 48000, 1, 32, 16 },
    Fixture { "decl-zero-md5-unknown-total", 48000, 1, 16, 16 },
    Fixture { "decl-depth4-rate-code12", 48000, 1, 4, 32 },
    Fixture { "decl-depth12-rate-code14", 44100, 1, 12, 32 },
    Fixture { "decl-depth20", 88200, 1, 20, 32 },
    Fixture { "decl-eight-channel", 48000, 8, 16, 16 },
};

std::filesystem::path root()
{
    return std::filesystem::path(DSPARK_TEST_FIXTURE_DIR) / "flac";
}

std::vector<int32_t> readPcm32(const std::filesystem::path& path)
{
    const std::vector<uint8_t> bytes = midi_test::readBytes(path);
    if ((bytes.size() & 3u) != 0) return {};
    std::vector<int32_t> samples;
    samples.reserve(bytes.size() / 4);
    for (size_t offset = 0; offset < bytes.size(); offset += 4)
    {
        const uint32_t representation = static_cast<uint32_t>(bytes[offset])
            | (static_cast<uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<uint32_t>(bytes[offset + 2]) << 16)
            | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        samples.push_back(std::bit_cast<int32_t>(representation));
    }
    return samples;
}

bool verifyFixture(const Fixture& fixture)
{
    const auto fixtureRoot = root();
    const auto source = readPcm32(fixtureRoot / (std::string(fixture.name) + ".source.pcm"));
    const auto reference = readPcm32(
        fixtureRoot / (std::string(fixture.name) + ".reference.pcm"));
    if (source != reference
        || source.size() != static_cast<size_t>(fixture.frames) * fixture.channels)
        return false;

    FlacFile file;
    if (!file.openRead(fixtureRoot / (std::string(fixture.name) + ".flac")))
        return false;
    const AudioFileInfo info = file.getInfo();
    if (info.sampleRate != fixture.sampleRate || info.numChannels != fixture.channels
        || info.bitsPerSample != fixture.bits || info.numSamples != fixture.frames
        || info.isFloatingPoint)
        return false;

    AudioBuffer<float> output;
    output.resize(static_cast<int>(fixture.channels), static_cast<int>(fixture.frames));
    if (!file.readSamples(output.toView())) return false;
    for (int64_t sample = 0; sample < fixture.frames; ++sample)
    {
        for (uint32_t channel = 0; channel < fixture.channels; ++channel)
        {
            const int32_t integer = source[static_cast<size_t>(sample) * fixture.channels + channel];
            const float expected = std::ldexp(static_cast<float>(integer),
                                               1 - static_cast<int>(fixture.bits));
            if (std::bit_cast<uint32_t>(output.getChannel(static_cast<int>(channel))[sample])
                != std::bit_cast<uint32_t>(expected))
                return false;
        }
    }
    return true;
}

const Fixture* named(const std::string& name)
{
    const auto found = std::find_if(fixtures.begin(), fixtures.end(),
        [&](const Fixture& fixture) { return name == fixture.name; });
    return found == fixtures.end() ? nullptr : &*found;
}

bool verifyNamed(const char* name)
{
    const Fixture* fixture = named(name);
    return fixture != nullptr && verifyFixture(*fixture);
}

bool rejectsNamed(const char* name)
{
    FlacFile file;
    return !file.openRead(root() / "malformed" / (std::string(name) + ".flac"))
        && !file.isOpen() && file.getInfo().numChannels == 0;
}

uint8_t crc8(std::span<const uint8_t> bytes)
{
    uint8_t value = 0;
    for (uint8_t byte : bytes)
    {
        value ^= byte;
        for (unsigned bit = 0; bit < 8; ++bit)
            value = static_cast<uint8_t>((value & 0x80u) != 0
                ? static_cast<uint8_t>((value << 1) ^ 0x07u)
                : static_cast<uint8_t>(value << 1));
    }
    return value;
}

uint16_t crc16(std::span<const uint8_t> bytes)
{
    uint16_t value = 0;
    for (uint8_t byte : bytes)
    {
        value ^= static_cast<uint16_t>(byte) << 8;
        for (unsigned bit = 0; bit < 8; ++bit)
            value = static_cast<uint16_t>((value & 0x8000u) != 0
                ? static_cast<uint16_t>((value << 1) ^ 0x8005u)
                : static_cast<uint16_t>(value << 1));
    }
    return value;
}

std::vector<uint8_t> noncanonicalCodedNumber()
{
    std::vector<uint8_t> bytes = midi_test::readBytes(root() / "decl-constant.flac");
    constexpr size_t frame = 42;
    constexpr size_t coded = frame + 4;
    bytes[coded] = 0xc0;
    bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(coded + 1), 0x80);
    constexpr size_t crcPosition = frame + 7;
    bytes[crcPosition] = crc8(std::span<const uint8_t>(bytes.data() + frame,
                                                       crcPosition - frame));
    const uint16_t footer = crc16(std::span<const uint8_t>(
        bytes.data() + frame, bytes.size() - frame - 2));
    bytes[bytes.size() - 2] = static_cast<uint8_t>(footer >> 8);
    bytes.back() = static_cast<uint8_t>(footer);
    return bytes;
}

} // namespace flac_test

DSPARK_TEST(FlacFile_complete_positive_corpus_is_integer_and_float_exact)
{
    bool all = true;
    for (const auto& fixture : flac_test::fixtures)
        all = midi_test::record(flac_test::verifyFixture(fixture), fixture.name) && all;
    EXPECT_TRUE(all);
}

DSPARK_TEST(FlacFile_indexed_range_and_untouched_excess_semantics)
{
    const auto* fixture = flac_test::named("decl-variable-uncommon-rate");
    EXPECT_TRUE(fixture != nullptr);
    FlacFile file;
    EXPECT_TRUE(file.openRead(flac_test::root() / "decl-variable-uncommon-rate.flac"));
    const auto source = flac_test::readPcm32(
        flac_test::root() / "decl-variable-uncommon-rate.source.pcm");

    AudioBuffer<float> range;
    range.resize(2, 12);
    for (int channel = 0; channel < 2; ++channel)
        std::fill_n(range.getChannel(channel), 12, 123.25f);
    EXPECT_TRUE(file.readSamples(range.toView(), 13, 9));
    for (int sample = 0; sample < 9; ++sample)
    {
        const float expected = std::ldexp(static_cast<float>(source[13 + sample]), -15);
        EXPECT_EQ(std::bit_cast<uint32_t>(range.getChannel(0)[sample]),
                  std::bit_cast<uint32_t>(expected));
    }
    for (int sample = 9; sample < 12; ++sample)
        EXPECT_EQ(range.getChannel(0)[sample], 123.25f);
    for (int sample = 0; sample < 12; ++sample)
        EXPECT_EQ(range.getChannel(1)[sample], 123.25f);

    AudioBuffer<float> empty;
    EXPECT_TRUE(file.readSamples(empty.toView(), 0, 1));
    EXPECT_TRUE(!file.readSamples(range.toView(), -1, 1));
    EXPECT_TRUE(!file.readSamples(range.toView(), 32, 2));
    EXPECT_TRUE(!file.readSamples(range.toView(), 0, 0));

    AudioBuffer<float> repeat;
    repeat.resize(1, 9);
    EXPECT_TRUE(file.readSamples(repeat.toView(), 13, 9));
    for (int sample = 0; sample < 9; ++sample)
        EXPECT_EQ(std::bit_cast<uint32_t>(repeat.getChannel(0)[sample]),
                  std::bit_cast<uint32_t>(range.getChannel(0)[sample]));
}

DSPARK_TEST(FlacFile_decode_only_and_transactional_lifecycle)
{
    midi_test::TempDirectory temp("flac-lifecycle");
    const auto target = temp.file("preserve.bin");
    const std::vector<uint8_t> sentinel { 1, 3, 3, 7 };
    EXPECT_TRUE(midi_test::writeBytes(target, sentinel));

    FlacFile file;
    EXPECT_TRUE(file.openRead(flac_test::root() / "rfc9639-d1.flac"));
    AudioFileInfo info;
    info.sampleRate = 48000;
    info.numChannels = 2;
    info.bitsPerSample = 16;
    EXPECT_TRUE(!file.openWrite(target, info));
    EXPECT_TRUE(!file.isOpen());
    EXPECT_TRUE(midi_test::readBytes(target) == sentinel);

    AudioBuffer<float> samples;
    samples.resize(1, 1);
    EXPECT_TRUE(!file.writeSamples(std::as_const(samples).toView()));
    EXPECT_TRUE(file.openRead(flac_test::root() / "rfc9639-d3.flac"));
    EXPECT_TRUE(!file.openRead(flac_test::root() / "malformed" / "marker-byte-0.flac"));
    EXPECT_TRUE(!file.isOpen());
    EXPECT_EQ(file.getInfo().numChannels, 0u);
    EXPECT_TRUE(!file.readSamples(samples.toView()));
    file.close();
    file.close();
}

DSPARK_TEST(FlacFile_surgical_malformed_corpus)
{
    size_t cases = 0;
    bool all = true;
    for (const auto& entry : std::filesystem::directory_iterator(
             flac_test::root() / "malformed"))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".flac")
            continue;
        ++cases;
        FlacFile file;
        const bool rejected = !file.openRead(entry.path()) && !file.isOpen()
            && file.getInfo().numChannels == 0;
        all = midi_test::record(rejected, entry.path().filename().string()) && all;
    }
    EXPECT_TRUE(cases >= 48);
    EXPECT_TRUE(all);
}

DSPARK_TEST(FlacFile_public_resource_policy_boundaries)
{
    static_assert(FlacFile::kMaxInputBytes == 256ull * 1024 * 1024);
    static_assert(FlacFile::kMaxMetadataBlocks == uint64_t(65536));
    static_assert(FlacFile::kMaxFrames == uint64_t(1048576));
    static_assert(FlacFile::kMaxInterchannelSamples == uint64_t(1) << 31);
    static_assert(FlacFile::kMaxDecodedPcmBytes == uint64_t(8) << 30);
    static_assert(FlacFile::kMaxRiceUnaryZeros == uint64_t(1) << 20);

    midi_test::TempDirectory temp("flac-resource");
    const auto inputOver = temp.file("input-over.flac");
    {
        std::ofstream output(inputOver, std::ios::binary | std::ios::trunc);
        output.seekp(static_cast<std::streamoff>(FlacFile::kMaxInputBytes));
        output.put(0);
    }
    FlacFile file;
    EXPECT_TRUE(!file.openRead(inputOver));

    const auto base = midi_test::readBytes(flac_test::root() / "decl-constant.flac");
    auto metadataFile = [&](uint64_t blockCount, const char* name)
        -> std::filesystem::path {
        std::vector<uint8_t> bytes(base.begin(), base.begin() + 42);
        bytes[4] = 0;
        for (uint64_t block = 1; block < blockCount; ++block)
        {
            const uint8_t type = static_cast<uint8_t>(
                (block + 1 == blockCount ? 0x80u : 0u) | 1u);
            bytes.insert(bytes.end(), { type, 0, 0, 0 });
        }
        bytes.insert(bytes.end(), base.begin() + 42, base.end());
        const auto path = temp.file(name);
        return midi_test::writeBytes(path, bytes) ? path : std::filesystem::path {};
    };
    const auto exact = metadataFile(FlacFile::kMaxMetadataBlocks, "metadata-exact.flac");
    EXPECT_TRUE(!exact.empty());
    EXPECT_TRUE(file.openRead(exact));
    file.close();
    const auto over = metadataFile(FlacFile::kMaxMetadataBlocks + 1,
                                   "metadata-over.flac");
    EXPECT_TRUE(!over.empty());
    EXPECT_TRUE(!file.openRead(over));
    EXPECT_TRUE(flac_test::verifyNamed("decl-rice-unary-cap"));
    EXPECT_TRUE(flac_test::rejectsNamed("rice-unary-cap-over"));
}

DSPARK_TEST(FlacFile_MUT_F01_crc8_is_load_bearing)
{
    EXPECT_TRUE(flac_test::rejectsNamed("frame-crc8"));
}

DSPARK_TEST(FlacFile_MUT_F02_crc16_is_load_bearing)
{
    EXPECT_TRUE(flac_test::rejectsNamed("frame-crc16"));
}

DSPARK_TEST(FlacFile_MUT_F03_side_width_is_load_bearing)
{
    EXPECT_TRUE(flac_test::verifyNamed("decl-left-side"));
    EXPECT_TRUE(flac_test::verifyNamed("decl-side-right"));
}

DSPARK_TEST(FlacFile_MUT_F04_mid_side_odd_rounding_is_load_bearing)
{
    EXPECT_TRUE(flac_test::verifyNamed("decl-mid-side-negative-odd"));
}

DSPARK_TEST(FlacFile_MUT_F05_lpc32_checked_width_is_load_bearing)
{
    EXPECT_TRUE(flac_test::verifyNamed("decl-lpc32-mid-side"));
}

DSPARK_TEST(FlacFile_MUT_F06_rice_unfolding_is_load_bearing)
{
    EXPECT_TRUE(flac_test::verifyNamed("decl-fixed0-rice4"));
}

DSPARK_TEST(FlacFile_MUT_F07_escape_dispatch_is_load_bearing)
{
    EXPECT_TRUE(flac_test::verifyNamed("decl-fixed3-escape31"));
}

DSPARK_TEST(FlacFile_MUT_F08_wasted_restore_is_load_bearing)
{
    EXPECT_TRUE(flac_test::verifyNamed("decl-verbatim-wasted"));
}

DSPARK_TEST(FlacFile_MUT_F09_coded_number_minimality_is_load_bearing)
{
    midi_test::TempDirectory temp("flac-coded-number");
    const auto path = temp.file("overlong.flac");
    EXPECT_TRUE(midi_test::writeBytes(path, flac_test::noncanonicalCodedNumber()));
    FlacFile file;
    EXPECT_TRUE(!file.openRead(path));
}

DSPARK_TEST(FlacFile_MUT_F10_metadata_bounds_precede_use)
{
    EXPECT_TRUE(flac_test::rejectsNamed("streaminfo-length-16777215"));
}

DSPARK_TEST(FlacFile_MUT_F11_nonzero_md5_is_load_bearing)
{
    EXPECT_TRUE(flac_test::rejectsNamed("nonzero-md5-mismatch"));
}

DSPARK_TEST(FlacFile_hostile_mutation_campaign_20000)
{
    const std::vector<uint8_t> corpus[] = {
        midi_test::readBytes(flac_test::root() / "rfc9639-d1.flac"),
        midi_test::readBytes(flac_test::root() / "decl-lpc32-mid-side.flac"),
        midi_test::readBytes(flac_test::root() / "decl-variable-uncommon-rate.flac")
    };
    EXPECT_TRUE(!corpus[0].empty() && !corpus[1].empty() && !corpus[2].empty());

    uint64_t state = 0x464c41435f323031ull;
    uint64_t mutationHash = 1469598103934665603ull;
    uint64_t accepted = 0, rejected = 0;
    int64_t worstMicros = 0;
    const auto campaignStart = std::chrono::steady_clock::now();
    for (size_t caseIndex = 0; caseIndex < 20000; ++caseIndex)
    {
        auto next = [&]() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        };
        std::vector<uint8_t> bytes = corpus[caseIndex % 3];
        const unsigned edits = 1u + static_cast<unsigned>(next() % 4u);
        for (unsigned edit = 0; edit < edits; ++edit)
        {
            const size_t position = static_cast<size_t>(next() % bytes.size());
            bytes[position] ^= static_cast<uint8_t>(1u << (next() & 7u));
        }
        mutationHash = midi_test::fnv1a(mutationHash, bytes);

        midi_test::TempDirectory temp("flac-mutation-" + std::to_string(caseIndex));
        const auto path = temp.file("mutated.flac");
        EXPECT_TRUE(midi_test::writeBytes(path, bytes));
        FlacFile file;
        const auto start = std::chrono::steady_clock::now();
        const bool ok = file.openRead(path);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        worstMicros = std::max(worstMicros, elapsed);
        EXPECT_TRUE(elapsed < 2000000);
        if (ok) ++accepted;
        else
        {
            ++rejected;
            EXPECT_TRUE(!file.isOpen());
            EXPECT_EQ(file.getInfo().numChannels, 0u);
        }
    }
    const auto campaignSeconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - campaignStart).count();
    EXPECT_TRUE(campaignSeconds < 180);
    EXPECT_EQ(accepted + rejected, uint64_t(20000));
    std::cout << "FLAC mutation seed=0x464c41435f323031 cases=20000 accepted="
              << accepted << " rejected=" << rejected << " corpus-fnv64="
              << mutationHash << " worst-us=" << worstMicros << "\n";
}
