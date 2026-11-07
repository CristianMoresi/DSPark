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
constexpr const char* testIOArrivalTimeoutVariable =
    "DSPARK_TESTIO_COLLISION_ARRIVAL_TIMEOUT_MS";
constexpr const char* testIOSyncMutantVariable =
    "DSPARK_TESTIO_COLLISION_SYNC_MUTANT";

struct TestIOCollisionSynchronization
{
    std::filesystem::path root;
    uint64_t identity = 0;
    uint64_t participants = 0;
    std::chrono::milliseconds timeout { 0 };
    std::chrono::milliseconds arrivalTimeout { 0 };
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
    const uint64_t arrivalMilliseconds = parseTestIOSyncUnsigned(
        std::getenv(testIOArrivalTimeoutVariable), testIOArrivalTimeoutVariable);
    if (sync.participants < 2 || sync.participants > 64
        || sync.identity >= sync.participants
        || timeoutMilliseconds == 0 || timeoutMilliseconds > 60000
        || arrivalMilliseconds == 0
        || arrivalMilliseconds > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("DSPARK_TESTIO_SYNC_ERROR: invalid synchronization bounds");
    sync.timeout = std::chrono::milliseconds(timeoutMilliseconds);
    sync.arrivalTimeout = std::chrono::milliseconds(arrivalMilliseconds);
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

size_t countTestIOSyncDirectories(const TestIOCollisionSynchronization& sync,
                                  const char* prefix, const char* phase)
{
    size_t observed = 0;
    for (uint64_t id = 0; id < sync.participants; ++id)
    {
        const bool exists = testIOSyncDirectoryExists(
            sync.root / (prefix + std::to_string(id)), phase);
        if (exists) ++observed;
    }
    return observed;
}

size_t countTestIOReady(const TestIOCollisionSynchronization& sync)
{
    return countTestIOSyncDirectories(sync, "ready-", "readiness");
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
                              Counter counter, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;)
    {
        const size_t observed = counter(sync);
        if (observed == sync.participants) return observed;
        if (std::chrono::steady_clock::now() >= deadline) return observed;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void waitForTestIOArrival(const TestIOCollisionSynchronization& sync)
{
    // Earlier tests can finish at different times in each process. Arrival
    // uses the runner's existing whole-suite budget; its process-tree watchdog
    // remains the outer limit. Start the short protocol deadlines only after
    // all participants have reached this test, including deliberate mutants.
    publishTestIOSyncDirectory(
        sync.root / ("arrived-" + std::to_string(sync.identity)),
        "DSPARK_TESTIO_SYNC_ERROR: duplicate process identity "
            + std::to_string(sync.identity));
    const size_t arrived = waitForTestIOSyncPhase(sync,
        [](const auto& context) {
            return countTestIOSyncDirectories(context, "arrived-", "arrival");
        }, sync.arrivalTimeout);
    if (arrived != sync.participants)
        throw std::runtime_error(
            "DSPARK_TESTIO_SYNC_TIMEOUT: arrival expected="
            + std::to_string(sync.participants) + " observed="
            + std::to_string(arrived));
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
    waitForTestIOArrival(sync);
    if (sync.mutant != "missing-readiness")
        publishTestIOSyncDirectory(
            sync.root / ("ready-" + std::to_string(sync.identity)),
            "DSPARK_TESTIO_SYNC_ERROR: duplicate process identity "
                + std::to_string(sync.identity));
    const size_t ready = waitForTestIOSyncPhase(sync, countTestIOReady, sync.timeout);
    if (ready != sync.participants)
        throw std::runtime_error(
            "DSPARK_TESTIO_SYNC_TIMEOUT: readiness expected="
            + std::to_string(sync.participants) + " observed="
            + std::to_string(ready));
#endif

#if defined(DSPARK_TESTIO_FIXED_NAME_MUTANT)
    // Only this deliberate claim is shared. Every WAV/MP3/MIDI/FLAC and
    // integration artifact remains inside the process-unique root above.
    const std::filesystem::path claim =
        sync.root / "dspark_process_exclusive_claim";
#else
    const std::filesystem::path claim = "dspark_process_exclusive_claim";
#endif
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
    const size_t attempted = waitForTestIOSyncPhase(sync, countTestIOAttempted, sync.timeout);
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
    EXPECT_EQ(n, N);   // gapless: exactly the samples written
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

// part2_3_length is a 12-bit field, but the encoder let a mono granule use the
// frame's whole budget: 4096 bits at 320 kbps / 44.1 kHz, and more at 32 kHz.
// 4096 wrapped to 0 and the decoder dropped that granule and the rest of its
// frame: dense material came back with whole frames missing (0 dB residual
// where the other frames sat near -38 dB).
DSPARK_TEST(Mp3File_mono_high_bitrate_granules_fit_their_12_bit_length)
{
    for (const double rate : { 44100.0, 32000.0 })
    {
        FileCleanup cleanup { "dspark_test_mono320.mp3" };
        const int N = static_cast<int>(rate);   // 1 s
        std::vector<float> orig(static_cast<size_t>(N));
        uint32_t rng = 1u;
        for (auto& v : orig)
        {
            rng = rng * 1664525u + 1013904223u;
            v = 0.2f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
        }
        {
            Mp3File w;
            AudioFileInfo info;
            info.sampleRate = rate;
            info.numChannels = 1;
            info.bitsPerSample = 320;
            info.numSamples = N;
            EXPECT_TRUE(w.openWrite("dspark_test_mono320.mp3", info));
            AudioBuffer<float> buf;
            buf.resize(1, N);
            std::copy(orig.begin(), orig.end(), buf.getChannel(0));
            EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
            w.close();
        }
        Mp3File r;
        EXPECT_TRUE(r.openRead("dspark_test_mono320.mp3"));
        const int n = static_cast<int>(r.getInfo().numSamples);
        AudioBuffer<float> dec;
        dec.resize(1, n);
        EXPECT_TRUE(r.readSamples(dec.toView()));
        r.close();
        const float* d = dec.getChannel(0);

        int lag = 0;
        double best = -1.0;
        for (int l = 0; l <= 2400; ++l)
        {
            double sab = 0.0;
            for (int i = 4000; i < 12000 && i + l < n; ++i)
                sab += static_cast<double>(orig[static_cast<size_t>(i)]) * d[i + l];
            if (sab > best) { best = sab; lag = l; }
        }
        double worstDb = -300.0;
        for (int b = 1; (b + 2) * 1152 < N && (b + 1) * 1152 + lag <= n; ++b)
        {
            double e = 0.0, ref = 0.0;
            for (int i = b * 1152; i < (b + 1) * 1152; ++i)
            {
                const double x = orig[static_cast<size_t>(i)];
                const double y = d[i + lag];
                e += (y - x) * (y - x);
                ref += x * x;
            }
            worstDb = std::max(worstDb, 10.0 * std::log10(e / ref));
        }
        EXPECT_LT(worstDb, -20.0);   // old: 0 dB in the frames that hit 4096 bits
    }
}

// The encoder wrote only the frames its input filled, so its 528-sample delay
// plus the decoder's 529 cut the last ~230 samples off, and the decoder
// returned 1057 samples of codec delay ahead of the music (plus a frame of
// silence for any Xing/Info tag frame, which it decoded as audio). The
// encoder now flushes the delay and writes an Info frame with a gapless tag,
// and the decoder trims by it: exactly the samples written come back,
// time-aligned. Any encoder's tag is honoured the same way when its CRC-16
// checks out; a tag whose CRC does not is skipped but not trusted.
DSPARK_TEST(Mp3File_round_trip_is_gapless)
{
    FileCleanup cleanup { "dspark_test_gapless.mp3" };
    constexpr int N = 30001;   // not a multiple of the 1152-sample frame
    std::vector<float> l(N), r(N);
    for (int i = 0; i < N; ++i)
    {
        const double t = static_cast<double>(i) / 44100.0;
        l[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2.0 * pi<double> * 330.0 * t)
                                                      + 0.2 * std::sin(2.0 * pi<double> * 2710.0 * t));
        r[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2.0 * pi<double> * 470.0 * t)
                                                      + 0.2 * std::sin(2.0 * pi<double> * 5130.0 * t));
    }
    {
        Mp3File w;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 2;
        info.bitsPerSample = 192;
        info.numSamples = N;
        EXPECT_TRUE(w.openWrite("dspark_test_gapless.mp3", info));
        AudioBuffer<float> buf;
        buf.resize(2, N);
        std::copy(l.begin(), l.end(), buf.getChannel(0));
        std::copy(r.begin(), r.end(), buf.getChannel(1));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
    }

    auto decode = [](AudioBuffer<float>& dec) {
        Mp3File rd;
        if (!rd.openRead("dspark_test_gapless.mp3")) return -1;
        const int n = static_cast<int>(rd.getInfo().numSamples);
        dec.resize(2, n);
        if (!rd.readSamples(dec.toView())) return -1;
        return n;
    };
    auto residualDb = [&](const AudioBuffer<float>& dec, int lag, int from, int to) {
        double e = 0.0, ref = 0.0;
        for (int ch = 0; ch < 2; ++ch)
            for (int i = from; i < to; ++i)
            {
                const double x = (ch == 0 ? l : r)[static_cast<size_t>(i)];
                const double y = dec.getChannel(ch)[i + lag];
                e += (y - x) * (y - x);
                ref += x * x;
            }
        return 10.0 * std::log10(e / ref);
    };

    AudioBuffer<float> dec;
    EXPECT_EQ(decode(dec), N);                                 // exact length
    EXPECT_LT(residualDb(dec, 0, 0, N), -25.0);                // aligned at lag 0
    EXPECT_LT(residualDb(dec, 0, N - 500, N), -25.0);          // the tail is there
    EXPECT_GT(residualDb(dec, 1, 1000, N - 1000), -10.0);      // and lag 1 is not it

    // The first frame is the Info tag frame, naming this encoder.
    std::vector<char> bytes;
    {
        std::ifstream in("dspark_test_gapless.mp3", std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const std::string head(bytes.begin(), bytes.begin() + std::min<size_t>(bytes.size(), 400));
    const auto tag = head.find("DSPark");
    EXPECT_TRUE(head.find("Info") != std::string::npos);
    EXPECT_TRUE(tag != std::string::npos);

    // The tag CRC covers the frame up to the CRC field, 34 bytes into the tag.
    const size_t crcAt = tag + 34;
    auto rewriteTagName = [&](const char* name, bool fixCrc) {
        std::copy(name, name + 9, bytes.begin() + static_cast<std::ptrdiff_t>(tag));
        uint16_t crc = 0;
        for (size_t i = 0; i < crcAt; ++i)
        {
            crc = static_cast<uint16_t>(crc ^ static_cast<uint8_t>(bytes[i]));
            for (int b = 0; b < 8; ++b)
                crc = static_cast<uint16_t>((crc & 1u) ? (crc >> 1) ^ 0xA001u : (crc >> 1));
        }
        if (!fixCrc) crc = static_cast<uint16_t>(crc ^ 0x5A5Au);
        bytes[crcAt] = static_cast<char>(crc >> 8);
        bytes[crcAt + 1] = static_cast<char>(crc & 0xFF);
        std::ofstream out("dspark_test_gapless.mp3", std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };
    rewriteTagName("Other1.00", true);                          // another encoder, valid CRC: same trim
    EXPECT_EQ(decode(dec), N);
    rewriteTagName("Other1.00", false);                         // CRC broken: untrusted, raw stream
    const int raw = decode(dec);
    EXPECT_EQ(raw % 1152, 0);
    EXPECT_GT(raw, N + 1057);
    EXPECT_LT(residualDb(dec, 1057, 0, N), -25.0);              // the codec delay, untrimmed
}

// A NaN written to the MP3 encoder reached the quantiser as NaN: its
// float-to-int conversion, and then the negation of the INT_MIN it produced
// on x86, are undefined behaviour (the sanitizer builds flag both).
// Non-finite input now encodes as silence and the stream carries on.
DSPARK_TEST(Mp3File_nonfinite_input_does_not_poison_the_stream)
{
    FileCleanup cleanup { "dspark_test_nonfinite.mp3" };
    constexpr int N = 22050;
    std::vector<float> orig(N);
    for (int i = 0; i < N; ++i)
        orig[static_cast<size_t>(i)] = 0.4f * std::sin(2.0f * pi<float> * 440.0f * float(i) / 44100.0f);
    {
        Mp3File w;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 1;
        info.bitsPerSample = 128;
        info.numSamples = N;
        EXPECT_TRUE(w.openWrite("dspark_test_nonfinite.mp3", info));
        AudioBuffer<float> buf;
        buf.resize(1, N);
        std::copy(orig.begin(), orig.end(), buf.getChannel(0));
        buf.getChannel(0)[5000] = std::numeric_limits<float>::quiet_NaN();
        buf.getChannel(0)[5001] = std::numeric_limits<float>::infinity();
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
    }
    Mp3File r;
    EXPECT_TRUE(r.openRead("dspark_test_nonfinite.mp3"));
    const int n = static_cast<int>(r.getInfo().numSamples);
    AudioBuffer<float> dec;
    dec.resize(1, n);
    EXPECT_TRUE(r.readSamples(dec.toView()));
    EXPECT_NO_NAN(dec.getChannel(0), n);
    double e = 0.0, ref = 0.0;   // the tone survives after the burst
    for (int i = 10000; i < N; ++i)
    {
        const double d = dec.getChannel(0)[i] - orig[static_cast<size_t>(i)];
        e += d * d;
        ref += static_cast<double>(orig[static_cast<size_t>(i)]) * orig[static_cast<size_t>(i)];
    }
    EXPECT_LT(10.0 * std::log10(e / ref), -30.0);
}

// 8-bit WAV is unsigned with 128 as zero and the reader scales by 128, but the
// writer scaled by 127. Every 8-bit round trip therefore came back 0.78% quiet
// - a whole quantisation step of SYSTEMATIC error, on top of quantisation. With
// both sides on the same grid, every value that the 128-step scale can express
// round-trips to within half a step and -1.0 is exact; only +1.0 clamps, which
// the scale itself forces.
// A NaN sample reached static_cast<int>(NaN) - undefined behaviour, which on
// x86 wrote the most negative code (a full-scale click) into integer WAVs.
// Integer formats now write NaN as silence and clamp +-Inf to full scale.
DSPARK_TEST(WavFile_nonfinite_samples_write_as_silence_or_full_scale)
{
    for (const int bits : { 8, 16, 24, 32 })
    {
        FileCleanup cleanup { "dspark_test_nonfinite.wav" };
        const float probes[] = { std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(), 0.5f };
        {
            WavFile w;
            AudioFileInfo info;
            info.sampleRate = 44100.0;
            info.numChannels = 1;
            info.bitsPerSample = bits;
            info.isFloatingPoint = false;
            info.numSamples = 4;
            EXPECT_TRUE(w.openWrite("dspark_test_nonfinite.wav", info));
            AudioBuffer<float> b;
            b.resize(1, 4);
            for (int i = 0; i < 4; ++i) b.getChannel(0)[i] = probes[i];
            EXPECT_TRUE(w.writeSamples(std::as_const(b).toView()));
            w.close();
        }
        WavFile r;
        EXPECT_TRUE(r.openRead("dspark_test_nonfinite.wav"));
        AudioBuffer<float> b;
        b.resize(1, 4);
        EXPECT_TRUE(r.readSamples(b.toView()));
        r.close();
        const float step = std::ldexp(1.0f, 1 - bits);
        EXPECT_EQ(b.getChannel(0)[0], 0.0f);                  // NaN -> silence
        EXPECT_NEAR(b.getChannel(0)[1], 1.0f, 1.01f * step);  // +Inf -> top code
        EXPECT_EQ(b.getChannel(0)[2], -1.0f);                 // -Inf -> bottom code
        EXPECT_NEAR(b.getChannel(0)[3], 0.5f, step);
    }
}

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
// the long region. Common encoders never emit mixed blocks, so no recorded fixture reaches
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

// ============================================================================
// 2026-11 IO audit: regression pins. Each case failed on the headers before
// the fixes it names; the comment states the defect and the measured number.
// ============================================================================

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <sys/resource.h>
#endif
#include <tuple>

namespace io_audit {

// A Layer III frame of any version: 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5.
// mode 3 is mono, 1 joint stereo. The side info and main data come from the
// callbacks; the header is written here and the frame zero-padded to size.
template <typename SideFn, typename MainFn>
std::vector<uint8_t> craftFrame(int version, int bitrateIndex, int rateIndex, int mode,
                                int modeExt, SideFn side, MainFn main)
{
    static constexpr int kMpeg1[16] = {0,32,40,48,56,64,80,96,112,128,160,192,224,256,320,0};
    static constexpr int kLsf[16] = {0,8,16,24,32,40,48,56,64,80,96,112,128,144,160,0};
    static constexpr int kRates[4][3] = { {11025,12000,8000}, {0,0,0},
                                          {22050,24000,16000}, {44100,48000,32000} };
    const bool lsf = version != 3;
    const int rate = kRates[version][rateIndex];
    const int bytes = (lsf ? 72 : 144) * (lsf ? kLsf : kMpeg1)[bitrateIndex] * 1000 / rate;
    const int sideBytes = lsf ? (mode == 3 ? 9 : 17) : (mode == 3 ? 17 : 32);
    Mp3TestBits b;
    b.put(0x7FF, 11); b.put(uint32_t(version), 2); b.put(1, 2); b.put(1, 1);
    b.put(uint32_t(bitrateIndex), 4); b.put(uint32_t(rateIndex), 2); b.put(0, 1); b.put(0, 1);
    b.put(uint32_t(mode), 2); b.put(uint32_t(modeExt), 2); b.put(0, 1); b.put(1, 1); b.put(0, 2);
    side(b);
    while (b.bitPos < size_t(4 + sideBytes) * 8) b.put(0, 1);
    main(b);
    std::vector<uint8_t> frame = b.bytes;
    frame.resize(size_t(bytes), 0);
    return frame;
}

void writeFrames(const std::filesystem::path& path, const std::vector<std::vector<uint8_t>>& frames)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    for (const auto& frame : frames)
        f.write(reinterpret_cast<const char*>(frame.data()), std::streamsize(frame.size()));
}

struct Decoded
{
    bool ok = false;
    double rate = 0.0;
    int channels = 0;
    std::vector<std::vector<float>> pcm;
};

Decoded decode(const std::filesystem::path& path)
{
    Decoded d;
    Mp3File r;
    if (!r.openRead(path)) return d;
    const auto info = r.getInfo();
    d.rate = info.sampleRate;
    d.channels = int(info.numChannels);
    AudioBuffer<float> buf;
    buf.resize(d.channels, int(info.numSamples));
    d.ok = r.readSamples(buf.toView());
    for (int ch = 0; ch < d.channels; ++ch)
        d.pcm.emplace_back(buf.getChannel(ch), buf.getChannel(ch) + info.numSamples);
    return d;
}

double energy(const std::vector<float>& v)
{
    double e = 0.0;
    for (float s : v) e += double(s) * double(s);
    return e;
}

// MPEG-1 granule side info (59 bits) or LSF granule side info (63 bits) for a
// long block whose region 0 covers the lines written with Huffman table 1.
void longGranule(Mp3TestBits& b, bool lsf, int part23, int bigValues, int gain,
                 int scalefacCompress, int table = 1, int count1Table = 0)
{
    b.put(uint32_t(part23), 12); b.put(uint32_t(bigValues), 9); b.put(uint32_t(gain), 8);
    b.put(uint32_t(scalefacCompress), lsf ? 9 : 4); b.put(0, 1);
    b.put(uint32_t(table), 5); b.put(uint32_t(table), 5); b.put(uint32_t(table), 5);
    b.put(15, 4); b.put(7, 3);
    if (!lsf) b.put(0, 1);              // preflag (LSF derives it)
    b.put(0, 1); b.put(uint32_t(count1Table), 1);
}

// Huffman table 1 pairs for `lines` (values -1, 0, 1); returns the bit count.
int table1Pairs(Mp3TestBits& b, const std::vector<int>& lines)
{
    int n = 0;
    for (size_t i = 0; i < lines.size(); i += 2)
    {
        const int x = lines[i], y = lines[i + 1];
        const int ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
        if (ax == 0 && ay == 0) { b.put(1, 1); n += 1; }
        else if (ax == 0) { b.put(0b001, 3); n += 3; }
        else if (ay == 0) { b.put(0b01, 2); n += 2; }
        else { b.put(0b000, 3); n += 3; }
        if (ax) { b.put(x < 0 ? 1u : 0u, 1); ++n; }
        if (ay) { b.put(y < 0 ? 1u : 0u, 1); ++n; }
    }
    return n;
}

int table1Bits(const std::vector<int>& lines)
{
    Mp3TestBits scratch;
    return table1Pairs(scratch, lines);
}

uint16_t crc16Arc(const uint8_t* data, size_t size, uint16_t crc)
{
    for (size_t i = 0; i < size; ++i)
    {
        crc = uint16_t(crc ^ data[i]);
        for (int k = 0; k < 8; ++k) crc = uint16_t((crc & 1u) ? (crc >> 1) ^ 0xA001u : (crc >> 1));
    }
    return crc;
}

void encodeTone(const std::filesystem::path& path, int channels, int frames)
{
    Mp3File w;
    AudioFileInfo info;
    info.sampleRate = 44100.0;
    info.numChannels = uint32_t(channels);
    info.bitsPerSample = 128;
    (void)w.openWrite(path, info);
    AudioBuffer<float> buf;
    buf.resize(channels, frames);
    for (int ch = 0; ch < channels; ++ch)
        for (int i = 0; i < frames; ++i)
            buf.getChannel(ch)[i] = 0.4f * std::sin(0.031f * float(i) + float(ch));
    (void)w.writeSamples(std::as_const(buf).toView());
    w.close();
}

// Makes the next file writes of this process fail: on Windows another handle
// locks the file's first gigabyte, elsewhere the file-size limit drops to
// 4 KiB (with SIGXFSZ ignored, so write() reports EFBIG).
class WriteFailure
{
public:
    explicit WriteFailure(const std::filesystem::path& path)
    {
#if defined(_WIN32)
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ != INVALID_HANDLE_VALUE)
        {
            OVERLAPPED region {};
            active_ = LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                                 0, 0x40000000, 0, &region) != 0;
        }
#else
        (void)path;
        previousHandler_ = std::signal(SIGXFSZ, SIG_IGN);
        if (getrlimit(RLIMIT_FSIZE, &previous_) == 0)
        {
            rlimit limited = previous_;
            limited.rlim_cur = 4096;
            active_ = setrlimit(RLIMIT_FSIZE, &limited) == 0;
        }
#endif
    }

    ~WriteFailure()
    {
#if defined(_WIN32)
        if (active_)
        {
            OVERLAPPED region {};
            UnlockFileEx(handle_, 0, 0x40000000, 0, &region);
        }
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if (active_) setrlimit(RLIMIT_FSIZE, &previous_);
        std::signal(SIGXFSZ, previousHandler_);
#endif
    }

    WriteFailure(const WriteFailure&) = delete;
    WriteFailure& operator=(const WriteFailure&) = delete;

    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    bool active_ = false;
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    rlimit previous_ {};
    void (*previousHandler_)(int) = SIG_DFL;
#endif
};

} // namespace io_audit

// Any 12-bit sync pattern with plausible header bits was decoded as a frame,
// so tags, cover art and other binary data after the audio became bursts of
// noise. A 300 KB binary APEv2 item measured 17 phantom frames ending in a
// full-scale burst (peak 1.0, RMS 0.98), and a file of random bytes "opened"
// as 27648 samples. The frame below is silent, so any phantom is visible.
DSPARK_TEST(Mp3File_binary_data_is_not_decoded_as_frames)
{
    midi_test::TempDirectory temp("mp3-phantom");
    auto silent = [](Mp3TestBits& b) {
        for (int gr = 0; gr < 2; ++gr) io_audit::longGranule(b, false, 0, 0, 210, 0);
    };
    const auto frame = mp3CraftFrame(silent, [](Mp3TestBits&) {});
    std::vector<uint8_t> bytes;
    for (int i = 0; i < 40; ++i) bytes.insert(bytes.end(), frame.begin(), frame.end());
    uint32_t lcg = 12345u;
    std::vector<uint8_t> noise(1u << 18);
    for (auto& v : noise) { lcg = lcg * 1664525u + 1013904223u; v = uint8_t(lcg >> 24); }
    bytes.insert(bytes.end(), noise.begin(), noise.end());
    const auto withTail = temp.file("tail.mp3");
    EXPECT_TRUE(midi_test::writeBytes(withTail, bytes));

    const auto d = io_audit::decode(withTail);
    EXPECT_TRUE(d.ok);
    EXPECT_EQ(d.pcm.empty() ? size_t(0) : d.pcm[0].size(), size_t(40 * 1152));
    EXPECT_NEAR(d.pcm.empty() ? -1.0 : io_audit::energy(d.pcm[0]), 0.0, 1e-12);

    const auto random = temp.file("random.mp3");
    EXPECT_TRUE(midi_test::writeBytes(random, noise));
    Mp3File r;
    EXPECT_FALSE(r.openRead(random));
}

// Intensity stereo applies only ABOVE the highest band where the right
// channel has a non-zero line. Every all-zero right band was treated as
// intensity coded instead, so a left band below that bound was panned by a
// position the encoder never meant (here position 0 moved it entirely into
// the right channel: the left output measured exactly 0). Files from another
// common encoder at 48-64 kbps matched the reference decoders at only 32.8 dB
// because of it.
// With no left content above the bound, the intensity frame must decode
// exactly like the same frame without intensity stereo.
DSPARK_TEST(Mp3File_intensity_stereo_starts_above_the_right_channel_bound)
{
    midi_test::TempDirectory temp("mp3-intensity");
    std::vector<int> left(24, 0), right(54, 0);
    for (int i = 20; i < 24; ++i) left[size_t(i)] = (i & 1) ? -1 : 1;   // sfb 5 at 48 kHz
    for (int i = 50; i < 54; ++i) right[size_t(i)] = 1;                  // sfb 10
    const int lBits = io_audit::table1Bits(left), rBits = io_audit::table1Bits(right);
    auto make = [&](int modeExt) {
        return io_audit::craftFrame(3, 9, 1, 1, modeExt,      // MPEG-1 128 kbps 48 kHz joint
            [&](Mp3TestBits& b) {
                b.put(0, 9); b.put(0, 3); b.put(0, 4); b.put(0, 4);
                io_audit::longGranule(b, false, lBits, 12, 180, 0);
                io_audit::longGranule(b, false, rBits, 27, 180, 0);
                io_audit::longGranule(b, false, 0, 0, 210, 0);
                io_audit::longGranule(b, false, 0, 0, 210, 0);
            },
            [&](Mp3TestBits& b) { io_audit::table1Pairs(b, left); io_audit::table1Pairs(b, right); });
    };
    const auto isPath = temp.file("is.mp3"), lrPath = temp.file("lr.mp3");
    io_audit::writeFrames(isPath, { make(1), make(1), make(1) });
    io_audit::writeFrames(lrPath, { make(0), make(0), make(0) });
    const auto is = io_audit::decode(isPath), lr = io_audit::decode(lrPath);
    EXPECT_TRUE(is.ok && lr.ok && is.channels == 2 && lr.channels == 2);
    if (!is.ok || !lr.ok || is.channels != 2 || lr.channels != 2) return;
    EXPECT_GT(io_audit::energy(lr.pcm[0]), 1e-6);
    double maxDiff = 0.0;
    for (int ch = 0; ch < 2; ++ch)
        for (size_t i = 0; i < is.pcm[size_t(ch)].size(); ++i)
            maxDiff = std::max(maxDiff, double(std::fabs(is.pcm[size_t(ch)][i] - lr.pcm[size_t(ch)][i])));
    EXPECT_LT(maxDiff, 1e-9);   // before the fix the left channel was silent
}

// MPEG-2 and MPEG-2.5 Layer III (ISO 13818-3 low sampling frequencies) were
// not decoded: an LSF file was either rejected or, when MPEG-1 false syncs
// turned up in it, "opened" as noise (22.05 kHz LAME file: SNR -13 dB, 45%
// clipped). Three properties are pinned:
//  1. An LSF granule decodes to exactly what the same spectral lines decode
//     to in an MPEG-1 granule (the filterbank is shared), at 22.05 kHz and at
//     the MPEG-2.5 8 kHz rate.
//  2. LSF scalefactors use the ISO 13818-3 partitions: scalefactor 1 in band
//     0 with scalefac_scale 0 halves the energy of that band.
//  3. LSF intensity positions: position 3 scales the left channel by
//     i0^2, i0 = 2^-1/4 (intensity_scale 0) or 2^-1/2 (intensity_scale 1),
//     so the left/right energy ratio is 1/2 or 1/4.
DSPARK_TEST(Mp3File_decodes_mpeg2_and_mpeg25_layer3)
{
    midi_test::TempDirectory temp("mp3-lsf");
    std::vector<int> lines = { 1, 0, -1, 1, 0, 1, 1, -1 };
    const int bits = io_audit::table1Bits(lines);

    // 1. Equivalence with MPEG-1 (44.1 kHz mono, 128 kbps; granule 1 silent).
    const auto mpeg1 = io_audit::craftFrame(3, 9, 0, 3, 0,
        [&](Mp3TestBits& b) {
            b.put(0, 9); b.put(0, 5); b.put(0, 4);
            io_audit::longGranule(b, false, bits, 4, 190, 0);
            io_audit::longGranule(b, false, 0, 0, 210, 0);
        },
        [&](Mp3TestBits& b) { io_audit::table1Pairs(b, lines); });
    const auto p1 = temp.file("mpeg1.mp3");
    io_audit::writeFrames(p1, { mpeg1 });
    const auto ref = io_audit::decode(p1);
    EXPECT_TRUE(ref.ok);
    for (const auto& [version, rateIndex, rate] : { std::tuple { 2, 0, 22050.0 },
                                                   std::tuple { 0, 2, 8000.0 } })
    {
        auto lsfFrame = [&](bool data) {
            return io_audit::craftFrame(version, 8, rateIndex, 3, 0,
                [&](Mp3TestBits& b) {
                    b.put(0, 8); b.put(0, 1);
                    io_audit::longGranule(b, true, data ? bits : 0, data ? 4 : 0, data ? 190 : 210, 0);
                },
                [&](Mp3TestBits& b) { if (data) io_audit::table1Pairs(b, lines); });
        };
        const auto p = temp.file("lsf" + std::to_string(version) + ".mp3");
        io_audit::writeFrames(p, { lsfFrame(true), lsfFrame(false) });
        const auto d = io_audit::decode(p);
        EXPECT_TRUE(d.ok);
        if (!d.ok || !ref.ok) continue;
        EXPECT_NEAR(d.rate, rate, 1e-9);
        EXPECT_EQ(d.pcm[0].size(), ref.pcm[0].size());
        double maxDiff = 0.0;
        for (size_t i = 0; i < std::min(d.pcm[0].size(), ref.pcm[0].size()); ++i)
            maxDiff = std::max(maxDiff, double(std::fabs(d.pcm[0][i] - ref.pcm[0][i])));
        EXPECT_LT(maxDiff, 1e-7);
        EXPECT_GT(io_audit::energy(d.pcm[0]), 1e-6);
    }

    // 2. LSF scalefactors: scalefac_compress 80 -> slen 1,0,0,0 (six 1-bit
    //    scalefactors for bands 0..5). Lines 0..1 lie in band 0.
    std::vector<int> pair = { 1, 1 };
    const int pairBits = io_audit::table1Bits(pair);
    auto scaled = [&](uint32_t scalefactor) {
        return io_audit::craftFrame(2, 8, 0, 3, 0,
            [&](Mp3TestBits& b) {
                b.put(0, 8); b.put(0, 1);
                io_audit::longGranule(b, true, 6 + pairBits, 1, 190, 80);
            },
            [&](Mp3TestBits& b) {
                b.put(scalefactor, 1);
                for (int i = 0; i < 5; ++i) b.put(0, 1);
                io_audit::table1Pairs(b, pair);
            });
    };
    const auto s0 = temp.file("sf0.mp3"), s1 = temp.file("sf1.mp3");
    io_audit::writeFrames(s0, { scaled(0), scaled(0) });
    io_audit::writeFrames(s1, { scaled(1), scaled(1) });
    const auto d0 = io_audit::decode(s0), d1 = io_audit::decode(s1);
    EXPECT_TRUE(d0.ok && d1.ok);
    if (d0.ok && d1.ok)
        EXPECT_NEAR(io_audit::energy(d1.pcm[0]) / io_audit::energy(d0.pcm[0]), 0.5, 1e-6);

    // 3. LSF intensity stereo. The right channel carries no lines, so every
    //    band is intensity coded; its scalefac_compress 258/259 selects the
    //    intensity partition table 3 with three 3-bit fields (21 positions).
    for (const auto& [intensityScale, expected] : { std::pair { 0, 0.5 }, std::pair { 1, 0.25 } })
    {
        auto frame = [&](bool data) {
            return io_audit::craftFrame(2, 8, 0, 1, 1,
                [&](Mp3TestBits& b) {
                    b.put(0, 8); b.put(0, 2);
                    io_audit::longGranule(b, true, data ? bits : 0, data ? 4 : 0, 190, 0);
                    io_audit::longGranule(b, true, 63, 0, 190, 258 + intensityScale);
                },
                [&](Mp3TestBits& b) {
                    if (data) io_audit::table1Pairs(b, lines);
                    for (int i = 0; i < 21; ++i) b.put(3, 3);
                });
        };
        const auto p = temp.file("is" + std::to_string(intensityScale) + ".mp3");
        io_audit::writeFrames(p, { frame(true), frame(false) });
        const auto d = io_audit::decode(p);
        EXPECT_TRUE(d.ok && d.channels == 2);
        if (!d.ok || d.channels != 2) continue;
        EXPECT_NEAR(io_audit::energy(d.pcm[0]) / io_audit::energy(d.pcm[1]), expected, 1e-6);
    }
}

// A gapless tag counting fewer frames than the stream holds (concatenated or
// re-muxed files) cut the end of the audio: two concatenated copies decoded
// to the length of one (1,323,000 samples where a reference decoder returns
// 2,648,495), and the output of one encoder lost 72% of its frames.
DSPARK_TEST(Mp3File_stale_gapless_frame_count_keeps_the_audio)
{
    midi_test::TempDirectory temp("mp3-stale-tag");
    const auto one = temp.file("one.mp3");
    io_audit::encodeTone(one, 1, 22050);
    auto bytes = midi_test::readBytes(one);
    const auto twice = bytes;
    bytes.insert(bytes.end(), twice.begin(), twice.end());
    const auto both = temp.file("both.mp3");
    EXPECT_TRUE(midi_test::writeBytes(both, bytes));
    const auto a = io_audit::decode(one), b = io_audit::decode(both);
    EXPECT_TRUE(a.ok && b.ok);
    if (!a.ok || !b.ok) return;
    EXPECT_EQ(a.pcm[0].size(), size_t(22050));
    EXPECT_GT(b.pcm[0].size(), size_t(2 * 22050));
}

// A frame whose main data begins in a bit reservoir that is not in the file (a
// stream cut in the middle) used to be decoded from the start of whatever
// bytes were there, every field read from the wrong position: a noise burst
// at program level (error RMS 0.24 against the uncut decode, where a
// reference decoder mutes). Such a frame must decode as silence.
DSPARK_TEST(Mp3File_frame_without_its_reservoir_is_muted)
{
    midi_test::TempDirectory temp("mp3-reservoir");
    auto orphan = io_audit::craftFrame(3, 5, 1, 3, 0,          // 64 kbps 48 kHz mono
        [](Mp3TestBits& b) {
            b.put(120, 9); b.put(0, 5); b.put(0, 4);           // main_data_begin = 120
            io_audit::longGranule(b, false, 900, 200, 200, 15);
            io_audit::longGranule(b, false, 900, 200, 200, 15);
        },
        [](Mp3TestBits& b) {
            uint32_t lcg = 7u;
            for (int i = 0; i < 160; ++i) { lcg = lcg * 1664525u + 1013904223u; b.put(lcg >> 24, 8); }
        });
    auto silent = mp3CraftFrame(
        [](Mp3TestBits& b) { for (int gr = 0; gr < 2; ++gr) io_audit::longGranule(b, false, 0, 0, 210, 0); },
        [](Mp3TestBits&) {});
    const auto p = temp.file("orphan.mp3");
    io_audit::writeFrames(p, { orphan, silent, silent, silent });
    const auto d = io_audit::decode(p);
    EXPECT_TRUE(d.ok);
    EXPECT_NEAR(d.ok ? io_audit::energy(d.pcm[0]) : -1.0, 0.0, 1e-12);
}

// The decoded size follows from the frame count alone: 96-byte frames (32
// kbps, 48 kHz, stereo) decode to 96 times their size, so a 4 MB file peaked
// at 393 MB and the 256 MB input cap allowed about 25 GB. On a 32-bit size_t
// (x86, WASM) nch * numSamples also wrapped, and the decoder wrote past a
// short buffer (a 182 MB file crashed). 932,068 such frames need just over
// kMaxDecodedPcmBytes (8 GiB) of float PCM: openRead must refuse them.
DSPARK_TEST(Mp3File_decoded_size_is_bounded)
{
    midi_test::TempDirectory temp("mp3-bound");
    std::vector<uint8_t> frame(96, 0);
    frame[0] = 0xFF; frame[1] = 0xFB; frame[2] = 0x14; frame[3] = 0x00;
    const auto p = temp.file("flood.mp3");
    {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        std::vector<uint8_t> chunk;
        for (int i = 0; i < 4096; ++i) chunk.insert(chunk.end(), frame.begin(), frame.end());
        int written = 0;
        while (written < 932068)
        {
            const int n = std::min(4096, 932068 - written);
            f.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(n * 96));
            written += n;
        }
    }
    Mp3File r;
    EXPECT_FALSE(r.openRead(p));
    EXPECT_FALSE(r.isOpen());
}

// readSamples() on a writer copied from the empty decode buffer (a crash),
// and the range check startFrame + numFrames > numSamples overflowed int64:
// readSamples(view, INT64_MAX - 5, 100) returned true, reading outside the
// buffer (MP3) or from the RIFF header (WAV).
DSPARK_TEST(AudioFile_readers_reject_invalid_ranges_and_writers)
{
    midi_test::TempDirectory temp("io-ranges");
    AudioBuffer<float> buf;
    buf.resize(2, 256);
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    {
        const auto p = temp.file("w.mp3");
        Mp3File w;
        AudioFileInfo info;
        info.sampleRate = 44100.0; info.numChannels = 2; info.bitsPerSample = 128;
        info.numSamples = 1000;
        EXPECT_TRUE(w.openWrite(p, info));
        EXPECT_FALSE(w.readSamples(buf.toView(), 0, 100));
        w.close();
    }
    {
        const auto p = temp.file("r.mp3");
        io_audit::encodeTone(p, 2, 4410);
        Mp3File r;
        EXPECT_TRUE(r.openRead(p));
        EXPECT_FALSE(r.readSamples(buf.toView(), kMax - 5, 100));
        EXPECT_FALSE(r.readSamples(buf.toView(), int64_t(1) << 62, kMax));
    }
    {
        const auto p = temp.file("r.wav");
        WavFile w;
        AudioFileInfo info;
        info.sampleRate = 48000.0; info.numChannels = 2; info.bitsPerSample = 16;
        EXPECT_TRUE(w.openWrite(p, info));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
        WavFile r;
        EXPECT_TRUE(r.openRead(p));
        EXPECT_FALSE(r.readSamples(buf.toView(), kMax - 5, 100));
        EXPECT_FALSE(r.readSamples(buf.toView(), int64_t(1) << 62, kMax));
    }
}

// Writers reported success while nothing reached the disk: Mp3File::
// writeSamples() never checked its stream (0 of 8 failed writes reported),
// MidiFile::write() returned true with a 0-byte file (good() was read before
// the buffered bytes were flushed), and close() gave no way to learn that
// the header patch failed. hasWriteError() now carries that status.
DSPARK_TEST(AudioFile_write_failures_are_reported)
{
    midi_test::TempDirectory temp("io-write-failure");
    AudioBuffer<float> buf;
    buf.resize(2, 44100);
    for (int i = 0; i < 44100; ++i) buf.getChannel(0)[i] = buf.getChannel(1)[i] = 0.25f;
    AudioFileInfo info;
    info.sampleRate = 44100.0; info.numChannels = 2;
    {
        const auto p = temp.file("fail.mp3");
        Mp3File w;
        info.bitsPerSample = 320;
        EXPECT_TRUE(w.openWrite(p, info));
        io_audit::WriteFailure failure(p);
        EXPECT_TRUE(failure.active());
        bool reported = false;
        for (int k = 0; k < 4; ++k) reported = !w.writeSamples(std::as_const(buf).toView()) || reported;
        w.close();
        EXPECT_TRUE(reported);
        EXPECT_TRUE(w.hasWriteError());
    }
    {
        const auto p = temp.file("fail.wav");
        WavFile w;
        info.bitsPerSample = 16;
        EXPECT_TRUE(w.openWrite(p, info));
        io_audit::WriteFailure failure(p);
        EXPECT_TRUE(failure.active());
        EXPECT_FALSE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
        EXPECT_TRUE(w.hasWriteError());
    }
    {
        const auto p = temp.file("ok.wav");
        WavFile w;
        EXPECT_TRUE(w.openWrite(p, info));
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
        EXPECT_FALSE(w.hasWriteError());
    }
    {
        const auto p = temp.file("fail.mid");
        MidiFile m;
        EXPECT_TRUE(m.create(0, 480, 1));
        for (int i = 0; i < 2000; ++i)
            EXPECT_TRUE(m.addChannelEvent(0, 10, 0x90, uint8_t(i % 128), 100));
        io_audit::WriteFailure failure(p);
        EXPECT_TRUE(failure.active());
        EXPECT_FALSE(m.write(p));
    }
}

// The last count1 quad of a granule must end within part2_3_length; one that
// runs past it is discarded by ISO 11172-3 and the reference decoders. A
// two-bit tolerance kept such a quad as a spurious coefficient (a file from
// a fixed-point encoder matched the reference decoders at 77.7 dB instead of
// 136.7 dB). Here
// part2_3_length is 3 bits and the only quad needs 5.
DSPARK_TEST(Mp3File_count1_quad_past_part2_3_length_is_discarded)
{
    midi_test::TempDirectory temp("mp3-count1");
    auto side = [](Mp3TestBits& b) {
        io_audit::longGranule(b, false, 3, 0, 200, 0);
        io_audit::longGranule(b, false, 0, 0, 210, 0);
    };
    const auto frame = mp3CraftFrame(side, [](Mp3TestBits& b) { b.put(0b01110, 5); });
    const auto p = temp.file("count1.mp3");
    io_audit::writeFrames(p, { frame, frame, frame });
    const auto d = io_audit::decode(p);
    EXPECT_TRUE(d.ok);
    EXPECT_NEAR(d.ok ? io_audit::energy(d.pcm[0]) : -1.0, 0.0, 1e-12);
}

// The gapless tag CRC was accepted only as LAME computes it (over the frame
// up to the CRC field). FFmpeg checksums 190 bytes with the field still zero;
// when the tag lies early in the frame (mono, MPEG-2/2.5) that span includes
// the field, so the tag of every such file was rejected and the delay and
// padding kept (ffmpeg-made LAME mono files decoded 1152 samples too long,
// MPEG-2 ones 1476).
DSPARK_TEST(Mp3File_gapless_tag_with_zeroed_crc_span_is_trusted)
{
    midi_test::TempDirectory temp("mp3-tag-crc");
    const auto p = temp.file("mono.mp3");
    io_audit::encodeTone(p, 1, 22050);
    auto bytes = midi_test::readBytes(p);
    // Info tag of this writer: header 4 + side info 17 + Xing fields 120;
    // the gapless tag's CRC field sits 34 bytes further, at offset 175.
    EXPECT_TRUE(bytes.size() > 200);
    if (bytes.size() <= 200) return;
    bytes[175] = 0; bytes[176] = 0;
    const uint16_t crc = io_audit::crc16Arc(bytes.data(), 190, 0);
    bytes[175] = uint8_t(crc >> 8); bytes[176] = uint8_t(crc & 0xFF);
    EXPECT_TRUE(midi_test::writeBytes(p, bytes));
    const auto d = io_audit::decode(p);
    EXPECT_TRUE(d.ok);
    EXPECT_EQ(d.ok ? d.pcm[0].size() : size_t(0), size_t(22050));
}

// RF64/BW64 (EBU Tech 3306) files were rejected, and the writer could not go
// past 4 GiB. A ds64 chunk now supplies the 64-bit data size on read, and
// Rf64Mode::Always writes RF64; Auto keeps small files classic RIFF.
DSPARK_TEST(WavFile_rf64_is_read_and_written)
{
    midi_test::TempDirectory temp("wav-rf64");
    std::vector<uint8_t> bytes;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) bytes.push_back(uint8_t(v >> (8 * i))); };
    auto u64 = [&](uint64_t v) { u32(uint32_t(v)); u32(uint32_t(v >> 32)); };
    auto tag = [&](const char* t) { bytes.insert(bytes.end(), t, t + 4); };
    const int16_t samples[4] = { 16384, -16384, 8192, -32768 };
    tag("RF64"); u32(0xFFFFFFFFu); tag("WAVE");
    tag("ds64"); u32(28); u64(0); u64(8); u64(4); u32(0);
    tag("fmt "); u32(16); u32(0x00010001u); u32(48000); u32(96000); u32(0x00100002u);
    tag("data"); u32(0xFFFFFFFFu);
    for (int16_t s : samples) { bytes.push_back(uint8_t(s & 0xFF)); bytes.push_back(uint8_t((s >> 8) & 0xFF)); }
    const auto crafted = temp.file("crafted.wav");
    EXPECT_TRUE(midi_test::writeBytes(crafted, bytes));
    {
        WavFile r;
        EXPECT_TRUE(r.openRead(crafted));
        EXPECT_EQ(r.getInfo().numSamples, int64_t(4));
        AudioBuffer<float> buf;
        buf.resize(1, 4);
        EXPECT_TRUE(r.readSamples(buf.toView()));
        EXPECT_NEAR(buf.getChannel(0)[0], 0.5f, 0.0f);
        EXPECT_NEAR(buf.getChannel(0)[3], -1.0f, 0.0f);
    }
    for (const auto mode : { WavFile::Rf64Mode::Always, WavFile::Rf64Mode::Auto })
    {
        const auto p = temp.file(mode == WavFile::Rf64Mode::Always ? "always.wav" : "auto.wav");
        WavFile w;
        w.setRf64Mode(mode);
        AudioFileInfo info;
        info.sampleRate = 48000.0; info.numChannels = 1; info.bitsPerSample = 24;
        EXPECT_TRUE(w.openWrite(p, info));
        AudioBuffer<float> buf;
        buf.resize(1, 101);
        for (int i = 0; i < 101; ++i) buf.getChannel(0)[i] = float(i - 50) / 64.0f;
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
        const auto out = midi_test::readBytes(p);
        EXPECT_TRUE(out.size() > 12);
        if (out.size() <= 12) continue;
        EXPECT_EQ(std::memcmp(out.data(), mode == WavFile::Rf64Mode::Always ? "RF64" : "RIFF", 4), 0);
        WavFile r;
        EXPECT_TRUE(r.openRead(p));
        EXPECT_EQ(r.getInfo().numSamples, int64_t(101));
        AudioBuffer<float> back;
        back.resize(1, 101);
        EXPECT_TRUE(r.readSamples(back.toView()));
        for (int i = 0; i < 101; ++i) EXPECT_NEAR(back.getChannel(0)[i], buf.getChannel(0)[i], 1e-6f);
    }
}

// RIFF chunks are word-aligned, but the writer ended an odd-sized data chunk
// (8-bit or 24-bit mono with an odd frame count) without its pad byte: three
// 8-bit frames made a 47-byte file whose RIFF size was odd.
DSPARK_TEST(WavFile_odd_data_chunk_gets_its_pad_byte)
{
    midi_test::TempDirectory temp("wav-pad");
    const auto p = temp.file("odd.wav");
    WavFile w;
    AudioFileInfo info;
    info.sampleRate = 48000.0; info.numChannels = 1; info.bitsPerSample = 8;
    EXPECT_TRUE(w.openWrite(p, info));
    AudioBuffer<float> buf;
    buf.resize(1, 3);
    EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
    w.close();
    const auto bytes = midi_test::readBytes(p);
    EXPECT_EQ(bytes.size(), size_t(48));
    if (bytes.size() < 8) return;
    const uint32_t riffSize = uint32_t(bytes[4]) | (uint32_t(bytes[5]) << 8)
                            | (uint32_t(bytes[6]) << 16) | (uint32_t(bytes[7]) << 24);
    EXPECT_EQ(riffSize, 40u);
    WavFile r;
    EXPECT_TRUE(r.openRead(p));
    EXPECT_EQ(r.getInfo().numSamples, int64_t(3));
}

// RFC 9639 B.1: before 2007 a variable-block stream was marked only by
// STREAMINFO min != max, with the strategy bit 0 and sample numbers coded;
// such frames were rejected (IETF vector subset/27 was the only one of 62 not
// decoded). A modern fixed stream whose STREAMINFO lists unequal bounds is
// legal too and was rejected as well. Both are built here from the
// declarative variable fixture (blocks of 16 and 17 samples) and must decode
// to its reference PCM.
DSPARK_TEST(FlacFile_legacy_variable_and_unequal_bound_fixed_streams_decode)
{
    const auto original = midi_test::readBytes(flac_test::root() / "decl-variable-uncommon-rate.flac");
    EXPECT_TRUE(original.size() > 44);
    if (original.size() <= 44) return;
    // Frames start after "fLaC" + one 34-byte STREAMINFO block: offset 42.
    std::vector<size_t> starts { 42 };
    for (size_t e = 44; e + 2 <= original.size() && starts.size() < 2; ++e)
    {
        const uint16_t stored = uint16_t((original[e] << 8) | original[e + 1]);
        if (flac_test::crc16(std::span<const uint8_t>(original.data() + 42, e - 42)) == stored
            && e + 2 < original.size() && original[e + 2] == 0xFF)
            starts.push_back(e + 2);
    }
    EXPECT_EQ(starts.size(), size_t(2));
    if (starts.size() != 2) return;
    const std::vector<size_t> ends { starts[1], original.size() };

    FlacFile reference;
    EXPECT_TRUE(reference.openRead(flac_test::root() / "decl-variable-uncommon-rate.flac"));
    AudioBuffer<float> expected;
    expected.resize(1, int(reference.getInfo().numSamples));
    EXPECT_TRUE(reference.readSamples(expected.toView()));

    // codedSecond: the coded number of frame 1 (16 = sample number, 1 = frame number)
    for (const uint8_t codedSecond : { uint8_t(16), uint8_t(1) })
    {
        auto bytes = original;
        for (size_t f = 0; f < 2; ++f)
        {
            const size_t s = starts[f];
            bytes[s + 1] = 0xF8;                                   // strategy bit 0
            if (f == 1) bytes[s + 4] = codedSecond;
            // Header: sync and codes (4 bytes), a one-byte coded number, the
            // optional block size and sample rate fields, then the CRC-8.
            size_t crcPos = s + 5;
            const uint8_t blockCode = uint8_t(bytes[s + 2] >> 4), rateCode = uint8_t(bytes[s + 2] & 15);
            if (blockCode == 6) crcPos += 1; else if (blockCode == 7) crcPos += 2;
            if (rateCode == 12) crcPos += 1; else if (rateCode == 13 || rateCode == 14) crcPos += 2;
            bytes[crcPos] = flac_test::crc8(std::span<const uint8_t>(bytes.data() + s, crcPos - s));
            const uint16_t crc = flac_test::crc16(std::span<const uint8_t>(bytes.data() + s, ends[f] - 2 - s));
            bytes[ends[f] - 2] = uint8_t(crc >> 8);
            bytes[ends[f] - 1] = uint8_t(crc & 0xFF);
        }
        midi_test::TempDirectory temp("flac-legacy-" + std::to_string(codedSecond));
        const auto p = temp.file("legacy.flac");
        EXPECT_TRUE(midi_test::writeBytes(p, bytes));
        FlacFile file;
        EXPECT_TRUE(file.openRead(p));
        EXPECT_EQ(file.getInfo().numSamples, reference.getInfo().numSamples);
        AudioBuffer<float> got;
        got.resize(1, int(std::max<int64_t>(1, file.getInfo().numSamples)));
        if (!file.isOpen()) continue;
        EXPECT_TRUE(file.readSamples(got.toView()));
        for (int i = 0; i < int(reference.getInfo().numSamples); ++i)
            EXPECT_NEAR(got.getChannel(0)[i], expected.getChannel(0)[i], 0.0f);
    }
}

// ============================================================================
// Mp3File - Layer I, Layer II, free format and the vector transform kernels
// ============================================================================

namespace {

struct mp3R2_Probe { int index; float value; };

const char* const mp3R2_kLayer2Hex =
    "fffd60406675b6cb642b023dfe56bb8f3d7459bf84d751cd395a0cf86938385959367158b859f7df7df7dfbf7ef7bdef"
    "7bdef7bdeeeef7bdef7bbbbbbc0ec8fbefbf80f5ef8c1f07bdcf835dddf07bdf07777773bba1f6df80afa43d75f38e62"
    "ea0733165b97d44ad0f1b1c105d659540414dbb28bcaedb9ef336e384dcb7b486c1ced8d32b7ea910c74bd086733c8a0"
    "f0f881b8e544ccada75cd1a34e7d225cee3ded6bd948b565e316364587dd6549cb59a5d51f6115ef741efd6d716a6756"
    "63442204ccd0f23346e36028f83bed17c00fbc18f985126107085e1a17828e6cd38d12e6f8dbb465c95a1968c947868c"
    "704253b75388aa685b04e2511bd2b6423074291a4c73137b95946839521446fcc4248fa8a1b5680d47ed3c5af5ca4eac"
    "f5a36591e7b78378e53a6f7a4cab28c070385a8f6099155dc0fffd600019c71c61851858aaaab8d44e3d2515556556aa"
    "b1be9340bf829751c8e4ca2b49a3254574e45725252c7d14d43e0d00024c7e63249a30a9c5860b1c95963ae9b0ba18c1"
    "0d400914471a2dd47142eb6a2999960a5b4e7663ac2f04b90927d42b1b02c97fd5537101c454d91e281982d19c113cb7"
    "85ef6d76ff775a03e8f346e38b254913b28d4f61a5552d7bb1f6da68acdfab8a3a4bba115c999277e89517561cdee44a"
    "a9446a91b31ff0ea877cd81c2534047e94a18e28b991d0b12950997d6406605ff7a6aaa4e31e1d93a6182b9546aa6e5d"
    "6e11b150da54a107d05ca626d1f75d4e236625a6ee3774a96c9084399ff5344dd3c94c330de3343126a65659aeed4ad4"
    "26f8405f8f8fcb87500cbe7ad33aba2d54fb1f186e8b352ac4bb0821219a8b7ce4dae6d986788317c7960bee7bd70c92"
    "c9e8fffd600019c7186346145aaaeaae38f4d4536dd65565e45048ba892f9e2e35939c3723045335715e1d30bb28b59e"
    "2a6d3edf60b0df9ff901dd1a1dbc969c6ab6d911812089eae2bcd60cf65b08dc1c95ac63d1503d71a92b7a944d770a49"
    "8a1f31f89f098ee9fa77ae2909646ef3c8f7cb491c6b9237bdf47ad50bde66380074b9592c2ac8978c69d78e43c94f92"
    "5b594b321f231f408bd61c4a7682b5664849709206dcdabd409633275a193dc2c817ab1b5859e9b0b38eeb250bd68422"
    "19ae272eb205e2f37fa1654906a948dca1b24ead56e7bf0e4e8b49ac581dd00e2c829c15b82ef553004f5a0bc04552f4"
    "4948c0db27bd7854261469b51b2baa661f89dec908b1358e6757cfb2178f02a9f01d735ad0046a7c1094d535f5cf1bd7"
    "0985679cfcdbfeff12b1ba02a624b86959eed80315197e895b5c00";

// 6912 interleaved samples; every 191th of the last two thirds, from a
// reference decoder.
const mp3R2_Probe mp3R2_kLayer2Probes[] = {
    { 2304, 0.0548375025f },
    { 2495, -0.0295396876f },
    { 2686, 0.00436188839f },
    { 2877, -0.0113057941f },
    { 3068, -0.0586682856f },
    { 3259, -0.0812272951f },
    { 3450, 0.0439098999f },
    { 3641, 0.0547870286f },
    { 3832, 0.022255186f },
    { 4023, 0.0370997153f },
    { 4214, -0.062348038f },
    { 4405, -0.0382896215f },
    { 4596, 0.0295080952f },
    { 4787, 0.0160588417f },
    { 4978, 0.0379188955f },
    { 5169, -0.0520958342f },
    { 5360, -0.060896486f },
    { 5551, -0.0129230088f },
    { 5742, 0.0126423305f },
    { 5933, 0.034145128f },
    { 6124, 0.050426323f },
    { 6315, 0.0234931298f },
    { 6506, -0.0544773899f },
    { 6697, 0.022095751f }
};

const char* const mp3R2_kLayer1Hex =
    "fffe84748861080a04000060000000000000080000000000000000000003b348fbc66bdfa3c27746556442a04f126f9b"
    "5df6345dd253fafd425f48676f11c595cbf5d19a7c0d32aae6010dba6905fda0f84b24e5730d80fdc23fed029ac76a1e"
    "44286c440523dc0000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "00000000000000000000000000000000fffe8454bb6f02001a00a226600070000563000c000c00bd0060e70babf26f59"
    "5bc73a8fdfb666e6d7d7d57355e045c0125277c04c180683725d3a74dc6f5e3a835682a5f66bd07dc0019fdcb58bc2e8"
    "4f742f025bbdf24fc51a8b630fb8914d489887040ed176adb2323f20092ba696403e640a3132588cbc33ec04eb815f28"
    "f6e3b6a057c28bd4db2a5080bb09dd7b13ae7c280c4fd7671a11e3fb9f0fdc6709a8ee53194b4d47d4e12484ebd689e6"
    "7c253a98b2426e075155ad38732952d8372677a35e191623898e2129aae4be8dceb5011895b721ce3cf4a807e0cafe31"
    "6ec5279974d244023e08801cb2fd9ac522ec0000000000000000000000000000fffe8444c42000e60007100000270700"
    "0000a4880013000c91cef6631f9e7b7c37df08d56efeab6da9775bcba76d84d8a6d8179ce800d2b09e674cd68c9d4d02"
    "48c38f192768626739fa69e3d2b0a1093ebf64d27e18f6b0d0c8b7750d8dec11fe896b5822b89a281ba82179a12fdd04"
    "b177f6f970ce2934d9f1a83353748e6bf9586b29bc0042bd61aaccdccf67024d2516d37521b392052b974cc157343d32"
    "b9a516a334fec3e2129053fc0361e0cb38ba99d167c8e7f3db009f9e1d60b6a89652912e65531c000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

// 2304 interleaved samples; every 63th of the last two thirds, from a
// reference decoder.
const mp3R2_Probe mp3R2_kLayer1Probes[] = {
    { 768, 0.00023808828f },
    { 831, -0.000379322213f },
    { 894, 0.000266827963f },
    { 957, 0.000718105934f },
    { 1020, 0.000108089822f },
    { 1083, 0.00100534107f },
    { 1146, -0.00159751938f },
    { 1209, -0.00090863643f },
    { 1272, 0.0154903978f },
    { 1335, -0.00928503927f },
    { 1398, -0.0134385554f },
    { 1461, -0.00608294364f },
    { 1524, -0.0296050161f },
    { 1587, 0.00306604523f },
    { 1650, 0.0212647226f },
    { 1713, -0.00335809449f },
    { 1776, 0.0162993073f },
    { 1839, 0.000179787807f },
    { 1902, -0.00811650045f },
    { 1965, 0.00744489208f },
    { 2028, 0.0122812157f },
    { 2091, 0.011782839f },
    { 2154, 0.00186728826f },
    { 2217, -0.0179273803f }
};

const char* const mp3R2_kLayer2FreeHex =
    "fffd046400000007000c000000000008800002018196057574516d93784d576ffbadf15b4e1397149e190fdfba52c226"
    "5c21435b00c08b863cf9297eecbcfae6e848ca51e64da2314498c4c3ab12b43cd1feedba4c885a2f0d8dcd195c20dbec"
    "a867e13f0ee9c7f738cde5de16ed4eed511964aa0959b302ce30779e232523163d4457acd126884b354e07b3dc4e5dfb"
    "493403bc48e125d8638a21a9ba5fdac65486e94423cd6d64fa5c0b542df59b7c3119c03a716cbfff58f2dbb655fd1bd3"
    "326f9080d36f0f3d335e6097461f6998bf4d4b7d472b97e11b5b66000000000000000000000000000000000000000000"
    "fffd04540000090000040010006000030a0880b5aca5dae7082dbe8ad859ac96bb1931897f2fd7cdd93914a078031ba9"
    "5d04be2d4ab026d292c1bf37138a72d58746cd25cd1937ce04ddadca275e8a93091a2bd99d80391464fa2a65231911b4"
    "035e9dd3257c4e37a2306f94c22b737938b4bf63e9afbc53621acf9a57bcf7db772a489d4a48966ea2a447da5fc428c6"
    "012a31d5b54eafdaf43425d4655ee9b85d2fc94ffe389e75514c2624d138e7880036091490488572e304118000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "fffd046402000000000000800000a0000000000000c339b7f7b1b04591bf2386291670f105ff028984e76378aa88dc4a"
    "0a66a1403f57a493da18113505c3610b605e2cffd6abddf5ac3caa0eb8f33408e55e1b66476aa220653d1105e5f97db4"
    "657196ef136754fd855eea6758c338880000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

// 6912 interleaved samples; every 191th of the last two thirds, from a
// reference decoder.
const mp3R2_Probe mp3R2_kLayer2FreeProbes[] = {
    { 2304, -0.000197731963f },
    { 2495, 1.32257428e-05f },
    { 2686, -0.0015833003f },
    { 2877, 0.00478105247f },
    { 3068, 0.00453008944f },
    { 3259, -0.015682023f },
    { 3450, 0.00255409256f },
    { 3641, -0.00338956341f },
    { 3832, 0.00806494895f },
    { 4023, -0.0104602128f },
    { 4214, 0.001730402f },
    { 4405, -0.00429509301f },
    { 4596, -0.00095198839f },
    { 4787, -0.00829491299f },
    { 4978, -0.000775594788f },
    { 5169, -0.00549798366f },
    { 5360, 0.0016896677f },
    { 5551, -0.0019900254f },
    { 5742, 0.000376064796f },
    { 5933, 0.00254127453f },
    { 6124, -0.00178212626f },
    { 6315, 0.00463826023f },
    { 6506, 0.000188134494f },
    { 6697, 0.000147649072f }
};

std::vector<uint8_t> mp3R2_fromHex(const char* hex)
{
    std::vector<uint8_t> out;
    auto nibble = [](char c) { return (c <= '9') ? c - '0' : c - 'a' + 10; };
    for (const char* p = hex; p[0] != '\0' && p[1] != '\0'; p += 2)
        out.push_back(static_cast<uint8_t>((nibble(p[0]) << 4) | nibble(p[1])));
    return out;
}

void mp3R2_write(const char* path, const std::vector<uint8_t>& bytes)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

/// Opens path and returns its interleaved samples (empty when rejected).
std::vector<float> mp3R2_decode(const char* path, AudioFileInfo* infoOut = nullptr)
{
    Mp3File f;
    if (!f.openRead(path)) return {};
    const AudioFileInfo info = f.getInfo();
    if (infoOut != nullptr) *infoOut = info;
    const int nch = static_cast<int>(info.numChannels);
    const int n = static_cast<int>(info.numSamples);
    AudioBuffer<float> buf;
    buf.resize(nch, n);
    if (!f.readSamples(buf.toView())) return {};
    std::vector<float> out(static_cast<size_t>(n) * static_cast<size_t>(nch));
    for (int i = 0; i < n; ++i)
        for (int ch = 0; ch < nch; ++ch)
            out[static_cast<size_t>(i * nch + ch)] = buf.getChannel(ch)[i];
    return out;
}

/// A 44.1 kHz stereo Layer III file written by Mp3File, with its bytes.
std::vector<uint8_t> mp3R2_encodeLayer3(const char* path, int frames)
{
    Mp3File w;
    AudioFileInfo info;
    info.sampleRate = 44100.0;
    info.numChannels = 2;
    info.bitsPerSample = 128;
    info.numSamples = frames;
    if (!w.openWrite(path, info)) return {};
    AudioBuffer<float> buf;
    buf.resize(2, frames);
    for (int i = 0; i < frames; ++i)
    {
        const double t = static_cast<double>(i) / 44100.0;
        buf.getChannel(0)[i] = static_cast<float>(0.4 * std::sin(2.0 * pi<double> * 440.0 * t)
                                                  + 0.1 * std::sin(2.0 * pi<double> * 6150.0 * t));
        buf.getChannel(1)[i] = static_cast<float>(0.3 * std::sin(2.0 * pi<double> * 659.0 * t));
    }
    if (!w.writeSamples(std::as_const(buf).toView())) return {};
    w.close();
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

} // namespace

DSPARK_TEST(Mp3File_decodes_layer1_and_layer2_frames)
{
    // Three Layer II frames of a reference encoder (44.1 kHz joint stereo,
    // 96 kbps) and three Layer I frames covering every allocation (48 kHz
    // joint stereo, 256 kbps, CRC protected). The probes come from a
    // reference decoder; the two agree to 3e-8 (139 and 132 dB).
    struct Case { const char* path; const char* hex; const mp3R2_Probe* probes; int numProbes;
                  double rate; int64_t frames; };
    const Case cases[] = {
        { "dspark_test_layer2.mp2", mp3R2_kLayer2Hex, mp3R2_kLayer2Probes,
          static_cast<int>(std::size(mp3R2_kLayer2Probes)), 44100.0, 3 * 1152 },
        { "dspark_test_layer1.mp1", mp3R2_kLayer1Hex, mp3R2_kLayer1Probes,
          static_cast<int>(std::size(mp3R2_kLayer1Probes)), 48000.0, 3 * 384 },
    };
    for (const Case& c : cases)
    {
        FileCleanup cleanup { c.path };
        mp3R2_write(c.path, mp3R2_fromHex(c.hex));
        AudioFileInfo info;
        const std::vector<float> pcm = mp3R2_decode(c.path, &info);
        EXPECT_FALSE(pcm.empty());
        EXPECT_EQ(info.numChannels, 2u);
        EXPECT_NEAR(info.sampleRate, c.rate, 0.0);
        EXPECT_EQ(info.numSamples, c.frames);
        for (int k = 0; k < c.numProbes; ++k)
            EXPECT_NEAR(pcm[static_cast<size_t>(c.probes[k].index)], c.probes[k].value, 2e-6f);
    }
}

DSPARK_TEST(Mp3File_free_format_streams_decode)
{
    // (a) Three free-format Layer II frames (48 kHz joint stereo, 240 bytes):
    // the frame length is in no header and is found from the next two.
    {
        FileCleanup cleanup { "dspark_test_free.mp2" };
        mp3R2_write("dspark_test_free.mp2", mp3R2_fromHex(mp3R2_kLayer2FreeHex));
        AudioFileInfo info;
        const std::vector<float> pcm = mp3R2_decode("dspark_test_free.mp2", &info);
        EXPECT_FALSE(pcm.empty());
        EXPECT_EQ(info.numSamples, int64_t { 3 * 1152 });
        for (const mp3R2_Probe& p : mp3R2_kLayer2FreeProbes)
            EXPECT_NEAR(pcm[static_cast<size_t>(p.index)], p.value, 2e-6f);
    }

    // (b) A Layer III stream rewritten to free format (bitrate index 0 in
    // every audio frame, the Info tag frame dropped) decodes to exactly the
    // same audio; without the tag nothing is trimmed, so it is 529 + 528
    // samples later.
    FileCleanup cbrCleanup { "dspark_test_cbr.mp3" };
    FileCleanup freeCleanup { "dspark_test_free.mp3" };
    const std::vector<uint8_t> cbr = mp3R2_encodeLayer3("dspark_test_cbr.mp3", 20000);
    EXPECT_FALSE(cbr.empty());
    static constexpr int kKbps[16] = { 0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0 };
    std::vector<uint8_t> freeFormat;
    bool tagFrame = true;
    for (size_t p = 0; p + 4 <= cbr.size();)
    {
        EXPECT_TRUE(cbr[p] == 0xFF && (cbr[p + 1] & 0xE0) == 0xE0);
        const int index = cbr[p + 2] >> 4;
        const size_t bytes = static_cast<size_t>(144 * kKbps[index] * 1000 / 44100 + ((cbr[p + 2] >> 1) & 1));
        EXPECT_TRUE(index > 0 && index < 15 && p + bytes <= cbr.size());
        if (!tagFrame)
        {
            const size_t at = freeFormat.size();
            freeFormat.insert(freeFormat.end(), cbr.begin() + static_cast<ptrdiff_t>(p),
                              cbr.begin() + static_cast<ptrdiff_t>(p + bytes));
            freeFormat[at + 2] = static_cast<uint8_t>(freeFormat[at + 2] & 0x0F);
        }
        tagFrame = false;
        p += bytes;
    }
    mp3R2_write("dspark_test_free.mp3", freeFormat);
    const std::vector<float> original = mp3R2_decode("dspark_test_cbr.mp3");
    const std::vector<float> rewritten = mp3R2_decode("dspark_test_free.mp3");
    EXPECT_EQ(original.size(), size_t { 2 * 20000 });
    EXPECT_TRUE(rewritten.size() >= original.size() + 2 * 1057);
    for (size_t i = 0; i < original.size(); ++i)
        EXPECT_NEAR(rewritten[i + 2 * 1057], original[i], 0.0f);

    // (c) A free-format length that no third header confirms is not
    // guessed: two frames alone are rejected.
    {
        FileCleanup cleanup { "dspark_test_free2.mp2" };
        std::vector<uint8_t> two = mp3R2_fromHex(mp3R2_kLayer2FreeHex);
        two.resize(2 * 240 - 7);   // the second frame is cut short
        mp3R2_write("dspark_test_free2.mp2", two);
        Mp3File f;
        EXPECT_FALSE(f.openRead("dspark_test_free2.mp2"));
    }
}

DSPARK_TEST(Mp3File_layer12_damaged_frames_stay_bounded)
{
    // A forbidden Layer I allocation (15), a Layer II stream cut in the
    // middle of its last frame and random byte damage: every file that opens
    // gives finite samples within [-1, 1], with the frame count it declares.
    const std::vector<uint8_t> layer1 = mp3R2_fromHex(mp3R2_kLayer1Hex);
    const std::vector<uint8_t> layer2 = mp3R2_fromHex(mp3R2_kLayer2Hex);
    auto check = [](const std::vector<float>& pcm) {
        for (float v : pcm)
            if (!(std::isfinite(v) && std::abs(v) <= 1.0f)) return false;
        return true;
    };
    FileCleanup cleanup { "dspark_test_damaged.mp2" };
    {
        std::vector<uint8_t> b = layer1;
        b[256 + 6] = static_cast<uint8_t>(0xF0 | (b[256 + 6] & 0x0F));   // frame 2, first allocation
        mp3R2_write("dspark_test_damaged.mp2", b);
        AudioFileInfo info;
        const std::vector<float> pcm = mp3R2_decode("dspark_test_damaged.mp2", &info);
        EXPECT_EQ(info.numSamples, int64_t { 3 * 384 });
        EXPECT_TRUE(check(pcm));
    }
    {
        std::vector<uint8_t> b = layer2;
        b.resize(b.size() - 100);
        mp3R2_write("dspark_test_damaged.mp2", b);
        AudioFileInfo info;
        const std::vector<float> pcm = mp3R2_decode("dspark_test_damaged.mp2", &info);
        EXPECT_EQ(info.numSamples, int64_t { 3 * 1152 });
        EXPECT_TRUE(check(pcm));
    }
    std::mt19937 rng(20261107u);
    for (int round = 0; round < 200; ++round)
    {
        std::vector<uint8_t> b = (round & 1) ? layer1 : layer2;
        for (int k = 0; k < 8; ++k)
            b[rng() % b.size()] = static_cast<uint8_t>(rng());
        mp3R2_write("dspark_test_damaged.mp2", b);
        EXPECT_TRUE(check(mp3R2_decode("dspark_test_damaged.mp2")));
    }
}

DSPARK_TEST(Mp3File_vector_kernels_agree_with_the_baseline_copy)
{
    // The transforms run on vectors, with an AVX2+FMA copy selected at run
    // time where the build and the CPU allow it. Both copies must give the
    // same audio to well below float resolution; where only one copy exists
    // the two decodes are identical.
    FileCleanup cleanup { "dspark_test_kernels.mp3" };
    EXPECT_FALSE(mp3R2_encodeLayer3("dspark_test_kernels.mp3", 30000).empty());
    const bool wasActive = simd::isRuntimeDispatchActive();
    simd::setRuntimeDispatch(false);
    const std::vector<float> baseline = mp3R2_decode("dspark_test_kernels.mp3");
    simd::setRuntimeDispatch(true);
    const std::vector<float> dispatched = mp3R2_decode("dspark_test_kernels.mp3");
    simd::setRuntimeDispatch(wasActive);
    EXPECT_EQ(baseline.size(), dispatched.size());
    EXPECT_FALSE(baseline.empty());
    double worst = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::abs(baseline[i] - dispatched[i])));
    EXPECT_TRUE(worst <= 1e-6);
    if (!simd::runtimeDispatchSupported()) EXPECT_NEAR(worst, 0.0, 0.0);
}

// readSamples() decodes only the frames a range covers: a read that does not
// continue the previous one restarts the decoder two frames early and
// rebuilds the bit reservoir from the frames before them. Every range must
// still return exactly the samples of the whole-file read. The crafted
// Layer III stream makes every frame start its main data 300 bytes back
// (four frames of 75 bytes), deeper than the restart, so it fails if the
// reservoir is not rebuilt; the Layer II stream has no reservoir and needs
// the restart to refill the synthesis window only, and the 12 slots of a
// Layer I frame are fewer than the window spans, so it fails with a restart
// of one frame.
DSPARK_TEST(Mp3File_any_range_reads_the_samples_of_the_whole_file)
{
    midi_test::TempDirectory temp("mp3-ranges");
    const auto deep = temp.file("deep_reservoir.mp3");
    {
        constexpr int kFrames = 40, kSlot = 75, kBack = 300, kSilent = 4;
        std::vector<uint8_t> stream(size_t(kFrames) * kSlot, 0);
        std::vector<std::array<int, 4>> sides(kFrames, std::array<int, 4> {});   // part23, bigValues x2
        std::vector<int> gains(kFrames, 210);
        uint32_t lcg = 12345u;
        for (int i = kSilent; i < kFrames; ++i)
        {
            Mp3TestBits g;
            for (int gr = 0; gr < 2; ++gr)
            {
                std::vector<int> lines(96);
                for (int& v : lines) { lcg = lcg * 1664525u + 1013904223u; v = int(lcg >> 30) - 1; if (v > 1) v = 0; }
                sides[size_t(i)][size_t(2 * gr)] = io_audit::table1Pairs(g, lines);
                sides[size_t(i)][size_t(2 * gr + 1)] = 48;
            }
            gains[size_t(i)] = 190 + (i % 7) * 2;
            const size_t at = size_t(i) * kSlot - kBack;
            std::copy(g.bytes.begin(), g.bytes.end(), stream.begin() + std::ptrdiff_t(at));
        }
        std::vector<std::vector<uint8_t>> frames;
        for (int i = 0; i < kFrames; ++i)
            frames.push_back(io_audit::craftFrame(3, 1, 1, 3, 0,       // 32 kbps 48 kHz mono
                [&](Mp3TestBits& b) {
                    b.put(i >= kSilent ? uint32_t(kBack) : 0u, 9); b.put(0, 5); b.put(0, 4);
                    for (int gr = 0; gr < 2; ++gr)
                        io_audit::longGranule(b, false, sides[size_t(i)][size_t(2 * gr)],
                                              sides[size_t(i)][size_t(2 * gr + 1)], gains[size_t(i)], 0);
                },
                [&](Mp3TestBits& b) {
                    for (int k = 0; k < kSlot; ++k) b.put(stream[size_t(i) * kSlot + size_t(k)], 8);
                }));
        io_audit::writeFrames(deep, frames);
    }
    auto repeated = [&](const char* name, const char* hex) {
        const std::vector<uint8_t> three = mp3R2_fromHex(hex);
        std::vector<uint8_t> bytes;
        for (int k = 0; k < 8; ++k) bytes.insert(bytes.end(), three.begin(), three.end());
        const auto p = temp.file(name);
        mp3R2_write(p.string().c_str(), bytes);
        return p;
    };
    const auto layer1 = repeated("layer1.mp1", mp3R2_kLayer1Hex);
    const auto layer2 = repeated("layer2.mp2", mp3R2_kLayer2Hex);
    const auto layer3 = temp.file("tone.mp3");
    io_audit::encodeTone(layer3, 2, 30000);

    for (const auto& path : { deep, layer1, layer2, layer3 })
    {
        Mp3File whole;
        EXPECT_TRUE(whole.openRead(path));
        const auto info = whole.getInfo();
        const int ch = int(info.numChannels);
        const int64_t n = info.numSamples;
        EXPECT_TRUE(n > 4000);
        AudioBuffer<float> all;
        all.resize(ch, int(n));
        EXPECT_TRUE(whole.readSamples(all.toView()));
        double e = 0.0;
        for (int c = 0; c < ch; ++c)
            for (int64_t i = 0; i < n; ++i) e += double(all.getChannel(c)[i]) * all.getChannel(c)[i];
        EXPECT_TRUE(e > 1.0);

        int mismatches = 0;
        AudioBuffer<float> part;
        part.resize(ch, 5000);
        auto same = [&](Mp3File& f, int64_t start, int64_t count) {
            if (!f.readSamples(part.toView(), start, count)) { ++mismatches; return; }
            for (int c = 0; c < ch; ++c)
                if (std::memcmp(part.getChannel(c), all.getChannel(c) + start, size_t(count) * sizeof(float)) != 0)
                    ++mismatches;
        };
        {
            Mp3File f;
            EXPECT_TRUE(f.openRead(path));
            for (int64_t s = 0; s < n; s += 1000) same(f, s, std::min<int64_t>(1000, n - s));
        }
        {
            Mp3File f;
            EXPECT_TRUE(f.openRead(path));
            for (int64_t s = ((n - 1) / 777) * 777; s >= 0; s -= 777) same(f, s, std::min<int64_t>(777, n - s));
        }
        {
            Mp3File f;
            EXPECT_TRUE(f.openRead(path));
            uint32_t lcg = 99u;
            for (int k = 0; k < 60; ++k)
            {
                lcg = lcg * 1664525u + 1013904223u;
                const int64_t s = int64_t(lcg % uint32_t(n));
                lcg = lcg * 1664525u + 1013904223u;
                same(f, s, 1 + int64_t(lcg % uint32_t(std::min<int64_t>(5000, n - s))));
            }
            same(f, n - 1, 1);
            same(f, 0, 1);
        }
        EXPECT_EQ(mismatches, 0);
    }
}
// ---------------------------------------------------------------------------
// WAV: G.711, IMA/Microsoft ADPCM and 12/20-bit PCM reading
// ---------------------------------------------------------------------------
// Every stream below is built byte by byte and decoded again here with an
// independent reference written from the specifications (ITU-T G.711, the
// IMA ADPCM reference algorithm, Microsoft ADPCM with truncating division).

namespace {

void wavR2_put16(std::vector<uint8_t>& v, uint32_t x)
{
    v.push_back(uint8_t(x & 0xFF));
    v.push_back(uint8_t((x >> 8) & 0xFF));
}

void wavR2_put32(std::vector<uint8_t>& v, uint32_t x)
{
    wavR2_put16(v, x & 0xFFFF);
    wavR2_put16(v, x >> 16);
}

void wavR2_chunk(std::vector<uint8_t>& riff, const char* id, const std::vector<uint8_t>& body)
{
    riff.insert(riff.end(), id, id + 4);
    wavR2_put32(riff, uint32_t(body.size()));
    riff.insert(riff.end(), body.begin(), body.end());
    if (body.size() & 1) riff.push_back(0);
}

std::vector<uint8_t> wavR2_fmt(uint16_t tag, uint16_t ch, uint32_t rate, uint16_t blockAlign,
                               uint16_t bits, const std::vector<uint8_t>* ext = nullptr)
{
    std::vector<uint8_t> f;
    wavR2_put16(f, tag); wavR2_put16(f, ch); wavR2_put32(f, rate);
    wavR2_put32(f, rate * blockAlign); wavR2_put16(f, blockAlign); wavR2_put16(f, bits);
    if (ext != nullptr)
    {
        wavR2_put16(f, uint32_t(ext->size()));
        f.insert(f.end(), ext->begin(), ext->end());
    }
    return f;
}

// fmt, optional fact, an optional odd-sized foreign chunk, then data.
void wavR2_writeWav(const char* path, const std::vector<uint8_t>& fmt,
                    const std::vector<uint8_t>& data, const std::vector<uint8_t>* fact = nullptr,
                    const char* extraChunkBeforeData = nullptr)
{
    std::vector<uint8_t> body { 'W', 'A', 'V', 'E' };
    wavR2_chunk(body, "fmt ", fmt);
    if (fact != nullptr) wavR2_chunk(body, "fact", *fact);
    if (extraChunkBeforeData != nullptr) wavR2_chunk(body, extraChunkBeforeData, { 'x', 'y', 'z' });
    wavR2_chunk(body, "data", data);
    std::vector<uint8_t> riff { 'R', 'I', 'F', 'F' };
    wavR2_put32(riff, uint32_t(body.size()));
    riff.insert(riff.end(), body.begin(), body.end());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(riff.data()), std::streamsize(riff.size()));
}

struct wavR2_Lcg
{
    uint32_t s;
    uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    uint8_t byte() { return uint8_t(next()); }
};

// ITU-T G.711 decoding through the segment/mantissa form of the tables.
int wavR2_aLaw(uint8_t code)
{
    const int a = code ^ 0x55;
    const int e = (a >> 4) & 7, m = a & 15;
    const int mag = 8 * ((e == 0) ? (2 * m + 1) : ((2 * m + 33) << (e - 1)));
    return (a & 0x80) ? mag : -mag;
}

int wavR2_muLaw(uint8_t code)
{
    const int u = ~code & 0xFF;
    const int e = (u >> 4) & 7, m = u & 15;
    const int mag = 4 * (((2 * m + 33) << e) - 33);
    return (u & 0x80) ? -mag : mag;
}

// IMA ADPCM reference decoder of one channel's nibbles.
struct wavR2_Ima
{
    int pred = 0, index = 0;
    int decode(int code)
    {
        static const int steps[89] = {
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
            253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
            1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
            3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
            12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767 };
        static const int indexTable[16] = { -1, -1, -1, -1, 2, 4, 6, 8,
                                            -1, -1, -1, -1, 2, 4, 6, 8 };
        const int step = steps[index];
        int vpdiff = step >> 3;
        if (code & 4) vpdiff += step;
        if (code & 2) vpdiff += step >> 1;
        if (code & 1) vpdiff += step >> 2;
        pred = (code & 8) ? pred - vpdiff : pred + vpdiff;
        pred = std::max(-32768, std::min(32767, pred));
        index = std::max(0, std::min(88, index + indexTable[code]));
        return pred;
    }
};

// Builds an IMA stream of whole blocks plus a short tail block of tailBytes,
// decoding it with the reference. Returns interleaved 16-bit frames.
std::vector<int> wavR2_buildIma(std::vector<uint8_t>& data, int ch, int blockAlign,
                                int blocks, int tailBytes, uint32_t seed, int badIndexBlock = -1)
{
    wavR2_Lcg rng { seed };
    std::vector<int> out;
    for (int b = 0; b <= blocks; ++b)
    {
        const int bytes = (b < blocks) ? blockAlign : tailBytes;
        if (bytes < 4 * ch) break;
        std::vector<uint8_t> blk(static_cast<size_t>(bytes));
        for (auto& x : blk) x = rng.byte();
        std::vector<wavR2_Ima> st(static_cast<size_t>(ch));
        for (int c = 0; c < ch; ++c)
        {
            const int pred = int(int16_t(uint16_t(rng.next())));
            const int index = (b == badIndexBlock && c == ch - 1) ? 89 : int(rng.next() % 89);
            blk[size_t(4 * c)] = uint8_t(pred & 0xFF);
            blk[size_t(4 * c + 1)] = uint8_t((pred >> 8) & 0xFF);
            blk[size_t(4 * c + 2)] = uint8_t(index);
            blk[size_t(4 * c + 3)] = 0;
            st[size_t(c)].pred = pred;
            st[size_t(c)].index = std::min(index, 88);
        }
        const int words = (bytes - 4 * ch) / (4 * ch);
        std::vector<int> frames(size_t(1 + 8 * words) * size_t(ch));
        for (int c = 0; c < ch; ++c) frames[size_t(c)] = st[size_t(c)].pred;
        for (int w = 0; w < words; ++w)
            for (int c = 0; c < ch; ++c)
                for (int k = 0; k < 8; ++k)
                {
                    const uint8_t byte = blk[size_t(4 * ch + w * 4 * ch + 4 * c + k / 2)];
                    const int code = (k & 1) ? (byte >> 4) : (byte & 15);
                    frames[size_t(1 + 8 * w + k) * size_t(ch) + size_t(c)] = st[size_t(c)].decode(code);
                }
        data.insert(data.end(), blk.begin(), blk.end());
        out.insert(out.end(), frames.begin(), frames.end());
    }
    return out;
}

// Microsoft ADPCM reference decoder of one block.
std::vector<int> wavR2_decodeMs(const uint8_t* blk, int bytes, int ch,
                                const std::vector<std::pair<int, int>>& coefs)
{
    static const int adapt[16] = { 230, 230, 230, 230, 307, 409, 512, 614,
                                   768, 614, 512, 409, 307, 230, 230, 230 };
    auto s16 = [&](int off) { return int(int16_t(uint16_t(blk[off] | (blk[off + 1] << 8)))); };
    std::vector<int64_t> c1(static_cast<size_t>(ch)), c2(c1), delta(c1), s1(c1), s2(c1);
    std::vector<int> out;
    for (int c = 0; c < ch; ++c)
    {
        c1[size_t(c)] = coefs[blk[c]].first;
        c2[size_t(c)] = coefs[blk[c]].second;
        delta[size_t(c)] = s16(ch + 2 * c);
        s1[size_t(c)] = s16(3 * ch + 2 * c);
        s2[size_t(c)] = s16(5 * ch + 2 * c);
    }
    for (int c = 0; c < ch; ++c) out.push_back(int(s2[size_t(c)]));
    for (int c = 0; c < ch; ++c) out.push_back(int(s1[size_t(c)]));
    const int nibbles = ((bytes - 7 * ch) * 2 / ch) * ch;
    for (int i = 0; i < nibbles; ++i)
    {
        const size_t c = size_t(i % ch);
        const int code = (i & 1) ? (blk[7 * ch + i / 2] & 15) : (blk[7 * ch + i / 2] >> 4);
        const int64_t predicted = (s1[c] * c1[c] + s2[c] * c2[c]) / 256; // truncating
        const int64_t v = std::max<int64_t>(-32768, std::min<int64_t>(32767,
            predicted + int64_t(code >= 8 ? code - 16 : code) * delta[c]));
        s2[c] = s1[c];
        s1[c] = v;
        delta[c] = std::max<int64_t>(16, adapt[code] * delta[c] / 256);
        out.push_back(int(v));
    }
    return out;
}

std::vector<int> wavR2_buildMs(std::vector<uint8_t>& data, int ch, int blockAlign, int blocks,
                               int tailBytes, const std::vector<std::pair<int, int>>& coefs,
                               uint32_t seed, int badPredictorBlock = -1)
{
    wavR2_Lcg rng { seed };
    static const int deltas[6] = { 16, 40, 300, 1500, 9000, -5 };
    std::vector<int> out;
    for (int b = 0; b <= blocks; ++b)
    {
        const int bytes = (b < blocks) ? blockAlign : tailBytes;
        if (bytes < 7 * ch) break;
        std::vector<uint8_t> blk(static_cast<size_t>(bytes));
        for (auto& x : blk) x = rng.byte();
        // Codes as an encoder emits them: mostly small steps (which shrink
        // the step size), now and then a large one. Uniform random nibbles
        // would grow the step size without bound.
        static const int smallCodes[7] = { 0, 1, 2, 3, 13, 14, 15 };
        for (size_t i = size_t(7 * ch); i < blk.size(); ++i)
        {
            int nib[2];
            for (int& n : nib)
            {
                const uint32_t r = rng.next();
                n = (r % 8 == 0) ? int(4 + (r / 8) % 9) : smallCodes[(r / 8) % 7];
            }
            blk[i] = uint8_t((nib[0] << 4) | nib[1]);
        }
        for (int c = 0; c < ch; ++c)
        {
            blk[size_t(c)] = (b == badPredictorBlock) ? uint8_t(coefs.size())
                                                      : uint8_t(rng.next() % coefs.size());
            const int d = deltas[rng.next() % 6];
            blk[size_t(ch + 2 * c)] = uint8_t(d & 0xFF);
            blk[size_t(ch + 2 * c + 1)] = uint8_t((d >> 8) & 0xFF);
        }
        const auto frames = (b != badPredictorBlock)
            ? wavR2_decodeMs(blk.data(), bytes, ch, coefs)
            : std::vector<int>(size_t(2 + ((bytes - 7 * ch) * 2) / ch) * size_t(ch), 0);
        out.insert(out.end(), frames.begin(), frames.end());
        data.insert(data.end(), blk.begin(), blk.end());
    }
    return out;
}

std::vector<uint8_t> wavR2_msExt(int ch, int blockAlign, const std::vector<std::pair<int, int>>& coefs)
{
    std::vector<uint8_t> ext;
    wavR2_put16(ext, uint32_t(2 + ((blockAlign - 7 * ch) * 2) / ch));
    wavR2_put16(ext, uint32_t(coefs.size()));
    for (const auto& c : coefs)
    {
        wavR2_put16(ext, uint16_t(int16_t(c.first)));
        wavR2_put16(ext, uint16_t(int16_t(c.second)));
    }
    return ext;
}

const std::vector<std::pair<int, int>>& wavR2_msCoefs()
{
    // The seven standard pairs plus two custom ones the streams also use.
    static const std::vector<std::pair<int, int>> c = {
        { 256, 0 }, { 512, -256 }, { 0, 0 }, { 192, 64 }, { 240, 0 }, { 460, -208 },
        { 392, -232 }, { 300, -100 }, { -500, 700 } };
    return c;
}

// Mismatches between decoded frames and expected interleaved 16-bit values.
int wavR2_mismatches(const AudioBuffer<float>& got, const std::vector<int>& expected,
                     int ch, int64_t firstFrame, int64_t frames)
{
    int bad = 0;
    for (int64_t f = 0; f < frames; ++f)
        for (int c = 0; c < ch; ++c)
            if (got.getChannel(c)[f] != float(expected[size_t((firstFrame + f) * ch + c)]) / 32768.0f)
                ++bad;
    return bad;
}

bool wavR2_readAll(const char* path, AudioBuffer<float>& out, AudioFileInfo& info)
{
    WavFile r;
    if (!r.openRead(path)) return false;
    info = r.getInfo();
    out.resize(int(info.numChannels), int(std::max<int64_t>(info.numSamples, 1)));
    return r.readSamples(out.toView());
}

} // namespace

DSPARK_TEST(WavFile_reads_g711_alaw_and_mulaw)
{
    // Every code of both laws, decoded to the exact 16-bit linear value of
    // ITU-T G.711. The old reader rejected formats 6 and 7.
    FileCleanup c1 { "dspark_r2_alaw.wav" };
    FileCleanup c2 { "dspark_r2_mulaw.wav" };
    FileCleanup c3 { "dspark_r2_alaw_ext.wav" };

    std::vector<uint8_t> mono(256);
    for (int i = 0; i < 256; ++i) mono[size_t(i)] = uint8_t(i);
    wavR2_writeWav("dspark_r2_alaw.wav", wavR2_fmt(6, 1, 8000, 1, 8), mono);

    // Stereo mu-law, odd number of bytes (a stray trailing byte and the pad
    // byte), a fact chunk and an odd-sized foreign chunk before the data.
    std::vector<uint8_t> stereo(2 * 129 + 1);
    for (size_t i = 0; i < stereo.size(); ++i) stereo[i] = uint8_t(i * 7 + 3);
    std::vector<uint8_t> fact; wavR2_put32(fact, 129);
    wavR2_writeWav("dspark_r2_mulaw.wav", wavR2_fmt(7, 2, 8000, 2, 8), stereo, &fact, "odd ");

    // A-law carried by WAVE_FORMAT_EXTENSIBLE.
    std::vector<uint8_t> ext;
    wavR2_put16(ext, 8); wavR2_put32(ext, 4); wavR2_put16(ext, 6);
    const uint8_t guid[14] = { 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71 };
    ext.insert(ext.end(), guid, guid + 14);
    wavR2_writeWav("dspark_r2_alaw_ext.wav", wavR2_fmt(0xFFFE, 1, 8000, 1, 8, &ext), mono);

    for (const char* path : { "dspark_r2_alaw.wav", "dspark_r2_alaw_ext.wav" })
    {
        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll(path, got, info));
        EXPECT_EQ(int(info.numSamples), 256);
        EXPECT_EQ(int(info.bitsPerSample), 16);
        int bad = 0;
        for (int i = 0; i < 256 && got.getNumSamples() >= 256; ++i)
            if (got.getChannel(0)[i] != float(wavR2_aLaw(uint8_t(i))) / 32768.0f) ++bad;
        EXPECT_EQ(bad, 0);
    }
    {
        WavFile r;
        EXPECT_TRUE(r.openRead("dspark_r2_alaw.wav"));
        EXPECT_TRUE(r.getEncoding() == WavFile::Encoding::ALaw);
    }
    {
        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll("dspark_r2_mulaw.wav", got, info));
        EXPECT_EQ(int(info.numSamples), 129);   // the stray trailing byte is not a frame
        int bad = 0;
        for (int f = 0; f < 129 && got.getNumSamples() >= 129; ++f)
            for (int c = 0; c < 2; ++c)
                if (got.getChannel(c)[f] != float(wavR2_muLaw(stereo[size_t(2 * f + c)])) / 32768.0f) ++bad;
        EXPECT_EQ(bad, 0);
        EXPECT_EQ(wavR2_muLaw(0x00), -32124);   // full scale of the 14-bit law
        EXPECT_EQ(wavR2_aLaw(0xAA), 32256);     // full scale of the 13-bit law
    }
}

DSPARK_TEST(WavFile_reads_ima_adpcm_bit_exact)
{
    // IMA/DVI ADPCM (format 0x11) against the reference algorithm, with a
    // short last block that ends half way through a data word (the half word
    // holds no complete group and is ignored), and ranged reads that start
    // inside a block. The old reader rejected the format.
    FileCleanup c1 { "dspark_r2_ima1.wav" };
    FileCleanup c2 { "dspark_r2_ima2.wav" };
    FileCleanup c3 { "dspark_r2_ima_fact.wav" };

    struct Case { const char* path; int ch; int blockAlign; int blocks; int tail; int frames; };
    const Case cases[2] = {
        { "dspark_r2_ima1.wav", 1, 256, 3, 4 + 4 * 5 + 2, 3 * 505 + 41 },
        { "dspark_r2_ima2.wav", 2, 512, 2, 8 + 8 * 3, 2 * 505 + 25 } };
    for (const auto& k : cases)
    {
        std::vector<uint8_t> data;
        const auto expected = wavR2_buildIma(data, k.ch, k.blockAlign, k.blocks, k.tail,
                                             0x1234u + uint32_t(k.ch));
        std::vector<uint8_t> ext; wavR2_put16(ext, 505);
        wavR2_writeWav(k.path, wavR2_fmt(0x11, uint16_t(k.ch), 22050, uint16_t(k.blockAlign), 4, &ext), data);

        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll(k.path, got, info));
        EXPECT_EQ(int(info.numSamples), k.frames);
        EXPECT_EQ(int(info.bitsPerSample), 16);
        EXPECT_EQ(int(expected.size()), k.frames * k.ch);
        if (int(info.numSamples) == k.frames)
            EXPECT_EQ(wavR2_mismatches(got, expected, k.ch, 0, k.frames), 0);

        WavFile r;
        EXPECT_TRUE(r.openRead(k.path));
        EXPECT_TRUE(r.getEncoding() == WavFile::Encoding::ImaAdpcm);
        AudioBuffer<float> part; part.resize(k.ch, 600);
        EXPECT_TRUE(r.readSamples(part.toView(), 300, 600));   // crosses a block boundary
        EXPECT_EQ(wavR2_mismatches(part, expected, k.ch, 300, 600), 0);
    }

    // Encoders pad the last block and store the true length in a fact
    // chunk: it is trusted when it ends inside the last block, and ignored
    // when it would cut whole blocks.
    std::vector<uint8_t> data;
    (void)wavR2_buildIma(data, 2, 512, 2, 0, 77u);
    std::vector<uint8_t> ext; wavR2_put16(ext, 505);
    for (uint32_t fact : { 910u, 100u })
    {
        std::vector<uint8_t> f; wavR2_put32(f, fact);
        wavR2_writeWav("dspark_r2_ima_fact.wav", wavR2_fmt(0x11, 2, 22050, 512, 4, &ext), data, &f);
        WavFile r;
        EXPECT_TRUE(r.openRead("dspark_r2_ima_fact.wav"));
        EXPECT_EQ(int(r.getInfo().numSamples), fact == 910u ? 910 : 1010);
    }
}

DSPARK_TEST(WavFile_reads_ms_adpcm_bit_exact)
{
    // Microsoft ADPCM (format 2) with every predictor of a nine-pair table
    // (the seven standard pairs and two custom ones, which only the table in
    // the fmt chunk defines), random block headers including a negative
    // delta, a short last block, and ranged reads. The prediction truncates
    // toward zero; flooring it moves these blocks by a few LSB. The old
    // reader rejected the format.
    FileCleanup c1 { "dspark_r2_ms1.wav" };
    FileCleanup c2 { "dspark_r2_ms2.wav" };

    struct Case { const char* path; int ch; int blockAlign; int blocks; int tail; int frames; };
    const Case cases[2] = {
        { "dspark_r2_ms1.wav", 1, 128, 4, 7 + 30, 4 * 244 + 62 },
        { "dspark_r2_ms2.wav", 2, 256, 3, 14 + 41, 3 * 244 + 43 } };
    for (const auto& k : cases)
    {
        std::vector<uint8_t> data;
        const auto expected = wavR2_buildMs(data, k.ch, k.blockAlign, k.blocks, k.tail,
                                            wavR2_msCoefs(), 0xBEEFu + uint32_t(k.ch));
        const auto ext = wavR2_msExt(k.ch, k.blockAlign, wavR2_msCoefs());
        wavR2_writeWav(k.path, wavR2_fmt(2, uint16_t(k.ch), 22050, uint16_t(k.blockAlign), 4, &ext), data);

        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll(k.path, got, info));
        EXPECT_EQ(int(info.numSamples), k.frames);
        EXPECT_EQ(int(expected.size()), k.frames * k.ch);
        if (int(info.numSamples) == k.frames)
            EXPECT_EQ(wavR2_mismatches(got, expected, k.ch, 0, k.frames), 0);

        WavFile r;
        EXPECT_TRUE(r.openRead(k.path));
        EXPECT_TRUE(r.getEncoding() == WavFile::Encoding::MsAdpcm);
        AudioBuffer<float> part; part.resize(k.ch, 300);
        EXPECT_TRUE(r.readSamples(part.toView(), 200, 300));
        EXPECT_EQ(wavR2_mismatches(part, expected, k.ch, 200, 300), 0);
    }
}

DSPARK_TEST(WavFile_reads_12_and_20_bit_pcm_in_block_align_containers)
{
    // Bit depths that are not whole bytes sit left-justified in the
    // container nBlockAlign defines. The old reader rejected them.
    struct Case { const char* path; int ch; int bits; int container; };
    const Case cases[4] = {
        { "dspark_r2_pcm12m.wav", 1, 12, 2 }, { "dspark_r2_pcm12s.wav", 2, 12, 2 },
        { "dspark_r2_pcm20s.wav", 2, 20, 3 }, { "dspark_r2_pcm20in32.wav", 2, 20, 4 } };
    for (const auto& k : cases)
    {
        FileCleanup cleanup { k.path };
        const int frames = 301;
        const int full = 1 << (k.bits - 1);
        std::vector<int> q(size_t(frames * k.ch));
        wavR2_Lcg rng { uint32_t(k.bits * 31 + k.ch) };
        std::vector<uint8_t> data;
        for (size_t i = 0; i < q.size(); ++i)
        {
            q[i] = (i == 0) ? -full : (i == 1) ? full - 1 : int(rng.next() % uint32_t(2 * full)) - full;
            const uint32_t word = uint32_t(q[i]) << (8 * k.container - k.bits);
            for (int b = 0; b < k.container; ++b) data.push_back(uint8_t(word >> (8 * b)));
        }
        wavR2_writeWav(k.path, wavR2_fmt(1, uint16_t(k.ch), 44100, uint16_t(k.ch * k.container),
                                         uint16_t(k.bits)), data);

        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll(k.path, got, info));
        EXPECT_EQ(int(info.numSamples), frames);
        EXPECT_EQ(int(info.bitsPerSample), k.bits);
        int bad = 0;
        for (int f = 0; f < frames && got.getNumSamples() >= frames; ++f)
            for (int c = 0; c < k.ch; ++c)
                if (got.getChannel(c)[f] != float(q[size_t(f * k.ch + c)]) / float(full)) ++bad;
        EXPECT_EQ(bad, 0);
    }

    // Containers the block alignment cannot describe stay rejected.
    FileCleanup bad { "dspark_r2_pcm_bad.wav" };
    const std::vector<uint8_t> some(12, 0);
    for (const auto& f : { wavR2_fmt(1, 2, 44100, 3, 12),    // 1.5 bytes per sample
                           wavR2_fmt(1, 1, 44100, 2, 20),    // 20 bits in 2 bytes
                           wavR2_fmt(1, 1, 44100, 5, 20),    // 5-byte container
                           wavR2_fmt(1, 1, 44100, 0, 12) })  // no alignment
    {
        wavR2_writeWav("dspark_r2_pcm_bad.wav", f, some);
        WavFile r;
        EXPECT_FALSE(r.openRead("dspark_r2_pcm_bad.wav"));
    }
}

DSPARK_TEST(WavFile_malformed_adpcm_is_rejected_or_bounded)
{
    FileCleanup cleanup { "dspark_r2_bad.wav" };
    const char* path = "dspark_r2_bad.wav";
    const std::vector<uint8_t> payload(600, 0x5A);
    auto opens = [&](const std::vector<uint8_t>& fmt, const std::vector<uint8_t>& data)
    {
        wavR2_writeWav(path, fmt, data);
        WavFile r;
        return r.openRead(path);
    };
    std::vector<uint8_t> imaExt; wavR2_put16(imaExt, 505);
    std::vector<uint8_t> imaExtBig; wavR2_put16(imaExtBig, 506);   // above 505
    std::vector<uint8_t> msExtSpb1; wavR2_put16(msExtSpb1, 1);     // below the header's 2
    const auto msExt = wavR2_msExt(1, 256, wavR2_msCoefs());
    std::vector<uint8_t> msNoCoef; wavR2_put16(msNoCoef, 500); wavR2_put16(msNoCoef, 0);
    std::vector<uint8_t> msTooMany; wavR2_put16(msTooMany, 500); wavR2_put16(msTooMany, 300);
    std::vector<uint8_t> msShortTable; wavR2_put16(msShortTable, 500); wavR2_put16(msShortTable, 7);
    for (int i = 0; i < 10; ++i) msShortTable.push_back(0);        // 7 pairs need 28 bytes

    EXPECT_FALSE(opens(wavR2_fmt(0x11, 1, 22050, 0, 4, &imaExt), payload));    // no block size
    EXPECT_FALSE(opens(wavR2_fmt(0x11, 1, 22050, 254, 4, &imaExt), payload));  // not 4-byte words
    EXPECT_FALSE(opens(wavR2_fmt(0x11, 1, 22050, 256, 3, &imaExt), payload));  // 3-bit IMA
    EXPECT_FALSE(opens(wavR2_fmt(0x11, 3, 22050, 768, 4, &imaExt), payload));  // three channels
    EXPECT_FALSE(opens(wavR2_fmt(0x11, 0, 22050, 256, 4, &imaExt), payload));  // no channel
    EXPECT_FALSE(opens(wavR2_fmt(0x11, 1, 22050, 256, 4, &imaExtBig), payload));
    EXPECT_FALSE(opens(wavR2_fmt(2, 1, 22050, 6, 4, &msExt), payload));        // under the header
    EXPECT_FALSE(opens(wavR2_fmt(2, 1, 22050, 256, 4, &msExtSpb1), payload));
    EXPECT_FALSE(opens(wavR2_fmt(2, 1, 22050, 256, 4, &msNoCoef), payload));
    EXPECT_FALSE(opens(wavR2_fmt(2, 1, 22050, 256, 4, &msTooMany), payload));
    EXPECT_FALSE(opens(wavR2_fmt(2, 1, 22050, 256, 4, &msShortTable), payload));
    EXPECT_FALSE(opens(wavR2_fmt(2, 1, 22050, 256, 8, &msExt), payload));      // 8-bit MS ADPCM
    EXPECT_FALSE(opens(wavR2_fmt(6, 1, 8000, 2, 16), payload));                // 16-bit A-law
    {
        // WAVE_FORMAT_EXTENSIBLE cannot carry the ADPCM parameters.
        std::vector<uint8_t> ext;
        wavR2_put16(ext, 4); wavR2_put32(ext, 4); wavR2_put16(ext, 0x11);
        const uint8_t guid[14] = { 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71 };
        ext.insert(ext.end(), guid, guid + 14);
        EXPECT_FALSE(opens(wavR2_fmt(0xFFFE, 1, 22050, 256, 4, &ext), payload));
    }

    // A block header out of range fails the reads that touch that block
    // only; the blocks before and after it still decode exactly.
    {
        std::vector<uint8_t> data;
        const auto expected = wavR2_buildIma(data, 2, 512, 3, 0, 99u, 1);
        std::vector<uint8_t> ext; wavR2_put16(ext, 505);
        wavR2_writeWav(path, wavR2_fmt(0x11, 2, 22050, 512, 4, &ext), data);
        WavFile r;
        EXPECT_TRUE(r.openRead(path));
        AudioBuffer<float> all; all.resize(2, 3 * 505);
        EXPECT_FALSE(r.readSamples(all.toView()));
        AudioBuffer<float> part; part.resize(2, 505);
        EXPECT_TRUE(r.readSamples(part.toView(), 0, 505));
        EXPECT_EQ(wavR2_mismatches(part, expected, 2, 0, 505), 0);
        EXPECT_TRUE(r.readSamples(part.toView(), 1010, 505));
        EXPECT_EQ(wavR2_mismatches(part, expected, 2, 1010, 505), 0);
    }
    {
        std::vector<uint8_t> data;
        const auto expected = wavR2_buildMs(data, 1, 128, 3, 0, wavR2_msCoefs(), 5u, 1);
        const auto ext = wavR2_msExt(1, 128, wavR2_msCoefs());
        wavR2_writeWav(path, wavR2_fmt(2, 1, 22050, 128, 4, &ext), data);
        WavFile r;
        EXPECT_TRUE(r.openRead(path));
        EXPECT_EQ(int(r.getInfo().numSamples), 3 * 244);
        AudioBuffer<float> all; all.resize(1, 3 * 244);
        EXPECT_FALSE(r.readSamples(all.toView()));
        AudioBuffer<float> part; part.resize(1, 244);
        EXPECT_TRUE(r.readSamples(part.toView(), 0, 244));
        EXPECT_EQ(wavR2_mismatches(part, expected, 1, 0, 244), 0);
        EXPECT_TRUE(r.readSamples(part.toView(), 488, 244));
        EXPECT_EQ(wavR2_mismatches(part, expected, 1, 488, 244), 0);
    }

    // The largest legal block (64 KiB) over a few bytes of data, a data
    // chunk shorter than one block header, and a wSamplesPerBlock smaller
    // than the block holds: all bounded and decoded as far as they go.
    {
        std::vector<uint8_t> data;
        const auto expected = wavR2_buildMs(data, 1, 65535, 0, 100, wavR2_msCoefs(), 11u);
        std::vector<uint8_t> ext; wavR2_put16(ext, 0);   // derived from nBlockAlign
        wavR2_put16(ext, uint32_t(wavR2_msCoefs().size()));
        for (const auto& c : wavR2_msCoefs())
        {
            wavR2_put16(ext, uint16_t(int16_t(c.first)));
            wavR2_put16(ext, uint16_t(int16_t(c.second)));
        }
        wavR2_writeWav(path, wavR2_fmt(2, 1, 22050, 65535, 4, &ext), data);
        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll(path, got, info));
        EXPECT_EQ(int(info.numSamples), 2 + 93 * 2);
        if (int(info.numSamples) == 2 + 93 * 2)
            EXPECT_EQ(wavR2_mismatches(got, expected, 1, 0, 2 + 93 * 2), 0);
    }
    {
        wavR2_writeWav(path, wavR2_fmt(2, 1, 22050, 256, 4, &msExt), std::vector<uint8_t>(5, 1));
        WavFile r;
        EXPECT_TRUE(r.openRead(path));
        EXPECT_EQ(int(r.getInfo().numSamples), 0);
        AudioBuffer<float> b; b.resize(1, 4);
        EXPECT_TRUE(r.readSamples(b.toView()));
    }
    {
        std::vector<uint8_t> data;
        const auto expected = wavR2_buildIma(data, 1, 256, 2, 0, 3u);
        std::vector<uint8_t> ext; wavR2_put16(ext, 100);
        wavR2_writeWav(path, wavR2_fmt(0x11, 1, 22050, 256, 4, &ext), data);
        AudioBuffer<float> got; AudioFileInfo info;
        EXPECT_TRUE(wavR2_readAll(path, got, info));
        EXPECT_EQ(int(info.numSamples), 200);
        int bad = 0;
        for (int b = 0; b < 2 && int(info.numSamples) == 200; ++b)
            for (int f = 0; f < 100; ++f)
                if (got.getChannel(0)[b * 100 + f] != float(expected[size_t(b * 505 + f)]) / 32768.0f) ++bad;
        EXPECT_EQ(bad, 0);
    }
}

// ============================================================================
// MidiFile - opt-in tolerant reading and SMPTE time division
// ============================================================================

namespace {

std::vector<uint8_t> midiR2_cat(std::initializer_list<std::vector<uint8_t>> parts)
{
    std::vector<uint8_t> out;
    for (const auto& part : parts)
        out.insert(out.end(), part.begin(), part.end());
    return out;
}

std::vector<uint8_t> midiR2_header(uint16_t format, uint16_t count, uint16_t division)
{
    std::vector<uint8_t> out;
    midi_test::tag(out, "MThd");
    midi_test::be32(out, 6);
    midi_test::be16(out, format);
    midi_test::be16(out, count);
    midi_test::be16(out, division);
    return out;
}

// A track chunk whose length field may disagree with its data.
std::vector<uint8_t> midiR2_chunk(const std::vector<uint8_t>& data, int64_t lengthDelta = 0)
{
    std::vector<uint8_t> out;
    midi_test::tag(out, "MTrk");
    midi_test::be32(out, static_cast<uint32_t>(static_cast<int64_t>(data.size()) + lengthDelta));
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

// count note-on/note-off pairs, 120 ticks apart, explicit status bytes.
std::vector<uint8_t> midiR2_notes(int count)
{
    std::vector<uint8_t> out;
    for (int i = 0; i < count; ++i)
    {
        const auto key = static_cast<uint8_t>(60 + i % 12);
        out.insert(out.end(), { uint8_t(0x78), uint8_t(0x90), key, uint8_t(100),
                                uint8_t(0x78), uint8_t(0x80), key, uint8_t(0) });
    }
    return out;
}

const std::vector<uint8_t> midiR2_eot { 0x00, 0xff, 0x2f, 0x00 };
const std::vector<uint8_t> midiR2_tempo { 0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20 };

class midiR2_Reader
{
public:
    explicit midiR2_Reader(const std::string& label) : dir_(label) {}

    bool read(const std::vector<uint8_t>& bytes, const MidiReadOptions& options,
              MidiFile& out)
    {
        const auto path = dir_.file("in-" + std::to_string(serial_++) + ".mid");
        return midi_test::writeBytes(path, bytes) && out.read(path, options);
    }

    bool rejects(const std::vector<uint8_t>& bytes, const MidiReadOptions& options)
    {
        MidiFile file;
        if (!file.create(1, 96, 1) || read(bytes, options, file))
            return false;
        return file.format() == -1 && file.tracks().empty()
            && file.recoveries() == MidiReadRecovery::None;
    }

    [[nodiscard]] std::filesystem::path file(const std::string& name) const
    {
        return dir_.file(name);
    }

private:
    midi_test::TempDirectory dir_;
    int serial_ = 0;
};

MidiReadOptions midiR2_allow(MidiReadRecovery mask)
{
    MidiReadOptions options;
    options.tolerate = mask;
    return options;
}

} // namespace

DSPARK_TEST(MidiFile_tolerant_read_recovers_each_named_defect)
{
    using R = MidiReadRecovery;
    midiR2_Reader io("r2-recover");
    const auto notes = midiR2_notes(20);
    const auto t0 = midiR2_cat({ midiR2_tempo, notes, midiR2_eot });
    const std::vector<uint8_t> name { 0x00, 0xff, 0x03, 0x04, 'b', 'a', 's', 's' };
    const auto t1 = midiR2_cat({ name, notes, midiR2_eot });
    const auto clean0 = midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(t0) });
    const auto clean1 = midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(t0), midiR2_chunk(t1) });

    // Overlong VLQs: a zero delta as 80 00, a 240-tick delta as 80 81 70, and
    // a meta length as 80 03; the clean equivalent uses canonical forms.
    const std::vector<uint8_t> noteAt240 { 0x81, 0x70, 0x90, 0x3c, 0x40 };
    const auto overlongEvents = midiR2_cat({ { 0x80, 0x00, 0xff, 0x51, 0x80, 0x03, 0x07, 0xa1, 0x20,
                                               0x80, 0x81, 0x70, 0x90, 0x3c, 0x40 },
                                             notes, midiR2_eot });
    const auto overlongClean = midiR2_cat({ midiR2_tempo, noteAt240, notes, midiR2_eot });

    // F0 packet without its closing F7, followed by other events.
    const std::vector<uint8_t> openSysEx { 0x00, 0xf0, 0x05, 0x7e, 0x7f, 0x09, 0x01, 0x00 };
    const std::vector<uint8_t> closedSysEx { 0x00, 0xf0, 0x06, 0x7e, 0x7f, 0x09, 0x01, 0x00, 0xf7 };

    // A file cut one byte into its last note-off: the incomplete event is dropped.
    auto truncated = clean0;
    truncated.resize(truncated.size() - 7);
    const auto truncatedClean = midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(midiR2_cat({
        midiR2_tempo, std::vector<uint8_t>(notes.begin(), notes.end() - 4), midiR2_eot })) });

    struct Case
    {
        const char* name;
        std::vector<uint8_t> bytes;
        R expected;
        std::vector<uint8_t> clean;
    };
    const std::vector<uint8_t> padding(3, 0);
    const Case cases[] = {
        { "trailing-zero-4", midiR2_cat({ clean0, std::vector<uint8_t>(4, 0) }), R::TrailingBytes, clean0 },
        { "trailing-0x1a-60", midiR2_cat({ clean0, std::vector<uint8_t>(60, 0x1a) }), R::TrailingBytes, clean0 },
        { "padding-after-eot-last",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(midiR2_cat({ t0, padding })) }),
          R::DataAfterEndOfTrack, clean0 },
        { "padding-after-eot-first-of-two",
          midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(midiR2_cat({ t0, padding })), midiR2_chunk(t1) }),
          R::DataAfterEndOfTrack, clean1 },
        { "missing-eot-last",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(midiR2_cat({ midiR2_tempo, notes })) }),
          R::MissingEndOfTrack, clean0 },
        { "missing-eot-first-of-two",
          midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(midiR2_cat({ midiR2_tempo, notes })), midiR2_chunk(t1) }),
          R::MissingEndOfTrack, clean1 },
        { "empty-track-chunk",
          midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(t0), midiR2_chunk({}) }),
          R::MissingEndOfTrack,
          midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(t0), midiR2_chunk(midiR2_eot) }) },
        { "overlong-vlq",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(overlongEvents) }),
          R::NoncanonicalVlq, midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(overlongClean) }) },
        { "length-understated-last",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(t0, -10) }), R::TrackLengthMismatch, clean0 },
        { "length-understated-first-of-two",
          midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(t0, -10), midiR2_chunk(t1) }),
          R::TrackLengthMismatch, clean1 },
        { "length-zero-last",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(t0, -static_cast<int64_t>(t0.size())) }),
          R::TrackLengthMismatch, clean0 },
        { "length-past-end-of-file",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(t0, 100) }), R::TrackLengthMismatch, clean0 },
        { "length-overstated-into-next-track",
          midiR2_cat({ midiR2_header(1, 2, 480), midiR2_chunk(t0, 5), midiR2_chunk(t1) }),
          R::TrackLengthMismatch, clean1 },
        { "track-count-too-small",
          midiR2_cat({ midiR2_header(1, 1, 480), midiR2_chunk(t0), midiR2_chunk(t1) }),
          R::TrackCountMismatch, clean1 },
        { "track-count-too-large",
          midiR2_cat({ midiR2_header(1, 3, 480), midiR2_chunk(t0), midiR2_chunk(t1) }),
          R::TrackCountMismatch, clean1 },
        { "format0-two-track-chunks",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(t0), midiR2_chunk(t1) }),
          R::TrackCountMismatch, clean0 },
        { "unterminated-sysex",
          midiR2_cat({ midiR2_header(0, 1, 480),
                       midiR2_chunk(midiR2_cat({ midiR2_tempo, openSysEx, notes, midiR2_eot })) }),
          R::UnterminatedSysEx,
          midiR2_cat({ midiR2_header(0, 1, 480),
                       midiR2_chunk(midiR2_cat({ midiR2_tempo, closedSysEx, notes, midiR2_eot })) }) },
        { "unterminated-sysex-at-missing-eot",
          midiR2_cat({ midiR2_header(0, 1, 480), midiR2_chunk(midiR2_cat({ midiR2_tempo, notes, openSysEx })) }),
          R::UnterminatedSysEx | R::MissingEndOfTrack,
          midiR2_cat({ midiR2_header(0, 1, 480),
                       midiR2_chunk(midiR2_cat({ midiR2_tempo, notes, closedSysEx, midiR2_eot })) }) },
        { "truncated-file", truncated, R::MissingEndOfTrack | R::TrackLengthMismatch, truncatedClean },
    };

    const R flags[] = { R::NoncanonicalVlq, R::DataAfterEndOfTrack, R::TrackLengthMismatch,
                        R::MissingEndOfTrack, R::UnterminatedSysEx, R::TrackCountMismatch,
                        R::TrailingBytes };
    bool all = true;
    for (const Case& test : cases)
    {
        const std::string label = test.name;
        MidiFile reference;
        bool ok = io.read(test.clean, {}, reference);
        // The strict default contract is unchanged.
        ok = midi_test::record(io.rejects(test.bytes, {}), label + "/strict-rejects") && ok;

        // Exactly the case's recoveries suffice, are reported, and give the
        // clean document.
        MidiFile tolerant;
        ok = ok && io.read(test.bytes, midiR2_allow(test.expected), tolerant)
            && tolerant.recoveries() == test.expected
            && tolerant.tracks() == reference.tracks()
            && tolerant.format() == reference.format();
        // Allowing everything applies no further recovery.
        MidiFile lenient;
        ok = ok && io.read(test.bytes, MidiReadOptions::tolerant(), lenient)
            && lenient.recoveries() == test.expected
            && lenient.tracks() == reference.tracks();
        // Each of the case's recoveries is needed on its own.
        for (const R flag : flags)
        {
            if (!hasRecovery(test.expected, flag))
                continue;
            const R without = static_cast<R>(static_cast<uint32_t>(test.expected)
                                             & ~static_cast<uint32_t>(flag));
            ok = midi_test::record(io.rejects(test.bytes, midiR2_allow(without)),
                                   label + "/needs-flag") && ok;
        }
        all = midi_test::record(ok, label) && all;
    }
    EXPECT_TRUE(all);
    EXPECT_FALSE(hasRecovery(R::All, R::None));
    EXPECT_TRUE(hasRecovery(R::All, R::TrailingBytes | R::NoncanonicalVlq));
}

DSPARK_TEST(MidiFile_tolerant_document_writes_back_and_rereads_strictly)
{
    using R = MidiReadRecovery;
    midiR2_Reader io("r2-writeback");
    const auto notes = midiR2_notes(8);
    // One file with six defects: header count 3 for 2 tracks, an overlong VLQ,
    // an open SysEx, a missing EOT, padding after an EOT and 0x1A padding.
    const auto track0 = midiR2_cat({ midiR2_tempo, { 0x80, 0x00, 0xf0, 0x02, 0x43, 0x10 }, notes });
    const auto track1 = midiR2_cat({ { 0x00, 0xc3, 0x05 }, notes, midiR2_eot, { 0, 0 } });
    const auto bytes = midiR2_cat({ midiR2_header(1, 3, 960), midiR2_chunk(track0), midiR2_chunk(track1),
                                    std::vector<uint8_t>(5, 0x1a) });
    const R expected = R::TrackCountMismatch | R::NoncanonicalVlq | R::UnterminatedSysEx
                     | R::MissingEndOfTrack | R::DataAfterEndOfTrack | R::TrailingBytes;

    MidiFile recovered;
    EXPECT_TRUE(io.read(bytes, MidiReadOptions::tolerant(), recovered));
    EXPECT_TRUE(recovered.recoveries() == expected);
    EXPECT_EQ(recovered.tracks().size(), size_t(2));
    EXPECT_EQ(recovered.ticksPerQuarter(), uint16_t(960));
    if (recovered.tracks().size() != 2)
        return;
    // The recovered track 0: tempo, SysEx closed by F7, 16 notes, appended EOT.
    const auto& events0 = recovered.tracks()[0].events;
    EXPECT_EQ(events0.size(), size_t(1 + 1 + 16 + 1));
    EXPECT_TRUE(events0[1].kind == MidiEventKind::SysExF0
                && events0[1].payload == std::vector<uint8_t>({ 0x43, 0x10, 0xf7 }));
    EXPECT_TRUE(events0.back().kind == MidiEventKind::Meta && events0.back().metaType == 0x2f
                && events0.back().deltaTicks == 0);

    const auto path = io.file("recovered.mid");
    EXPECT_TRUE(recovered.write(path));
    MidiFile strict;
    EXPECT_TRUE(strict.read(path));
    EXPECT_TRUE(strict.recoveries() == R::None);
    EXPECT_TRUE(strict.tracks() == recovered.tracks());
    EXPECT_EQ(strict.format(), 1);
    EXPECT_EQ(*strict.tickToMicroseconds(960), uint64_t(500000));
}

DSPARK_TEST(MidiFile_tolerant_read_keeps_malformed_input_fatal)
{
    using midi_test::smf;
    using midi_test::withEot;
    midiR2_Reader io("r2-fatal");
    const auto tolerant = MidiReadOptions::tolerant();
    const auto valid = smf(0, 96, { withEot({}) });
    std::vector<std::pair<std::string, std::vector<uint8_t>>> cases;
    cases.push_back({ "zero-tempo", smf(0, 96, { withEot({ 0, 0xff, 0x51, 3, 0, 0, 0 }) }) });
    cases.push_back({ "illegal-status-f1", smf(0, 96, { withEot({ 0x00, 0xf1, 0x10 }) }) });
    cases.push_back({ "data-byte-high", smf(0, 96, { withEot({ 0x00, 0x90, 60, 0x80 }) }) });
    cases.push_back({ "vlq-five-significant-bytes",
                      smf(0, 96, { { 0x81, 0x80, 0x80, 0x80, 0x00, 0xff, 0x2f, 0 } }) });
    cases.push_back({ "running-status-at-start", smf(0, 96, { withEot({ 0x00, 60, 64 }) }) });
    cases.push_back({ "running-after-meta",
                      smf(0, 96, { withEot({ 0x00, 0x90, 60, 64, 0x00, 0xff, 1, 0, 0x00, 61, 64 }) }) });
    cases.push_back({ "meta-type-high", smf(0, 96, { withEot({ 0, 0xff, 0x80, 0 }) }) });
    cases.push_back({ "nonempty-eot", smf(0, 96, { { 0x00, 0xff, 0x2f, 1, 0 } }) });
    auto format3 = valid; format3[9] = 3; cases.push_back({ "format-3", format3 });
    auto shortHeader = valid; shortHeader[7] = 5; cases.push_back({ "header-length-5", shortHeader });
    auto duplicate = valid;
    duplicate.insert(duplicate.end(), { 'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 0, 0, 1, 0, 96 });
    cases.push_back({ "duplicate-mthd", duplicate });
    auto zeroDivision = valid; zeroDivision[12] = 0; zeroDivision[13] = 0;
    cases.push_back({ "zero-division", zeroDivision });
    cases.push_back({ "smpte-rate-26", smf(0, uint16_t(((256 - 26) << 8) | 40), { withEot({}) }) });
    cases.push_back({ "smpte-zero-ticks-per-frame", smf(0, uint16_t((256 - 25) << 8), { withEot({}) }) });
    // An alien chunk whose length runs past the end of the file swallows it:
    // the trailing bytes are dropped and no track chunk is left.
    auto alienOverrun = smf(0, 96, { withEot({}) }, {}, { 0 });
    alienOverrun[18] = 0x08; alienOverrun[19] = 0; alienOverrun[20] = 0; alienOverrun[21] = 1;
    cases.push_back({ "alien-overrun-hides-every-track", alienOverrun });
    cases.push_back({ "no-track-chunk", midiR2_header(1, 1, 96) });

    bool all = true;
    for (const auto& test : cases)
        all = midi_test::record(io.rejects(test.second, tolerant), test.first) && all;
    EXPECT_TRUE(all);

    // Mutation campaign: tolerant reading accepts a strict superset, and every
    // recovered document passes the strict model, writes and rereads strictly.
    namespace fs = std::filesystem;
    const fs::path root = fs::path(DSPARK_TEST_FIXTURE_DIR) / "midi";
    const std::vector<uint8_t> corpus[] = {
        midi_test::readBytes(root / "rp001-format0.mid"),
        midi_test::readBytes(root / "rp001-format1.mid")
    };
    EXPECT_TRUE(!corpus[0].empty() && !corpus[1].empty());
    if (corpus[0].empty() || corpus[1].empty())
        return;
    uint64_t state = 0x4d494449523254ull;
    uint64_t strictAccepted = 0, tolerantAccepted = 0, rereadMismatch = 0, notSuperset = 0;
    for (size_t index = 0; index < 1500; ++index)
    {
        auto next = [&]() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        };
        std::vector<uint8_t> bytes = corpus[index & 1u];
        const unsigned edits = 1u + static_cast<unsigned>(next() % 3u);
        for (unsigned edit = 0; edit < edits; ++edit)
        {
            const size_t pos = static_cast<size_t>(next() % bytes.size());
            switch (next() % 4u)
            {
                case 0: bytes[pos] ^= static_cast<uint8_t>(1u << (next() & 7u)); break;
                case 1: bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(pos), uint8_t(0x80)); break;
                case 2: bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(pos)); break;
                default: bytes.resize(std::max<size_t>(pos, 14)); break;
            }
        }
        MidiFile strict, lenient;
        const bool strictOk = io.read(bytes, {}, strict);
        const bool lenientOk = io.read(bytes, tolerant, lenient);
        strictAccepted += strictOk ? 1u : 0u;
        tolerantAccepted += lenientOk ? 1u : 0u;
        if (strictOk && (!lenientOk || lenient.tracks() != strict.tracks()
                         || lenient.recoveries() != MidiReadRecovery::None))
            ++notSuperset;
        if (lenientOk && lenient.format() != 2)
        {
            const auto path = io.file("rewrite.mid");
            MidiFile reread;
            if (!lenient.write(path) || !reread.read(path)
                || reread.tracks() != lenient.tracks())
                ++rereadMismatch;
        }
    }
    EXPECT_EQ(notSuperset, uint64_t(0));
    EXPECT_EQ(rereadMismatch, uint64_t(0));
    EXPECT_GT(tolerantAccepted, strictAccepted + 100);
    std::cout << "MIDI tolerant mutation cases=1500 strict-accepted=" << strictAccepted
              << " tolerant-accepted=" << tolerantAccepted << "\n";
}

DSPARK_TEST(MidiFile_SMPTE_division_timing_is_exact_and_ignores_tempo)
{
    midiR2_Reader io("r2-smpte");
    MidiReadOptions smpte;
    smpte.acceptSmpteDivision = true;
    const auto notes = midiR2_notes(20);                       // 4800 ticks
    const std::vector<uint8_t> tempoAt960 { 0x87, 0x40, 0xff, 0x51, 0x03, 0x03, 0x0d, 0x40 };
    const auto withTempo = midiR2_cat({ midiR2_tempo, tempoAt960, notes, midiR2_eot });
    const auto withoutTempo = midiR2_cat({ { 0x87, 0x40, 0xff, 0x01, 0x00 }, notes, midiR2_eot });

    struct Rate
    {
        int fps;
        uint8_t ticksPerFrame;
        double endSeconds;   // at tick 5760
        uint64_t numerator;  // ticks per second = numerator / denominator
        uint64_t denominator;
    };
    // 29 is 30 drop frame: 30000/1001 frames per second.
    const Rate rates[] = {
        { 24, 4, 60.0, 96, 1 },
        { 25, 40, 5.76, 1000, 1 },
        { 29, 80, 2.4024, 30000ull * 80, 1001 },
        { 30, 100, 1.92, 3000, 1 },
    };
    bool all = true;
    for (const Rate& rate : rates)
    {
        const std::string label = "smpte-" + std::to_string(rate.fps);
        const auto division = static_cast<uint16_t>(((256 - rate.fps) << 8) | rate.ticksPerFrame);
        for (const uint16_t format : { uint16_t(1), uint16_t(2) })
        {
            const auto bytes = midiR2_cat({ midiR2_header(format, 2, division), midiR2_chunk(withTempo),
                                            midiR2_chunk(withoutTempo) });
            bool ok = midi_test::record(io.rejects(bytes, {}), label + "/default-rejects");
            MidiFile file;
            ok = ok && io.read(bytes, smpte, file) && file.isSmpteDivision()
                && file.division() == division && file.smpteFramesPerSecond() == rate.fps
                && file.ticksPerFrame() == rate.ticksPerFrame && file.ticksPerQuarter() == 0
                && !file.tempoMap(0).has_value() && file.recoveries() == MidiReadRecovery::None;
            // Track 0 holds two tempo events, track 1 none: the time base is
            // the same whichever track supplies it.
            for (const size_t track : { size_t(0), size_t(1) })
            {
                ok = ok && file.tickToSeconds(5760, track) == rate.endSeconds;
                for (const uint64_t tick : { uint64_t(0), uint64_t(1), uint64_t(7), uint64_t(960),
                                             uint64_t(5760), uint64_t(24000), uint64_t(1000000007) })
                {
                    const uint64_t exact = tick * 1000000ull * rate.denominator / rate.numerator;
                    ok = ok && file.tickToMicroseconds(tick, track) == exact;
                }
            }
            all = midi_test::record(ok, label + "/format" + std::to_string(format)) && all;
        }
    }
    EXPECT_TRUE(all);

    // 30 drop frame runs at 29.97 frames per second, not 29 or 30.
    const auto dropFrame = midiR2_cat({ midiR2_header(0, 1, 0xe350), midiR2_chunk(withTempo) });
    MidiFile file;
    EXPECT_TRUE(io.read(dropFrame, MidiReadOptions::tolerant(), file));
    EXPECT_TRUE(file.tickToMicroseconds(24000) == uint64_t(10010000));
    EXPECT_TRUE(file.tickToSeconds(24000) == 10.01);
}

DSPARK_TEST(MidiFile_SMPTE_authoring_round_trips)
{
    midiR2_Reader io("r2-smpte-write");
    MidiFile midi;
    EXPECT_TRUE(midi.createSmpte(1, 29, 80, 2));
    EXPECT_EQ(midi.division(), uint16_t(0xe350));
    EXPECT_TRUE(midi.isSmpteDivision());
    EXPECT_EQ(midi.ticksPerQuarter(), uint16_t(0));
    EXPECT_TRUE(midi.addMetaEvent(0, 0, 0x51, std::array<uint8_t, 3> { 0x07, 0xa1, 0x20 }));
    EXPECT_TRUE(midi.addChannelEvent(1, 2400, 0x90, 60, 100));
    EXPECT_TRUE(midi.addChannelEvent(1, 2400, 0x80, 60, 0));
    EXPECT_TRUE(midi.tickToMicroseconds(4800, 1) == uint64_t(2002000));

    const auto path = io.file("smpte.mid");
    EXPECT_TRUE(midi.write(path));
    MidiFile strict;
    EXPECT_FALSE(strict.read(path));
    MidiReadOptions smpte;
    smpte.acceptSmpteDivision = true;
    MidiFile decoded;
    EXPECT_TRUE(decoded.read(path, smpte));
    EXPECT_EQ(decoded.division(), uint16_t(0xe350));
    EXPECT_EQ(decoded.format(), 1);
    if (decoded.tracks().size() == 2 && decoded.tracks()[1].events.size() == 3)
    {
        // The writer appends the missing EOTs; the authored events match.
        EXPECT_TRUE(decoded.tracks()[1].events[0] == midi.tracks()[1].events[0]);
        EXPECT_TRUE(decoded.tracks()[1].events[1] == midi.tracks()[1].events[1]);
        EXPECT_TRUE(decoded.tracks()[0].events[0] == midi.tracks()[0].events[0]);
    }
    else
    {
        EXPECT_TRUE(false);
    }

    const MidiTrack before = midi.tracks()[1];
    EXPECT_FALSE(midi.createSmpte(1, 26, 80));
    EXPECT_FALSE(midi.createSmpte(1, 25, 0));
    EXPECT_FALSE(midi.createSmpte(2, 25, 40));
    EXPECT_FALSE(midi.createSmpte(0, 25, 40, 2));
    EXPECT_TRUE(midi.tracks()[1] == before);
    EXPECT_EQ(midi.division(), uint16_t(0xe350));
    EXPECT_TRUE(midi.create(0, 480));
    EXPECT_FALSE(midi.isSmpteDivision());
    EXPECT_EQ(midi.smpteFramesPerSecond(), 0);
    EXPECT_EQ(midi.ticksPerQuarter(), uint16_t(480));
}

// ============================================================================
// 2026-11 round 2: FLAC Rice decoding. The decoder takes Rice codes from a
// 64-bit register, restoring each sample as its residual arrives, and falls
// back to the code-by-code reader for codes longer than the bits in hand and
// for the end of the buffer. This pins both paths, every Rice parameter and
// every bit alignment against the values that were encoded, and the residual
// bound on both sides of the fall-back.
// ============================================================================

namespace {

class flacR2_BitWriter
{
public:
    void bit(bool one)
    {
        if (used_ == 0) bytes_.push_back(0);
        if (one) bytes_.back() = static_cast<uint8_t>(bytes_.back() | (0x80u >> used_));
        used_ = (used_ + 1u) & 7u;
    }
    void put(uint64_t value, unsigned count)
    {
        for (unsigned i = count; i-- > 0;) bit(((value >> i) & 1u) != 0);
    }
    void unary(uint64_t zeros)
    {
        for (uint64_t i = 0; i < zeros; ++i) bit(false);
        bit(true);
    }
    void align() { used_ = 0; }   // the rest of the last byte is already zero
    [[nodiscard]] std::vector<uint8_t>& bytes() { return bytes_; }

private:
    std::vector<uint8_t> bytes_;
    unsigned used_ = 0;
};

// A mono 32-bit stream with fixed blocking and STREAMINFO MD5 zero. Frame f is
// one FIXED order-0 subframe (the prediction is zero, so the samples are the
// residuals) with a single Rice partition of parameter parameters[f] that
// codes the folded values folded[f].
std::vector<uint8_t> flacR2_riceStream(const std::vector<unsigned>& parameters,
                                       const std::vector<std::vector<uint64_t>>& folded,
                                       uint32_t blockSize)
{
    const uint64_t total = static_cast<uint64_t>(blockSize) * parameters.size();
    std::vector<uint8_t> out { 'f', 'L', 'a', 'C', 0x80, 0, 0, 34 };
    flacR2_BitWriter info;
    info.put(blockSize, 16);
    info.put(blockSize, 16);
    info.put(0, 24);
    info.put(0, 24);
    info.put(48000, 20);
    info.put(0, 3);        // channels - 1
    info.put(31, 5);       // bits per sample - 1
    info.put(total, 36);
    info.put(0, 64);       // MD5 not computed
    info.put(0, 64);
    out.insert(out.end(), info.bytes().begin(), info.bytes().end());

    for (size_t f = 0; f < parameters.size(); ++f)
    {
        flacR2_BitWriter frame;
        frame.put(0xfff8, 16);                 // sync, fixed blocking
        frame.put(7, 4);                       // block size: 16-bit field
        frame.put(0, 4);                       // rate from STREAMINFO
        frame.put(0, 4);                       // mono
        frame.put(7, 3);                       // 32 bits per sample
        frame.put(0, 1);
        frame.put(f, 8);                       // frame number (< 128)
        frame.put(blockSize - 1u, 16);
        frame.put(flac_test::crc8(frame.bytes()), 8);
        frame.put(0, 1);
        frame.put(8, 6);                       // FIXED, order 0
        frame.put(0, 1);                       // no wasted bits
        frame.put(1, 2);                       // 5-bit Rice parameters
        frame.put(0, 4);                       // one partition
        const unsigned k = parameters[f];
        frame.put(k, 5);
        for (uint64_t value : folded[f])
        {
            frame.unary(value >> k);
            frame.put(value & ((uint64_t { 1 } << k) - 1u), k);
        }
        frame.align();
        const uint16_t crc = flac_test::crc16(frame.bytes());
        frame.put(crc, 16);
        out.insert(out.end(), frame.bytes().begin(), frame.bytes().end());
    }
    return out;
}

int64_t flacR2_unfold(uint64_t folded)
{
    return (folded & 1u) == 0 ? static_cast<int64_t>(folded >> 1)
                              : -static_cast<int64_t>(folded >> 1) - 1;
}

} // namespace

DSPARK_TEST(FlacFile_rice_codes_decode_exactly_at_every_parameter_and_alignment)
{
    constexpr uint32_t kBlock = 1024;
    constexpr uint64_t kMaxFolded = 4294967294ull;   // residual +-(2^31 - 1)
    std::vector<unsigned> parameters;
    std::vector<std::vector<uint64_t>> folded;
    uint64_t state = 0x52494345u;
    auto next = [&]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (unsigned k = 0; k <= 30; ++k)
    {
        // Mostly short quotients (the register path), some of 40..100 zeros
        // (beyond the 56 bits in hand for large k: the code-by-code path),
        // and for large k the extreme residuals.
        std::vector<uint64_t> values(kBlock);
        for (uint64_t& value : values)
        {
            const uint64_t pick = next() % 10;
            const uint64_t quotient = pick < 7 ? next() % 4
                                    : pick < 9 ? next() % 41 : 40 + next() % 61;
            value = std::min((quotient << k) | (next() & ((uint64_t { 1 } << k) - 1u)),
                             kMaxFolded);
        }
        values[kBlock / 2] = std::min(kMaxFolded, (uint64_t { 100 } << k) | 1u);
        if (k >= 26)
        {
            values[3] = kMaxFolded;
            values[kBlock - 2] = kMaxFolded - 1;
        }
        parameters.push_back(k);
        folded.push_back(std::move(values));
    }

    midi_test::TempDirectory temp("flac-r2-rice");
    const auto path = temp.file("rice.flac");
    EXPECT_TRUE(midi_test::writeBytes(path, flacR2_riceStream(parameters, folded, kBlock)));
    FlacFile file;
    EXPECT_TRUE(file.openRead(path));
    EXPECT_EQ(file.getInfo().numSamples, int64_t(kBlock) * 31);
    if (!file.isOpen()) return;
    AudioBuffer<float> decoded;
    decoded.resize(1, static_cast<int>(kBlock * 31));
    EXPECT_TRUE(file.readSamples(decoded.toView()));
    const float scale = std::ldexp(1.0f, -31);
    size_t mismatches = 0;
    for (size_t f = 0; f < folded.size(); ++f)
        for (uint32_t i = 0; i < kBlock; ++i)
        {
            const float expected = static_cast<float>(
                static_cast<int32_t>(flacR2_unfold(folded[f][i]))) * scale;
            if (std::bit_cast<uint32_t>(decoded.getChannel(0)[f * kBlock + i])
                != std::bit_cast<uint32_t>(expected))
                ++mismatches;
        }
    EXPECT_EQ(mismatches, size_t(0));

    // One folded value past the bound (residual -2^31) rejects the stream:
    // with k = 30 the code is short (register path), with k = 20 its 4095
    // zeros take the code-by-code path. 2^32 - 2 is still accepted.
    for (const unsigned k : { 30u, 20u })
        for (const uint64_t extreme : { kMaxFolded, kMaxFolded + 1 })
        {
            std::vector<uint64_t> values(64, 5);
            values[17] = extreme;
            const auto bytes = flacR2_riceStream({ k }, { values }, 64);
            const auto bad = temp.file("bound-" + std::to_string(k) + "-"
                                       + std::to_string(extreme) + ".flac");
            EXPECT_TRUE(midi_test::writeBytes(bad, bytes));
            FlacFile boundFile;
            EXPECT_EQ(boundFile.openRead(bad), extreme == kMaxFolded);
        }
}
