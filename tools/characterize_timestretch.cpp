// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

// Offline characterisation harness for Effects/TimeStretch.h.
//
// Measures the acceptance criteria of the time stretcher and writes one
// machine-generated artifact per criterion plus a summary table. It is a
// measurement tool, not part of the library: the WSOLA reference and the
// consistency analyser below exist only so the phase vocoder can be compared
// against something independent of itself.
//
//   g++ -std=c++20 -O2 -Wall -Wextra -I. tools/characterize_timestretch.cpp -o ts
//   ./ts <output-directory>
//
// Criterion tags used in the artifacts (defined here, used here):
//   C1: stationary fidelity (log-spectral distance, spectral convergence)
//   C2: spurious / alias floor
//   C3: transient preservation against a WSOLA reference
//   C4: vertical coherence (STFT consistency)
//   C5: stereo integrity (level difference, coherence)
//   C6: exact ratio and drift
//   C7: unity passthrough
//   C8: determinism under block chopping

#include "../Core/FFT.h"
#include "../Effects/TimeStretch.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kRate = 48000.0;
constexpr int kBlock = 512;

constexpr std::array<const char*, 9> kExpectedOutputs = {
    "c1-stationary-fidelity.csv",
    "c2-spurious-floor.csv",
    "c3-transient-preservation.csv",
    "c4-vertical-coherence.csv",
    "c5-stereo-integrity.csv",
    "c6-exact-ratio-drift.csv",
    "c7-unity-passthrough.csv",
    "c8-chopping-determinism.csv",
    "timestretch-metrics.md",
};

constexpr const char* kTransactionDirectory =
    ".dspark-timestretch-transaction";

using Signal = std::vector<std::vector<double>>;   // [channel][sample]

// -- deterministic sources -----------------------------------------------------

struct Lcg
{
    uint32_t s = 987654321u;
    double next() { s = s * 1664525u + 1013904223u; return static_cast<double>(s >> 8) / 8388608.0 - 1.0; }
};

/// 12-partial harmonic bed, f0 = 110 Hz, 1/k amplitudes, fixed phases.
Signal harmonicBed(double seconds, int channels)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    for (int k = 1; k <= 12; ++k)
    {
        const double f = 110.0 * k;
        const double a = 0.35 / k;
        const double ph = 0.7 * k;
        for (size_t i = 0; i < n; ++i)
        {
            const double v = a * std::sin(2.0 * kPi * f * static_cast<double>(i) / kRate + ph);
            for (auto& ch : sig) ch[i] += v;
        }
    }
    return sig;
}

/// The same bed with 3 Hz / 30 cent vibrato on every partial. Phase locking
/// only has something to do when the bins of one partial disagree, which a
/// perfectly stationary partial never makes them do.
Signal vibratoBed(double seconds, int channels)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    for (int k = 1; k <= 12; ++k)
    {
        const double a = 0.35 / k;
        double phase = 0.7 * k;
        for (size_t i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(i) / kRate;
            const double f = 110.0 * k * std::pow(2.0, 0.30 / 12.0 * std::sin(2.0 * kPi * 3.0 * t));
            phase += 2.0 * kPi * f / kRate;
            const double v = a * std::sin(phase);
            for (auto& ch : sig) ch[i] += v;
        }
    }
    return sig;
}

/// Sine at an exact analysis bin, so the analyser contributes no leakage.
Signal sine(double freq, double seconds, int channels, double amp = 0.5)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
        const double v = amp * std::sin(2.0 * kPi * freq * static_cast<double>(i) / kRate);
        for (auto& ch : sig) ch[i] = v;
    }
    return sig;
}

/// Pink noise by the Voss-McCartney octave sum (deterministic).
Signal pinkNoise(double seconds, int channels)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    Lcg rng;
    double rows[16] = { 0 };
    for (size_t i = 0; i < n; ++i)
    {
        uint32_t counter = static_cast<uint32_t>(i);
        for (int b = 0; b < 16; ++b)
            if (((counter >> b) & 1u) == 0u) { rows[b] = rng.next(); break; }
        double sum = 0.0;
        for (double r : rows) sum += r;
        const double v = 0.08 * sum;
        for (auto& ch : sig) ch[i] = v;
    }
    return sig;
}

// -- durable nine-artifact publication ----------------------------------------

// This is deliberately self-contained: the characterisation tool is built on
// every supported C++20 platform and cannot depend on an OS transaction API or
// a crypto library.  The journal checksum and artifact identities therefore use
// this compact SHA-256 implementation.
class TransactionSha256
{
public:
    void update(const unsigned char* bytes, size_t count) noexcept
    {
        bitCount_ += static_cast<uint64_t>(count) * 8u;
        while (count > 0)
        {
            const size_t copied = std::min(count, block_.size() - used_);
            std::copy_n(bytes, copied, block_.begin() + static_cast<long>(used_));
            bytes += copied;
            count -= copied;
            used_ += copied;
            if (used_ == block_.size())
            {
                transform(block_.data());
                used_ = 0;
            }
        }
    }

    void update(std::string_view text) noexcept
    {
        update(reinterpret_cast<const unsigned char*>(text.data()), text.size());
    }

    std::string finish() noexcept
    {
        const uint64_t originalBits = bitCount_;
        const unsigned char marker = 0x80u;
        update(&marker, 1);
        const unsigned char zero = 0;
        while (used_ != 56)
            update(&zero, 1);
        std::array<unsigned char, 8> length{};
        for (size_t index = 0; index < length.size(); ++index)
            length[7 - index] = static_cast<unsigned char>(
                (originalBits >> (index * 8u)) & 0xffu);
        update(length.data(), length.size());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t word : state_)
            output << std::setw(8) << word;
        return output.str();
    }

private:
    static uint32_t rotate(uint32_t value, unsigned count) noexcept
    {
        return (value >> count) | (value << (32u - count));
    }

    static uint32_t load(const unsigned char* bytes) noexcept
    {
        return (static_cast<uint32_t>(bytes[0]) << 24u)
             | (static_cast<uint32_t>(bytes[1]) << 16u)
             | (static_cast<uint32_t>(bytes[2]) << 8u)
             | static_cast<uint32_t>(bytes[3]);
    }

    void transform(const unsigned char* bytes) noexcept
    {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        std::array<uint32_t, 64> schedule{};
        for (size_t index = 0; index < 16; ++index)
            schedule[index] = load(bytes + index * 4);
        for (size_t index = 16; index < schedule.size(); ++index)
        {
            const uint32_t a = schedule[index - 15];
            const uint32_t b = schedule[index - 2];
            const uint32_t s0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3u);
            const uint32_t s1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10u);
            schedule[index] = schedule[index - 16] + s0
                            + schedule[index - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t index = 0; index < schedule.size(); ++index)
        {
            const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const uint32_t choose = (e & f) ^ (~e & g);
            const uint32_t first = h + s1 + choose
                                 + constants[index] + schedule[index];
            const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t second = s0 + majority;
            h = g; g = f; f = e; e = d + first;
            d = c; c = b; b = a; a = first + second;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<unsigned char, 64> block_{};
    size_t used_ = 0;
    uint64_t bitCount_ = 0;
};

std::string transactionSha256(std::string_view text) noexcept
{
    TransactionSha256 hash;
    hash.update(text);
    return hash.finish();
}

constexpr std::string_view kZeroSha256 =
    "0000000000000000000000000000000000000000000000000000000000000000";

struct TransactionFileIdentity
{
    bool present = false;
    bool valid = false;
    uintmax_t size = 0;
    std::string sha256;
};

TransactionFileIdentity transactionFileIdentity(
    const std::filesystem::path& path, bool requireNonempty = true) noexcept
{
    TransactionFileIdentity result;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory)
        return result;
    if (error || !std::filesystem::exists(status))
        return result;
    result.present = true;
    if (!std::filesystem::is_regular_file(status))
        return result;
    result.size = std::filesystem::file_size(path, error);
    if (error || (requireNonempty && result.size == 0))
        return result;
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
        return result;
    TransactionSha256 hash;
    std::array<char, 8192> buffer{};
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0)
            hash.update(reinterpret_cast<const unsigned char*>(buffer.data()),
                        static_cast<size_t>(count));
    }
    if (!input.eof())
        return result;
    result.sha256 = hash.finish();
    result.valid = true;
    return result;
}

enum class TransactionState
{
    Empty,
    Setup,
    Staging,
    Staged,
    BackupInProgress,
    PublishInProgress,
    CommitReady,
    RollbackRequired,
    RecoveryRequiredRollback,
    CommittedCleanupPending,
    RecoveryRequiredCleanup,
    Complete,
    ForeignOrMalformedCollision,
};

enum class TransactionPending
{
    None,
    Backup,
    Publish,
    RemoveNew,
    RestoreOld,
    CleanupBackup,
};

constexpr std::array<std::string_view, 13> kTransactionStateNames = {
    "EMPTY", "SETUP", "STAGING", "STAGED", "BACKUP_IN_PROGRESS",
    "PUBLISH_IN_PROGRESS", "COMMIT_READY", "ROLLBACK_REQUIRED",
    "RECOVERY_REQUIRED_ROLLBACK", "COMMITTED_CLEANUP_PENDING",
    "RECOVERY_REQUIRED_CLEANUP", "COMPLETE",
    "FOREIGN_OR_MALFORMED_COLLISION",
};

constexpr std::array<std::string_view, 6> kTransactionPendingNames = {
    "NONE", "BACKUP", "PUBLISH", "REMOVE_NEW", "RESTORE_OLD",
    "CLEANUP_BACKUP",
};

struct TransactionEntry
{
    bool oldPresent = false;
    uintmax_t oldSize = 0;
    std::string oldSha256 = std::string(kZeroSha256);
    std::string oldLocation = "ABSENT";
    uintmax_t newSize = 0;
    std::string newSha256 = std::string(kZeroSha256);
    std::string newLocation = "NONE";
};

struct TransactionJournal
{
    std::string transactionId;
    std::string rootBindingSha256;
    uint64_t generation = 0;
    std::string previousPayloadSha256 = std::string(kZeroSha256);
    std::string transitionId = "BEGIN";
    uint64_t transitionIndex = 0;
    TransactionState state = TransactionState::Empty;
    TransactionPending pending = TransactionPending::None;
    size_t pendingIndex = kExpectedOutputs.size();
    size_t rollbackCursor = 0;
    size_t cleanupCursor = 0;
    size_t finalizeCursor = 0;
    std::array<TransactionEntry, kExpectedOutputs.size()> entries{};
};

constexpr std::array<std::string_view, 32> kTransactionTransitionIds = {
    "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08",
    "T09", "T10", "T11", "T12", "T13", "T14", "T15", "T16",
    "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24",
    "T25", "T26", "T27", "T28", "T29", "T30", "T31", "T32",
};

std::string twoDigitIndex(size_t index)
{
    std::ostringstream output;
    output << std::setw(2) << std::setfill('0') << index;
    return output.str();
}

bool transactionLowerHex(std::string_view value, size_t size) noexcept
{
    return value.size() == size
        && std::all_of(value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9')
                || (character >= 'a' && character <= 'f');
        });
}

bool transactionDecimal(std::string_view value, uint64_t& result) noexcept
{
    if (value.empty() || (value.size() > 1 && value.front() == '0'))
        return false;
    uint64_t parsed = 0;
    for (char character : value)
    {
        if (character < '0' || character > '9')
            return false;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10u)
            return false;
        parsed = parsed * 10u + digit;
    }
    result = parsed;
    return true;
}

std::vector<std::string_view> transactionSplit(
    std::string_view value, char delimiter)
{
    std::vector<std::string_view> fields;
    size_t begin = 0;
    while (true)
    {
        const size_t end = value.find(delimiter, begin);
        fields.push_back(value.substr(begin, end - begin));
        if (end == std::string_view::npos)
            break;
        begin = end + 1;
    }
    return fields;
}

std::string serializeTransactionJournal(const TransactionJournal& journal)
{
    std::ostringstream payload;
    payload << "magic=DSPARK_TIMESTRETCH_TXN\n"
            << "version=2\n"
            << "transaction_id=" << journal.transactionId << '\n'
            << "root_binding_sha256=" << journal.rootBindingSha256 << '\n'
            << "generation=" << journal.generation << '\n'
            << "previous_payload_sha256=" << journal.previousPayloadSha256 << '\n'
            << "transition_id=" << journal.transitionId << '\n'
            << "transition_index=" << journal.transitionIndex << '\n'
            << "state=" << kTransactionStateNames[static_cast<size_t>(journal.state)] << '\n'
            << "pending_kind=" << kTransactionPendingNames[static_cast<size_t>(journal.pending)] << '\n'
            << "pending_index=" << journal.pendingIndex << '\n'
            << "rollback_cursor=" << journal.rollbackCursor << '\n'
            << "cleanup_cursor=" << journal.cleanupCursor << '\n'
            << "finalize_cursor=" << journal.finalizeCursor << '\n';
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        const auto& entry = journal.entries[index];
        payload << "output" << index << '=' << index << '|'
                << kExpectedOutputs[index] << '|'
                << (entry.oldPresent ? 1 : 0) << '|'
                << entry.oldSize << '|' << entry.oldSha256 << '|'
                << entry.oldLocation << '|' << entry.newSize << '|'
                << entry.newSha256 << '|' << entry.newLocation << '\n';
    }
    const std::string bytes = payload.str();
    return bytes + "payload_sha256=" + transactionSha256(bytes) + "\n";
}

bool transactionStateFromName(std::string_view name,
                              TransactionState& state) noexcept
{
    for (size_t index = 0; index < kTransactionStateNames.size(); ++index)
        if (name == kTransactionStateNames[index])
        {
            state = static_cast<TransactionState>(index);
            return true;
        }
    return false;
}

bool transactionPendingFromName(std::string_view name,
                                TransactionPending& pending) noexcept
{
    for (size_t index = 0; index < kTransactionPendingNames.size(); ++index)
        if (name == kTransactionPendingNames[index])
        {
            pending = static_cast<TransactionPending>(index);
            return true;
        }
    return false;
}

bool transactionTransitionId(std::string_view value) noexcept
{
    return std::find(kTransactionTransitionIds.begin(),
                     kTransactionTransitionIds.end(), value)
        != kTransactionTransitionIds.end();
}

bool parseTransactionJournal(std::string_view bytes,
                             TransactionJournal& journal) noexcept
{
    if (bytes.empty() || bytes.size() > 65536 || bytes.back() != '\n'
        || bytes.find('\r') != std::string_view::npos
        || bytes.find('\0') != std::string_view::npos)
        return false;
    for (unsigned char character : bytes)
        if (character != '\n' && (character < 0x20u || character > 0x7eu))
            return false;
    std::vector<std::string_view> lines;
    size_t begin = 0;
    while (begin < bytes.size())
    {
        const size_t end = bytes.find('\n', begin);
        if (end == std::string_view::npos || end == begin)
            return false;
        lines.push_back(bytes.substr(begin, end - begin));
        begin = end + 1;
    }
    if (lines.size() != 24)
        return false;
    auto value = [&lines](size_t index, std::string_view key,
                          std::string_view& output) {
        const std::string prefix = std::string(key) + '=';
        if (!lines[index].starts_with(prefix))
            return false;
        output = lines[index].substr(prefix.size());
        return true;
    };
    if (lines[0] != "magic=DSPARK_TIMESTRETCH_TXN"
        || lines[1] != "version=2")
        return false;
    std::string_view field;
    if (!value(2, "transaction_id", field)
        || !transactionLowerHex(field, 32))
        return false;
    journal.transactionId = field;
    if (!value(3, "root_binding_sha256", field)
        || !transactionLowerHex(field, 64))
        return false;
    journal.rootBindingSha256 = field;
    uint64_t number = 0;
    if (!value(4, "generation", field) || !transactionDecimal(field, number)
        || number == 0)
        return false;
    journal.generation = number;
    if (!value(5, "previous_payload_sha256", field)
        || !transactionLowerHex(field, 64))
        return false;
    journal.previousPayloadSha256 = field;
    if (!value(6, "transition_id", field)
        || (field != "BEGIN" && !transactionTransitionId(field)))
        return false;
    journal.transitionId = field;
    if (!value(7, "transition_index", field)
        || !transactionDecimal(field, number))
        return false;
    journal.transitionIndex = number;
    if ((journal.generation == 1
         && (journal.previousPayloadSha256 != kZeroSha256
             || journal.transitionId != "BEGIN"
             || journal.transitionIndex != 0))
        || (journal.generation != 1
            && (journal.previousPayloadSha256 == kZeroSha256
                || journal.transitionId == "BEGIN"
                || journal.transitionIndex != journal.generation - 1)))
        return false;
    if (!value(8, "state", field) || !transactionStateFromName(field, journal.state)
        || journal.state == TransactionState::Empty
        || journal.state == TransactionState::Complete
        || journal.state == TransactionState::ForeignOrMalformedCollision)
        return false;
    if (!value(9, "pending_kind", field)
        || !transactionPendingFromName(field, journal.pending))
        return false;
    if (!value(10, "pending_index", field) || !transactionDecimal(field, number)
        || number > kExpectedOutputs.size())
        return false;
    journal.pendingIndex = static_cast<size_t>(number);
    if ((journal.pending == TransactionPending::None
         && journal.pendingIndex != kExpectedOutputs.size())
        || (journal.pending != TransactionPending::None
            && journal.pendingIndex >= kExpectedOutputs.size()))
        return false;
    if (!value(11, "rollback_cursor", field) || !transactionDecimal(field, number)
        || number > 2 * kExpectedOutputs.size())
        return false;
    journal.rollbackCursor = static_cast<size_t>(number);
    if (!value(12, "cleanup_cursor", field) || !transactionDecimal(field, number)
        || number > kExpectedOutputs.size())
        return false;
    journal.cleanupCursor = static_cast<size_t>(number);
    if (!value(13, "finalize_cursor", field) || !transactionDecimal(field, number)
        || number > 3)
        return false;
    journal.finalizeCursor = static_cast<size_t>(number);
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        const std::string key = "output" + std::to_string(index);
        if (!value(14 + index, key, field))
            return false;
        const auto fields = transactionSplit(field, '|');
        if (fields.size() != 9 || fields[0] != std::to_string(index)
            || fields[1] != kExpectedOutputs[index]
            || (fields[2] != "0" && fields[2] != "1"))
            return false;
        auto& entry = journal.entries[index];
        entry.oldPresent = fields[2] == "1";
        if (!transactionDecimal(fields[3], number))
            return false;
        entry.oldSize = static_cast<uintmax_t>(number);
        if (!transactionLowerHex(fields[4], 64))
            return false;
        entry.oldSha256 = fields[4];
        if (fields[5] != "FINAL" && fields[5] != "BACKUP"
            && fields[5] != "ABSENT")
            return false;
        entry.oldLocation = fields[5];
        if (!transactionDecimal(fields[6], number))
            return false;
        entry.newSize = static_cast<uintmax_t>(number);
        if (!transactionLowerHex(fields[7], 64))
            return false;
        entry.newSha256 = fields[7];
        if (fields[8] != "NONE" && fields[8] != "STAGED"
            && fields[8] != "FINAL")
            return false;
        entry.newLocation = fields[8];
        if ((!entry.oldPresent
             && (entry.oldSize != 0 || entry.oldSha256 != kZeroSha256
                 || entry.oldLocation != "ABSENT"))
            || (entry.oldPresent
                && (entry.oldSize == 0 || entry.oldSha256 == kZeroSha256
                    || (entry.oldLocation == "ABSENT"
                        && journal.state
                            != TransactionState::CommittedCleanupPending
                        && journal.state
                            != TransactionState::RecoveryRequiredCleanup)))
            || (entry.newSize == 0
                && (entry.newSha256 != kZeroSha256
                    || entry.newLocation != "NONE"))
            || (entry.newSize > 0 && entry.newSha256 == kZeroSha256))
            return false;
    }
    if (!value(23, "payload_sha256", field)
        || !transactionLowerHex(field, 64))
        return false;
    const size_t checksumLine = bytes.rfind("payload_sha256=");
    if (checksumLine == std::string_view::npos
        || transactionSha256(bytes.substr(0, checksumLine)) != field)
        return false;
    const bool committed = journal.state == TransactionState::CommittedCleanupPending
                        || journal.state == TransactionState::RecoveryRequiredCleanup;
    if (committed)
    {
        for (const auto& entry : journal.entries)
            if (entry.newSize == 0 || entry.newLocation != "FINAL")
                return false;
        if (journal.pending != TransactionPending::None
            && journal.pending != TransactionPending::CleanupBackup)
            return false;
    }
    else if (journal.pending == TransactionPending::CleanupBackup)
    {
        return false;
    }

    size_t established = 0;
    bool sawUnestablished = false;
    for (const auto& entry : journal.entries)
    {
        if (entry.newSize == 0)
        {
            sawUnestablished = true;
        }
        else
        {
            if (sawUnestablished)
                return false;
            ++established;
        }
    }
    const auto oldInitial = [&journal]() {
        return std::all_of(journal.entries.begin(), journal.entries.end(),
            [](const TransactionEntry& entry) {
                return entry.oldLocation == (entry.oldPresent ? "FINAL" : "ABSENT");
            });
    };
    const auto oldBackedUp = [&journal]() {
        return std::all_of(journal.entries.begin(), journal.entries.end(),
            [](const TransactionEntry& entry) {
                return entry.oldLocation == (entry.oldPresent ? "BACKUP" : "ABSENT");
            });
    };
    const auto newLocations = [&journal](std::string_view value) {
        return std::all_of(journal.entries.begin(), journal.entries.end(),
            [value](const TransactionEntry& entry) {
                return entry.newSize == 0
                    ? entry.newLocation == "NONE"
                    : entry.newLocation == value;
            });
    };
    const bool cursorsZero = journal.rollbackCursor == 0
                          && journal.cleanupCursor == 0
                          && journal.finalizeCursor == 0;
    switch (journal.state)
    {
        case TransactionState::Setup:
            if (established != 0 || journal.pending != TransactionPending::None
                || !cursorsZero || !oldInitial())
                return false;
            break;
        case TransactionState::Staging:
            if (established == 0 || established >= kExpectedOutputs.size()
                || journal.pending != TransactionPending::None || !cursorsZero
                || !oldInitial() || !newLocations("STAGED"))
                return false;
            break;
        case TransactionState::Staged:
            if (established != kExpectedOutputs.size()
                || journal.pending != TransactionPending::None || !cursorsZero
                || !oldInitial() || !newLocations("STAGED"))
                return false;
            break;
        case TransactionState::BackupInProgress:
            if (established != kExpectedOutputs.size() || !cursorsZero
                || !newLocations("STAGED")
                || (journal.pending != TransactionPending::None
                    && journal.pending != TransactionPending::Backup))
                return false;
            break;
        case TransactionState::PublishInProgress:
        {
            if (established != kExpectedOutputs.size() || !cursorsZero
                || !oldBackedUp()
                || (journal.pending != TransactionPending::None
                    && journal.pending != TransactionPending::Publish))
                return false;
            bool sawStaged = false;
            for (const auto& entry : journal.entries)
            {
                if (entry.newLocation == "STAGED")
                    sawStaged = true;
                else if (entry.newLocation != "FINAL" || sawStaged)
                    return false;
            }
            break;
        }
        case TransactionState::CommitReady:
            if (established != kExpectedOutputs.size()
                || journal.pending != TransactionPending::None || !cursorsZero
                || !oldBackedUp() || !newLocations("FINAL"))
                return false;
            break;
        case TransactionState::RollbackRequired:
        case TransactionState::RecoveryRequiredRollback:
            if (journal.cleanupCursor != 0
                || (journal.pending != TransactionPending::None
                    && journal.pending != TransactionPending::RemoveNew
                    && journal.pending != TransactionPending::RestoreOld)
                || (journal.finalizeCursor != 0 && journal.rollbackCursor != 18))
                return false;
            if (journal.pending == TransactionPending::RemoveNew
                && (journal.rollbackCursor >= 9
                    || journal.pendingIndex != 8 - journal.rollbackCursor))
                return false;
            if (journal.pending == TransactionPending::RestoreOld
                && (journal.rollbackCursor < 9 || journal.rollbackCursor >= 18
                    || journal.pendingIndex != 17 - journal.rollbackCursor))
                return false;
            break;
        case TransactionState::CommittedCleanupPending:
        case TransactionState::RecoveryRequiredCleanup:
            if (established != kExpectedOutputs.size()
                || journal.rollbackCursor != 0 || !newLocations("FINAL")
                || (journal.pending != TransactionPending::None
                    && journal.pending != TransactionPending::CleanupBackup)
                || (journal.pending == TransactionPending::CleanupBackup
                    && journal.pendingIndex != journal.cleanupCursor)
                || (journal.finalizeCursor != 0
                    && journal.cleanupCursor != kExpectedOutputs.size()))
                return false;
            for (size_t index = 0; index < journal.entries.size(); ++index)
            {
                const auto& entry = journal.entries[index];
                const std::string_view expected = !entry.oldPresent
                    || index < journal.cleanupCursor ? "ABSENT" : "BACKUP";
                if (entry.oldLocation != expected)
                    return false;
            }
            break;
        case TransactionState::Empty:
        case TransactionState::Complete:
        case TransactionState::ForeignOrMalformedCollision:
            return false;
    }
    return true;
}

struct ParsedTransactionJournal
{
    bool present = false;
    bool valid = false;
    std::string bytes;
    std::string payloadSha256;
    TransactionJournal journal;
};

ParsedTransactionJournal readTransactionJournal(
    const std::filesystem::path& path) noexcept
{
    ParsedTransactionJournal result;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory)
        return result;
    if (error || !std::filesystem::exists(status))
        return result;
    result.present = true;
    if (!std::filesystem::is_regular_file(status))
        return result;
    const uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > 65536)
        return result;
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
        return result;
    result.bytes.assign(std::istreambuf_iterator<char>(input), {});
    if (!input.eof() && input.fail())
        return result;
    result.valid = parseTransactionJournal(result.bytes, result.journal);
    if (result.valid)
    {
        const size_t checksumLine = result.bytes.rfind("payload_sha256=");
        result.payloadSha256 = transactionSha256(
            std::string_view(result.bytes).substr(0, checksumLine));
    }
    return result;
}

bool transactionEntryEqual(const TransactionEntry& first,
                           const TransactionEntry& second) noexcept
{
    return first.oldPresent == second.oldPresent
        && first.oldSize == second.oldSize
        && first.oldSha256 == second.oldSha256
        && first.oldLocation == second.oldLocation
        && first.newSize == second.newSize
        && first.newSha256 == second.newSha256
        && first.newLocation == second.newLocation;
}

bool transactionJournalBodyEqual(const TransactionJournal& first,
                                 const TransactionJournal& second) noexcept
{
    if (first.transactionId != second.transactionId
        || first.rootBindingSha256 != second.rootBindingSha256
        || first.state != second.state || first.pending != second.pending
        || first.pendingIndex != second.pendingIndex
        || first.rollbackCursor != second.rollbackCursor
        || first.cleanupCursor != second.cleanupCursor
        || first.finalizeCursor != second.finalizeCursor)
        return false;
    for (size_t index = 0; index < first.entries.size(); ++index)
        if (!transactionEntryEqual(first.entries[index], second.entries[index]))
            return false;
    return true;
}

bool transactionJournalEqual(const TransactionJournal& first,
                             const TransactionJournal& second) noexcept
{
    return transactionJournalBodyEqual(first, second)
        && first.generation == second.generation
        && first.previousPayloadSha256 == second.previousPayloadSha256
        && first.transitionId == second.transitionId
        && first.transitionIndex == second.transitionIndex;
}

bool transactionApplyTransition(const TransactionJournal& low,
                                const TransactionJournal& supplied,
                                std::string_view transitionId,
                                std::string_view lowPayloadSha256,
                                TransactionJournal& high) noexcept
{
    if (low.generation == std::numeric_limits<uint64_t>::max()
        || !transactionLowerHex(lowPayloadSha256, 64))
        return false;
    high = low;
    const auto noPending = [&low]() {
        return low.pending == TransactionPending::None
            && low.pendingIndex == kExpectedOutputs.size();
    };
    const auto allNewAt = [&low](std::string_view location) {
        return std::all_of(low.entries.begin(), low.entries.end(),
            [location](const TransactionEntry& entry) {
                return entry.newSize > 0 && entry.newLocation == location;
            });
    };
    const auto allOldBackedUp = [&low]() {
        return std::all_of(low.entries.begin(), low.entries.end(),
            [](const TransactionEntry& entry) {
                return entry.oldLocation == (entry.oldPresent ? "BACKUP" : "ABSENT");
            });
    };
    if (transitionId == "T01")
    {
        if (low.state != TransactionState::Setup)
            return false;
    }
    else if (transitionId == "T02")
    {
        if (low.state != TransactionState::Setup
            || low.entries[0].newSize != 0
            || supplied.entries[0].newSize == 0
            || supplied.entries[0].newSha256 == kZeroSha256
            || supplied.entries[0].newLocation != "STAGED")
            return false;
        high.state = TransactionState::Staging;
        high.entries[0].newSize = supplied.entries[0].newSize;
        high.entries[0].newSha256 = supplied.entries[0].newSha256;
        high.entries[0].newLocation = "STAGED";
    }
    else if (transitionId == "T03" || transitionId == "T04")
    {
        if (low.state != TransactionState::Staging)
            return false;
        const auto targetIterator = std::find_if(
            low.entries.begin(), low.entries.end(),
            [](const TransactionEntry& entry) { return entry.newSize == 0; });
        if (targetIterator == low.entries.end())
            return false;
        const size_t target = static_cast<size_t>(
            std::distance(low.entries.begin(), targetIterator));
        if ((transitionId == "T03" && (target == 0 || target == 8))
            || (transitionId == "T04" && target != 8)
            || supplied.entries[target].newSize == 0
            || supplied.entries[target].newSha256 == kZeroSha256
            || supplied.entries[target].newLocation != "STAGED")
            return false;
        high.entries[target].newSize = supplied.entries[target].newSize;
        high.entries[target].newSha256 = supplied.entries[target].newSha256;
        high.entries[target].newLocation = "STAGED";
        high.state = transitionId == "T04"
                   ? TransactionState::Staged : TransactionState::Staging;
    }
    else if (transitionId == "T05")
    {
        if (low.state != TransactionState::Staged || !noPending())
            return false;
        high.state = TransactionState::BackupInProgress;
        high.pending = TransactionPending::Backup;
        high.pendingIndex = 0;
    }
    else if (transitionId == "T06")
    {
        if (low.state != TransactionState::BackupInProgress
            || low.pending != TransactionPending::Backup
            || low.pendingIndex >= kExpectedOutputs.size())
            return false;
        const size_t target = low.pendingIndex;
        high.entries[target].oldLocation = high.entries[target].oldPresent
            ? "BACKUP" : "ABSENT";
        high.pending = TransactionPending::None;
        high.pendingIndex = kExpectedOutputs.size();
    }
    else if (transitionId == "T07")
    {
        if (low.state != TransactionState::BackupInProgress || !noPending())
            return false;
        const auto targetIterator = std::find_if(
            low.entries.begin(), low.entries.end(),
            [](const TransactionEntry& entry) {
                return entry.oldPresent && entry.oldLocation == "FINAL";
            });
        if (targetIterator == low.entries.end())
            return false;
        high.pending = TransactionPending::Backup;
        high.pendingIndex = static_cast<size_t>(
            std::distance(low.entries.begin(), targetIterator));
    }
    else if (transitionId == "T08")
    {
        if (low.state != TransactionState::BackupInProgress || !noPending()
            || !allOldBackedUp())
            return false;
        high.state = TransactionState::PublishInProgress;
        high.pending = TransactionPending::Publish;
        high.pendingIndex = 0;
    }
    else if (transitionId == "T09")
    {
        if (low.state != TransactionState::PublishInProgress
            || low.pending != TransactionPending::Publish
            || low.pendingIndex >= kExpectedOutputs.size())
            return false;
        high.entries[low.pendingIndex].newLocation = "FINAL";
        high.pending = TransactionPending::None;
        high.pendingIndex = kExpectedOutputs.size();
    }
    else if (transitionId == "T10")
    {
        if (low.state != TransactionState::PublishInProgress || !noPending())
            return false;
        const auto targetIterator = std::find_if(
            low.entries.begin(), low.entries.end(),
            [](const TransactionEntry& entry) {
                return entry.newLocation == "STAGED";
            });
        if (targetIterator == low.entries.end())
            return false;
        high.pending = TransactionPending::Publish;
        high.pendingIndex = static_cast<size_t>(
            std::distance(low.entries.begin(), targetIterator));
    }
    else if (transitionId == "T11")
    {
        if (low.state != TransactionState::PublishInProgress || !noPending()
            || !allNewAt("FINAL"))
            return false;
        high.state = TransactionState::CommitReady;
    }
    else if (transitionId == "T12")
    {
        if (low.state != TransactionState::CommitReady || !noPending()
            || !allNewAt("FINAL"))
            return false;
        high.state = TransactionState::CommittedCleanupPending;
    }
    else if (transitionId >= "T13" && transitionId <= "T18")
    {
        const std::array<TransactionState, 6> sources = {
            TransactionState::Setup, TransactionState::Staging,
            TransactionState::Staged, TransactionState::BackupInProgress,
            TransactionState::PublishInProgress, TransactionState::CommitReady,
        };
        const size_t offset = static_cast<size_t>(transitionId[2] - '3');
        if (offset >= sources.size() || low.state != sources[offset] || !noPending())
            return false;
        high.state = TransactionState::RollbackRequired;
        high.rollbackCursor = 0;
        high.cleanupCursor = 0;
        high.finalizeCursor = 0;
    }
    else if (transitionId == "T19")
    {
        if (low.state != TransactionState::RollbackRequired || !noPending()
            || low.rollbackCursor >= 9)
            return false;
        high.pending = TransactionPending::RemoveNew;
        high.pendingIndex = 8 - low.rollbackCursor;
    }
    else if (transitionId == "T20")
    {
        if (low.state != TransactionState::RollbackRequired
            || low.pending != TransactionPending::RemoveNew
            || low.rollbackCursor >= 9
            || low.pendingIndex != 8 - low.rollbackCursor)
            return false;
        high.entries[low.pendingIndex].newLocation = "NONE";
        high.pending = TransactionPending::None;
        high.pendingIndex = kExpectedOutputs.size();
        ++high.rollbackCursor;
    }
    else if (transitionId == "T21")
    {
        if (low.state != TransactionState::RollbackRequired || !noPending()
            || low.rollbackCursor < 9 || low.rollbackCursor >= 18)
            return false;
        high.pending = TransactionPending::RestoreOld;
        high.pendingIndex = 17 - low.rollbackCursor;
    }
    else if (transitionId == "T22")
    {
        if (low.state != TransactionState::RollbackRequired
            || low.pending != TransactionPending::RestoreOld
            || low.rollbackCursor < 9 || low.rollbackCursor >= 18
            || low.pendingIndex != 17 - low.rollbackCursor)
            return false;
        high.entries[low.pendingIndex].oldLocation =
            high.entries[low.pendingIndex].oldPresent ? "FINAL" : "ABSENT";
        high.pending = TransactionPending::None;
        high.pendingIndex = kExpectedOutputs.size();
        ++high.rollbackCursor;
    }
    else if (transitionId == "T23")
    {
        if (low.state != TransactionState::RollbackRequired)
            return false;
        high.state = TransactionState::RecoveryRequiredRollback;
    }
    else if (transitionId == "T24")
    {
        if (low.state != TransactionState::RecoveryRequiredRollback)
            return false;
    }
    else if (transitionId == "T25")
    {
        if (low.state != TransactionState::RecoveryRequiredRollback)
            return false;
        high.state = TransactionState::RollbackRequired;
    }
    else if (transitionId == "T26")
    {
        if (low.state != TransactionState::RollbackRequired || !noPending()
            || low.rollbackCursor != 18 || low.finalizeCursor >= 3)
            return false;
        ++high.finalizeCursor;
    }
    else if (transitionId == "T27")
    {
        if (low.state != TransactionState::CommittedCleanupPending || !noPending()
            || low.cleanupCursor >= kExpectedOutputs.size())
            return false;
        high.pending = TransactionPending::CleanupBackup;
        high.pendingIndex = low.cleanupCursor;
    }
    else if (transitionId == "T28")
    {
        if (low.state != TransactionState::CommittedCleanupPending
            || low.pending != TransactionPending::CleanupBackup
            || low.pendingIndex != low.cleanupCursor
            || low.cleanupCursor >= kExpectedOutputs.size())
            return false;
        high.entries[low.pendingIndex].oldLocation = "ABSENT";
        high.pending = TransactionPending::None;
        high.pendingIndex = kExpectedOutputs.size();
        ++high.cleanupCursor;
    }
    else if (transitionId == "T29")
    {
        if (low.state != TransactionState::CommittedCleanupPending)
            return false;
        high.state = TransactionState::RecoveryRequiredCleanup;
    }
    else if (transitionId == "T30")
    {
        if (low.state != TransactionState::RecoveryRequiredCleanup)
            return false;
    }
    else if (transitionId == "T31")
    {
        if (low.state != TransactionState::RecoveryRequiredCleanup)
            return false;
        high.state = TransactionState::CommittedCleanupPending;
    }
    else if (transitionId == "T32")
    {
        if (low.state != TransactionState::CommittedCleanupPending || !noPending()
            || low.cleanupCursor != kExpectedOutputs.size()
            || low.finalizeCursor >= 3)
            return false;
        ++high.finalizeCursor;
    }
    else
    {
        return false;
    }
    high.generation = low.generation + 1;
    high.previousPayloadSha256 = std::string(lowPayloadSha256);
    high.transitionId = std::string(transitionId);
    high.transitionIndex = low.transitionIndex + 1;
    if (!transactionJournalBodyEqual(high, supplied))
        return false;
    TransactionJournal parsed;
    return parseTransactionJournal(serializeTransactionJournal(high), parsed)
        && transactionJournalEqual(high, parsed);
}

bool transactionValidateAdjacent(const ParsedTransactionJournal& low,
                                 const ParsedTransactionJournal& high) noexcept
{
    if (!low.valid || !high.valid
        || low.journal.generation == std::numeric_limits<uint64_t>::max()
        || high.journal.generation != low.journal.generation + 1
        || high.journal.previousPayloadSha256 != low.payloadSha256
        || high.journal.transitionIndex != low.journal.transitionIndex + 1)
        return false;
    TransactionJournal expected;
    return transactionApplyTransition(
               low.journal, high.journal, high.journal.transitionId,
               low.payloadSha256, expected)
        && transactionJournalEqual(expected, high.journal);
}

bool transactionClassifyAuthority(
    const std::array<ParsedTransactionJournal, 2>& slots,
    const std::array<ParsedTransactionJournal, 2>& temporaries,
    std::string_view currentRoot,
    std::string_view currentTransaction,
    int& selected) noexcept
{
    for (const auto& slot : slots)
        if (slot.present
            && (!slot.valid || slot.bytes != serializeTransactionJournal(slot.journal)
                || slot.journal.rootBindingSha256 != currentRoot
                || slot.journal.transactionId != currentTransaction))
            return false;
    for (const auto& temporary : temporaries)
        if (temporary.present
            && (!temporary.valid
                || temporary.bytes != serializeTransactionJournal(temporary.journal)
                || temporary.journal.rootBindingSha256 != currentRoot
                || temporary.journal.transactionId != currentTransaction))
            return false;
    const size_t finalCount = static_cast<size_t>(slots[0].present)
                            + static_cast<size_t>(slots[1].present);
    const size_t temporaryCount = static_cast<size_t>(temporaries[0].present)
                                + static_cast<size_t>(temporaries[1].present);
    if (finalCount == 0 || temporaryCount > 1
        || (finalCount == 2 && temporaryCount != 0))
        return false;
    selected = slots[0].present ? 0 : 1;
    if (finalCount == 2)
    {
        if (slots[0].journal.generation == slots[1].journal.generation)
        {
            if (slots[0].bytes != slots[1].bytes)
                return false;
            selected = 0;
        }
        else
        {
            const int low = slots[0].journal.generation
                          < slots[1].journal.generation ? 0 : 1;
            const int high = low == 0 ? 1 : 0;
            if (!transactionValidateAdjacent(
                    slots[static_cast<size_t>(low)],
                    slots[static_cast<size_t>(high)]))
                return false;
            selected = high;
        }
    }
    if (slots[static_cast<size_t>(selected)].journal.generation
        == std::numeric_limits<uint64_t>::max())
        return false;
    if (temporaryCount == 1)
    {
        const size_t temporaryIndex = temporaries[0].present ? 0 : 1;
        if (slots[temporaryIndex].present
            || !transactionValidateAdjacent(
                slots[static_cast<size_t>(selected)],
                temporaries[temporaryIndex]))
            return false;
    }
    return true;
}

bool transactionIdentityMatches(const TransactionFileIdentity& identity,
                                uintmax_t size,
                                const std::string& sha256) noexcept
{
    return identity.present && identity.valid
        && identity.size == size && identity.sha256 == sha256;
}

class OutputTransaction
{
public:
    explicit OutputTransaction(std::filesystem::path outputDirectory);
    OutputTransaction(const OutputTransaction&) = delete;
    OutputTransaction& operator=(const OutputTransaction&) = delete;
    ~OutputTransaction() = default; // Recovery is explicit; destruction never mutates files.

    bool begin();
    bool openOutput(std::ofstream& output, const char* name);
    bool finishOutput(std::ofstream& output, const char* name) noexcept;
    bool commit();
    int abortAndReport() noexcept;
    int reportFailure() const noexcept;
    bool recoveredCommitted() const noexcept { return recoveredCommitted_; }

private:
    enum class PathKind { Missing, Regular, Directory, Other, Error };

    static int outputIndex(const std::string& name) noexcept;
    static PathKind pathKind(const std::filesystem::path& path) noexcept;
    static bool directoryEmpty(const std::filesystem::path& path) noexcept;

    bool readLegacyInjections();
    bool outerCensus(bool allowTransaction);
    bool internalCensus() const;
    bool initializeNewTransaction();
    bool recoverExistingTransaction();
    bool loadJournalAuthority();
    bool reconcilePending(bool committed);
    bool validateStagedCensus() const;
    bool validateOldAuthority() const;
    bool validateRecoveryPhysicalState(bool committed) const;
    bool validateCommittedFinals() const;
    bool validateTerminalFinalCensus(bool allowTransaction) const;
    bool discardUnjournaledStaged() noexcept;

    bool writeJournal(const std::string& logicalPoint,
                      std::string_view transitionId);
    bool writeJournalDirect(const std::filesystem::path& path,
                            const TransactionJournal& journal) const;
    bool removeKnownFile(const std::filesystem::path& path,
                         const std::string& faultPoint = {});
    bool removeKnownDirectory(const std::filesystem::path& path,
                              const std::string& faultPoint = {});
    bool renameKnown(const std::filesystem::path& source,
                     const std::filesystem::path& destination,
                     const std::string& faultPoint);
    bool rollbackAndClean() noexcept;
    bool committedCleanup() noexcept;
    bool cleanupTransactionTree(bool committed) noexcept;
    bool failPrecommit(const char* phase, const std::string& detail) noexcept;
    bool failRecoveryRollback(const std::string& detail) noexcept;
    bool failRecoveryCleanup(const std::string& detail) noexcept;
    bool failCollision(const std::string& detail) noexcept;
    void setFailure(const char* phase, const std::string& detail,
                    int code = 3, bool replace = false) noexcept;

#if defined(DSPARK_TIMESTRETCH_TRANSACTION_TESTING)
    bool readInjectionIndex(const char* variable, int& target);
    bool injectedFault(const std::string& point);
    void crashCut(const std::string& point) const noexcept;
    void trace(const std::string& event) const noexcept;
#else
    bool injectedFault(const std::string&) noexcept { return false; }
    void crashCut(const std::string&) const noexcept {}
    void trace(const std::string&) const noexcept {}
#endif

    std::filesystem::path outputDirectory_;
    std::filesystem::path transactionDirectory_;
    std::filesystem::path stagedDirectory_;
    std::filesystem::path backupDirectory_;
    std::array<std::filesystem::path, 2> journalPaths_;
    std::array<std::filesystem::path, 2> journalTempPaths_;
    TransactionJournal journal_;
    int activeJournalSlot_ = -1;
    std::array<bool, kExpectedOutputs.size()> opened_{};
    std::array<bool, kExpectedOutputs.size()> finished_{};
    std::array<TransactionFileIdentity, kExpectedOutputs.size()>
        stagedIdentities_{};
    std::string failurePhase_;
    std::string failureDetail_;
    int terminalCode_ = 3;
    bool journalInitialized_ = false;
    bool terminalFinalized_ = false;
    bool committed_ = false;
    bool recoveredCommitted_ = false;
    bool recovering_ = false;
#if defined(DSPARK_TIMESTRETCH_TRANSACTION_TESTING)
    int stageFailureIndex_ = -1;
    int commitFailureIndex_ = -1;
    bool faultConsumed_ = false;
#endif
};

OutputTransaction::OutputTransaction(std::filesystem::path outputDirectory)
    : outputDirectory_(std::move(outputDirectory)),
      transactionDirectory_(outputDirectory_ / kTransactionDirectory),
      stagedDirectory_(transactionDirectory_ / "staged"),
      backupDirectory_(transactionDirectory_ / "backup"),
      journalPaths_{transactionDirectory_ / "journal.a",
                    transactionDirectory_ / "journal.b"},
      journalTempPaths_{transactionDirectory_ / "journal.a.tmp",
                        transactionDirectory_ / "journal.b.tmp"}
{
}

int OutputTransaction::outputIndex(const std::string& name) noexcept
{
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
        if (name == kExpectedOutputs[index])
            return static_cast<int>(index);
    return -1;
}

OutputTransaction::PathKind OutputTransaction::pathKind(
    const std::filesystem::path& path) noexcept
{
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory)
        return PathKind::Missing;
    if (error)
        return PathKind::Error;
    if (!std::filesystem::exists(status))
        return PathKind::Missing;
    if (std::filesystem::is_regular_file(status))
        return PathKind::Regular;
    if (std::filesystem::is_directory(status))
        return PathKind::Directory;
    return PathKind::Other;
}

bool OutputTransaction::directoryEmpty(
    const std::filesystem::path& path) noexcept
{
    std::error_code error;
    const bool empty = std::filesystem::is_empty(path, error);
    return !error && empty;
}

void OutputTransaction::setFailure(const char* phase,
                                   const std::string& detail,
                                   int code, bool replace) noexcept
{
    if (failurePhase_.empty() || replace)
    {
        failurePhase_ = phase;
        failureDetail_ = detail;
        terminalCode_ = code;
    }
}

bool OutputTransaction::failRecoveryRollback(
    const std::string& detail) noexcept
{
    setFailure("RECOVERY_REQUIRED_ROLLBACK", detail, 4, true);
    terminalFinalized_ = true;
    trace("terminal:RECOVERY_REQUIRED_ROLLBACK:" + detail);
    return false;
}

bool OutputTransaction::failRecoveryCleanup(
    const std::string& detail) noexcept
{
    setFailure("RECOVERY_REQUIRED_CLEANUP", detail, 5, true);
    terminalFinalized_ = true;
    trace("terminal:RECOVERY_REQUIRED_CLEANUP:" + detail);
    return false;
}

bool OutputTransaction::failCollision(const std::string& detail) noexcept
{
    setFailure("RECOVERY_COLLISION", detail, 6, true);
    terminalFinalized_ = true;
    trace("terminal:RECOVERY_COLLISION:" + detail);
    return false;
}

int OutputTransaction::reportFailure() const noexcept
{
    const char* phase = failurePhase_.empty()
                      ? "UNKNOWN" : failurePhase_.c_str();
    const char* detail = failureDetail_.empty()
                       ? "unspecified" : failureDetail_.c_str();
    std::fprintf(stderr, "ERROR TIMESTRETCH_TRANSACTION_%s %s\n",
                 phase, detail);
    return terminalCode_;
}

#if defined(DSPARK_TIMESTRETCH_TRANSACTION_TESTING)
bool OutputTransaction::readInjectionIndex(const char* variable, int& target)
{
    const char* value = std::getenv(variable);
    if (value == nullptr)
        return true;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0
        || parsed >= static_cast<long>(kExpectedOutputs.size()))
    {
        setFailure("INJECTION_CONFIGURATION", variable);
        return false;
    }
    target = static_cast<int>(parsed);
    return true;
}

bool OutputTransaction::injectedFault(const std::string& point)
{
    const char* selected = std::getenv("DSPARK_TIMESTRETCH_DURABLE_FAULT");
    if (selected == nullptr || point != selected)
        return false;
    const char* mode = std::getenv("DSPARK_TIMESTRETCH_DURABLE_MODE");
    if (mode == nullptr
        || (std::strcmp(mode, "one-shot") != 0
            && std::strcmp(mode, "persistent") != 0))
    {
        setFailure("INJECTION_CONFIGURATION",
                   "DSPARK_TIMESTRETCH_DURABLE_MODE");
        return true;
    }
    if (std::strcmp(mode, "one-shot") == 0 && faultConsumed_)
        return false;
    faultConsumed_ = true;
    trace("fault:" + point + ':' + mode);
    return true;
}

void OutputTransaction::crashCut(const std::string& point) const noexcept
{
    const char* selected = std::getenv("DSPARK_TIMESTRETCH_CRASH_CUT");
    if (selected != nullptr && point == selected)
    {
        trace("crash:" + point);
        std::_Exit(86);
    }
}

void OutputTransaction::trace(const std::string& event) const noexcept
{
    const char* path = std::getenv("DSPARK_TIMESTRETCH_TRACE_FILE");
    if (path == nullptr)
        return;
    std::ofstream output(path, std::ios::out | std::ios::app);
    if (output.is_open())
        output << event << '\n';
}
#endif

bool OutputTransaction::readLegacyInjections()
{
#if defined(DSPARK_TIMESTRETCH_TRANSACTION_TESTING)
    if (!readInjectionIndex("DSPARK_TIMESTRETCH_FAIL_STAGE_INDEX",
                            stageFailureIndex_)
        || !readInjectionIndex("DSPARK_TIMESTRETCH_FAIL_COMMIT_INDEX",
                               commitFailureIndex_))
        return false;
    const char* durable = std::getenv("DSPARK_TIMESTRETCH_DURABLE_FAULT");
    const char* mode = std::getenv("DSPARK_TIMESTRETCH_DURABLE_MODE");
    if ((durable == nullptr) != (mode == nullptr))
    {
        setFailure("INJECTION_CONFIGURATION",
                   "DSPARK_TIMESTRETCH_DURABLE_FAULT");
        return false;
    }
#endif
    return true;
}

bool OutputTransaction::outerCensus(bool allowTransaction)
{
    std::array<bool, kExpectedOutputs.size()> seen{};
    std::error_code error;
    std::filesystem::directory_iterator iterator(outputDirectory_, error);
    const std::filesystem::directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error))
    {
        const std::string name = iterator->path().filename().string();
        if (allowTransaction && name == kTransactionDirectory)
        {
            if (pathKind(iterator->path()) != PathKind::Directory)
                return false;
            continue;
        }
        const int index = outputIndex(name);
        if (index < 0 || seen[static_cast<size_t>(index)]
            || pathKind(iterator->path()) != PathKind::Regular)
            return false;
        seen[static_cast<size_t>(index)] = true;
    }
    return !error;
}

bool OutputTransaction::internalCensus() const
{
    if (pathKind(transactionDirectory_) != PathKind::Directory)
        return false;
    std::error_code error;
    std::filesystem::directory_iterator iterator(transactionDirectory_, error);
    const std::filesystem::directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error))
    {
        const std::string name = iterator->path().filename().string();
        if (name == "staged" || name == "backup")
        {
            if (pathKind(iterator->path()) != PathKind::Directory)
                return false;
            std::array<bool, kExpectedOutputs.size()> seen{};
            std::error_code childError;
            std::filesystem::directory_iterator child(iterator->path(), childError);
            for (; !childError && child != end; child.increment(childError))
            {
                const int index = outputIndex(child->path().filename().string());
                if (index < 0 || seen[static_cast<size_t>(index)]
                    || pathKind(child->path()) != PathKind::Regular)
                    return false;
                seen[static_cast<size_t>(index)] = true;
            }
            if (childError)
                return false;
            continue;
        }
        if (name != "journal.a" && name != "journal.b"
            && name != "journal.a.tmp" && name != "journal.b.tmp")
            return false;
        if (pathKind(iterator->path()) != PathKind::Regular)
            return false;
    }
    return !error;
}

bool OutputTransaction::writeJournalDirect(
    const std::filesystem::path& path,
    const TransactionJournal& journal) const
{
    if (pathKind(path) != PathKind::Missing)
        return false;
    const std::string bytes = serializeTransactionJournal(journal);
    if (bytes.size() > 65536)
        return false;
    std::ofstream output(path, std::ios::out | std::ios::binary);
    if (!output.is_open())
        return false;
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.flush();
    const bool flushed = static_cast<bool>(output);
    output.close();
    if (!flushed || output.fail())
        return false;
    const ParsedTransactionJournal reread = readTransactionJournal(path);
    return reread.present && reread.valid && reread.bytes == bytes
        && reread.journal.generation == journal.generation
        && reread.journal.transactionId == journal.transactionId
        && reread.journal.rootBindingSha256 == journal.rootBindingSha256;
}

bool OutputTransaction::writeJournal(const std::string& logicalPoint,
                                     std::string_view transitionId)
{
    const int target = activeJournalSlot_ == 0 ? 1 : 0;
    if (activeJournalSlot_ < 0 && journal_.generation != 0)
        return false;
    const std::string currentRoot = transactionSha256(
        std::filesystem::absolute(outputDirectory_).lexically_normal().generic_string());
    const std::string currentTransaction = transactionSha256(
        "DSPark-TimeStretch-transaction-id|" + currentRoot).substr(0, 32);
    if (journal_.rootBindingSha256 != currentRoot
        || journal_.transactionId != currentTransaction)
        return false;

    ParsedTransactionJournal active;
    TransactionJournal next;
    bool temporaryOwnedByAttempt = false;
    const auto rejectWrite = [this, &active, &next, &temporaryOwnedByAttempt,
                              target]() {
        if (temporaryOwnedByAttempt
            && pathKind(journalPaths_[static_cast<size_t>(target)])
                == PathKind::Missing)
        {
            const ParsedTransactionJournal temporary = readTransactionJournal(
                journalTempPaths_[static_cast<size_t>(target)]);
            if (temporary.present && temporary.valid
                && transactionJournalEqual(temporary.journal, next)
                && temporary.bytes == serializeTransactionJournal(next))
            {
                std::error_code error;
                std::filesystem::remove(
                    journalTempPaths_[static_cast<size_t>(target)], error);
            }
        }
        if (active.present && active.valid)
            journal_ = active.journal;
        return false;
    };
    if (activeJournalSlot_ >= 0)
    {
        active =
            readTransactionJournal(journalPaths_[static_cast<size_t>(activeJournalSlot_)]);
        if (!active.present || !active.valid
            || active.bytes != serializeTransactionJournal(active.journal)
            || active.journal.transactionId != currentTransaction
            || active.journal.rootBindingSha256 != currentRoot
            || !transactionApplyTransition(
                active.journal, journal_, transitionId,
                active.payloadSha256, next))
            return rejectWrite();
    }
    else
    {
        if (journal_.state != TransactionState::Setup
            || transitionId != "T01")
            return rejectWrite();
        next = journal_;
        next.generation = 1;
        next.previousPayloadSha256 = std::string(kZeroSha256);
        next.transitionId = "BEGIN";
        next.transitionIndex = 0;
        TransactionJournal parsed;
        if (!parseTransactionJournal(serializeTransactionJournal(next), parsed)
            || !transactionJournalEqual(next, parsed))
            return rejectWrite();
    }
    if (next.generation == std::numeric_limits<uint64_t>::max())
        return rejectWrite();
    if (pathKind(journalTempPaths_[static_cast<size_t>(target)]) != PathKind::Missing)
    {
        const ParsedTransactionJournal temporary =
            readTransactionJournal(journalTempPaths_[static_cast<size_t>(target)]);
        if (activeJournalSlot_ < 0 || !temporary.present || !temporary.valid
            || pathKind(journalPaths_[static_cast<size_t>(target)])
                != PathKind::Missing
            || temporary.journal.transactionId != currentTransaction
            || temporary.journal.rootBindingSha256 != currentRoot
            || !transactionValidateAdjacent(active, temporary))
            return rejectWrite();
        std::error_code error;
        if (!std::filesystem::remove(
                journalTempPaths_[static_cast<size_t>(target)], error) || error)
            return rejectWrite();
    }
    if (pathKind(journalPaths_[static_cast<size_t>(target)]) != PathKind::Missing)
    {
        if (activeJournalSlot_ < 0)
            return rejectWrite();
        const ParsedTransactionJournal stale =
            readTransactionJournal(journalPaths_[static_cast<size_t>(target)]);
        if (!stale.present || !stale.valid
            || stale.journal.transactionId != currentTransaction
            || stale.journal.rootBindingSha256 != currentRoot
            || (stale.bytes != active.bytes
                && !transactionValidateAdjacent(stale, active)))
            return rejectWrite();
        std::error_code error;
        if (!std::filesystem::remove(journalPaths_[static_cast<size_t>(target)], error)
            || error)
            return rejectWrite();
    }

    const std::string writePoint = "journal-write:" + logicalPoint;
    if (injectedFault(writePoint))
        return rejectWrite();
    temporaryOwnedByAttempt = true;
    if (!writeJournalDirect(journalTempPaths_[static_cast<size_t>(target)], next))
        return rejectWrite();
    if (injectedFault("journal-replace:" + logicalPoint))
        return rejectWrite();
    std::error_code error;
    std::filesystem::rename(journalTempPaths_[static_cast<size_t>(target)],
                            journalPaths_[static_cast<size_t>(target)], error);
    if (error)
        return rejectWrite();
    const ParsedTransactionJournal promoted =
        readTransactionJournal(journalPaths_[static_cast<size_t>(target)]);
    if (!promoted.present || !promoted.valid
        || !transactionJournalEqual(promoted.journal, next)
        || promoted.bytes != serializeTransactionJournal(next))
        return rejectWrite();
    journal_ = std::move(next);
    activeJournalSlot_ = target;
    journalInitialized_ = true;
    trace("journal:" + logicalPoint + ":generation="
          + std::to_string(journal_.generation));
    return true;
}

bool OutputTransaction::removeKnownFile(const std::filesystem::path& path,
                                        const std::string& faultPoint)
{
    const PathKind kind = pathKind(path);
    if (kind == PathKind::Missing)
        return true;
    if (kind != PathKind::Regular || (!faultPoint.empty() && injectedFault(faultPoint)))
        return false;
    std::error_code error;
    const bool removed = std::filesystem::remove(path, error);
    if (!removed || error)
        return false;
    trace("remove-file:" + path.generic_string());
    return true;
}

bool OutputTransaction::removeKnownDirectory(
    const std::filesystem::path& path, const std::string& faultPoint)
{
    const PathKind kind = pathKind(path);
    if (kind == PathKind::Missing)
        return true;
    if (kind != PathKind::Directory || !directoryEmpty(path)
        || (!faultPoint.empty() && injectedFault(faultPoint)))
        return false;
    std::error_code error;
    const bool removed = std::filesystem::remove(path, error);
    if (!removed || error)
        return false;
    trace("remove-directory:" + path.generic_string());
    return true;
}

bool OutputTransaction::renameKnown(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    const std::string& faultPoint)
{
    if (pathKind(source) != PathKind::Regular
        || pathKind(destination) != PathKind::Missing
        || injectedFault(faultPoint))
        return false;
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error)
        return false;
    trace("rename:" + source.generic_string() + "->" + destination.generic_string());
    return true;
}

bool OutputTransaction::initializeNewTransaction()
{
    journal_ = {};
    const std::string rootSpelling = std::filesystem::absolute(outputDirectory_)
        .lexically_normal().generic_string();
    journal_.rootBindingSha256 = transactionSha256(rootSpelling);
    journal_.transactionId = transactionSha256(
        "DSPark-TimeStretch-transaction-id|" + journal_.rootBindingSha256).substr(0, 32);
    journal_.state = TransactionState::Setup;
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        const auto identity = transactionFileIdentity(
            outputDirectory_ / kExpectedOutputs[index]);
        if (identity.present && !identity.valid)
        {
            setFailure("PRECHECK", kExpectedOutputs[index]);
            return false;
        }
        auto& entry = journal_.entries[index];
        entry.oldPresent = identity.present;
        if (identity.present)
        {
            entry.oldSize = identity.size;
            entry.oldSha256 = identity.sha256;
            entry.oldLocation = "FINAL";
        }
    }

    if (injectedFault("setup:00"))
    {
        setFailure("SETUP", "transaction-directory");
        return false;
    }
    std::error_code error;
    if (!std::filesystem::create_directory(transactionDirectory_, error) || error)
    {
        setFailure("COLLISION", kTransactionDirectory);
        return false;
    }
    if (!writeJournal("setup:00", "T01"))
        return failPrecommit("SETUP", "journal.a");
    crashCut("setup:00");

    if (injectedFault("setup:01")
        || !std::filesystem::create_directory(stagedDirectory_, error) || error)
        return failPrecommit("SETUP", "staged");
    if (!writeJournal("setup:01", "T01"))
        return failPrecommit("SETUP", "journal-staged");
    crashCut("setup:01");

    error.clear();
    if (injectedFault("setup:02")
        || !std::filesystem::create_directory(backupDirectory_, error) || error)
        return failPrecommit("SETUP", "backup");
    if (!writeJournal("setup:02", "T01"))
        return failPrecommit("SETUP", "journal-backup");
    crashCut("setup:02");
    return true;
}

bool OutputTransaction::begin()
{
    if (!readLegacyInjections())
        return false;
    const PathKind transactionKind = pathKind(transactionDirectory_);
    if (transactionKind != PathKind::Missing)
    {
        if (transactionKind != PathKind::Directory || !outerCensus(true))
            return failCollision(kTransactionDirectory);
        return recoverExistingTransaction();
    }
    if (!outerCensus(false))
    {
        std::error_code error;
        std::filesystem::directory_iterator iterator(outputDirectory_, error);
        const std::filesystem::directory_iterator end;
        for (; !error && iterator != end; iterator.increment(error))
        {
            const std::string name = iterator->path().filename().string();
            const int index = outputIndex(name);
            if (index < 0)
            {
                setFailure("ARTIFACT_CENSUS", name);
                return false;
            }
            if (pathKind(iterator->path()) != PathKind::Regular)
            {
                setFailure("PRECHECK", name);
                return false;
            }
        }
        setFailure("ARTIFACT_CENSUS", "directory-iteration");
        return false;
    }
    return initializeNewTransaction();
}

bool OutputTransaction::openOutput(std::ofstream& output, const char* name)
{
    const int signedIndex = outputIndex(name);
    if (signedIndex < 0)
    {
        setFailure("STAGE", name);
        return false;
    }
    const size_t index = static_cast<size_t>(signedIndex);
    if (opened_[index] || !journalInitialized_ || committed_
        || pathKind(stagedDirectory_) != PathKind::Directory)
    {
        setFailure("STAGE", name);
        return false;
    }
#if defined(DSPARK_TIMESTRETCH_TRANSACTION_TESTING)
    if (signedIndex == stageFailureIndex_)
    {
        setFailure("STAGE", name);
        return false;
    }
#endif
    if (pathKind(stagedDirectory_ / name) != PathKind::Missing)
    {
        setFailure("STAGE", name);
        return false;
    }
    output.open(stagedDirectory_ / name, std::ios::out | std::ios::binary);
    if (!output.is_open())
    {
        setFailure("STAGE", name);
        return false;
    }
    opened_[index] = true;
    return true;
}

bool OutputTransaction::finishOutput(std::ofstream& output,
                                     const char* name) noexcept
{
    const int signedIndex = outputIndex(name);
    output.flush();
    const bool flushed = static_cast<bool>(output);
    output.close();
    const bool closed = !output.fail();
    if (signedIndex < 0 || !flushed || !closed)
    {
        setFailure("STAGE", name);
        return false;
    }
    const size_t index = static_cast<size_t>(signedIndex);
    const auto identity = transactionFileIdentity(stagedDirectory_ / name);
    if (!opened_[index] || finished_[index] || !identity.valid)
    {
        setFailure("STAGE", name);
        return false;
    }
    stagedIdentities_[index] = identity;
    finished_[index] = true;
    size_t established = static_cast<size_t>(std::count_if(
        journal_.entries.begin(), journal_.entries.end(),
        [](const TransactionEntry& candidate) {
            return candidate.newSize > 0;
        }));
    while (established < kExpectedOutputs.size() && finished_[established])
    {
        const auto& establishedIdentity = stagedIdentities_[established];
        const auto establishedPath =
            stagedDirectory_ / kExpectedOutputs[established];
        if (!transactionIdentityMatches(
                transactionFileIdentity(establishedPath),
                establishedIdentity.size, establishedIdentity.sha256))
        {
            setFailure("STAGE", kExpectedOutputs[established]);
            return false;
        }
        auto& entry = journal_.entries[established];
        entry.newSize = establishedIdentity.size;
        entry.newSha256 = establishedIdentity.sha256;
        entry.newLocation = "STAGED";
        const bool last = established + 1 == kExpectedOutputs.size();
        journal_.state = last ? TransactionState::Staged
                              : TransactionState::Staging;
        const std::string_view transitionId = established == 0 ? "T02"
                                            : (last ? "T04" : "T03");
        if (!writeJournal(
                "stage:" + twoDigitIndex(established), transitionId))
        {
            setFailure("STAGE", kExpectedOutputs[established]);
            return false;
        }
        crashCut("stage:" + twoDigitIndex(established));
        if (last)
            crashCut("staged-boundary");
        ++established;
    }
    return true;
}

bool OutputTransaction::discardUnjournaledStaged() noexcept
{
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        if (!finished_[index] || journal_.entries[index].newSize > 0)
            continue;
        const auto& identity = stagedIdentities_[index];
        const auto path = stagedDirectory_ / kExpectedOutputs[index];
        if (!transactionIdentityMatches(
                transactionFileIdentity(path), identity.size, identity.sha256)
            || !removeKnownFile(path))
            return false;
    }
    return true;
}

bool OutputTransaction::validateStagedCensus() const
{
    if (pathKind(stagedDirectory_) != PathKind::Directory)
        return false;
    std::array<bool, kExpectedOutputs.size()> seen{};
    std::error_code error;
    std::filesystem::directory_iterator iterator(stagedDirectory_, error);
    const std::filesystem::directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error))
    {
        const int signedIndex = outputIndex(iterator->path().filename().string());
        if (signedIndex < 0)
            return false;
        const size_t index = static_cast<size_t>(signedIndex);
        if (seen[index]
            || !transactionIdentityMatches(
                transactionFileIdentity(iterator->path()),
                journal_.entries[index].newSize,
                journal_.entries[index].newSha256))
            return false;
        seen[index] = true;
    }
    return !error && std::all_of(
        seen.begin(), seen.end(), [](bool value) { return value; });
}

bool OutputTransaction::validateOldAuthority() const
{
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        const auto& entry = journal_.entries[index];
        const auto finalIdentity = transactionFileIdentity(
            outputDirectory_ / kExpectedOutputs[index]);
        const auto backupIdentity = transactionFileIdentity(
            backupDirectory_ / kExpectedOutputs[index]);
        if (entry.oldPresent)
        {
            const bool oldFinal = transactionIdentityMatches(
                finalIdentity, entry.oldSize, entry.oldSha256);
            const bool oldBackup = transactionIdentityMatches(
                backupIdentity, entry.oldSize, entry.oldSha256);
            const bool newFinal = entry.newSize > 0 && transactionIdentityMatches(
                finalIdentity, entry.newSize, entry.newSha256);
            if ((oldFinal && oldBackup) || (!oldFinal && !oldBackup)
                || (finalIdentity.present && !oldFinal && !newFinal)
                || (backupIdentity.present && !oldBackup))
                return false;
        }
        else if (backupIdentity.present)
        {
            return false;
        }
    }
    return true;
}

bool OutputTransaction::validateRecoveryPhysicalState(bool committed) const
{
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        const auto& entry = journal_.entries[index];
        const auto finalIdentity = transactionFileIdentity(
            outputDirectory_ / kExpectedOutputs[index]);
        const auto stagedIdentity = transactionFileIdentity(
            stagedDirectory_ / kExpectedOutputs[index]);
        const auto backupIdentity = transactionFileIdentity(
            backupDirectory_ / kExpectedOutputs[index]);
        const bool oldFinal = entry.oldPresent && transactionIdentityMatches(
            finalIdentity, entry.oldSize, entry.oldSha256);
        const bool oldBackup = entry.oldPresent && transactionIdentityMatches(
            backupIdentity, entry.oldSize, entry.oldSha256);
        const bool newFinal = entry.newSize > 0 && transactionIdentityMatches(
            finalIdentity, entry.newSize, entry.newSha256);
        const bool newStaged = entry.newSize > 0 && transactionIdentityMatches(
            stagedIdentity, entry.newSize, entry.newSha256);
        if ((finalIdentity.present && !oldFinal && !newFinal)
            || (stagedIdentity.present && !newStaged)
            || (backupIdentity.present && !oldBackup)
            || (newFinal && newStaged))
            return false;
        if (committed)
        {
            if (!newFinal)
                return false;
        }
        else if ((entry.oldPresent && oldFinal == oldBackup)
                 || (!entry.oldPresent && backupIdentity.present))
        {
            return false;
        }
    }
    return true;
}

bool OutputTransaction::validateCommittedFinals() const
{
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        const auto& entry = journal_.entries[index];
        if (entry.newSize == 0
            || !transactionIdentityMatches(
                transactionFileIdentity(outputDirectory_ / kExpectedOutputs[index]),
                entry.newSize, entry.newSha256))
            return false;
    }
    return true;
}

bool OutputTransaction::validateTerminalFinalCensus(
    bool allowTransaction) const
{
    std::array<bool, kExpectedOutputs.size()> seen{};
    std::error_code error;
    std::filesystem::directory_iterator iterator(outputDirectory_, error);
    const std::filesystem::directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error))
    {
        const std::string name = iterator->path().filename().string();
        if (allowTransaction && name == kTransactionDirectory)
            continue;
        const int signedIndex = outputIndex(name);
        if (signedIndex < 0)
            return false;
        const size_t index = static_cast<size_t>(signedIndex);
        if (seen[index] || pathKind(iterator->path()) != PathKind::Regular)
            return false;
        seen[index] = true;
    }
    return !error && std::all_of(
        seen.begin(), seen.end(), [](bool value) { return value; });
}

bool OutputTransaction::loadJournalAuthority()
{
    if (!internalCensus())
        return false;
    std::array<ParsedTransactionJournal, 2> slots = {
        readTransactionJournal(journalPaths_[0]),
        readTransactionJournal(journalPaths_[1]),
    };
    std::array<ParsedTransactionJournal, 2> temporaries = {
        readTransactionJournal(journalTempPaths_[0]),
        readTransactionJournal(journalTempPaths_[1]),
    };
    const std::string currentRoot = transactionSha256(
        std::filesystem::absolute(outputDirectory_).lexically_normal().generic_string());
    const std::string currentTransaction = transactionSha256(
        "DSPark-TimeStretch-transaction-id|" + currentRoot).substr(0, 32);
    const size_t temporaryCount = static_cast<size_t>(temporaries[0].present)
                                + static_cast<size_t>(temporaries[1].present);
    int selected = -1;
    if (!transactionClassifyAuthority(
            slots, temporaries, currentRoot, currentTransaction, selected))
        return false;
    journal_ = slots[static_cast<size_t>(selected)].journal;
    activeJournalSlot_ = selected;
    if (temporaryCount == 1)
    {
        const size_t temporaryIndex = temporaries[0].present ? 0 : 1;
        std::error_code error;
        if (!std::filesystem::remove(journalTempPaths_[temporaryIndex], error)
            || error)
            return false;
    }
    journalInitialized_ = true;
    return true;
}

bool OutputTransaction::reconcilePending(bool committed)
{
    if (journal_.pending == TransactionPending::None)
        return true;
    const size_t index = journal_.pendingIndex;
    if (index >= kExpectedOutputs.size())
        return false;
    auto& entry = journal_.entries[index];
    const auto finalIdentity = transactionFileIdentity(
        outputDirectory_ / kExpectedOutputs[index]);
    const auto stagedIdentity = transactionFileIdentity(
        stagedDirectory_ / kExpectedOutputs[index]);
    const auto backupIdentity = transactionFileIdentity(
        backupDirectory_ / kExpectedOutputs[index]);
    const bool oldFinal = entry.oldPresent && transactionIdentityMatches(
        finalIdentity, entry.oldSize, entry.oldSha256);
    const bool oldBackup = entry.oldPresent && transactionIdentityMatches(
        backupIdentity, entry.oldSize, entry.oldSha256);
    const bool newFinal = entry.newSize > 0 && transactionIdentityMatches(
        finalIdentity, entry.newSize, entry.newSha256);
    const bool newStaged = entry.newSize > 0 && transactionIdentityMatches(
        stagedIdentity, entry.newSize, entry.newSha256);
    switch (journal_.pending)
    {
        case TransactionPending::Backup:
            if (entry.oldPresent)
            {
                if (oldFinal == oldBackup)
                    return false;
                if (oldFinal
                    && !renameKnown(outputDirectory_ / kExpectedOutputs[index],
                                    backupDirectory_ / kExpectedOutputs[index], {}))
                    return false;
                entry.oldLocation = "BACKUP";
            }
            else
            {
                if (finalIdentity.present || backupIdentity.present)
                    return false;
                entry.oldLocation = "ABSENT";
            }
            journal_.pending = TransactionPending::None;
            journal_.pendingIndex = kExpectedOutputs.size();
            return writeJournal("coherence:T06:reconcile", "T06");
        case TransactionPending::Publish:
            if (newStaged == newFinal)
                return false;
            if (newStaged
                && !renameKnown(stagedDirectory_ / kExpectedOutputs[index],
                                outputDirectory_ / kExpectedOutputs[index], {}))
                return false;
            entry.newLocation = "FINAL";
            journal_.pending = TransactionPending::None;
            journal_.pendingIndex = kExpectedOutputs.size();
            return writeJournal("coherence:T09:reconcile", "T09");
        case TransactionPending::RemoveNew:
            if (newFinal && entry.oldPresent && !oldBackup)
                return false;
            if (finalIdentity.present && !newFinal && !oldFinal)
                return false;
            if (newFinal
                && !removeKnownFile(
                    outputDirectory_ / kExpectedOutputs[index],
                    "new-final-remove:" + twoDigitIndex(index)))
                return false;
            if (newStaged
                && !removeKnownFile(stagedDirectory_ / kExpectedOutputs[index]))
                return false;
            entry.newLocation = "NONE";
            journal_.pending = TransactionPending::None;
            journal_.pendingIndex = kExpectedOutputs.size();
            ++journal_.rollbackCursor;
            return writeJournal(
                "recovery-reconcile:"
                    + twoDigitIndex(journal_.rollbackCursor - 1),
                "T20");
        case TransactionPending::RestoreOld:
            if (!entry.oldPresent || oldBackup == oldFinal)
            {
                if (!entry.oldPresent && !finalIdentity.present
                    && !backupIdentity.present)
                {
                    entry.oldLocation = "ABSENT";
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if (oldBackup
                    && !renameKnown(backupDirectory_ / kExpectedOutputs[index],
                                    outputDirectory_ / kExpectedOutputs[index],
                                    "restore-rename:" + twoDigitIndex(index)))
                    return false;
                entry.oldLocation = "FINAL";
            }
            journal_.pending = TransactionPending::None;
            journal_.pendingIndex = kExpectedOutputs.size();
            ++journal_.rollbackCursor;
            return writeJournal("coherence:T22:reconcile", "T22");
        case TransactionPending::CleanupBackup:
            if (!committed || !validateCommittedFinals()
                || (backupIdentity.present && !oldBackup))
                return false;
            if (oldBackup
                && !removeKnownFile(
                    backupDirectory_ / kExpectedOutputs[index],
                    "backup-cleanup:" + twoDigitIndex(index)))
                return false;
            entry.oldLocation = "ABSENT";
            journal_.pending = TransactionPending::None;
            journal_.pendingIndex = kExpectedOutputs.size();
            ++journal_.cleanupCursor;
            return writeJournal("coherence:T28:reconcile", "T28");
        case TransactionPending::None:
            break;
    }
    return true;
}

bool OutputTransaction::recoverExistingTransaction()
{
    recovering_ = true;
    if (!loadJournalAuthority())
        return failCollision(kTransactionDirectory);
    const bool committed =
        journal_.state == TransactionState::CommittedCleanupPending
        || journal_.state == TransactionState::RecoveryRequiredCleanup;
    if (!validateRecoveryPhysicalState(committed))
        return failCollision(kTransactionDirectory);
    if (committed)
    {
        committed_ = true;
        if (!committedCleanup())
            return false;
        recoveredCommitted_ = true;
        return true;
    }
    setFailure("RECOVERED_ROLLBACK", kTransactionDirectory);
    return rollbackAndClean();
}

bool OutputTransaction::failPrecommit(const char* phase,
                                      const std::string& detail) noexcept
{
    setFailure(phase, detail);
    if (pathKind(transactionDirectory_) == PathKind::Missing)
    {
        terminalFinalized_ = true;
        return false;
    }
    return rollbackAndClean();
}

bool OutputTransaction::rollbackAndClean() noexcept
{
    if (committed_)
        return failRecoveryCleanup("rollback-forbidden-after-commit");
    trace("rollback:begin");

    if (!recovering_ && !discardUnjournaledStaged())
        return failRecoveryRollback("unrecorded-staged");

    if (!journalInitialized_)
    {
        if (!removeKnownDirectory(stagedDirectory_)
            || !removeKnownDirectory(backupDirectory_)
            || !removeKnownDirectory(transactionDirectory_))
            return failRecoveryRollback("uninitialized-transaction");
        terminalFinalized_ = true;
        return false;
    }
    if (journal_.state == TransactionState::RecoveryRequiredRollback)
    {
        journal_.state = TransactionState::RollbackRequired;
        if (!writeJournal("coherence:T25", "T25"))
            return failRecoveryRollback("retry-rollback");
    }
    if (!reconcilePending(false))
        return failRecoveryRollback("pending-operation");
    if (journal_.state != TransactionState::RollbackRequired)
    {
        std::string_view transitionId;
        switch (journal_.state)
        {
            case TransactionState::Setup: transitionId = "T13"; break;
            case TransactionState::Staging: transitionId = "T14"; break;
            case TransactionState::Staged: transitionId = "T15"; break;
            case TransactionState::BackupInProgress: transitionId = "T16"; break;
            case TransactionState::PublishInProgress: transitionId = "T17"; break;
            case TransactionState::CommitReady: transitionId = "T18"; break;
            default: return failRecoveryRollback("rollback-source-state");
        }
        journal_.state = TransactionState::RollbackRequired;
        journal_.pending = TransactionPending::None;
        journal_.pendingIndex = kExpectedOutputs.size();
        journal_.rollbackCursor = 0;
        journal_.cleanupCursor = 0;
        journal_.finalizeCursor = 0;
        if (!writeJournal("coherence:" + std::string(transitionId), transitionId))
            return failRecoveryRollback("enter-rollback");
    }

    while (journal_.rollbackCursor < 9)
    {
        const size_t cursor = journal_.rollbackCursor;
        const size_t index = 8 - cursor;
        journal_.pending = TransactionPending::RemoveNew;
        journal_.pendingIndex = index;
        if (!writeJournal("rollback-remove:" + twoDigitIndex(index), "T19"))
            return failRecoveryRollback(kExpectedOutputs[index]);
        crashCut("rollback-remove-before:" + twoDigitIndex(index));
        auto& entry = journal_.entries[index];
        const auto finalPath = outputDirectory_ / kExpectedOutputs[index];
        const auto stagedPath = stagedDirectory_ / kExpectedOutputs[index];
        const auto backupPath = backupDirectory_ / kExpectedOutputs[index];
        const auto finalIdentity = transactionFileIdentity(finalPath);
        const auto stagedIdentity = transactionFileIdentity(stagedPath);
        const bool newFinal = entry.newSize > 0 && transactionIdentityMatches(
            finalIdentity, entry.newSize, entry.newSha256);
        const bool newStaged = entry.newSize > 0 && transactionIdentityMatches(
            stagedIdentity, entry.newSize, entry.newSha256);
        const bool oldFinal = entry.oldPresent && transactionIdentityMatches(
            finalIdentity, entry.oldSize, entry.oldSha256);
        if ((finalIdentity.present && !newFinal && !oldFinal)
            || (stagedIdentity.present && !newStaged) || (newFinal && newStaged))
            return failRecoveryRollback(kExpectedOutputs[index]);
        if (newFinal && entry.oldPresent
            && !transactionIdentityMatches(transactionFileIdentity(backupPath),
                                           entry.oldSize, entry.oldSha256))
            return failRecoveryRollback(kExpectedOutputs[index]);
        if (newFinal
            && !removeKnownFile(finalPath,
                                "new-final-remove:" + twoDigitIndex(index)))
            return failRecoveryRollback(kExpectedOutputs[index]);
        if (newStaged && !removeKnownFile(stagedPath))
            return failRecoveryRollback(kExpectedOutputs[index]);
        entry.newLocation = "NONE";
        crashCut("rollback-remove-after:" + twoDigitIndex(index));
        journal_.pending = TransactionPending::None;
        journal_.pendingIndex = kExpectedOutputs.size();
        ++journal_.rollbackCursor;
        if (!writeJournal("recovery-reconcile:" + twoDigitIndex(cursor), "T20"))
            return failRecoveryRollback(kExpectedOutputs[index]);
    }

    while (journal_.rollbackCursor < 18)
    {
        const size_t index = 17 - journal_.rollbackCursor;
        journal_.pending = TransactionPending::RestoreOld;
        journal_.pendingIndex = index;
        if (!writeJournal("rollback-restore:" + twoDigitIndex(index), "T21"))
            return failRecoveryRollback(kExpectedOutputs[index]);
        crashCut("rollback-restore-before:" + twoDigitIndex(index));
        auto& entry = journal_.entries[index];
        const auto finalPath = outputDirectory_ / kExpectedOutputs[index];
        const auto backupPath = backupDirectory_ / kExpectedOutputs[index];
        if (entry.oldPresent)
        {
            const auto finalIdentity = transactionFileIdentity(finalPath);
            const auto backupIdentity = transactionFileIdentity(backupPath);
            const bool oldFinal = transactionIdentityMatches(
                finalIdentity, entry.oldSize, entry.oldSha256);
            const bool oldBackup = transactionIdentityMatches(
                backupIdentity, entry.oldSize, entry.oldSha256);
            if (oldFinal == oldBackup)
                return failRecoveryRollback(kExpectedOutputs[index]);
            if (oldBackup
                && (finalIdentity.present
                    || !renameKnown(backupPath, finalPath,
                                    "restore-rename:" + twoDigitIndex(index))))
                return failRecoveryRollback(kExpectedOutputs[index]);
            entry.oldLocation = "FINAL";
        }
        else
        {
            if (transactionFileIdentity(finalPath).present
                || transactionFileIdentity(backupPath).present)
                return failRecoveryRollback(kExpectedOutputs[index]);
            entry.oldLocation = "ABSENT";
        }
        crashCut("rollback-restore-after:" + twoDigitIndex(index));
        journal_.pending = TransactionPending::None;
        journal_.pendingIndex = kExpectedOutputs.size();
        ++journal_.rollbackCursor;
        if (!writeJournal("coherence:T22:" + twoDigitIndex(index), "T22"))
            return failRecoveryRollback(kExpectedOutputs[index]);
    }

    if (!validateOldAuthority())
        return failRecoveryRollback("old-authority-census");
    journal_.pending = TransactionPending::None;
    journal_.pendingIndex = kExpectedOutputs.size();
    if (journal_.finalizeCursor > 3)
        return failRecoveryRollback("finalize-cursor");
    for (size_t index = journal_.finalizeCursor; index < 3; ++index)
    {
        journal_.finalizeCursor = index + 1;
        if (!writeJournal("finalize:" + twoDigitIndex(index), "T26"))
            return failRecoveryRollback("finalize-journal");
    }
    if (!cleanupTransactionTree(false))
        return false;
    terminalFinalized_ = true;
    journal_.state = TransactionState::Complete;
    trace("rollback:complete");
    return false;
}

bool OutputTransaction::committedCleanup() noexcept
{
    committed_ = true;
    trace("cleanup:begin");
    if (journal_.state == TransactionState::RecoveryRequiredCleanup)
    {
        journal_.state = TransactionState::CommittedCleanupPending;
        if (!writeJournal("coherence:T31", "T31"))
            return failRecoveryCleanup("retry-cleanup");
    }
    if (!reconcilePending(true))
        return failRecoveryCleanup("pending-operation");
    if (journal_.state != TransactionState::CommittedCleanupPending
        || journal_.cleanupCursor > kExpectedOutputs.size())
        return failRecoveryCleanup("cleanup-cursor");
    while (journal_.cleanupCursor < kExpectedOutputs.size())
    {
        const size_t index = journal_.cleanupCursor;
        if (!validateCommittedFinals())
            return failRecoveryCleanup(kExpectedOutputs[index]);
        auto& entry = journal_.entries[index];
        journal_.pending = TransactionPending::CleanupBackup;
        journal_.pendingIndex = index;
        if (!writeJournal("cleanup-backup:" + twoDigitIndex(index), "T27"))
            return failRecoveryCleanup(kExpectedOutputs[index]);
        crashCut("cleanup-backup-before:" + twoDigitIndex(index));
        const auto backupPath = backupDirectory_ / kExpectedOutputs[index];
        const auto identity = transactionFileIdentity(backupPath);
        if (identity.present)
        {
            if (!entry.oldPresent
                || !transactionIdentityMatches(identity, entry.oldSize,
                                                entry.oldSha256)
                || !removeKnownFile(
                    backupPath, "backup-cleanup:" + twoDigitIndex(index)))
                return failRecoveryCleanup(kExpectedOutputs[index]);
        }
        entry.oldLocation = "ABSENT";
        crashCut("cleanup-backup-after:" + twoDigitIndex(index));
        journal_.pending = TransactionPending::None;
        journal_.pendingIndex = kExpectedOutputs.size();
        ++journal_.cleanupCursor;
        if (!writeJournal("coherence:T28:" + twoDigitIndex(index), "T28"))
            return failRecoveryCleanup(kExpectedOutputs[index]);
    }
    journal_.pending = TransactionPending::None;
    journal_.pendingIndex = kExpectedOutputs.size();
    journal_.cleanupCursor = kExpectedOutputs.size();
    if (journal_.finalizeCursor > 3)
        return failRecoveryCleanup("finalize-cursor");
    for (size_t index = journal_.finalizeCursor; index < 3; ++index)
    {
        journal_.finalizeCursor = index + 1;
        if (!writeJournal("finalize:" + twoDigitIndex(index), "T32"))
            return failRecoveryCleanup("finalize-journal");
    }
    if (!cleanupTransactionTree(true))
        return false;
    terminalFinalized_ = true;
    journal_.state = TransactionState::Complete;
    trace("cleanup:complete");
    return true;
}

bool OutputTransaction::cleanupTransactionTree(bool committed) noexcept
{
    auto fail = [this, committed](const std::string& detail) {
        return committed ? failRecoveryCleanup(detail)
                         : failRecoveryRollback(detail);
    };
    if (committed && !validateCommittedFinals())
        return fail("final-census");
    for (const auto& temporary : journalTempPaths_)
    {
        const auto parsed = readTransactionJournal(temporary);
        if (parsed.present
            && (!parsed.valid
                || parsed.journal.transactionId != journal_.transactionId
                || parsed.journal.rootBindingSha256 != journal_.rootBindingSha256))
            return fail("journal-temp");
        if (parsed.present && !removeKnownFile(temporary))
            return fail("journal-temp");
    }
    const int inactive = activeJournalSlot_ == 0 ? 1 : 0;
    crashCut("journal-cleanup-before:00");
    if (committed && !validateCommittedFinals())
        return fail("final-census");
    if (!removeKnownFile(journalPaths_[static_cast<size_t>(inactive)],
                         "journal-cleanup:00"))
        return fail("journal-cleanup:00");
    crashCut("journal-cleanup-after:00");

    crashCut("directory-cleanup-before:00");
    if (committed && !validateCommittedFinals())
        return fail("final-census");
    if (!removeKnownDirectory(stagedDirectory_, "directory-cleanup:00"))
        return fail("directory-cleanup:00");
    crashCut("directory-cleanup-after:00");

    crashCut("directory-cleanup-before:01");
    if (committed && !validateCommittedFinals())
        return fail("final-census");
    if (!removeKnownDirectory(backupDirectory_, "directory-cleanup:01"))
        return fail("directory-cleanup:01");
    crashCut("directory-cleanup-after:01");

    crashCut("journal-cleanup-before:01");
    crashCut("directory-cleanup-before:02");
    if (committed && !validateCommittedFinals())
        return fail("final-census");
    if (injectedFault("journal-cleanup:01"))
        return fail("journal-cleanup:01");
    if (injectedFault("directory-cleanup:02"))
        return fail("directory-cleanup:02");
    const std::filesystem::path active =
        journalPaths_[static_cast<size_t>(activeJournalSlot_)];
    if (!removeKnownFile(active))
        return fail("journal-cleanup:01");
    std::error_code error;
    if (!std::filesystem::remove(transactionDirectory_, error) || error)
    {
        // A reported final rmdir error must not strand an unidentifiable empty
        // directory. Recreate the last valid committed/rollback journal so a
        // later invocation retains deterministic recovery authority.
        if (!writeJournalDirect(active, journal_))
            return fail("directory-cleanup:02-journal-recreate");
        return fail("directory-cleanup:02");
    }
    activeJournalSlot_ = -1;
    journalInitialized_ = false;
    crashCut("journal-cleanup-after:01");
    crashCut("directory-cleanup-after:02");
    return true;
}

bool OutputTransaction::commit()
{
    if (!std::all_of(finished_.begin(), finished_.end(),
                     [](bool value) { return value; })
        || journal_.state != TransactionState::Staged
        || !validateStagedCensus() || !validateOldAuthority())
        return failPrecommit("STAGE", "artifact-census");

    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        auto& entry = journal_.entries[index];
        const auto finalPath = outputDirectory_ / kExpectedOutputs[index];
        const auto backupPath = backupDirectory_ / kExpectedOutputs[index];
        if (index != 0 && !entry.oldPresent)
            continue;
        journal_.state = TransactionState::BackupInProgress;
        journal_.pending = TransactionPending::Backup;
        journal_.pendingIndex = index;
        if (!writeJournal("backup:" + twoDigitIndex(index),
                          index == 0 ? "T05" : "T07"))
            return failPrecommit("BACKUP", kExpectedOutputs[index]);
        crashCut("backup-before:" + twoDigitIndex(index));
        if (entry.oldPresent)
        {
            if (!transactionIdentityMatches(transactionFileIdentity(finalPath),
                                            entry.oldSize, entry.oldSha256)
                || pathKind(backupPath) != PathKind::Missing
                || !renameKnown(finalPath, backupPath,
                                "backup-rename:" + twoDigitIndex(index)))
                return failPrecommit("BACKUP", kExpectedOutputs[index]);
            entry.oldLocation = "BACKUP";
        }
        else if (pathKind(finalPath) != PathKind::Missing
                 || pathKind(backupPath) != PathKind::Missing)
        {
            return failPrecommit("PRECHECK", kExpectedOutputs[index]);
        }
        crashCut("backup-after:" + twoDigitIndex(index));
        journal_.pending = TransactionPending::None;
        journal_.pendingIndex = kExpectedOutputs.size();
        if (!writeJournal("coherence:T06:" + twoDigitIndex(index), "T06"))
            return failRecoveryRollback(kExpectedOutputs[index]);
    }

    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {
        auto& entry = journal_.entries[index];
        const auto stagedPath = stagedDirectory_ / kExpectedOutputs[index];
        const auto finalPath = outputDirectory_ / kExpectedOutputs[index];
        journal_.state = TransactionState::PublishInProgress;
        journal_.pending = TransactionPending::Publish;
        journal_.pendingIndex = index;
        if (!writeJournal("publish:" + twoDigitIndex(index),
                          index == 0 ? "T08" : "T10"))
            return failPrecommit("PUBLISH", kExpectedOutputs[index]);
        crashCut("publish-before:" + twoDigitIndex(index));
#if defined(DSPARK_TIMESTRETCH_TRANSACTION_TESTING)
        if (static_cast<int>(index) == commitFailureIndex_)
            return failPrecommit("COMMIT", kExpectedOutputs[index]);
#endif
        if (!transactionIdentityMatches(transactionFileIdentity(stagedPath),
                                        entry.newSize, entry.newSha256)
            || pathKind(finalPath) != PathKind::Missing
            || !renameKnown(stagedPath, finalPath,
                            "publish-rename:" + twoDigitIndex(index)))
            return failPrecommit("PUBLISH", kExpectedOutputs[index]);
        entry.newLocation = "FINAL";
        crashCut("publish-after:" + twoDigitIndex(index));
        journal_.pending = TransactionPending::None;
        journal_.pendingIndex = kExpectedOutputs.size();
        if (!writeJournal("coherence:T09:" + twoDigitIndex(index), "T09"))
            return failRecoveryRollback(kExpectedOutputs[index]);
    }

    journal_.state = TransactionState::CommitReady;
    journal_.pending = TransactionPending::None;
    journal_.pendingIndex = kExpectedOutputs.size();
    if (!validateCommittedFinals())
        return failPrecommit("COMMIT", "artifact-census");
    if (!writeJournal("coherence:T11", "T11"))
        return failPrecommit("COMMIT", "commit-ready");
    crashCut("commit-ready");
    journal_.state = TransactionState::CommittedCleanupPending;
    if (!writeJournal("commit-marker", "T12"))
        return failPrecommit("COMMIT", "commit-marker");
    committed_ = true; // Successful promotion above is the sole commit point.
    trace("commit-point:COMMITTED_CLEANUP_PENDING");
    crashCut("post-commit-marker");
    if (!committedCleanup())
        return false;
    return validateTerminalFinalCensus(false);
}

int OutputTransaction::abortAndReport() noexcept
{
    if (!terminalFinalized_ && !committed_
        && pathKind(transactionDirectory_) == PathKind::Directory)
        rollbackAndClean();
    return reportFailure();
}

/// Band-limited click: a windowed sinc, so it has no energy above the cutoff.
void addClick(std::vector<double>& dst, size_t at, double cutoffHz, double amp)
{
    const int half = 96;
    for (int k = -half; k <= half; ++k)
    {
        const auto idx = static_cast<long long>(at) + k;
        if (idx < 0 || idx >= static_cast<long long>(dst.size())) continue;
        const double x = 2.0 * cutoffHz * static_cast<double>(k) / kRate;
        const double s = (k == 0) ? 1.0 : std::sin(kPi * x) / (kPi * x);
        const double w = 0.5 + 0.5 * std::cos(kPi * static_cast<double>(k) / half);
        dst[static_cast<size_t>(idx)] += amp * s * w;
    }
}

/// Click train at `bpm`, band-limited to 16 kHz.
Signal clickTrain(double seconds, double bpm, int channels, std::vector<size_t>& onsets)
{
    const auto n = static_cast<size_t>(seconds * kRate);
    Signal sig(static_cast<size_t>(channels), std::vector<double>(n, 0.0));
    onsets.clear();
    const double period = 60.0 / bpm * kRate;
    for (double t = 4800.0; t < static_cast<double>(n) - 4800.0; t += period)
    {
        const auto at = static_cast<size_t>(t);
        onsets.push_back(at);
        for (auto& ch : sig) addClick(ch, at, 16000.0, 0.9);
    }
    return sig;
}

// -- analysis helpers ----------------------------------------------------------

std::vector<double> hannWindow(int n)
{
    std::vector<double> w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        w[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n);
    return w;
}

/// Frame-averaged magnitude spectrum (Hann, 75% overlap).
std::vector<double> averageSpectrum(const std::vector<double>& x, int n, int hop)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    const auto w = hannWindow(n);
    std::vector<double> frame(static_cast<size_t>(n)), spec(static_cast<size_t>(n + 2));
    const int bins = n / 2 + 1;
    std::vector<double> acc(static_cast<size_t>(bins), 0.0);
    int frames = 0;
    for (size_t start = 0; start + static_cast<size_t>(n) <= x.size();
         start += static_cast<size_t>(hop))
    {
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        for (int k = 0; k < bins; ++k)
        {
            const double re = spec[static_cast<size_t>(2 * k)];
            const double im = spec[static_cast<size_t>(2 * k + 1)];
            acc[static_cast<size_t>(k)] += re * re + im * im;
        }
        ++frames;
    }
    if (frames == 0) frames = 1;
    for (auto& a : acc) a = std::sqrt(a / frames);
    return acc;
}

/// Rectangular-window magnitude spectrum of one segment (coherent sampling).
std::vector<double> segmentSpectrum(const std::vector<double>& x, size_t start, int n)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    std::vector<double> frame(static_cast<size_t>(n), 0.0), spec(static_cast<size_t>(n + 2));
    for (int i = 0; i < n; ++i)
        if (start + static_cast<size_t>(i) < x.size())
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)];
    fft.forward(frame.data(), spec.data());
    const int bins = n / 2 + 1;
    std::vector<double> mag(static_cast<size_t>(bins));
    for (int k = 0; k < bins; ++k)
    {
        const double re = spec[static_cast<size_t>(2 * k)];
        const double im = spec[static_cast<size_t>(2 * k + 1)];
        mag[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
    }
    return mag;
}

double rms(const std::vector<double>& x, size_t from, size_t to)
{
    if (to > x.size()) to = x.size();
    if (from >= to) return 0.0;
    double s = 0.0;
    for (size_t i = from; i < to; ++i) s += x[i] * x[i];
    return std::sqrt(s / static_cast<double>(to - from));
}

double db(double v) { return 20.0 * std::log10(std::max(v, 1e-300)); }

// -- STFT consistency (Laroche-Dolson) -----------------------------------------

/// Consistency ratio: energy of the STFT over the energy it loses when it is
/// resynthesised and analysed again. Higher means the frames agree with each
/// other, which is the property phase locking is there to preserve.
double stftConsistency(const std::vector<double>& x, int n, int hop)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    const auto w = hannWindow(n);
    const int bins = n / 2 + 1;
    const size_t nFrames = (x.size() < static_cast<size_t>(n))
                         ? 0 : (x.size() - static_cast<size_t>(n)) / static_cast<size_t>(hop) + 1;
    if (nFrames < 4) return 0.0;

    std::vector<double> Z(nFrames * static_cast<size_t>(n + 2));
    std::vector<double> frame(static_cast<size_t>(n)), spec(static_cast<size_t>(n + 2));
    for (size_t f = 0; f < nFrames; ++f)
    {
        const size_t start = f * static_cast<size_t>(hop);
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        std::copy(spec.begin(), spec.end(), Z.begin() + static_cast<long>(f) * (n + 2));
    }

    // Inverse: weighted overlap-add, normalised by the summed squared window.
    const size_t len = (nFrames - 1) * static_cast<size_t>(hop) + static_cast<size_t>(n);
    std::vector<double> y(len, 0.0), norm(len, 0.0), timeBuf(static_cast<size_t>(n));
    for (size_t f = 0; f < nFrames; ++f)
    {
        std::copy(Z.begin() + static_cast<long>(f) * (n + 2),
                  Z.begin() + static_cast<long>(f + 1) * (n + 2), spec.begin());
        fft.inverse(spec.data(), timeBuf.data());
        const size_t start = f * static_cast<size_t>(hop);
        for (int i = 0; i < n; ++i)
        {
            y[start + static_cast<size_t>(i)] += timeBuf[static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
            norm[start + static_cast<size_t>(i)] += w[static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        }
    }
    for (size_t i = 0; i < len; ++i)
        if (norm[i] > 1e-12) y[i] /= norm[i];

    // Re-analyse and compare, skipping the first and last frame where the
    // overlap-add normalisation is incomplete.
    double num = 0.0, den = 0.0;
    for (size_t f = 1; f + 1 < nFrames; ++f)
    {
        const size_t start = f * static_cast<size_t>(hop);
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = y[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        for (int k = 0; k < bins; ++k)
        {
            const double zr = Z[static_cast<size_t>(f) * (n + 2) + static_cast<size_t>(2 * k)];
            const double zi = Z[static_cast<size_t>(f) * (n + 2) + static_cast<size_t>(2 * k + 1)];
            const double xr = spec[static_cast<size_t>(2 * k)];
            const double xi = spec[static_cast<size_t>(2 * k + 1)];
            num += zr * zr + zi * zi;
            den += (zr - xr) * (zr - xr) + (zi - xi) * (zi - xi);
        }
    }
    return num / std::max(den, 1e-300);
}

/**
 * Vertical phase coherence around spectral peaks.
 *
 * A single sinusoid seen through a Hann window puts a fixed phase relation
 * between a peak bin and its neighbours: the window kernel's first side lobes
 * are negative, so the neighbours sit half a turn away from the peak. Identity
 * phase locking exists precisely to keep that relation; a plain vocoder
 * advances each bin on its own and loses it, which is what "phasiness" is.
 * The measure is the energy-weighted mean of that relation over every peak of
 * every frame: 1.0 is perfect coherence, 0.0 none.
 *
 * This is reported beside the round-trip consistency ratio because the
 * round-trip ratio cannot separate the two cases: the transform pair used
 * there inverts exactly on any real signal, so the spectrogram of ANY output
 * is self-consistent and the ratio is pinned at its numerical ceiling.
 */
double verticalCoherence(const std::vector<double>& x, int n, int hop)
{
    dspark::FFTReal<double> fft(static_cast<size_t>(n));
    const auto w = hannWindow(n);
    const int bins = n / 2 + 1;
    std::vector<double> frame(static_cast<size_t>(n)), spec(static_cast<size_t>(n + 2));
    double sum = 0.0, weight = 0.0;
    for (size_t start = 0; start + static_cast<size_t>(n) <= x.size();
         start += static_cast<size_t>(hop))
    {
        for (int i = 0; i < n; ++i)
            frame[static_cast<size_t>(i)] = x[start + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
        fft.forward(frame.data(), spec.data());
        std::vector<double> mag(static_cast<size_t>(bins));
        double peak = 0.0;
        for (int k = 0; k < bins; ++k)
        {
            const double re = spec[static_cast<size_t>(2 * k)];
            const double im = spec[static_cast<size_t>(2 * k + 1)];
            mag[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
            peak = std::max(peak, mag[static_cast<size_t>(k)]);
        }
        if (peak < 1e-9) continue;
        const double floorMag = peak * 1e-3;
        for (int k = 2; k < bins - 2; ++k)
        {
            const double m = mag[static_cast<size_t>(k)];
            if (m < floorMag) continue;
            if (!(m > mag[static_cast<size_t>(k - 1)] && m >= mag[static_cast<size_t>(k + 1)]
                  && m > mag[static_cast<size_t>(k - 2)] && m >= mag[static_cast<size_t>(k + 2)]))
                continue;
            const double pr = spec[static_cast<size_t>(2 * k)];
            const double pi = spec[static_cast<size_t>(2 * k + 1)];
            for (int d : { -1, 1 })
            {
                const double nr = spec[static_cast<size_t>(2 * (k + d))];
                const double ni = spec[static_cast<size_t>(2 * (k + d) + 1)];
                const double cr = nr * pr + ni * pi;     // Re(neighbour * conj(peak))
                const double ci = ni * pr - nr * pi;
                const double amp = std::sqrt(cr * cr + ci * ci);
                if (amp < 1e-30) continue;
                sum += m * m * (-cr / amp);
                weight += m * m;
            }
        }
    }
    return (weight > 0.0) ? sum / weight : 0.0;
}

// -- WSOLA reference -----------------------------------------------------------

/// Independent time-domain time-scaler used only as the transient reference.
/// Overlap-add with a search for the segment that continues the output best,
/// which is what keeps a strike intact without any spectral processing.
std::vector<double> wsola(const std::vector<double>& x, double ratio, int frame, int seek)
{
    const int hopS = frame / 2;
    const auto hopA = static_cast<double>(hopS) / ratio;
    const auto outLen = static_cast<size_t>(std::lround(static_cast<double>(x.size()) * ratio));
    std::vector<double> y(outLen + static_cast<size_t>(frame), 0.0);
    std::vector<double> norm(y.size(), 0.0);
    const auto w = hannWindow(frame);

    double aPos = 0.0;
    size_t sPos = 0;
    std::vector<double> tail(static_cast<size_t>(hopS), 0.0);   // wanted continuation
    while (sPos + static_cast<size_t>(frame) < y.size()
           && static_cast<size_t>(aPos) + static_cast<size_t>(frame + seek) < x.size())
    {
        long best = 0;
        double bestScore = -1e300;
        const auto base = static_cast<long>(aPos);
        for (long d = -seek; d <= seek; ++d)
        {
            const long s = base + d;
            if (s < 0 || s + frame >= static_cast<long>(x.size())) continue;
            double num = 0.0, en = 1e-12;
            for (int i = 0; i < hopS; ++i)
            {
                const double v = x[static_cast<size_t>(s + i)];
                num += v * tail[static_cast<size_t>(i)];
                en += v * v;
            }
            const double score = num / std::sqrt(en);
            if (score > bestScore) { bestScore = score; best = s; }
        }
        for (int i = 0; i < frame; ++i)
        {
            const auto si = static_cast<size_t>(best + i);
            if (si >= x.size()) break;
            y[sPos + static_cast<size_t>(i)] += x[si] * w[static_cast<size_t>(i)];
            norm[sPos + static_cast<size_t>(i)] += w[static_cast<size_t>(i)];
        }
        for (int i = 0; i < hopS; ++i)
        {
            const auto si = static_cast<size_t>(best + hopS + i);
            tail[static_cast<size_t>(i)] = (si < x.size()) ? x[si] : 0.0;
        }
        sPos += static_cast<size_t>(hopS);
        aPos += hopA;
    }
    for (size_t i = 0; i < y.size(); ++i)
        if (norm[i] > 1e-9) y[i] /= norm[i];
    y.resize(outLen);
    return y;
}

// -- device under test ---------------------------------------------------------

struct Options
{
    double ratio = 1.0;
    bool phaseLock = true;
    bool transient = true;
    int fftSize = 2048;
};

Signal stretchOffline(const Signal& in, const Options& opt)
{
    const int nCh = static_cast<int>(in.size());
    const auto nS = static_cast<int>(in[0].size());

    dspark::AudioBuffer<float> src, dst;
    src.resize(nCh, nS);
    for (int ch = 0; ch < nCh; ++ch)
        for (int i = 0; i < nS; ++i)
            src.getChannel(ch)[i] = static_cast<float>(in[static_cast<size_t>(ch)][static_cast<size_t>(i)]);

    dspark::TimeStretch<float> ts;
    ts.prepare({ kRate, kBlock, nCh }, opt.fftSize);
    ts.setTimeRatio(static_cast<float>(opt.ratio));
    ts.setPhaseLock(opt.phaseLock);
    ts.setTransientPreserve(opt.transient);
    ts.process(src.toView(), dst);

    Signal out(static_cast<size_t>(nCh),
               std::vector<double>(static_cast<size_t>(dst.getNumSamples()), 0.0));
    for (int ch = 0; ch < nCh; ++ch)
        for (int i = 0; i < dst.getNumSamples(); ++i)
            out[static_cast<size_t>(ch)][static_cast<size_t>(i)] = dst.getChannel(ch)[i];
    return out;
}

/// Streaming run with a caller-chosen chopping pattern.
std::vector<double> stretchStreaming(const std::vector<double>& in, const Options& opt,
                                     const std::vector<int>& pattern)
{
    dspark::TimeStretch<float> ts;
    ts.prepare({ kRate, 4096, 1 }, opt.fftSize);
    ts.setTimeRatio(static_cast<float>(opt.ratio));

    std::vector<double> out;
    out.reserve(in.size());
    std::vector<float> scratch(4096);
    size_t pos = 0, p = 0;
    while (pos < in.size())
    {
        const auto want = static_cast<size_t>(pattern[p % pattern.size()]);
        const auto n = static_cast<int>(std::min(want, in.size() - pos));
        for (int i = 0; i < n; ++i)
            scratch[static_cast<size_t>(i)] = static_cast<float>(in[pos + static_cast<size_t>(i)]);
        float* ptrs[1] = { scratch.data() };
        dspark::AudioBufferView<float> view(ptrs, 1, n);
        ts.processBlock(view);
        for (int i = 0; i < n; ++i) out.push_back(scratch[static_cast<size_t>(i)]);
        pos += static_cast<size_t>(n);
        ++p;
    }
    return out;
}

// -- reporting -----------------------------------------------------------------

struct Row
{
    std::string label;
    double ratio = 1.0;
    double lsd = 0.0, sc = 0.0, spurious = 0.0, transientKeep = 0.0, attack = 0.0;
    double consistencyIn = 0.0, consistencyOut = 0.0, consistencyPlain = 0.0;
    double vcohIn = 0.0, vcohLocked = 0.0, vcohPlain = 0.0;
    double ild = 0.0, coherence = 0.0;
    double ratioError = 0.0;
    long long outLen = 0, wantLen = 0;
};

std::string fmt(double v, int prec = 3)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", prec, v);
    return buf;
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr,
                     "ERROR TIMESTRETCH_OUTPUT_DIRECTORY expected exactly one existing directory argument\n");
        return 2;
    }
    const std::filesystem::path outputDirectory(argv[1]);
    std::error_code directoryError;
    const auto outputStatus = std::filesystem::symlink_status(
        outputDirectory, directoryError);
    if (directoryError || !std::filesystem::is_directory(outputStatus))
    {
        std::fprintf(stderr,
                     "ERROR TIMESTRETCH_OUTPUT_DIRECTORY path is not an existing directory: %s\n",
                     argv[1]);
        return 2;
    }
    OutputTransaction transaction(outputDirectory);
    if (!transaction.begin())
        return transaction.abortAndReport();
    if (transaction.recoveredCommitted())
    {
        std::printf("characterisation written to %s\n",
                    outputDirectory.string().c_str());
        return 0;
    }
    const std::string dir =
        (outputDirectory / kTransactionDirectory / "staged").string();

    // The tone frequency is placed exactly on an analysis bin of the 32768
    // spectrum used for C2/C7, so a rectangular window leaks nothing and the
    // measured floor is the device's, not the analyser's.
    constexpr int kToneFft = 32768;
    const double toneHz = 683.0 * kRate / kToneFft;   // 1000.195 Hz

    struct Target { const char* label; double ratio; double lsdLimit; };
    const Target targets[] = {
        { "r=0.926 (-8%)",       0.926,               1.5 },
        { "r=0.950",             0.95,                1.0 },
        { "r=1.000",             1.0,                 1.0 },
        { "r=1.050",             1.05,                1.0 },
        { "r=1.081 (+8%)",       1.081,               1.5 },
        { "120->110 BPM",        120.0 / 110.0,       1.5 },
        { "120->130 BPM",        120.0 / 130.0,       1.5 },
    };

    std::vector<Row> rows;

    // ---------------------------------------------------------------- C6, C7
    // The regression tripwires run first, before any aesthetic measurement.
    {
        std::ofstream f;
        if (!transaction.openOutput(f, "c6-exact-ratio-drift.csv"))
            return transaction.abortAndReport();
        f << "target_ratio,input_samples,output_samples,expected_samples,length_error_samples,"
             "one_hop,realised_ratio_from_onsets,ratio_error_percent,onsets_used,"
             "worst_onset_error_samples\n";
        for (const auto& t : targets)
        {
            // 32 s of clicks: the realised ratio is measured from where the
            // clicks land, not from the buffer length the class allocates.
            std::vector<size_t> onsets;
            Signal in = clickTrain(32.0, 120.0, 1, onsets);
            Options opt; opt.ratio = t.ratio;
            Signal out = stretchOffline(in, opt);

            const auto n = static_cast<long long>(in[0].size());
            const auto m = static_cast<long long>(out[0].size());
            const long long want = std::llround(static_cast<double>(n) * t.ratio);

            // Locate each click in the output by peak search around where it
            // should be, then fit a straight line through the pairs.
            double sxy = 0.0, sxx = 0.0;
            double worst = 0.0;
            int used = 0;
            for (size_t o : onsets)
            {
                const auto expect = static_cast<long long>(std::llround(static_cast<double>(o) * t.ratio));
                const long long lo = std::max(0LL, expect - 2000);
                const long long hi = std::min(m, expect + 2000);
                long long peak = -1;
                double best = 0.0;
                for (long long i = lo; i < hi; ++i)
                {
                    const double v = std::fabs(out[0][static_cast<size_t>(i)]);
                    if (v > best) { best = v; peak = i; }
                }
                if (peak < 0 || best < 0.05) continue;
                sxy += static_cast<double>(o) * static_cast<double>(peak);
                sxx += static_cast<double>(o) * static_cast<double>(o);
                worst = std::max(worst, std::fabs(static_cast<double>(peak - expect)));
                ++used;
            }
            const double realised = (sxx > 0.0) ? sxy / sxx : 0.0;
            const double err = (realised - t.ratio) / t.ratio * 100.0;

            f << fmt(t.ratio, 6) << ',' << n << ',' << m << ',' << want << ','
              << (m - want) << ',' << (opt.fftSize / 4) << ','
              << fmt(realised, 8) << ',' << fmt(err, 6) << ',' << used << ','
              << fmt(worst, 1) << '\n';

            Row r;
            r.label = t.label; r.ratio = t.ratio;
            r.outLen = m; r.wantLen = want; r.ratioError = err;
            rows.push_back(r);
        }
        if (!transaction.finishOutput(f, "c6-exact-ratio-drift.csv"))
            return transaction.abortAndReport();
    }

    {
        std::ofstream f;
        if (!transaction.openOutput(f, "c7-unity-passthrough.csv"))
            return transaction.abortAndReport();
        f << "signal,path,residual_dbfs,thd_n_dbfs,peak_dbfs\n";
        Options unity; unity.ratio = 1.0;

        auto measure = [&](const char* name, const Signal& in, bool streaming)
        {
            std::vector<double> out;
            int latency = 0;
            if (streaming)
            {
                dspark::TimeStretch<float> probe;
                probe.prepare({ kRate, kBlock, 1 }, unity.fftSize);
                latency = probe.getLatency();
                out = stretchStreaming(in[0], unity, { kBlock });
            }
            else
            {
                out = stretchOffline(in, unity)[0];
            }

            // Residual against the aligned input over the settled region.
            const size_t start = static_cast<size_t>(latency) + 24000;
            double num = 0.0;
            size_t count = 0;
            for (size_t i = start; i < out.size() && i - static_cast<size_t>(latency) < in[0].size(); ++i)
            {
                const double d = out[i] - in[0][i - static_cast<size_t>(latency)];
                num += d * d; ++count;
            }
            const double residual = db(std::sqrt(num / std::max<size_t>(count, 1)));

            // THD+N of the tone: everything outside the fundamental bin group.
            // On a bed that is not a single tone the quantity is not defined,
            // and the field is written `n/a`. Writing 0.00 there would read as
            // a measurement of zero dB - a 60 dB failure - to anything checking
            // this file with a machine rather than an eye.
            bool thdnApplies = false;
            double thdn = 0.0;
            if (std::strcmp(name, "1 kHz tone") == 0)
            {
                thdnApplies = true;
                const auto mag = segmentSpectrum(out, start, kToneFft);
                double fund = 0.0, rest = 0.0;
                for (size_t k = 0; k < mag.size(); ++k)
                {
                    const double p = mag[k] * mag[k];
                    if (std::llabs(static_cast<long long>(k) - 683LL) <= 2) fund += p;
                    else rest += p;
                }
                thdn = 10.0 * std::log10(rest / std::max(fund, 1e-300));
            }
            double peak = 0.0;
            for (double v : out) peak = std::max(peak, std::fabs(v));
            f << name << ',' << (streaming ? "streaming" : "offline") << ','
              << fmt(residual, 2) << ',' << (thdnApplies ? fmt(thdn, 2) : std::string("n/a"))
              << ',' << fmt(db(peak), 2) << '\n';
        };

        const Signal tone = sine(toneHz, 4.0, 1);
        const Signal pink = pinkNoise(4.0, 1);
        measure("1 kHz tone", tone, false);
        measure("1 kHz tone", tone, true);
        measure("pink noise", pink, false);
        measure("pink noise", pink, true);
        if (!transaction.finishOutput(f, "c7-unity-passthrough.csv"))
            return transaction.abortAndReport();
    }

    // ------------------------------------------------------------------- C8
    {
        std::ofstream f;
        if (!transaction.openOutput(f, "c8-chopping-determinism.csv"))
            return transaction.abortAndReport();
        f << "ratio,pattern,samples_compared,differing_samples,first_divergence,"
             "max_abs_difference\n";
        const std::vector<std::vector<int>> patterns = {
            { 512 }, { 64 }, { 1, 7, 63, 512, 129, 4096 }, { 4096 }, { 333 }
        };
        Signal bed = harmonicBed(6.0, 1);
        for (double r : { 0.926, 1.0, 1.05, 1.081 })
        {
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = r;
                const auto ref = stretchStreaming(bed[0], opt, patterns[0]);
                for (size_t p = 1; p < patterns.size(); ++p)
                {
                    const auto got = stretchStreaming(bed[0], opt, patterns[p]);
                    const size_t n = std::min(ref.size(), got.size());
                    size_t diff = 0;
                    long long first = -1;
                    double worst = 0.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        const double d = std::fabs(ref[i] - got[i]);
                        if (d != 0.0)
                        {
                            ++diff;
                            if (first < 0) first = static_cast<long long>(i);
                            worst = std::max(worst, d);
                        }
                    }
                    std::string desc;
                    for (int b : patterns[p]) desc += std::to_string(b) + "|";
                    f << fmt(r, 3) << ',' << desc << ','
                      << n << ',' << diff << ',' << first << ',' << fmt(worst, 12) << '\n';
                }
            }
        }
        if (!transaction.finishOutput(f, "c8-chopping-determinism.csv"))
            return transaction.abortAndReport();
    }

    // --------------------------------------------------------------- C1, C4
    {
        std::ofstream f1;
        if (!transaction.openOutput(f1, "c1-stationary-fidelity.csv"))
            return transaction.abortAndReport();
        f1 << "# LSD is computed on both averaged spectra clipped to 80 dB below the\n"
              "# reference's own peak. The bed is 12 partials, so most bins of the\n"
              "# 20 Hz-16 kHz range hold nothing but the analyser's numerical residue\n"
              "# in BOTH spectra; without the clip their ratio is the dominant term\n"
              "# and the metric reports the analyser instead of the device. The\n"
              "# unclipped value is carried alongside so the difference is visible.\n";
        f1 << "ratio,lsd_db,lsd_limit_db,lsd_db_unclipped,spectral_convergence_db\n";
        std::ofstream f4;
        if (!transaction.openOutput(f4, "c4-vertical-coherence.csv"))
        {
            transaction.finishOutput(f1, "c1-stationary-fidelity.csv");
            return transaction.abortAndReport();
        }
        f4 << "# consistency_* is the round-trip ratio as specified. The transform\n"
              "# pair inverts exactly on any real signal, so every one of these is\n"
              "# pinned at the numerical ceiling and the comparison between them is\n"
              "# noise; vcoh_* is the discriminating measure (peak-to-neighbour phase\n"
              "# relation, 1.0 = fully coherent).\n";
        f4 << "ratio,bed,consistency_in,consistency_out_locked,consistency_out_plain,"
              "ratio_out_over_in,vcoh_in,vcoh_locked,vcoh_plain,locked_better_than_plain\n";

        const Signal bed = harmonicBed(5.0, 1);
        const auto sref = averageSpectrum(bed[0], 4096, 1024);
        const double cIn = stftConsistency(bed[0], 2048, 512);
        const double vIn = verticalCoherence(bed[0], 2048, 512);
        const Signal vib = vibratoBed(5.0, 1);
        const double cVibIn = stftConsistency(vib[0], 2048, 512);
        const double vVibIn = verticalCoherence(vib[0], 2048, 512);
        const int loBin = static_cast<int>(std::ceil(20.0 * 4096.0 / kRate));
        const int hiBin = static_cast<int>(std::floor(16000.0 * 4096.0 / kRate));
        double srefPeak = 0.0;
        for (double v : sref) srefPeak = std::max(srefPeak, v);
        const double clip = srefPeak * 1e-4;   // 80 dB below the reference peak

        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            const auto& t = targets[ti];
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = t.ratio;
                const auto out = stretchOffline(bed, opt)[0];
                const auto sout = averageSpectrum(out, 4096, 1024);

                double acc = 0.0, accRaw = 0.0, dn = 0.0, dd = 0.0;
                int used = 0;
                for (int k = loBin; k <= hiBin && k < static_cast<int>(sref.size()); ++k)
                {
                    const double a = sref[static_cast<size_t>(k)];
                    const double b = sout[static_cast<size_t>(k)];
                    const double d = 20.0 * std::log10(std::max(a, clip) / std::max(b, clip));
                    acc += d * d;
                    const double dRaw = 20.0 * std::log10(std::max(a, 1e-12) / std::max(b, 1e-12));
                    accRaw += dRaw * dRaw;
                    ++used;
                    const double e = a - b;
                    dn += e * e; dd += a * a;
                }
                const double lsd = std::sqrt(acc / std::max(used, 1));
                const double lsdRaw = std::sqrt(accRaw / std::max(used, 1));
                const double sc = 10.0 * std::log10(dn / std::max(dd, 1e-300));
                f1 << fmt(t.ratio, 6) << ','
                   << fmt(lsd, 4) << ',' << fmt(t.lsdLimit, 1) << ',' << fmt(lsdRaw, 2) << ','
                   << fmt(sc, 2) << '\n';
                { rows[ti].lsd = lsd; rows[ti].sc = sc; }
            }

            // Phase locking versus the plain vocoder on the same bed.
            Options locked; locked.ratio = t.ratio; locked.phaseLock = true;
            Options plain;  plain.ratio = t.ratio;  plain.phaseLock = false;
            const auto lockedOut = stretchOffline(bed, locked)[0];
            const auto plainOut  = stretchOffline(bed, plain)[0];
            const double cLocked = stftConsistency(lockedOut, 2048, 512);
            const double cPlain  = stftConsistency(plainOut, 2048, 512);
            const double vLocked = verticalCoherence(lockedOut, 2048, 512);
            const double vPlain  = verticalCoherence(plainOut, 2048, 512);
            f4 << fmt(t.ratio, 6) << ",stationary," << fmt(cIn, 3) << ',' << fmt(cLocked, 3) << ','
               << fmt(cPlain, 3) << ',' << fmt(cLocked / std::max(cIn, 1e-30), 4) << ','
               << fmt(vIn, 5) << ',' << fmt(vLocked, 5) << ',' << fmt(vPlain, 5) << ','
               << (vLocked > vPlain ? "yes" : "NO") << '\n';

            const auto vibLocked = stretchOffline(vib, locked)[0];
            const auto vibPlain  = stretchOffline(vib, plain)[0];
            const double cvLocked = stftConsistency(vibLocked, 2048, 512);
            const double cvPlain  = stftConsistency(vibPlain, 2048, 512);
            const double vvLocked = verticalCoherence(vibLocked, 2048, 512);
            const double vvPlain  = verticalCoherence(vibPlain, 2048, 512);
            f4 << fmt(t.ratio, 6) << ",vibrato," << fmt(cVibIn, 3) << ',' << fmt(cvLocked, 3) << ','
               << fmt(cvPlain, 3) << ',' << fmt(cvLocked / std::max(cVibIn, 1e-30), 4) << ','
               << fmt(vVibIn, 5) << ',' << fmt(vvLocked, 5) << ',' << fmt(vvPlain, 5) << ','
               << (vvLocked > vvPlain ? "yes" : "NO") << '\n';

            rows[ti].consistencyIn = cIn;
            rows[ti].consistencyOut = cLocked;
            rows[ti].consistencyPlain = cPlain;
            rows[ti].vcohIn = vVibIn;
            rows[ti].vcohLocked = vvLocked;
            rows[ti].vcohPlain = vvPlain;
        }
        const bool f1Finished = transaction.finishOutput(
            f1, "c1-stationary-fidelity.csv");
        const bool f4Finished = transaction.finishOutput(
            f4, "c4-vertical-coherence.csv");
        if (!f1Finished || !f4Finished)
            return transaction.abortAndReport();
    }

    // ------------------------------------------------------------------- C2
    {
        std::ofstream f;
        if (!transaction.openOutput(f, "c2-spurious-floor.csv"))
            return transaction.abortAndReport();
        f << "ratio,worst_spurious_db_re_fundamental,worst_bin,worst_hz\n";
        const Signal tone = sine(toneHz, 4.0, 1);
        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = targets[ti].ratio;
                const auto out = stretchOffline(tone, opt)[0];
                const auto mag = segmentSpectrum(out, 48000, kToneFft);
                double fund = 0.0;
                for (long long k = 681; k <= 685; ++k)
                    fund = std::max(fund, mag[static_cast<size_t>(k)]);
                double worst = 0.0;
                long long worstBin = 0;
                for (size_t k = 1; k < mag.size(); ++k)
                {
                    bool harmonic = false;
                    for (long long h = 1; h * 683 < static_cast<long long>(mag.size()); ++h)
                        if (std::llabs(static_cast<long long>(k) - h * 683) <= 2) { harmonic = true; break; }
                    if (harmonic) continue;
                    if (mag[k] > worst) { worst = mag[k]; worstBin = static_cast<long long>(k); }
                }
                const double rel = db(worst / std::max(fund, 1e-300));
                f << fmt(targets[ti].ratio, 6) << ','
                  << fmt(rel, 2) << ',' << worstBin << ','
                  << fmt(static_cast<double>(worstBin) * kRate / kToneFft, 1) << '\n';
                rows[ti].spurious = rel;
            }
        }
        if (!transaction.finishOutput(f, "c2-spurious-floor.csv"))
            return transaction.abortAndReport();
    }

    // ------------------------------------------------------------------- C3
    {
        std::ofstream f;
        if (!transaction.openOutput(f, "c3-transient-preservation.csv"))
            return transaction.abortAndReport();
        f << "ratio,onsets,mean_energy_ratio_vs_wsola,worst_energy_ratio_vs_wsola,"
             "source_attack_ms,output_attack_ms,attack_expansion_percent\n";

        std::vector<size_t> onsets;
        const Signal clicks = clickTrain(5.0, 120.0, 1, onsets);

        // Short-time envelope, 1 ms window, used for both attack measurements.
        auto envelope = [](const std::vector<double>& x)
        {
            const int w = 48;   // 1 ms at 48 kHz
            std::vector<double> e(x.size(), 0.0);
            double acc = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                acc += x[i] * x[i];
                if (i >= static_cast<size_t>(w)) acc -= x[i - static_cast<size_t>(w)] * x[i - static_cast<size_t>(w)];
                e[i] = std::sqrt(acc / w);
            }
            return e;
        };
        auto attackMs = [](const std::vector<double>& env, size_t peakAt)
        {
            const double peak = env[peakAt];
            const size_t lo = (peakAt > 2400) ? peakAt - 2400 : 0;
            size_t t10 = peakAt, t90 = peakAt;
            for (size_t i = peakAt; i > lo; --i) { if (env[i] < 0.9 * peak) { t90 = i; break; } }
            for (size_t i = t90; i > lo; --i)   { if (env[i] < 0.1 * peak) { t10 = i; break; } }
            return static_cast<double>(t90 - t10) / kRate * 1000.0;
        };

        const auto srcEnv = envelope(clicks[0]);
        double srcAttack = 0.0;
        {
            double s = 0.0;
            int c = 0;
            for (size_t o : onsets)
            {
                size_t peak = o;
                double best = 0.0;
                for (size_t i = (o > 200 ? o - 200 : 0); i < o + 200 && i < srcEnv.size(); ++i)
                    if (srcEnv[i] > best) { best = srcEnv[i]; peak = i; }
                s += attackMs(srcEnv, peak); ++c;
            }
            srcAttack = s / std::max(c, 1);
        }

        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            const double r = targets[ti].ratio;
            const auto ref = wsola(clicks[0], r, 2048, 512);
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = r;
                const auto out = stretchOffline(clicks, opt)[0];
                const auto outEnv = envelope(out);

                // Each method is allowed its own onset timing: a WSOLA
                // reference splices at correlation maxima, so its clicks land
                // up to a frame away from where the ratio alone would put
                // them, and comparing the two at the SAME absolute sample
                // would credit the vocoder for the reference's timing rather
                // than measure either one's concentration. Both windows are
                // therefore centred on the onset each output actually
                // produced, found within +-40 ms of the mapped time.
                const auto win = static_cast<size_t>(0.005 * kRate);   // +-5 ms
                const auto search = static_cast<size_t>(0.040 * kRate);
                auto locate = [&](const std::vector<double>& sig, size_t around) -> size_t
                {
                    const size_t lo = (around > search) ? around - search : 0;
                    const size_t hi = std::min(sig.size(), around + search);
                    size_t at = lo;
                    double best = 0.0;
                    for (size_t i = lo; i < hi; ++i)
                        if (std::fabs(sig[i]) > best) { best = std::fabs(sig[i]); at = i; }
                    return at;
                };
                auto energyAround = [&](const std::vector<double>& sig, size_t at)
                {
                    double e = 0.0;
                    const size_t lo = (at > win) ? at - win : 0;
                    const size_t hi = std::min(sig.size(), at + win);
                    for (size_t i = lo; i < hi; ++i) e += sig[i] * sig[i];
                    return e;
                };

                double sumRatio = 0.0, worstRatio = 1e300, sumAttack = 0.0;
                int used = 0;
                for (size_t o : onsets)
                {
                    const auto mapped = static_cast<size_t>(std::llround(static_cast<double>(o) * r));
                    if (mapped + search >= out.size() || mapped + search >= ref.size()
                        || mapped < search) continue;
                    const size_t pOut = locate(out, mapped);
                    const size_t pRef = locate(ref, mapped);
                    const double eOut = energyAround(out, pOut);
                    const double eRef = energyAround(ref, pRef);
                    if (eRef < 1e-12) continue;
                    const double q = eOut / eRef;
                    sumRatio += q;
                    worstRatio = std::min(worstRatio, q);
                    sumAttack += attackMs(outEnv, locate(outEnv, mapped));
                    ++used;
                }
                const double meanRatio = sumRatio / std::max(used, 1);
                const double outAttack = sumAttack / std::max(used, 1);
                const double expansion = (outAttack / std::max(srcAttack * r, 1e-9) - 1.0) * 100.0;
                f << fmt(r, 6) << ',' << used << ','
                  << fmt(meanRatio, 4) << ',' << fmt(worstRatio, 4) << ','
                  << fmt(srcAttack * r, 4) << ',' << fmt(outAttack, 4) << ','
                  << fmt(expansion, 2) << '\n';
                { rows[ti].transientKeep = meanRatio; rows[ti].attack = expansion; }
            }
        }
        if (!transaction.finishOutput(f, "c3-transient-preservation.csv"))
            return transaction.abortAndReport();
    }

    // ------------------------------------------------------------------- C5
    {
        std::ofstream f;
        if (!transaction.openOutput(f, "c5-stereo-integrity.csv"))
            return transaction.abortAndReport();
        f << "ratio,ild_in_db,ild_out_db,ild_delta_db,coherence_in,coherence_out\n";

        // Correlated stereo: the same bed in both channels, right delayed 3 ms.
        // The bed is pink noise rather than the 12 partials of the fidelity
        // test, because the criterion averages coherence over every bin from
        // 20 Hz to 16 kHz: a sparse spectrum leaves most of those bins empty
        // in both channels, and their coherence is then the ratio of two
        // numerical residues, which is what the average would end up
        // reporting.
        Signal st = pinkNoise(5.0, 2);
        const auto delay = static_cast<size_t>(0.003 * kRate);
        {
            std::vector<double> r(st[1].size(), 0.0);
            for (size_t i = delay; i < r.size(); ++i) r[i] = st[1][i - delay] * 0.85;
            st[1] = r;
        }

        auto coherence = [](const std::vector<double>& a, const std::vector<double>& b)
        {
            // Long segments: the two channels are 3 ms apart, and a window
            // only a few times that long makes the estimator report the
            // window mismatch rather than the signals' relation.
            constexpr int n = 8192;
            dspark::FFTReal<double> fft(n);
            const auto w = hannWindow(n);
            const int bins = n / 2 + 1;
            std::vector<double> saa(static_cast<size_t>(bins), 0.0), sbb(static_cast<size_t>(bins), 0.0);
            std::vector<double> sabRe(static_cast<size_t>(bins), 0.0), sabIm(static_cast<size_t>(bins), 0.0);
            std::vector<double> fa(n), fb(n), za(n + 2), zb(n + 2);
            const size_t len = std::min(a.size(), b.size());
            for (size_t s = 0; s + n <= len; s += n / 2)
            {
                for (int i = 0; i < n; ++i)
                {
                    fa[static_cast<size_t>(i)] = a[s + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
                    fb[static_cast<size_t>(i)] = b[s + static_cast<size_t>(i)] * w[static_cast<size_t>(i)];
                }
                fft.forward(fa.data(), za.data());
                fft.forward(fb.data(), zb.data());
                for (int k = 0; k < bins; ++k)
                {
                    const double ar = za[static_cast<size_t>(2 * k)], ai = za[static_cast<size_t>(2 * k + 1)];
                    const double br = zb[static_cast<size_t>(2 * k)], bi = zb[static_cast<size_t>(2 * k + 1)];
                    saa[static_cast<size_t>(k)] += ar * ar + ai * ai;
                    sbb[static_cast<size_t>(k)] += br * br + bi * bi;
                    sabRe[static_cast<size_t>(k)] += ar * br + ai * bi;
                    sabIm[static_cast<size_t>(k)] += ai * br - ar * bi;
                }
            }
            const int lo = static_cast<int>(std::ceil(20.0 * n / kRate));
            const int hi = static_cast<int>(std::floor(16000.0 * n / kRate));
            double sum = 0.0;
            int used = 0;
            for (int k = lo; k <= hi && k < bins; ++k)
            {
                const double den = saa[static_cast<size_t>(k)] * sbb[static_cast<size_t>(k)];
                if (den < 1e-24) continue;
                const double num = sabRe[static_cast<size_t>(k)] * sabRe[static_cast<size_t>(k)]
                                 + sabIm[static_cast<size_t>(k)] * sabIm[static_cast<size_t>(k)];
                sum += num / den; ++used;
            }
            return sum / std::max(used, 1);
        };

        const double ildIn = db(rms(st[0], 0, st[0].size())) - db(rms(st[1], 0, st[1].size()));
        const double cohIn = coherence(st[0], st[1]);
        for (size_t ti = 0; ti < std::size(targets); ++ti)
        {
            // one engine path: the split has no public switch here
            {
                Options opt; opt.ratio = targets[ti].ratio;
                const auto out = stretchOffline(st, opt);
                const size_t skip = 8192;
                const double ildOut = db(rms(out[0], skip, out[0].size() - skip))
                                    - db(rms(out[1], skip, out[1].size() - skip));
                const double cohOut = coherence(out[0], out[1]);
                f << fmt(targets[ti].ratio, 6) << ','
                  << fmt(ildIn, 4) << ',' << fmt(ildOut, 4) << ','
                  << fmt(ildOut - ildIn, 4) << ',' << fmt(cohIn, 5) << ',' << fmt(cohOut, 5) << '\n';
                { rows[ti].ild = ildOut - ildIn; rows[ti].coherence = cohOut; }
            }
        }
        if (!transaction.finishOutput(f, "c5-stereo-integrity.csv"))
            return transaction.abortAndReport();
    }

    // -------------------------------------------------------------- summary
    {
        std::ofstream f;
        if (!transaction.openOutput(f, "timestretch-metrics.md"))
            return transaction.abortAndReport();
        f << "# TimeStretch characterisation (48 kHz, 2048 frame)\n\n"
             "Machine-generated by `tools/characterize_timestretch.cpp`. Every number\n"
             "here is at 48 kHz with the default 2048-sample frame; the CSV files\n"
             "beside this one carry the plain-vocoder control as well.\n\n"
             "| Target | out/in length | ratio error % | LSD dB | SC dB | spurious dB |"
             " transient vs WSOLA | attack exp % | vertical coherence | dILD dB | stereo coherence |\n"
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& r : rows)
        {
            f << "| " << r.label << " | " << r.outLen << "/" << r.wantLen << " | "
              << fmt(r.ratioError, 4) << " | " << fmt(r.lsd, 3) << " | " << fmt(r.sc, 1) << " | "
              << fmt(r.spurious, 1) << " | " << fmt(r.transientKeep, 3) << " | "
              << fmt(r.attack, 1) << " | "
              << fmt(r.vcohLocked, 4) << " | "
              << fmt(r.ild, 3) << " | " << fmt(r.coherence, 4) << " |\n";
        }
        f << "\nPhase locking against the plain vocoder on the same bed. The\n"
             "round-trip consistency ratio is reported because it is the stated\n"
             "measure, and the peak-to-neighbour phase coherence beside it because\n"
             "the round-trip is pinned at its numerical ceiling for every real\n"
             "signal and cannot tell the two apart.\n\n"
             "| Target | consistency out/in | vertical coherence locked | plain | locked better |\n"
             "|---|---:|---:|---:|---|\n";
        for (const auto& r : rows)
            f << "| " << r.label << " | "
              << fmt(r.consistencyOut / std::max(r.consistencyIn, 1e-30), 4) << " | "
              << fmt(r.vcohLocked, 5) << " | " << fmt(r.vcohPlain, 5) << " | "
              << (r.vcohLocked > r.vcohPlain ? "yes" : "NO") << " |\n";
        if (!transaction.finishOutput(f, "timestretch-metrics.md"))
            return transaction.abortAndReport();
    }

    if (!transaction.commit())
        return transaction.reportFailure();

    std::printf("characterisation written to %s\n",
                outputDirectory.string().c_str());
    return 0;
}
