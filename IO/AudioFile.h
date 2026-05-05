// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AudioFile.h
 * @brief Abstract interface for reading and writing audio files.
 *
 * Defines a common contract for audio file I/O implementations (WAV, AIFF, etc.).
 * Focuses on offline processing and buffer preparation.
 *
 * @warning DANGER: Disk I/O is non-deterministic and blocking. NEVER call any
 * methods from this class inside the real-time audio thread. Do all file 
 * operations in a secondary thread or during the initialization phase.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"

#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <limits>

namespace dspark {

/**
 * @struct AudioFileInfo
 * @brief Metadata describing an audio file's format and dimensions.
 */
struct AudioFileInfo
{
    /** @brief Sample rate in Hz (e.g., 44100.0, 48000.0, 96000.0). */
    double sampleRate = 44100.0;

    /** @brief Number of audio channels (1 = mono, 2 = stereo). */
    uint32_t numChannels = 0;

    /** @brief Total number of sample frames in the file. */
    int64_t numSamples = 0;

    /** @brief Bits per sample in the stored format (8, 16, 24, 32, 64). */
    uint32_t bitsPerSample = 16;

    /** @brief True if the file stores floating-point samples (IEEE 754). */
    bool isFloatingPoint = false;

    /**
     * @brief Converts this file info to an AudioSpec for processor preparation.
     *
     * @warning If processing offline in a single block, passing 0 will attempt 
     * to use the full file length. This can cause massive memory allocations 
     * for large files.
     *
     * @param defaultBlockSize Desired block size. If 0, uses the entire file length (offline mode).
     * @return AudioSpec configured with the file's sample rate and channels.
     */
    [[nodiscard]] AudioSpec toSpec(int defaultBlockSize = 0) const noexcept
    {
        // Clamp to avoid integer overflow if numSamples exceeds 2.14B (INT_MAX)
        int safeFullLength = static_cast<int>(std::min<int64_t>(numSamples, std::numeric_limits<int>::max()));
        
        return {
            .sampleRate   = sampleRate,
            .maxBlockSize = (defaultBlockSize > 0) ? defaultBlockSize : safeFullLength,
            .numChannels  = static_cast<int>(numChannels)
        };
    }
};

/**
 * @class AudioFile
 * @brief Abstract base class for audio file readers and writers.
 *
 * Provides a uniform API to transfer audio to/from AudioBufferView objects.
 * Samples are automatically normalized to the [-1.0, 1.0] float range.
 * 
 * @note Reading 32-bit integer PCM files into float buffers will result in a 
 * loss of the 8 least significant bits due to 24-bit mantissa limitations. 
 */
class AudioFile
{
public:
    virtual ~AudioFile() = default;

    /**
     * @brief Opens a file for reading.
     *
     * @param path File path using C++20 std::filesystem (handles UTF-8/cross-platform safely).
     * @return True if opened and header parsed successfully. False otherwise.
     */
    [[nodiscard]] virtual bool openRead(const std::filesystem::path& path) = 0;

    /**
     * @brief Opens a file for writing, creating it or overwriting if it exists.
     *
     * @param path File path.
     * @param info Desired output format metadata.
     * @return True if created and header written successfully. False otherwise.
     */
    [[nodiscard]] virtual bool openWrite(const std::filesystem::path& path, const AudioFileInfo& info) = 0;

    /**
     * @brief Retrieves metadata of the currently opened file.
     * @return AudioFileInfo struct. Returns default values if no file is open.
     */
    [[nodiscard]] virtual AudioFileInfo getInfo() const = 0;

    /**
     * @brief Reads all samples from the file into the destination buffer view.
     *
     * Excess buffer space is left untouched. It is the caller's responsibility 
     * to ensure the view has adequate capacity.
     *
     * @param dest Buffer view to receive the audio data.
     * @return True if read completely. False if an I/O error occurred.
     */
    [[nodiscard]] virtual bool readSamples(AudioBufferView<float> dest) = 0;

    /**
     * @brief Reads a specific range of sample frames. Useful for chunked streaming.
     *
     * @param dest Buffer view to receive the audio data.
     * @param startFrame The absolute frame index in the file to start reading from.
     * @param numFrames The number of frames to read.
     * @return True if the range was read successfully.
     */
    [[nodiscard]] virtual bool readSamples(AudioBufferView<float> dest,
                                           int64_t startFrame, int64_t numFrames) = 0;

    /**
     * @brief Writes samples from the view to the file.
     *
     * Converts float [-1.0, 1.0] back to the native bit-depth defined in openWrite().
     *
     * @param src Buffer view containing the data to write.
     * @return True on success, False if disk is full or an I/O error occurred.
     */
    [[nodiscard]] virtual bool writeSamples(AudioBufferView<const float> src) = 0;

    /**
     * @brief Finalizes file headers and releases system handles.
     * Safe to call multiple times. Automatically called on destruction.
     */
    virtual void close() = 0;

    /** @brief Checks if a valid file handle is currently open. */
    [[nodiscard]] virtual bool isOpen() const noexcept = 0;
};

} // namespace dspark