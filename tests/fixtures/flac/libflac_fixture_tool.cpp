// DSPark FLAC fixture oracle - generation use only
// Copyright (c) 2026 Cristian Moresi - MIT License

#include <FLAC/stream_decoder.h>
#include <FLAC/stream_encoder.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct DecodeState
{
    std::ofstream output;
    bool failed = false;
};

void writeLe32(std::ostream& output, std::int32_t value)
{
    const auto bits = static_cast<std::uint32_t>(value);
    const std::array<char, 4> bytes {
        static_cast<char>(bits), static_cast<char>(bits >> 8),
        static_cast<char>(bits >> 16), static_cast<char>(bits >> 24)
    };
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::int32_t readLe32(const std::array<unsigned char, 4>& bytes)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
    return static_cast<std::int32_t>(bits);
}

FLAC__StreamDecoderWriteStatus writeCallback(
    const FLAC__StreamDecoder*, const FLAC__Frame* frame,
    const FLAC__int32* const channels[], void* clientData)
{
    auto& state = *static_cast<DecodeState*>(clientData);
    for (unsigned sample = 0; sample < frame->header.blocksize; ++sample)
        for (unsigned channel = 0; channel < frame->header.channels; ++channel)
            writeLe32(state.output, channels[channel][sample]);
    if (!state.output)
    {
        state.failed = true;
        return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

void metadataCallback(const FLAC__StreamDecoder*, const FLAC__StreamMetadata*, void*) {}

void errorCallback(const FLAC__StreamDecoder*, FLAC__StreamDecoderErrorStatus, void* clientData)
{
    static_cast<DecodeState*>(clientData)->failed = true;
}

bool decode(const char* inputPath, const char* outputPath)
{
    DecodeState state;
    state.output.open(outputPath, std::ios::binary | std::ios::trunc);
    if (!state.output) return false;

    FLAC__StreamDecoder* decoder = FLAC__stream_decoder_new();
    if (decoder == nullptr) return false;
    const auto init = FLAC__stream_decoder_init_file(
        decoder, inputPath, writeCallback, metadataCallback, errorCallback, &state);
    const bool ok = init == FLAC__STREAM_DECODER_INIT_STATUS_OK
        && FLAC__stream_decoder_process_until_end_of_stream(decoder)
        && FLAC__stream_decoder_get_state(decoder) == FLAC__STREAM_DECODER_END_OF_STREAM
        && !state.failed;
    FLAC__stream_decoder_finish(decoder);
    FLAC__stream_decoder_delete(decoder);
    state.output.close();
    return ok && state.output.good();
}

bool encode(const char* inputPath, const char* outputPath,
            unsigned sampleRate, unsigned channels, unsigned bitsPerSample)
{
    if (channels == 0 || channels > 8 || bitsPerSample < 4 || bitsPerSample > 32)
        return false;
    std::ifstream input(inputPath, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff byteCount = input.tellg();
    const std::uint64_t stride = static_cast<std::uint64_t>(channels) * 4;
    if (byteCount < 0 || static_cast<std::uint64_t>(byteCount) % stride != 0)
        return false;
    const std::uint64_t samples = static_cast<std::uint64_t>(byteCount) / stride;
    if (samples == 0 || samples > std::numeric_limits<unsigned>::max()) return false;

    input.seekg(0);
    std::vector<FLAC__int32> pcm(static_cast<std::size_t>(samples) * channels);
    for (auto& value : pcm)
    {
        std::array<unsigned char, 4> bytes {};
        input.read(reinterpret_cast<char*>(bytes.data()), 4);
        if (!input) return false;
        value = readLe32(bytes);
    }

    FLAC__StreamEncoder* encoder = FLAC__stream_encoder_new();
    if (encoder == nullptr) return false;
    bool ok = FLAC__stream_encoder_set_channels(encoder, channels)
        && FLAC__stream_encoder_set_bits_per_sample(encoder, bitsPerSample)
        && FLAC__stream_encoder_set_sample_rate(encoder, sampleRate)
        && FLAC__stream_encoder_set_total_samples_estimate(encoder, samples)
        && FLAC__stream_encoder_set_compression_level(encoder, 8);
    if (ok && samples >= 16)
        ok = FLAC__stream_encoder_set_blocksize(
            encoder, static_cast<unsigned>(samples > 64 ? 64 : samples));
    if (ok)
        ok = FLAC__stream_encoder_init_file(encoder, outputPath, nullptr, nullptr)
            == FLAC__STREAM_ENCODER_INIT_STATUS_OK;
    if (ok)
        ok = FLAC__stream_encoder_process_interleaved(
            encoder, pcm.data(), static_cast<unsigned>(samples));
    if (ok) ok = FLAC__stream_encoder_finish(encoder);
    FLAC__stream_encoder_delete(encoder);
    return ok;
}

unsigned parseUnsigned(const char* text)
{
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<unsigned>::max())
        return 0;
    return static_cast<unsigned>(value);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--version")
    {
        std::cout << FLAC__VERSION_STRING << '\n';
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "decode")
        return decode(argv[2], argv[3]) ? 0 : 1;
    if (argc == 8 && std::string(argv[1]) == "encode")
    {
        const unsigned sampleRate = parseUnsigned(argv[4]);
        const unsigned channels = parseUnsigned(argv[5]);
        const unsigned bitsPerSample = parseUnsigned(argv[6]);
        const unsigned expectedSamples = parseUnsigned(argv[7]);
        if (sampleRate == 0 || expectedSamples == 0) return 2;
        std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
        if (!input || static_cast<std::uint64_t>(input.tellg())
                != static_cast<std::uint64_t>(expectedSamples) * channels * 4)
            return 2;
        return encode(argv[2], argv[3], sampleRate, channels, bitsPerSample) ? 0 : 1;
    }
    std::cerr << "usage: libflac_fixture_tool decode INPUT.flac OUTPUT.pcm\n"
                 "       libflac_fixture_tool encode INPUT.pcm OUTPUT.flac RATE CHANNELS BITS SAMPLES\n"
                 "       libflac_fixture_tool --version\n";
    return 2;
}
