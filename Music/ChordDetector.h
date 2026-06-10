// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file ChordDetector.h
 * @brief Real-time chord detection: Goertzel chroma + template matching.
 *
 * Classic music-information-retrieval pipeline, allocation-free:
 *
 * 1. A mono sum is windowed (Hann) every hop and analyzed with one exact
 *    Goertzel per note over MIDI 36..83 (four octaves) — no FFT-grid
 *    compromise at low pitches.
 * 2. Note energies fold into a 12-bin chroma vector.
 * 3. The chroma is cosine-matched against chord templates (major, minor,
 *    diminished, augmented, sus2, sus4, dom7, maj7, min7, half-dim7) at all
 *    12 roots; the winner and its margin over the runner-up produce a
 *    confidence in [0, 1].
 *
 * The reading is gated: while confidence is below the threshold the last
 * confident chord is held, so brief transients and silences do not flicker
 * the display. Readout is lock-free from any thread.
 *
 * Dependencies: Goertzel.h, HarmonyConstants.h, AudioSpec.h, AudioBuffer.h,
 * WindowFunctions.h, DspMath.h.
 */

#include "../Analysis/Goertzel.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/WindowFunctions.h"
#include "HarmonyConstants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class ChordDetector
 * @brief Monophonic-buffer chord recognition with confidence gating.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class ChordDetector
{
public:
    /** @brief Recognized chord families. */
    enum class ChordType : uint8_t
    {
        None = 0, Major, Minor, Diminished, Augmented,
        Sus2, Sus4, Dominant7, Major7, Minor7, HalfDim7
    };

    /** @brief One detection result. */
    struct Result
    {
        int rootPitchClass = -1;       ///< 0 = C ... 11 = B; -1 = none.
        ChordType type = ChordType::None;
        float confidence = 0.0f;       ///< [0, 1].
    };

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the analysis pipeline.
     * @param spec       Audio environment specification.
     * @param windowSize Analysis window (default 4096: ~85 ms at 48 kHz).
     */
    void prepare(const AudioSpec& spec, int windowSize = 4096)
    {
        if (spec.sampleRate <= 0.0) return;
        sampleRate_ = spec.sampleRate;
        windowSize_ = std::clamp(windowSize, 1024, 16384);
        hopSize_ = windowSize_ / 2;

        ring_.assign(static_cast<size_t>(windowSize_), T(0));
        writePos_ = 0;
        sinceHop_ = 0;

        window_.resize(static_cast<size_t>(windowSize_));
        WindowFunctions<T>::hann(window_.data(), windowSize_, true);
        scratch_.resize(static_cast<size_t>(windowSize_));

        for (int n = 0; n < kNumNotes; ++n)
        {
            const double freq = 440.0 * std::exp2((kFirstMidi + n - 69) / 12.0);
            notes_[static_cast<size_t>(n)].prepare(sampleRate_, freq, windowSize_);
        }

        prepared_ = true;
        reset();
    }

    /** @brief Forgets the held chord. RT-safe. */
    void reset() noexcept
    {
        std::fill(ring_.begin(), ring_.end(), T(0));
        writePos_ = 0;
        sinceHop_ = 0;
        packed_.store(pack(Result {}), std::memory_order_relaxed);
    }

    /** @brief Confidence below which the previous chord is held (default 0.55). */
    void setConfidenceThreshold(float threshold) noexcept
    {
        threshold_.store(std::clamp(threshold, 0.0f, 1.0f), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Feeds a block (channels averaged to mono). */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_) return;
        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh <= 0) return;

        const T invCh = T(1) / static_cast<T>(nCh);
        for (int i = 0; i < nS; ++i)
        {
            T m = T(0);
            for (int ch = 0; ch < nCh; ++ch)
                m += buffer.getChannel(ch)[i] * invCh;
            push(m);
        }
    }

    /** @brief Feeds mono samples directly. */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_) return;
        for (const T s : samples)
            push(s);
    }

    // -- Readout (lock-free, any thread) ------------------------------------------------

    /** @return The current (possibly held) chord. */
    [[nodiscard]] Result getChord() const noexcept
    {
        return unpack(packed_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Writes a human-readable chord name ("C", "F#m7", "Bbsus4"...).
     * @param result Chord to name.
     * @param dest   Destination buffer.
     * @param size   Capacity of dest (8+ recommended).
     * @return Number of characters written (excluding the terminator).
     */
    static int getChordName(const Result& result, char* dest, int size) noexcept
    {
        if (size <= 0) return 0;
        if (result.rootPitchClass < 0 || result.type == ChordType::None)
        {
            dest[0] = '\0';
            return 0;
        }
        static constexpr const char* kRoots[12] = {
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
        };
        static constexpr const char* kSuffix[11] = {
            "", "", "m", "dim", "aug", "sus2", "sus4", "7", "maj7", "m7", "m7b5"
        };
        int len = 0;
        for (const char* p = kRoots[result.rootPitchClass]; *p && len < size - 1; ++p)
            dest[len++] = *p;
        for (const char* p = kSuffix[static_cast<int>(result.type)]; *p && len < size - 1; ++p)
            dest[len++] = *p;
        dest[len] = '\0';
        return len;
    }

private:
    static constexpr int kFirstMidi = 36;   ///< C2.
    static constexpr int kNumNotes = 48;    ///< Four octaves.
    static constexpr int kNumTemplates = 10;

    struct Template
    {
        ChordType type;
        harmony::NoteSet mask;     ///< Intervals from the root, bit 0 = root.
        int count;
    };

    static constexpr std::array<Template, kNumTemplates> kTemplates { {
        { ChordType::Major,      0b000010010001, 3 },   // 0 4 7
        { ChordType::Minor,      0b000010001001, 3 },   // 0 3 7
        { ChordType::Diminished, 0b000001001001, 3 },   // 0 3 6
        { ChordType::Augmented,  0b000100010001, 3 },   // 0 4 8
        { ChordType::Sus2,       0b000010000101, 3 },   // 0 2 7
        { ChordType::Sus4,       0b000010100001, 3 },   // 0 5 7
        { ChordType::Dominant7,  0b010010010001, 4 },   // 0 4 7 10
        { ChordType::Major7,     0b100010010001, 4 },   // 0 4 7 11
        { ChordType::Minor7,     0b010010001001, 4 },   // 0 3 7 10
        { ChordType::HalfDim7,   0b010001001001, 4 },   // 0 3 6 10
    } };

    void push(T sample) noexcept
    {
        ring_[static_cast<size_t>(writePos_)] = sample;
        writePos_ = (writePos_ + 1) % windowSize_;
        if (++sinceHop_ >= hopSize_)
        {
            sinceHop_ = 0;
            analyze();
        }
    }

    void analyze() noexcept
    {
        // Window the ring (oldest sample first).
        for (int i = 0; i < windowSize_; ++i)
        {
            const int idx = (writePos_ + i) % windowSize_;
            scratch_[static_cast<size_t>(i)] = ring_[static_cast<size_t>(idx)]
                                             * window_[static_cast<size_t>(i)];
        }

        // Note energies -> chroma, tracking the lowest sounding note: the
        // bass is the standard root disambiguator (e.g. Dsus4 and Gsus2 are
        // the same pitch-class set; the bass decides which one you played).
        std::array<double, 12> chroma {};
        std::array<double, static_cast<size_t>(kNumNotes)> noteE {};
        double total = 0.0, maxNote = 0.0;
        for (int n = 0; n < kNumNotes; ++n)
        {
            auto& g = notes_[static_cast<size_t>(n)];
            g.reset();
            g.processBlock(scratch_.data(), windowSize_);
            const double e = static_cast<double>(g.getMagnitude());
            noteE[static_cast<size_t>(n)] = e * e;
            chroma[static_cast<size_t>((kFirstMidi + n) % 12)] += e * e;
            total += e * e;
            maxNote = std::max(maxNote, e * e);
        }
        // Lowest LOCAL maximum: window-lobe leakage spreads energy onto
        // neighbouring semitones, so a plain threshold would pick a sidelobe.
        int bassPc = -1;
        for (int n = 0; n < kNumNotes; ++n)
        {
            const double e = noteE[static_cast<size_t>(n)];
            const double prev = (n > 0) ? noteE[static_cast<size_t>(n - 1)] : 0.0;
            const double next = (n + 1 < kNumNotes) ? noteE[static_cast<size_t>(n + 1)] : 0.0;
            if (e > 0.15 * maxNote && e >= prev && e >= next)
            {
                bassPc = (kFirstMidi + n) % 12;
                break;
            }
        }
        if (total < 1e-12)
        {
            // Silence: drop confidence but keep the last chord displayed.
            Result held = unpack(packed_.load(std::memory_order_relaxed));
            held.confidence = 0.0f;
            packed_.store(pack(held), std::memory_order_relaxed);
            return;
        }

        double norm = 0.0;
        for (const double c : chroma) norm += c * c;
        norm = std::sqrt(norm);

        // Cosine match against every template at every root.
        double best = 0.0, second = 0.0;
        int bestRoot = -1;
        ChordType bestType = ChordType::None;
        for (int root = 0; root < 12; ++root)
        {
            for (const auto& tpl : kTemplates)
            {
                double inSum = 0.0;
                for (int iv = 0; iv < 12; ++iv)
                    if (tpl.mask & (1u << iv))
                        inSum += chroma[static_cast<size_t>((root + iv) % 12)];
                double score = inSum / (norm * std::sqrt(static_cast<double>(tpl.count)));
                if (root == bassPc)
                    score *= 1.25;   // the bass note names the chord
                if (score > best)
                {
                    second = best;
                    best = score;
                    bestRoot = root;
                    bestType = tpl.type;
                }
                else if (score > second)
                {
                    second = score;
                }
            }
        }

        // Confidence: absolute quality times the margin over the runner-up.
        const double margin = (best > 1e-9) ? std::clamp((best - second) / best * 4.0, 0.0, 1.0)
                                            : 0.0;
        const auto confidence = static_cast<float>(std::clamp(best, 0.0, 1.0) * (0.5 + 0.5 * margin));

        Result out;
        if (confidence >= threshold_.load(std::memory_order_relaxed))
        {
            out.rootPitchClass = bestRoot;
            out.type = bestType;
            out.confidence = confidence;
        }
        else
        {
            out = unpack(packed_.load(std::memory_order_relaxed));   // hold
            out.confidence = confidence;
        }
        packed_.store(pack(out), std::memory_order_relaxed);
    }

    // Pack the result into one atomic word (no torn reads cross-thread).
    [[nodiscard]] static uint64_t pack(const Result& r) noexcept
    {
        const auto conf = static_cast<uint32_t>(std::clamp(r.confidence, 0.0f, 1.0f) * 65535.0f);
        return (static_cast<uint64_t>(static_cast<uint8_t>(r.rootPitchClass + 1)) << 24)
             | (static_cast<uint64_t>(static_cast<uint8_t>(r.type)) << 16)
             | conf;
    }

    [[nodiscard]] static Result unpack(uint64_t v) noexcept
    {
        Result r;
        r.rootPitchClass = static_cast<int>((v >> 24) & 0xFF) - 1;
        r.type = static_cast<ChordType>((v >> 16) & 0xFF);
        r.confidence = static_cast<float>(v & 0xFFFF) / 65535.0f;
        return r;
    }

    // -- Members --------------------------------------------------------------------
    double sampleRate_ = 48000.0;
    int windowSize_ = 4096;
    int hopSize_ = 2048;
    bool prepared_ = false;

    std::vector<T> ring_, window_, scratch_;
    int writePos_ = 0;
    int sinceHop_ = 0;

    std::array<Goertzel<T>, static_cast<size_t>(kNumNotes)> notes_;

    std::atomic<uint64_t> packed_ { 0 };
    std::atomic<float> threshold_ { 0.55f };
};

} // namespace dspark
