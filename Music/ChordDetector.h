// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file ChordDetector.h
 * @brief Real-time chord detection: Goertzel chroma + template matching.
 *
 * Classic music-information-retrieval pipeline, allocation-free:
 *
 * 1. A mono sum is windowed (Hann) every hop and analyzed with one exact
 *    Goertzel per note over MIDI 36..83 (four octaves, C2..B5): each bin sits
 *    exactly on its tempered frequency, avoiding the FFT's fixed-grid
 *    quantization. Frequency RESOLUTION, however, equals an FFT of the same
 *    length: the Hann main lobe spans ~4*fs/windowSize -- 46.9 Hz at 48 kHz
 *    with a 4096 window, but 93.8 Hz at 96 kHz and 187.5 Hz at 192 kHz for
 *    that same window -- so every register bound scales with fs/windowSize
 *    and is meaningless without its sample rate. By default prepare()
 *    therefore derives the window from the rate (constant ~85 ms span:
 *    4096 at 44.1/48 kHz, 8192 at 88.2/96 kHz, 16384 at 176.4/192 kHz),
 *    which keeps the reliable register at ~F#3..E5 for root-position triads
 *    from 44.1 to 192 kHz. See prepare() for the measured register table
 *    and the rules for explicit window sizes.
 * 2. Note energies fold into a 12-bin chroma vector.
 * 3. The chroma is cosine-matched against chord templates (major, minor,
 *    diminished, augmented, sus2, sus4, dom7, maj7, min7, half-dim7) at all
 *    12 roots; the winner and its margin over the runner-up produce a
 *    confidence in [0, 1].
 *
 * The reading is gated: while confidence is below the threshold the last
 * confident chord is held, so brief transients and silences do not flicker
 * the display.
 *
 * Threading: prepare() is setup-thread only (allocates). processBlock /
 * pushSamples and reset() belong to the thread that owns the stream.
 * setConfidenceThreshold() may be called from any thread, and getChord()
 * is a lock-free readout safe from any thread (single packed atomic word,
 * never torn). getChroma() and getFrameCount() expose the shared front end
 * to a second consumer and are NOT cross-thread readouts: getChroma() is a
 * stream-owner reference readout, valid on the thread that pushes samples and
 * between that thread's own calls, and a caller on any other thread would be
 * reading it while the owning thread writes it.
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
#include <cstddef>
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
    enum class ChordType : std::uint8_t
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
     * @param windowSize Analysis window in samples. Values <= 0 (the default)
     *                   select the AUTOMATIC window: the smallest power of
     *                   two in [1024, 16384] spanning at least 4096/48000 s
     *                   (~85 ms) at spec.sampleRate -- 4096 at 44.1/48 kHz,
     *                   8192 at 88.2/96 kHz, 16384 at 176.4/192 kHz.
     *                   Explicit values are clamped to [1024, 16384].
     *                   Rates that cannot place the highest analysed note
     *                   (MIDI 83, B5) below Nyquist are rejected and leave a
     *                   previous valid configuration untouched.
     *
     * REGISTER BOUNDS -- every number below is measured (a sweep of
     * root-position pure-tone major triads, roots MIDI 36..84);
     * "reliable" means correct root AND chord type through the 0.55
     * confidence gate:
     *
     * - Upper bound (every configuration): the analysis bins stop at
     *   MIDI 83 (B5), so chord tones above B5 are invisible and
     *   root-position triads with roots above E5 (MIDI 76) are NEVER
     *   detected. Above that ceiling, exactly as below the F#3 floor, the
     *   window-fill transient can still pass the confidence gate with a WRONG
     *   chord (measured: a C6 root-position major triad latches B Maj7 at
     *   ~0.58 during the first window, then never again), and the
     *   hold-last-confident rule keeps that reading on display indefinitely.
     *   Call reset() on programme change, and treat a Result whose confidence
     *   is below the threshold as STALE rather than as a current reading.
     * - Lower bound: adjacent-semitone leakage from the Hann main lobe,
     *   ~4*fs/windowSize Hz wide. The floor is a property of that RATIO,
     *   not of the rate alone; measured floors:
     *     4*fs/N =  46.9 Hz (48k/4096, 96k/8192, 192k/16384) -> F#3..E5
     *     4*fs/N =  43.1 Hz (44.1k/4096)                     -> E3..E5
     *     4*fs/N =  93.8 Hz (96k/4096, 192k/8192)            -> F#4..E5
     *     4*fs/N >= 187.5 Hz (48k/1024, 192k/4096)           -> EMPTY
     *     4*fs/N =  23.4 Hz (48k/8192, 96k/16384)            -> C2..E5, gaps
     *     4*fs/N <= 11.7 Hz (44.1k or 48k with 16384)        -> C2..E5
     * - Below the floor the detector does not merely lose confidence:
     *   leakage adds phantom adjacent semitones that match a richer
     *   template and can pass the 0.55 gate, producing a CONFIDENTLY WRONG
     *   reading (measured: C3 major at 48 kHz/4096 reads C Maj7 at ~0.58;
     *   C4 major at 96 kHz with a forced 4096 window reads C Maj7 at
     *   ~0.57). Treat the bounds as hard usage limits, not soft advice.
     * - The AUTOMATIC window keeps 4*fs/windowSize <= 46.9 Hz for every
     *   rate up to 192 kHz, so the F#3..E5 register (use ~G3 up as a
     *   comfortable margin) holds at 44.1, 48, 88.2, 96, 176.4 and
     *   192 kHz alike. Above 192 kHz the 16384 ceiling widens the lobe
     *   again (93.8 Hz at 384 kHz: reliable only F#4..E5, and C4 again
     *   misreads as Maj7).
     * - Small explicit windows at professional rates have NO reliable
     *   register: 1024 at 48 kHz detects 0 of 49 swept roots (several
     *   confidently wrong) because its 187.5 Hz lobe exceeds the semitone
     *   spacing of every note below the bin ceiling. Explicit 1024/2048
     *   windows are only meaningful at low rates (1024 spans 128 ms at
     *   8 kHz and resolves G3/C4 majors exactly); requests below 1024
     *   clamp INTO 1024 and inherit all of this. Prefer the automatic
     *   window.
     */
    void prepare(const AudioSpec& spec, int windowSize = 0)
    {
        // Conservative no-op on invalid specs (NaN rate included): a hot
        // detector keeps its previous configuration instead of going deaf.
        if (!spec.isValid()) return;
        const double highestNote =
            440.0 * std::exp2((kFirstMidi + kNumNotes - 1 - 69) / 12.0);
        if (!(highestNote < spec.sampleRate * 0.5)) return;
        sampleRate_ = spec.sampleRate;
        if (windowSize <= 0)
        {
            // Automatic: hold the analysis TIME SPAN constant across sample
            // rates (the span 4096 samples cover at 48 kHz). The Hann main
            // lobe spans ~4/timeSpan Hz, so a constant span pins the
            // frequency resolution -- and with it the reliable register --
            // instead of letting it degrade as fs rises. Rounding up to a
            // power of two only narrows the lobe further.
            const double target = sampleRate_ * (4096.0 / 48000.0);
            int n = 1024;
            while (n < 16384 && static_cast<double>(n) < target) n *= 2;
            windowSize_ = n;
        }
        else
        {
            windowSize_ = std::clamp(windowSize, 1024, 16384);
        }
        hopSize_ = windowSize_ / 2;

        ring_.assign(static_cast<std::size_t>(windowSize_), T(0));
        writePos_ = 0;
        sinceHop_ = 0;

        window_.resize(static_cast<std::size_t>(windowSize_));
        WindowFunctions<T>::hann(window_.data(), windowSize_, true);
        scratch_.resize(static_cast<std::size_t>(windowSize_));

        for (int n = 0; n < kNumNotes; ++n)
        {
            const double freq = 440.0 * std::exp2((kFirstMidi + n - 69) / 12.0);
            notes_[static_cast<std::size_t>(n)].prepare(sampleRate_, freq, windowSize_);
        }

        prepared_.store(true, std::memory_order_release);
        reset();
    }

    /**
     * @brief Clears the analysis ring and forgets the held chord.
     *
     * Allocation-free, but it rewrites the stream state: call it from the
     * thread that owns the stream (or while processing is stopped), not
     * concurrently with processBlock()/pushSamples().
     */
    void reset() noexcept
    {
        std::fill(ring_.begin(), ring_.end(), T(0));
        writePos_ = 0;
        sinceHop_ = 0;
        chromaFrame_.fill(T(0));
        frameCount_ = 0;
        packed_.store(pack(Result {}), std::memory_order_relaxed);
    }

    /**
     * @brief Confidence below which the previous chord is held (default 0.55).
     *
     * Callable from any thread. Non-finite values are ignored (a NaN would
     * make every comparison false and freeze the detector on the held chord
     * forever).
     */
    void setConfidenceThreshold(float threshold) noexcept
    {
        if (!std::isfinite(threshold)) return;
        threshold_.store(std::clamp(threshold, 0.0f, 1.0f), std::memory_order_relaxed);
    }

    /** @return The current confidence-gating threshold. */
    [[nodiscard]] float getConfidenceThreshold() const noexcept
    {
        return threshold_.load(std::memory_order_relaxed);
    }

    /**
     * @return The analysis window (samples) in effect: the automatic choice
     *         if prepare() was called with windowSize <= 0, the clamped
     *         explicit request otherwise (member default 4096 before the
     *         first successful prepare()). Latency is one full window;
     *         readings update every windowSize/2 samples.
     */
    [[nodiscard]] int getWindowSize() const noexcept { return windowSize_; }

    /**
     * @return Samples between consecutive analysis frames (windowSize/2).
     *
     * A second consumer of this front end feeds it in chunks of at most this
     * many samples and polls getFrameCount(): frames are exactly this far
     * apart, so a chunk that long spans at most one of them and none can be
     * missed.
     */
    [[nodiscard]] int getHopSize() const noexcept { return hopSize_; }

    // -- Shared chroma front end (stream-owner thread) ---------------------------

    /**
     * @brief Number of analysis frames produced since prepare()/reset().
     *
     * Increments once per hop, so a caller can tell a fresh chroma frame from
     * the one it already consumed.
     *
     * Reads plain state owned by the thread that pushes samples: call it from
     * that thread only. It is NOT a cross-thread readout -- getChord() is the
     * one that is.
     */
    [[nodiscard]] std::uint64_t getFrameCount() const noexcept { return frameCount_; }

    /**
     * @brief The chroma vector of the most recent analysis frame.
     *
     * Twelve bins of summed note ENERGY (Goertzel magnitude squared) over
     * MIDI 36..83, folded by pitch class with index 0 = C. Raw, unnormalized
     * and in the units the analysis produces, because a consumer that
     * accumulates frames needs to choose its own weighting -- normalizing here
     * would destroy the frame-to-frame level information and pre-empt that
     * choice. All zeros before the first frame.
     *
     * The register the numbers are trustworthy in is the register documented
     * on prepare(): energy below the leakage floor or above the MIDI 83 bin
     * ceiling is not present in these bins, and adjacent-semitone leakage is.
     *
     * stream-owner reference readout: the reference is to state owned by the
     * thread that calls processBlock()/pushSamples(), and it is valid on that
     * thread only, between that thread's own calls. A caller on any other
     * thread would be reading these words while the owning thread writes them.
     * getChord() is the readout for any other thread.
     */
    [[nodiscard]] const std::array<T, 12>& getChroma() const noexcept { return chromaFrame_; }

    // -- Processing -------------------------------------------------------------------

    /** @brief Feeds a block (channels averaged to mono). */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_acquire)) return;
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
        if (!prepared_.load(std::memory_order_acquire)) return;
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
        ring_[static_cast<std::size_t>(writePos_)] = sample;
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
            scratch_[static_cast<std::size_t>(i)] = ring_[static_cast<std::size_t>(idx)]
                                             * window_[static_cast<std::size_t>(i)];
        }

        // Note energies -> chroma, tracking the lowest sounding note: the
        // bass is the standard root disambiguator (e.g. Dsus4 and Gsus2 are
        // the same pitch-class set; the bass decides which one you played).
        std::array<double, 12> chroma {};
        std::array<double, static_cast<std::size_t>(kNumNotes)> noteE {};
        double total = 0.0, maxNote = 0.0;
        for (int n = 0; n < kNumNotes; ++n)
        {
            auto& g = notes_[static_cast<std::size_t>(n)];
            g.reset();
            g.processBlock(scratch_.data(), windowSize_);
            const double e = static_cast<double>(g.getMagnitude());
            noteE[static_cast<std::size_t>(n)] = e * e;
            chroma[static_cast<std::size_t>((kFirstMidi + n) % 12)] += e * e;
            total += e * e;
            maxNote = std::max(maxNote, e * e);
        }
        // Publish the frame for consumers of this front end before any of the
        // chord-specific work, and count it even when the frame turns out to
        // be silent: a consumer polls the count to tell a new frame from the
        // previous one, so a frame that fails to increment it would be lost.
        for (int pc = 0; pc < 12; ++pc)
            chromaFrame_[static_cast<std::size_t>(pc)] =
                static_cast<T>(chroma[static_cast<std::size_t>(pc)]);
        ++frameCount_;

        // Lowest LOCAL maximum: window-lobe leakage spreads energy onto
        // neighbouring semitones, so a plain threshold would pick a sidelobe.
        int bassPc = -1;
        for (int n = 0; n < kNumNotes; ++n)
        {
            const double e = noteE[static_cast<std::size_t>(n)];
            const double prev = (n > 0) ? noteE[static_cast<std::size_t>(n - 1)] : 0.0;
            const double next = (n + 1 < kNumNotes) ? noteE[static_cast<std::size_t>(n + 1)] : 0.0;
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
                        inSum += chroma[static_cast<std::size_t>((root + iv) % 12)];
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
    [[nodiscard]] static std::uint64_t pack(const Result& r) noexcept
    {
        const auto conf = static_cast<std::uint32_t>(std::clamp(r.confidence, 0.0f, 1.0f) * 65535.0f);
        return (static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.rootPitchClass + 1)) << 24)
             | (static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.type)) << 16)
             | conf;
    }

    [[nodiscard]] static Result unpack(std::uint64_t v) noexcept
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
    std::atomic<bool> prepared_ { false };

    std::vector<T> ring_, window_, scratch_;
    int writePos_ = 0;
    int sinceHop_ = 0;

    std::array<Goertzel<T>, static_cast<std::size_t>(kNumNotes)> notes_;

    // Latest analysis frame, owned by the thread that pushes samples.
    std::array<T, 12> chromaFrame_ {};
    std::uint64_t frameCount_ = 0;

    std::atomic<std::uint64_t> packed_ { 0 };
    std::atomic<float> threshold_ { 0.55f };
};

} // namespace dspark
