// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file ChordDetector.h
 * @brief Real-time chord detection: multirate note chroma + template matching.
 *
 * Classic music-information-retrieval pipeline, allocation-free:
 *
 * 1. A mono sum is analysed every hop (half a window) at the 48 tempered
 *    notes of MIDI 36..83 (four octaves, C2..B5): each note is the Hann-
 *    windowed DTFT evaluated exactly on its tempered frequency, the value a
 *    Goertzel filter over the window would return, avoiding the FFT's
 *    fixed-grid quantization. Frequency RESOLUTION, however, equals an FFT of
 *    the same length: the Hann main lobe spans ~4*fs/windowSize -- 46.9 Hz at
 *    48 kHz with a 4096 window, but 93.8 Hz at 96 kHz and 187.5 Hz at 192 kHz
 *    for that same window -- so every register bound scales with
 *    fs/windowSize and is meaningless without its sample rate. By default
 *    prepare() therefore derives the window from the rate (constant ~85 ms
 *    span: 4096 at 44.1/48 kHz, 8192 at 88.2/96 kHz, 16384 at 176.4/192 kHz),
 *    which keeps the reliable register at ~F#3..E5 for root-position triads
 *    from 44.1 to 192 kHz. See prepare() for the measured register table and
 *    the rules for explicit window sizes.
 * 2. Note energies fold into a 12-bin chroma vector.
 * 3. The chroma is cosine-matched against chord templates (major, minor,
 *    diminished, augmented, sus2, sus4, dom7, maj7, min7, half-dim7) at all
 *    12 roots; the winner and its margin over the runner-up produce a
 *    confidence in [0, 1].
 *
 * HOW STEP 1 IS COMPUTED. Every analysed note lies below 1 kHz, so the
 * window holds far more samples than its content needs. The input runs
 * through a cascade of linear-phase half-band decimators, and each octave is
 * evaluated at the lowest rate that keeps its top note, plus eight bins of
 * the window's leakage, below a quarter of that rate: two, four or five
 * times fewer samples per octave and level. The window of an octave is the
 * same span of time sampled at its own rate, with the same periodic Hann
 * taken at those points, and each note is one dot product against a table
 * that folds in the window, the tempered frequency and the decimators' gain
 * at that note. Measured on three ballroom recordings against the direct
 * full-rate evaluation of the same span, notes within 30 dB of a frame's
 * loudest differ by at most 0.005 dB at 44.1 kHz with the automatic window
 * (0.008 dB worst over 44.1, 48 and 96 kHz and windows of 8192 and 16384),
 * and the held chord and its confidence are the same on every frame. The cost
 * per frame falls from 304 us to 2.1 us (float, 44.1 kHz, 4096), from 614 us
 * to 2.9 us at 96 kHz with 8192, and from 1214 us to 5.7 us with 16384. The
 * decimators delay the analysis: each octave's window is read further back
 * in its own buffer until all four describe the same span, which ends
 * getAnalysisDelaySamples() before the frame boundary -- 770 samples
 * (17.5 ms) at 44.1 kHz with the automatic window.
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
 * Dependencies: HarmonyConstants.h, AudioSpec.h, AudioBuffer.h, DspMath.h,
 * SimdOps.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SimdOps.h"
#include "HarmonyConstants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class ChordDetector
 * @brief Monophonic-buffer chord recognition with confidence gating.
 *
 * Threading: getChord() is the atomic foreign-thread publication. getChroma()
 * is a stream-owner reference readout and is valid only on the thread that
 * owns processBlock() / pushSamples(), between that thread's own calls. Calling
 * getChroma() from another thread while processing runs would race the
 * non-atomic chroma storage.
 *
 * @tparam T Sample type (float or double). The note bank runs in T: the
 *           decimated rates keep every analysed note at least a twentieth of
 *           its own rate away from DC, where float dot products over at most
 *           1024 samples lose nothing the chroma resolves.
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
     *
     * @return true when the specification was accepted and state was reset;
     *         false when the complete previous state was preserved.
     */
    bool prepare(const AudioSpec& spec, int windowSize = 0)
    {
        // Conservative no-op on invalid specs (NaN rate included): a hot
        // detector keeps its previous configuration instead of going deaf.
        if (!spec.isValid() || !std::isfinite(spec.sampleRate)) return false;
        const double highestNote = noteHz(kFirstMidi + kNumNotes - 1);
        if (!(spec.sampleRate > highestNote * 2.0)) return false;
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

        designBank();

        prepared_.store(true, std::memory_order_release);
        reset();
        return true;
    }

    /**
     * @brief Clears the analysis state and forgets the held chord.
     *
     * Allocation-free, but it rewrites the stream state: call it from the
     * thread that owns the stream (or while processing is stopped), not
     * concurrently with processBlock()/pushSamples().
     */
    void reset() noexcept
    {
        std::fill(input_.begin(), input_.end(), T(0));
        inputFill_ = 0;
        for (auto& s : stages_)
        {
            std::fill(s.even.begin(), s.even.end(), T(0));
            std::fill(s.odd.begin(), s.odd.end(), T(0));
        }
        for (auto& o : octaves_) std::fill(o.history.begin(), o.history.end(), T(0));
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
     *         first successful prepare()). Readings update every
     *         windowSize/2 samples.
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

    /**
     * @return How far before the frame boundary the analysed span ends, in
     *         input samples: the group delay of the decimators that feed the
     *         slowest octave (see the file header). A frame computed when the
     *         stream reaches sample n describes the window ending this many
     *         samples before n, to within half a sample of that octave's
     *         rate. Fixed for a given rate and window; 0 before prepare().
     */
    [[nodiscard]] int getAnalysisDelaySamples() const noexcept { return analysisDelay_; }

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
     * Twelve bins of summed note ENERGY (the squared magnitude of the
     * Hann-windowed DTFT at each tempered note, normalised as a Goertzel
     * magnitude is, by 2 over the window length) over MIDI 36..83, folded by
     * pitch class with index 0 = C. Raw, unnormalized and in the units the
     * analysis produces, because a consumer that accumulates frames needs to
     * choose its own weighting -- normalizing here would destroy the
     * frame-to-frame level information and pre-empt that choice. All zeros
     * before the first frame.
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
        int done = 0;
        while (done < nS)
        {
            const int take = std::min(nS - done, hopSize_ - inputFill_);
            T* dst = input_.data() + inputFill_;
            for (int i = 0; i < take; ++i)
            {
                T m = T(0);
                for (int ch = 0; ch < nCh; ++ch)
                    m += buffer.getChannel(ch)[done + i] * invCh;
                dst[i] = m;
            }
            done += take;
            advance(take);
        }
    }

    /** @brief Feeds mono samples directly. */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_.load(std::memory_order_acquire)) return;
        size_t done = 0;
        while (done < samples.size())
        {
            const size_t take = std::min(samples.size() - done,
                                         static_cast<size_t>(hopSize_ - inputFill_));
            std::memcpy(input_.data() + inputFill_, samples.data() + done, take * sizeof(T));
            done += take;
            advance(static_cast<int>(take));
        }
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
        static constexpr const char* kSuffix[11] = {
            "", "", "m", "dim", "aug", "sus2", "sus4", "7", "maj7", "m7", "m7b5"
        };
        int len = 0;
        for (const char c : harmony::noteName(result.rootPitchClass, result.rootPitchClass))
            if (len < size - 1) dest[len++] = c;
        for (const char* p = kSuffix[static_cast<int>(result.type)]; *p && len < size - 1; ++p)
            dest[len++] = *p;
        dest[len] = '\0';
        return len;
    }

private:
    static constexpr int kFirstMidi = 36;   ///< C2.
    static constexpr int kNumNotes = 48;    ///< Four octaves.
    static constexpr int kNumOctaves = kNumNotes / 12;
    static constexpr int kNumTemplates = 10;

    /// Leakage protected around the band of each octave, in analysis bins
    /// (fs / windowSize). Folded decimator junk lands at least this far from
    /// every note of the octave, where the Hann sidelobes are below -60 dB.
    static constexpr double kMarginBins = 8.0;
    /// An octave is decimated while the next rate stays at least this many
    /// times its protected band edge: a half-band stage then needs only a
    /// short filter, since its transition spans a fifth of its input rate.
    static constexpr double kRateSeparation = 2.5;
    /// Stopband of every half-band stage, dB.
    static constexpr double kStopbandDb = 80.0;
    /// Shortest decimated window an octave may be read over.
    static constexpr int kMinOctaveWindow = 32;
    /// Deepest decimation (input rate / 2^kMaxLevels).
    static constexpr int kMaxLevels = 9;

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

    /**
     * @brief One half-band decimate-by-two stage, run a hop at a time in
     *        polyphase form.
     *
     * The filter has 4K+3 taps: a centre tap and K+1 odd taps either side,
     * every even tap but the centre being zero. Split into its two phases
     * the output is the centre tap on one phase plus a symmetric
     * (2K+2)-tap filter on the other, and both run over contiguous arrays
     * through the block FIR kernel of SimdOps.h.
     */
    struct Stage
    {
        int pairs = 1;             ///< K + 1.
        T centre = T(0.5);
        std::vector<T> taps;       ///< The K + 1 odd taps, nearest first.
        std::vector<T> kernel;     ///< taps mirrored: h_K .. h_0, h_0 .. h_K.
        std::vector<T> even, odd;  ///< Phases of history + one hop of input.
        [[nodiscard]] int historyHalf() const noexcept { return 2 * pairs - 1; } ///< (4K+2)/2.
        [[nodiscard]] int delay() const noexcept { return 2 * pairs - 1; }       ///< Centre, input samples.
    };

    /// One octave's analysis at its own rate.
    struct Octave
    {
        int level = 0;             ///< Decimation level (rate fs / 2^level).
        int length = 0;            ///< Window length at that rate.
        int offset = 0;            ///< Samples read back so all octaves align.
        std::vector<T> history;    ///< Last length + offset samples, oldest first.
        std::vector<T> rows;       ///< 24 rows x length: scaled Hann*cos, Hann*sin.
    };

    [[nodiscard]] static double noteHz(int midi) noexcept
    {
        return 440.0 * std::exp2((midi - 69) / 12.0);
    }

    [[nodiscard]] static double besselI0(double x) noexcept
    {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k < 64; ++k)
        {
            term *= (0.5 * x) / static_cast<double>(k);
            sum += term * term;
            if (term * term < 1e-17 * sum) break;
        }
        return sum;
    }

    /// Kaiser-window half-band whose band below @p passHz is protected from
    /// aliasing when decimating from @p rate (stopband from rate/2 - pass).
    static void designStage(Stage& s, double passHz, double rate)
    {
        const double width = std::max(0.02, (0.5 * rate - 2.0 * passHz) / rate);
        const double beta = 0.1102 * (kStopbandDb - 8.7);
        const double length = (kStopbandDb - 8.0) / (2.285 * twoPi<double> * width) + 1.0;
        const int k = std::clamp(static_cast<int>(std::ceil((length - 3.0) / 4.0)), 0, 64);
        s.pairs = k + 1;
        const double half = static_cast<double>(2 * k + 1);   // (L - 1) / 2
        std::vector<double> h(static_cast<size_t>(s.pairs));
        double sum = 0.5;
        for (int j = 0; j < s.pairs; ++j)
        {
            const int n = 2 * j + 1;
            const double r = static_cast<double>(n) / half;
            const double w = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r)))
                           / besselI0(beta);
            const double sinc = ((j & 1) ? -1.0 : 1.0) / (pi<double> * static_cast<double>(n));
            h[static_cast<size_t>(j)] = sinc * w;
            sum += 2.0 * h[static_cast<size_t>(j)];
        }
        s.taps.resize(static_cast<size_t>(s.pairs));
        for (int j = 0; j < s.pairs; ++j)
            s.taps[static_cast<size_t>(j)] = static_cast<T>(h[static_cast<size_t>(j)] / sum);
        s.centre = static_cast<T>(0.5 / sum);
        s.kernel.resize(static_cast<size_t>(2 * s.pairs));
        for (int j = 0; j < s.pairs; ++j)
        {
            s.kernel[static_cast<size_t>(s.pairs - 1 - j)] = s.taps[static_cast<size_t>(j)];
            s.kernel[static_cast<size_t>(s.pairs + j)] = s.taps[static_cast<size_t>(j)];
        }
    }

    /// Zero-phase magnitude of a stage at @p hz for input rate @p rate.
    [[nodiscard]] static double stageGain(const Stage& s, double hz, double rate) noexcept
    {
        double a = static_cast<double>(s.centre);
        for (int j = 0; j < s.pairs; ++j)
            a += 2.0 * static_cast<double>(s.taps[static_cast<size_t>(j)])
               * std::cos(twoPi<double> * hz / rate * static_cast<double>(2 * j + 1));
        return std::abs(a);
    }

    void designBank()
    {
        const double fs = sampleRate_;
        const double bin = fs / static_cast<double>(windowSize_);
        std::array<double, kNumOctaves> pass {};
        levels_ = 0;
        for (int o = 0; o < kNumOctaves; ++o)
        {
            pass[static_cast<size_t>(o)] = noteHz(kFirstMidi + 12 * o + 11) + kMarginBins * bin;
            int d = 0;
            while (d < kMaxLevels
                   && fs / std::exp2(d + 1) >= kRateSeparation * pass[static_cast<size_t>(o)]
                   && (windowSize_ >> (d + 1)) >= kMinOctaveWindow)
                ++d;
            octaves_[static_cast<size_t>(o)].level = d;
            levels_ = std::max(levels_, d);
        }

        stages_.assign(static_cast<size_t>(levels_), Stage {});
        for (int k = 0; k < levels_; ++k)
        {
            double p = 0.0;
            for (int o = 0; o < kNumOctaves; ++o)
                if (octaves_[static_cast<size_t>(o)].level > k)
                    p = std::max(p, pass[static_cast<size_t>(o)]);
            Stage& s = stages_[static_cast<size_t>(k)];
            designStage(s, p, fs / std::exp2(k));
            const size_t phase = static_cast<size_t>(s.historyHalf() + (hopSize_ >> (k + 1)));
            s.even.assign(phase, T(0));
            s.odd.assign(phase, T(0));
        }

        // Group delay to each level, in input samples, and the alignment of
        // every octave's window onto the slowest one's span. The window of
        // level d ends (2^d - 1) - delay(d) samples from the frame boundary,
        // so reading `offset` samples further back moves it 2^d each.
        auto levelDelay = [&](int d) {
            int g = 0;
            for (int s = 0; s < d; ++s) g += stages_[static_cast<size_t>(s)].delay() << s;
            return g - ((1 << d) - 1);
        };
        int target = 0;
        for (const Octave& o : octaves_) target = std::max(target, levelDelay(o.level));
        analysisDelay_ = target;

        for (int o = 0; o < kNumOctaves; ++o)
        {
            Octave& oc = octaves_[static_cast<size_t>(o)];
            const int d = oc.level;
            oc.length = windowSize_ >> d;
            const double step = std::exp2(d);
            oc.offset = std::max(0, static_cast<int>(std::lround(
                static_cast<double>(target - levelDelay(d)) / step)));
            oc.history.assign(static_cast<size_t>(oc.length + oc.offset), T(0));

            const double rate = fs / step;
            const double norm = 2.0 / static_cast<double>(oc.length);
            oc.rows.assign(static_cast<size_t>(24 * oc.length), T(0));
            for (int k = 0; k < 12; ++k)
            {
                const double hz = noteHz(kFirstMidi + 12 * o + k);
                double gain = 1.0;
                for (int s = 0; s < d; ++s)
                    gain *= stageGain(stages_[static_cast<size_t>(s)], hz, fs / std::exp2(s));
                const double scale = norm / std::max(gain, 1e-6);
                const double w = twoPi<double> * hz / rate;
                T* cosRow = oc.rows.data() + static_cast<size_t>(k) * static_cast<size_t>(oc.length);
                T* sinRow = oc.rows.data() + static_cast<size_t>(12 + k) * static_cast<size_t>(oc.length);
                for (int i = 0; i < oc.length; ++i)
                {
                    const double hann = 0.5 - 0.5 * std::cos(twoPi<double> * static_cast<double>(i)
                                                             / static_cast<double>(oc.length));
                    const double a = w * static_cast<double>(i);
                    cosRow[i] = static_cast<T>(scale * hann * std::cos(a));
                    sinRow[i] = static_cast<T>(scale * hann * std::sin(a));
                }
            }
        }

        input_.assign(static_cast<size_t>(hopSize_), T(0));
        for (int d = 1; d <= levels_; ++d)
            levelBlock_[static_cast<size_t>(d)].assign(static_cast<size_t>(hopSize_ >> d), T(0));
    }

    /// Accounts for @p count samples just written into the hop being collected.
    void advance(int count) noexcept
    {
        inputFill_ += count;
        if (inputFill_ >= hopSize_)
        {
            inputFill_ = 0;
            analyze();
        }
    }

    /// Decimates one hop of input (n samples, even) through @p s into @p y.
    static void runStage(Stage& s, const T* x, int n, T* DSPARK_RESTRICT y) noexcept
    {
        constexpr int W = simd::kVecWidth<T>;
        using O = simd::Vec<T, W>;
        const int hist = s.historyHalf();
        T* e = s.even.data();
        T* o = s.odd.data();
        const int outN = n / 2;
        int m = 0;
        for (; m + W <= outN; m += W)
        {
            typename O::V ev, od;
            O::loadDeinterleave(x + 2 * m, ev, od);
            O::store(e + hist + m, ev);
            O::store(o + hist + m, od);
        }
        for (; m < outN; ++m)
        {
            e[hist + m] = x[2 * m];
            o[hist + m] = x[2 * m + 1];
        }
        // Output j: the symmetric odd taps over odd samples j .. j + 2K + 1,
        // plus the centre tap on the even sample between them, j + K + 1.
        simd::firCorrelate(o, s.kernel.data(), 2 * s.pairs, y, outN, T(1));
        const T* centre = e + s.pairs;
        const auto c = O::set1(s.centre);
        int k = 0;
        for (; k + W <= outN; k += W)
            O::store(y + k, O::madd(c, O::load(centre + k), O::load(y + k)));
        for (; k < outN; ++k)
        {
            // The vector multiply-add's rounding at the tail too.
            T lanes[W];
            O::store(lanes, O::madd(c, O::set1(centre[k]), O::set1(y[k])));
            y[k] = lanes[0];
        }
        std::memmove(e, e + outN, static_cast<size_t>(hist) * sizeof(T));
        std::memmove(o, o + outN, static_cast<size_t>(hist) * sizeof(T));
    }

    void analyze() noexcept
    {
        // Decimate the hop through every level that an octave reads.
        const T* source = input_.data();
        int n = hopSize_;
        for (int d = 0; d < levels_; ++d)
        {
            T* out = levelBlock_[static_cast<size_t>(d + 1)].data();
            runStage(stages_[static_cast<size_t>(d)], source, n, out);
            source = out;
            n /= 2;
        }

        // Note energies -> chroma, tracking the lowest sounding note: the
        // bass is the standard root disambiguator (e.g. Dsus4 and Gsus2 are
        // the same pitch-class set; the bass decides which one you played).
        std::array<double, 12> chroma {};
        std::array<double, static_cast<std::size_t>(kNumNotes)> noteE {};
        double total = 0.0, maxNote = 0.0;
        for (int o = 0; o < kNumOctaves; ++o)
        {
            Octave& oc = octaves_[static_cast<size_t>(o)];
            const int fresh = hopSize_ >> oc.level;
            const int keep = oc.length + oc.offset - fresh;
            T* h = oc.history.data();
            std::memmove(h, h + fresh, static_cast<size_t>(keep) * sizeof(T));
            std::memcpy(h + keep, oc.level == 0 ? input_.data()
                                                : levelBlock_[static_cast<size_t>(oc.level)].data(),
                        static_cast<size_t>(fresh) * sizeof(T));
            const T* window = h;   // the oldest `length` of length + offset
            for (int k = 0; k < 12; ++k)
            {
                const T* cosRow = oc.rows.data() + static_cast<size_t>(k) * static_cast<size_t>(oc.length);
                const T* sinRow = oc.rows.data() + static_cast<size_t>(12 + k) * static_cast<size_t>(oc.length);
                const double re = static_cast<double>(simd::dotProduct(cosRow, window, oc.length));
                const double im = static_cast<double>(simd::dotProduct(sinRow, window, oc.length));
                const double e = re * re + im * im;
                const int n12 = 12 * o + k;
                noteE[static_cast<std::size_t>(n12)] = e;
                chroma[static_cast<std::size_t>((kFirstMidi + n12) % 12)] += e;
                total += e;
                maxNote = std::max(maxNote, e);
            }
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
        for (int n12 = 0; n12 < kNumNotes; ++n12)
        {
            const double e = noteE[static_cast<std::size_t>(n12)];
            const double prev = (n12 > 0) ? noteE[static_cast<std::size_t>(n12 - 1)] : 0.0;
            const double next = (n12 + 1 < kNumNotes) ? noteE[static_cast<std::size_t>(n12 + 1)] : 0.0;
            if (e > 0.15 * maxNote && e >= prev && e >= next)
            {
                bassPc = (kFirstMidi + n12) % 12;
                break;
            }
        }
        // A NaN sample travels through the decimators and the dot products
        // into the energies; the comparisons below are false for NaN, so it
        // is caught here like silence rather than published.
        if (!(total >= 1e-12) || !std::isfinite(total))
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
    int levels_ = 0;
    int analysisDelay_ = 0;
    std::atomic<bool> prepared_ { false };

    std::vector<T> input_;     ///< The hop being collected.
    int inputFill_ = 0;
    std::vector<Stage> stages_;
    std::array<std::vector<T>, static_cast<std::size_t>(kMaxLevels + 1)> levelBlock_;
    std::array<Octave, static_cast<std::size_t>(kNumOctaves)> octaves_;

    // Latest analysis frame, owned by the thread that pushes samples.
    std::array<T, 12> chromaFrame_ {};
    std::uint64_t frameCount_ = 0;

    std::atomic<std::uint64_t> packed_ { 0 };
    std::atomic<float> threshold_ { 0.55f };
};

} // namespace dspark
