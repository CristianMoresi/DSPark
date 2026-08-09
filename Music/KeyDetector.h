// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file KeyDetector.h
 * @brief Musical key detection: chroma accumulation + key-profile correlation.
 *
 * The Krumhansl-Schmuckler method. A 12-bin chroma vector is accumulated over
 * the programme and correlated (Pearson) against a published key profile
 * rotated to all 12 tonics in both modes; the best of those 24 correlations
 * names the key.
 *
 * The chroma comes from `ChordDetector`, which owns the framework's Goertzel
 * chroma front end: one exact Goertzel per note over MIDI 36..83, folded by
 * pitch class. This class holds one and consumes its frames rather than
 * running a second bank, so the analysis window, the automatic window law and
 * the measured register bounds are the same numbers in both places and cannot
 * drift apart. Everything prepare() documents about the reliable register
 * applies here unchanged: material below the leakage floor or above the
 * MIDI 83 bin ceiling is not analysed accurately, and a key estimate over such
 * material is an estimate over a chroma that is already wrong.
 *
 * Frames are accumulated with EQUAL WEIGHT, not by energy. Krumhansl's input
 * vector is the total DURATION each pitch class sounds, so a loud passage must
 * not outvote a long one; each analysis frame is normalized before it is added.
 * Frames far below the loudest frame seen are dropped instead of normalized,
 * because normalizing near-silence turns its noise floor into a full-strength
 * chroma vote.
 *
 * What the estimate cannot do, stated up front rather than absorbed into a
 * tolerance:
 *
 * - A key and its RELATIVE (A minor against C major) share all seven pitch
 *   classes. Only the profile's weighting of tonic, third and fifth separates
 *   them, so material that does not dwell on its tonic triad -- and modal
 *   material, which dwells on a different one -- lands on the relative reading.
 *   getKey() therefore also reports the runner-up, and a caller that sees the
 *   relative pair as the top two should treat the mode as unresolved rather
 *   than trusting it.
 * - Modal material has no correct answer in a 24-key output. D Dorian is the
 *   white notes with D as the centre; this class can only offer D minor or
 *   F major, and which one wins is decided by how much the passage dwells on
 *   the raised sixth, not by anything the method knows about modes.
 *
 * Threading: prepare() is setup-thread only (allocates). processBlock /
 * pushSamples and reset() belong to the thread that owns the stream.
 * setProfile() may be called from any thread and takes effect at the next
 * analysis frame. getKey() is a lock-free readout safe from any thread: the
 * whole result travels in one packed atomic word, so a reader can never see a
 * tonic from one estimate beside a confidence from another. chroma() is NOT
 * such a readout -- it hands back a reference to state owned by the thread
 * that pushes samples, and belongs to that thread.
 *
 * Dependencies: ChordDetector.h, HarmonyConstants.h, AudioSpec.h,
 * AudioBuffer.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "ChordDetector.h"
#include "HarmonyConstants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace dspark {

/**
 * @class KeyDetector
 * @brief Key estimation by profile correlation over an accumulated chroma.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class KeyDetector final
{
public:
    /**
     * @brief Which published key profile to correlate against.
     *
     * The two are different objects, not two spellings of one:
     *
     * - KrumhanslKessler: the probe-tone ratings of Krumhansl and Kessler
     *   (1982), listed as the "K-S model" profiles in Temperley, "Bayesian
     *   Models of Musical Structure and Cognition", Musicae Scientiae 8(2)
     *   (2004), Table 1. Perceptual data, unequal spacing, and the widest
     *   spread between diatonic and chromatic degrees.
     * - Temperley: the revised profiles of Temperley, "What's Key for Key?
     *   The Krumhansl-Schmuckler Key-Finding Algorithm Reconsidered", Music
     *   Perception 17(1) (1999), Figure 4, reprinted as the "CBMS model"
     *   column of Temperley 2004 Table 1. Hand-adjusted rather than measured,
     *   with every chromatic degree flattened to one value and the leading
     *   tone raised.
     *
     * Beware the name in other software: several widely used libraries label
     * their "Temperley" profile with the corpus-derived Kostka-Payne
     * probabilities of Temperley, "Music and Probability" (2007), which are a
     * different vector entirely. The values here are the 1999 ones the
     * architecture's citation names.
     */
    enum class Profile : std::uint8_t
    {
        KrumhanslKessler = 0,   ///< Krumhansl & Kessler 1982 probe-tone ratings.
        Temperley               ///< Temperley 1999 revised profiles.
    };

    /** @brief One key estimate. */
    struct Key
    {
        int tonicPitchClass = -1;      ///< 0 = C ... 11 = B; -1 = nothing yet.
        bool isMinor = false;          ///< true = minor mode.
        T confidence = T(0);           ///< [0, 1]; see getKey().
        int runnerUpPitchClass = -1;   ///< Second-best key, same encoding.
        bool runnerUpMinor = false;    ///< Mode of the second-best key.
    };

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the analysis pipeline.
     * @param spec       Audio environment specification.
     * @param windowSize Analysis window in samples, passed straight to the
     *                   shared chroma front end. Values <= 0 (the default)
     *                   select the AUTOMATIC window: the smallest power of two
     *                   in [1024, 16384] spanning at least 4096/48000 s
     *                   (~85 ms) at spec.sampleRate -- 4096 at 44.1/48 kHz,
     *                   8192 at 88.2/96 kHz, 16384 at 176.4/192 kHz. Explicit
     *                   values are clamped to [1024, 16384].
     *
     * The register the estimate is trustworthy in is the front end's, because
     * the chroma is the front end's. At the automatic window the Hann main lobe
     * stays at or below 46.9 Hz for every rate up to 192 kHz, which keeps notes
     * from about F#3 to E5 resolved; below that floor adjacent-semitone leakage
     * adds pitch classes that were never played, and above MIDI 83 (B5) there
     * are no bins at all, so an octave-6 melody contributes nothing. A key
     * estimate is a sum over those bins and inherits every one of those limits.
     * See ChordDetector::prepare() for the measured register table.
     */
    void prepare(const AudioSpec& spec, int windowSize = 0)
    {
        if (!spec.isValid()) return;
        front_.prepare(spec, windowSize);
        prepared_.store(true, std::memory_order_release);
        reset();
    }

    /**
     * @brief Clears the accumulated chroma and forgets the estimate.
     *
     * Allocation-free, but it rewrites the stream state: call it from the
     * thread that owns the stream (or while processing is stopped), not
     * concurrently with processBlock()/pushSamples().
     */
    void reset() noexcept
    {
        front_.reset();
        accum_.fill(0.0);
        chromaOut_.fill(T(0));
        maxFrameTotal_ = 0.0;
        lastFrame_ = 0;
        framesUsed_ = 0;
        packed_.store(pack(Key {}), std::memory_order_relaxed);
    }

    /**
     * @brief Selects the key profile (default KrumhanslKessler).
     *
     * Callable from any thread; it takes effect at the next analysis frame,
     * like every other parameter in the framework. It does not disturb the
     * accumulated chroma, so switching profiles re-reads the same evidence.
     */
    void setProfile(Profile p) noexcept
    {
        profile_.store(static_cast<std::uint8_t>(p), std::memory_order_relaxed);
    }

    /** @return The profile in effect. */
    [[nodiscard]] Profile getProfile() const noexcept
    {
        return static_cast<Profile>(profile_.load(std::memory_order_relaxed));
    }

    /**
     * @return The analysis window (samples) in effect -- the automatic choice
     *         if prepare() was given windowSize <= 0, the clamped explicit
     *         request otherwise. The estimate updates every windowSize/2
     *         samples; it is an accumulation, so it also keeps improving for
     *         as long as material is fed.
     */
    [[nodiscard]] int getWindowSize() const noexcept { return front_.getWindowSize(); }

    // -- Processing -------------------------------------------------------------------

    /**
     * @brief Feeds a block (channels averaged to mono). Allocation-free.
     *
     * The block is handed to the front end in chunks no longer than one hop, so
     * every analysis frame it produces is seen: frames are exactly one hop
     * apart, and a chunk that long can contain at most one of them.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_acquire)) return;
        const int hop = front_.getHopSize();
        const int n = buffer.getNumSamples();
        if (hop <= 0 || n <= 0) return;
        for (int off = 0; off < n; off += hop)
        {
            front_.processBlock(buffer.getSubView(off, std::min(hop, n - off)));
            consumeFrames();
        }
    }

    /** @brief Feeds mono samples directly. Allocation-free. */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_.load(std::memory_order_acquire)) return;
        const int hop = front_.getHopSize();
        if (hop <= 0) return;
        const auto step = static_cast<std::size_t>(hop);
        for (std::size_t off = 0; off < samples.size(); off += step)
        {
            front_.pushSamples(samples.subspan(off, std::min(step, samples.size() - off)));
            consumeFrames();
        }
    }

    // -- Readout ----------------------------------------------------------------------

    /**
     * @return The current key estimate. Lock-free and safe from any thread.
     *
     * `confidence` is the margin of the winning correlation over the runner-up,
     * (top - runnerUp) / top, clamped to [0, 1]. It answers one question --
     * how far ahead the winner is -- and deliberately not "is this the right
     * key": a passage that is genuinely between two keys reports a low margin
     * whether or not the winner is correct, and a short excerpt that happens to
     * fit one profile well reports a high one. Where the top two are a relative
     * pair the margin is the one number that says so, together with the
     * runner-up fields themselves.
     *
     * The clamp is not cosmetic. The runner-up correlation can be negative, in
     * which case the raw ratio exceeds 1; and where the winning correlation is
     * itself zero or negative -- silence, noise, or a chroma with no tonal
     * structure at all -- the ratio has no meaning and the confidence is
     * reported as 0 rather than as whatever the division produced.
     */
    [[nodiscard]] Key getKey() const noexcept
    {
        return unpack(packed_.load(std::memory_order_relaxed));
    }

    /**
     * @brief The accumulated chroma the estimate was formed from.
     *
     * Twelve bins, index 0 = C, normalized to sum to 1 so it can be compared
     * against a profile directly and does not grow with programme length. All
     * zeros before the first frame is accumulated.
     *
     * Returns a reference to state owned by the thread that pushes samples: it
     * is valid on that thread only, between its own calls. A caller on another
     * thread would be reading it while that thread writes it.
     */
    [[nodiscard]] const std::array<T, 12>& chroma() const noexcept { return chromaOut_; }

    /** @return Analysis frames accumulated since prepare()/reset(). */
    [[nodiscard]] std::uint64_t getFrameCount() const noexcept { return framesUsed_; }

    /**
     * @brief Writes a key name ("C", "F#m", "Bbm") into `dest`.
     * @param key  Key to name.
     * @param dest Destination buffer.
     * @param size Capacity of dest (8+ recommended).
     * @return Characters written, excluding the terminator.
     *
     * Spelling follows the framework's own convention for the tonic, so a key
     * whose tonic is usually written flat is written flat.
     */
    static int getKeyName(const Key& key, char* dest, int size) noexcept
    {
        if (dest == nullptr || size <= 0) return 0;
        if (key.tonicPitchClass < 0 || key.tonicPitchClass > 11)
        {
            dest[0] = '\0';
            return 0;
        }
        const std::string_view root =
            harmony::noteName(key.tonicPitchClass, key.tonicPitchClass);
        int len = 0;
        for (char c : root)
        {
            if (len >= size - 1) break;
            dest[len++] = c;
        }
        if (key.isMinor && len < size - 1) dest[len++] = 'm';
        dest[len] = '\0';
        return len;
    }

    // -- Published key profiles -------------------------------------------------------

    /**
     * @brief Krumhansl & Kessler (1982) probe-tone ratings, major.
     *
     * Index 0 is the tonic, ascending chromatically. Verified digit for digit
     * against Temperley, Musicae Scientiae 8(2) (2004), Table 1, column
     * "K-S model / major", which reprints them; the same twelve numbers appear
     * again in that paper's worked example of the Krumhansl-Schmuckler model.
     */
    static constexpr std::array<double, 12> kKrumhanslKesslerMajor {
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    };

    /** @brief Krumhansl & Kessler (1982) probe-tone ratings, minor.
     *
     * Same source and column "K-S model / minor". Note the two features that
     * make the mode decidable at all: the raised third of the relative major
     * (index 4) drops to 2.60 while the minor third (index 3) carries 5.38, and
     * the subtonic (index 10, 3.34) outranks the leading tone (index 11, 3.17),
     * which is the natural-minor scale asserting itself over the harmonic one.
     */
    static constexpr std::array<double, 12> kKrumhanslKesslerMinor {
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    };

    /** @brief Temperley (1999) revised profiles, major.
     *
     * From "What's Key for Key? The Krumhansl-Schmuckler Key-Finding Algorithm
     * Reconsidered", Music Perception 17(1), Figure 4; reprinted as the
     * "CBMS model / major" column of Temperley 2004 Table 1. Every chromatic
     * degree sits at 2.0 except the subtonic at 1.5, and the leading tone is
     * raised to 4.0 -- the deliberate departure from the measured ratings.
     */
    static constexpr std::array<double, 12> kTemperleyMajor {
        5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0
    };

    /** @brief Temperley (1999) revised profiles, minor.
     *
     * Same source, minor panel. It assumes the HARMONIC minor scale, so the
     * flat sixth (index 8) is diatonic at 3.5 and the natural sixth (index 9)
     * is chromatic at 2.0 -- the opposite of the Krumhansl-Kessler minor
     * ordering, and the reason the two profiles disagree most on modal
     * material.
     */
    static constexpr std::array<double, 12> kTemperleyMinor {
        5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0
    };

private:
    /**
     * @brief Frames quieter than this fraction of the loudest frame seen are
     *        dropped instead of accumulated.
     *
     * Equal-weight accumulation normalizes every frame to the same total, which
     * is exactly what turns a near-silent frame into a full-strength vote for
     * whatever its noise floor happens to look like. The gate is relative to
     * the programme's own loudest frame rather than absolute, so it means the
     * same thing at any recording level.
     */
    static constexpr double kFrameGateFraction = 1.0e-3;

    /// Absolute floor below which a frame carries no usable direction at all.
    static constexpr double kFrameEnergyFloor = 1.0e-12;

    void consumeFrames() noexcept
    {
        const std::uint64_t fc = front_.getFrameCount();
        if (fc == lastFrame_) return;
        lastFrame_ = fc;

        const std::array<T, 12>& frame = front_.getChroma();
        double total = 0.0;
        for (const T v : frame) total += static_cast<double>(v);
        if (!(total > kFrameEnergyFloor)) return;

        if (total > maxFrameTotal_) maxFrameTotal_ = total;
        if (total < kFrameGateFraction * maxFrameTotal_) return;

        // Equal weight per frame: the input to the method is how LONG each
        // pitch class sounds, so a frame's own level must not scale its vote.
        const double inv = 1.0 / total;
        for (int pc = 0; pc < 12; ++pc)
            accum_[static_cast<std::size_t>(pc)] +=
                static_cast<double>(frame[static_cast<std::size_t>(pc)]) * inv;
        ++framesUsed_;

        estimate();
    }

    void estimate() noexcept
    {
        double sum = 0.0;
        for (const double v : accum_) sum += v;
        if (!(sum > 0.0)) return;

        std::array<double, 12> x {};
        for (int pc = 0; pc < 12; ++pc)
        {
            x[static_cast<std::size_t>(pc)] = accum_[static_cast<std::size_t>(pc)] / sum;
            chromaOut_[static_cast<std::size_t>(pc)] =
                static_cast<T>(x[static_cast<std::size_t>(pc)]);
        }

        const bool temperley = (getProfile() == Profile::Temperley);
        const auto& major = temperley ? kTemperleyMajor : kKrumhanslKesslerMajor;
        const auto& minor = temperley ? kTemperleyMinor : kKrumhanslKesslerMinor;

        double best = -2.0, second = -2.0;
        int bestPc = -1, secondPc = -1;
        bool bestMinor = false, secondMinor = false;

        for (int mode = 0; mode < 2; ++mode)
        {
            const auto& p = (mode == 0) ? major : minor;
            for (int tonic = 0; tonic < 12; ++tonic)
            {
                const double r = pearsonRotated(x, p, tonic);
                if (r > best)
                {
                    second = best; secondPc = bestPc; secondMinor = bestMinor;
                    best = r; bestPc = tonic; bestMinor = (mode == 1);
                }
                else if (r > second)
                {
                    second = r; secondPc = tonic; secondMinor = (mode == 1);
                }
            }
        }

        Key key;
        key.tonicPitchClass = bestPc;
        key.isMinor = bestMinor;
        key.runnerUpPitchClass = secondPc;
        key.runnerUpMinor = secondMinor;
        // Undefined where the winner is not itself a positive correlation:
        // dividing by a zero or negative top would turn "no tonal structure"
        // into a large number of the wrong sign.
        key.confidence = (best > 0.0)
                       ? static_cast<T>(std::clamp((best - second) / best, 0.0, 1.0))
                       : T(0);
        packed_.store(pack(key), std::memory_order_relaxed);
    }

    /**
     * @brief Pearson correlation of `x` against `profile` rotated to `tonic`.
     *
     * The profile's own mean and spread are recomputed here rather than cached:
     * a rotation permutes the vector, so they are the same for all twelve
     * rotations, and this is 24 twelve-element passes per analysis frame.
     */
    [[nodiscard]] static double pearsonRotated(const std::array<double, 12>& x,
                                               const std::array<double, 12>& profile,
                                               int tonic) noexcept
    {
        double mx = 0.0, mp = 0.0;
        for (int i = 0; i < 12; ++i)
        {
            mx += x[static_cast<std::size_t>(i)];
            mp += profile[static_cast<std::size_t>(i)];
        }
        mx /= 12.0;
        mp /= 12.0;

        double num = 0.0, dx = 0.0, dp = 0.0;
        for (int pc = 0; pc < 12; ++pc)
        {
            const double a = x[static_cast<std::size_t>(pc)] - mx;
            const double b = profile[static_cast<std::size_t>((pc - tonic + 12) % 12)] - mp;
            num += a * b;
            dx  += a * a;
            dp  += b * b;
        }
        const double den = std::sqrt(dx * dp);
        return (den > 0.0) ? (num / den) : 0.0;
    }

    // One packed word so a reader never sees a tonic from one estimate beside
    // a confidence from another: bits 0-15 confidence, 16 mode, 17-20 tonic+1,
    // 21 runner-up mode, 22-25 runner-up tonic+1.
    [[nodiscard]] static std::uint64_t pack(const Key& k) noexcept
    {
        const auto q = static_cast<std::uint64_t>(
            std::clamp(static_cast<double>(k.confidence), 0.0, 1.0) * 65535.0 + 0.5);
        const auto t  = static_cast<std::uint64_t>(
            std::clamp(k.tonicPitchClass, -1, 11) + 1);
        const auto ru = static_cast<std::uint64_t>(
            std::clamp(k.runnerUpPitchClass, -1, 11) + 1);
        return q
             | (static_cast<std::uint64_t>(k.isMinor ? 1 : 0) << 16)
             | (t << 17)
             | (static_cast<std::uint64_t>(k.runnerUpMinor ? 1 : 0) << 21)
             | (ru << 22);
    }

    [[nodiscard]] static Key unpack(std::uint64_t v) noexcept
    {
        Key k;
        k.confidence = static_cast<T>(static_cast<double>(v & 0xFFFFu) / 65535.0);
        k.isMinor = ((v >> 16) & 1u) != 0u;
        k.tonicPitchClass = static_cast<int>((v >> 17) & 0xFu) - 1;
        k.runnerUpMinor = ((v >> 21) & 1u) != 0u;
        k.runnerUpPitchClass = static_cast<int>((v >> 22) & 0xFu) - 1;
        return k;
    }

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "audio-thread stores must not lock");

    ChordDetector<T> front_;
    std::atomic<bool> prepared_ { false };

    std::array<double, 12> accum_ {};
    std::array<T, 12> chromaOut_ {};
    double maxFrameTotal_ = 0.0;
    std::uint64_t lastFrame_ = 0;
    std::uint64_t framesUsed_ = 0;

    std::atomic<std::uint64_t> packed_ { 0 };
    std::atomic<std::uint8_t> profile_ { static_cast<std::uint8_t>(Profile::KrumhanslKessler) };
};

} // namespace dspark
