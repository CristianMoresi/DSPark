// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Sampler.h
 * @brief Polyphonic multi-zone sampler: alias-free transposition, crossfaded
 *        loops, voice stealing without clicks, sample-accurate events.
 *
 * A zone is one recording mapped over a key range and a velocity range, with
 * its root key, tuning, gain, pan and loop. Every zone that matches a note
 * sounds (layering). Each voice reads its zone through a windowed-sinc
 * interpolator whose cutoff follows the playback rate:
 *
 * - At or below the recording's own rate the kernel is a 64-tap Kaiser sinc
 *   with its passband to 0.40 and its stopband from 0.50 of the recording's
 *   sample rate at 100 dB, so a note played down keeps the recording's top
 *   octave and adds no images.
 * - Played up by a ratio r, each recording sample is instead scattered over
 *   the output samples around the instant it falls on, with the same kernel
 *   laid out in OUTPUT time, so its stopband starts at the output Nyquist
 *   frequency: whatever the recording holds above it is removed before it
 *   can fold. The weights come from the same contiguous polyphase rows, so a
 *   voice costs about r times an untransposed one. Transposition is capped
 *   at 8x (+36 semitones).
 *
 * Measured at 48 kHz: -118 to -130 dB THD+N on a 1 kHz tone from -12 to +19
 * semitones, with the amplitude exact; -110 dB of alias for an 18 kHz tone
 * played an octave up and -115 dB for a 20 kHz tone a fifth up. A pitch bend
 * that carries a voice across its recording's rate keeps the same guarantees.
 *
 * Loops are read through: a read past the loop end wraps to the loop start,
 * so a loop point needs no alignment to the kernel and no unrolled copy. A
 * loop crossfade is baked once, at load, into a separate copy of the region
 * that precedes the loop end, so the first pass and a release in Sustain mode
 * still play the recording untouched.
 *
 * Voices: prepare() fixes the polyphony. A note that finds every voice busy
 * steals one - the oldest released voice, else the quietest, else the oldest
 * - and the stolen voice fades out over 3 ms in one of the spare voices kept
 * for that purpose instead of being cut, so stealing never clicks. A note
 * already sounding on the same key is released, not cut, when it retriggers.
 *
 * Events carry a sample offset into the next processBlock() and are applied
 * at that sample, so a host passes its MIDI through unchanged and timing is
 * exact whatever the block size.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with anything else).
 * - setZones() / loadSample() / clearZones(): control thread, one at a time.
 *   They allocate and build the new zone set off the audio thread and hand it
 *   over through a lock-free exchange; the audio thread adopts it at the start
 *   of a block, lets the voices of the previous set finish, and hands that set
 *   back to be freed by the next control call (or the destructor). They never
 *   block the audio thread and the audio thread never frees memory.
 * - setEnvelope() / setVelocityRange() / setGain(): any thread, lock free
 *   (relaxed atomics read when a note starts or is released).
 * - noteOn() / noteOff() / setSustainPedal() / setPitchBend() / allNotesOff()
 *   and processBlock() / reset(): audio thread, the stream owner. Events are
 *   queued, allocation free, for the next processBlock().
 * - getActiveVoiceCount(): any thread, lock free.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O. The shared interpolation tables (about 1 MB) are
 * built once per process, on the first prepare().
 *
 * Dependencies: DspMath.h, AudioBuffer.h, AudioSpec.h, EnvelopeGenerator.h.
 */

#include "DspMath.h"
#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "EnvelopeGenerator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class Sampler
 * @brief Polyphonic multi-zone sample player.
 *
 * @code
 *   dspark::Sampler<float> sampler;
 *   sampler.prepare(spec, 32);                     // 32 voices
 *   sampler.loadSample(piano, 48000.0, 60);        // one zone, root C4
 *   // audio thread, per block:
 *   sampler.noteOn(64, 0.8f, offsetInBlock);
 *   sampler.processBlock(output);
 * @endcode
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Sampler
{
public:
    /** @brief How a zone loops. */
    enum class LoopMode
    {
        None,        ///< Play once to the end of the recording.
        Continuous,  ///< Loop for as long as the voice sounds, release included.
        Sustain      ///< Loop while the key is held, then play on past the loop end.
    };

    /** @brief One zone, as the caller describes it. The audio is copied. */
    struct ZoneSpec
    {
        AudioBufferView<const T> audio;  ///< The recording (1 or more channels).
        double sampleRate = 48000.0;     ///< Its sample rate.
        int rootKey = 60;                ///< MIDI key that plays it untransposed.
        double tuneCents = 0.0;          ///< Fine tuning added to every note.
        int keyLow = 0;                  ///< Lowest key it answers (inclusive).
        int keyHigh = 127;               ///< Highest key it answers (inclusive).
        double velocityLow = 0.0;        ///< Lowest velocity it answers, 0..1 (inclusive).
        double velocityHigh = 1.0;       ///< Highest velocity it answers, 0..1 (inclusive).
        double gainDb = 0.0;             ///< Zone gain.
        double pan = 0.0;                ///< -1 left .. +1 right (balance for multichannel).
        int64_t startOffset = 0;         ///< First sample played.
        LoopMode loopMode = LoopMode::None;
        int64_t loopStart = 0;           ///< First sample of the loop.
        int64_t loopEnd = 0;             ///< One past the last sample of the loop.
        int64_t loopCrossfade = 0;       ///< Crossfade length in samples (0 = hard loop).
        bool equalPowerCrossfade = false;///< Equal-power fade (uncorrelated material)
                                         ///< instead of linear (phase-matched material).
    };

    Sampler() = default;
    ~Sampler()
    {
        delete pending_.exchange(nullptr, std::memory_order_acq_rel);
        delete retired_.exchange(nullptr, std::memory_order_acq_rel);
        delete fading_;
        delete active_;
    }
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Allocates the voices and the event queue.
     * @param spec      Output sample rate and channel count.
     * @param maxVoices Polyphony, 1..256. Spare voices for stealing fades are
     *                  added on top.
     */
    void prepare(const AudioSpec& spec, int maxVoices = 32)
    {
        if (!(spec.sampleRate > 0.0) || !std::isfinite(spec.sampleRate) || spec.numChannels < 1)
            return;
        sampleRate_ = spec.sampleRate;
        polyphony_ = std::clamp(maxVoices, 1, 256);
        voices_.assign(static_cast<size_t>(polyphony_ + polyphony_ / 4 + 2), Voice {});
        for (auto& v : voices_)
        {
            v.env.prepare(sampleRate_);
            for (auto& a : v.acc) a.assign(static_cast<size_t>(kAccLength), T(0));
        }
        weights_.assign(static_cast<size_t>(2 * kMaxHalfTaps + 2), T(0));
        (void)polyphaseTable();   // build the shared tables here, never on the audio thread
        stealFadeSamples_ = std::max(1, static_cast<int>(std::lround(0.003 * sampleRate_)));
        reset();
        prepared_ = true;
    }

    /** @brief Silences every voice and drops queued events. Audio thread. */
    void reset() noexcept
    {
        for (auto& v : voices_) { v.active = false; v.env.reset(); }
        eventCount_ = 0;
        sustainPedal_ = false;
        pitchBend_ = 0.0;
        activeVoices_.store(0, std::memory_order_relaxed);
    }

    // -- Zones (control thread) --------------------------------------------------

    /**
     * @brief Replaces the zone set.
     *
     * Validates and copies every zone, bakes loop crossfades, and publishes the
     * set without blocking the audio thread. A zone with no audio, an invalid
     * rate or an empty key or velocity range is rejected, and so is the whole
     * call. Loop points are clamped into the recording, and the crossfade to
     * the samples available before the loop start and inside the loop.
     * @return True when the set was accepted.
     */
    bool setZones(std::span<const ZoneSpec> zones)
    {
        reclaim();
        auto set = std::make_unique<ZoneSet>();
        set->zones.reserve(zones.size());
        for (const ZoneSpec& z : zones)
        {
            ZoneData d;
            if (!buildZone(z, d)) return false;
            set->zones.push_back(std::move(d));
        }
        publish(set.release());
        return true;
    }

    /** @brief One zone over the whole keyboard and velocity range. */
    bool loadSample(AudioBufferView<const T> audio, double sampleRate, int rootKey = 60)
    {
        ZoneSpec z;
        z.audio = audio;
        z.sampleRate = sampleRate;
        z.rootKey = rootKey;
        return setZones(std::span<const ZoneSpec>(&z, 1));
    }

    /** @brief Publishes an empty zone set: sounding voices finish, new notes are silent. */
    void clearZones() { reclaim(); publish(new ZoneSet {}); }

    // -- Parameters (any thread) --------------------------------------------------

    /** @brief ADSR for notes that start from now on (release: also for notes
     *         released from now on). Times in ms, sustain 0..1. */
    void setEnvelope(T attackMs, T decayMs, T sustain, T releaseMs) noexcept
    {
        if (!std::isfinite(attackMs) || !std::isfinite(decayMs)
            || !std::isfinite(sustain) || !std::isfinite(releaseMs)) return;
        attackMs_.store(std::max(0.01, static_cast<double>(attackMs)), std::memory_order_relaxed);
        decayMs_.store(std::max(0.01, static_cast<double>(decayMs)), std::memory_order_relaxed);
        sustain_.store(std::clamp(static_cast<double>(sustain), 0.0, 1.0), std::memory_order_relaxed);
        releaseMs_.store(std::max(0.01, static_cast<double>(releaseMs)), std::memory_order_relaxed);
    }

    /** @brief Velocity sensitivity: velocity 0 plays rangeDb below velocity 1
     *         (0..96 dB, default 24; 0 ignores velocity). */
    void setVelocityRange(T rangeDb) noexcept
    {
        if (!std::isfinite(rangeDb)) return;
        velocityRangeDb_.store(std::clamp(static_cast<double>(rangeDb), 0.0, 96.0),
                               std::memory_order_relaxed);
    }

    /** @brief Output gain in dB. */
    void setGain(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        outputGain_.store(std::pow(10.0, std::clamp(static_cast<double>(dB), -120.0, 24.0) / 20.0),
                          std::memory_order_relaxed);
    }

    // -- Events (audio thread) ----------------------------------------------------

    /** @brief Starts a note at `offset` samples into the next block. Velocity 0
     *         is a note-off, as in MIDI. */
    void noteOn(int key, T velocity, int offset = 0) noexcept
    {
        if (!std::isfinite(velocity)) return;
        if (velocity <= T(0)) { noteOff(key, offset); return; }
        push({ EventType::NoteOn, std::clamp(key, 0, 127),
               std::clamp(static_cast<double>(velocity), 0.0, 1.0), offset });
    }

    /** @brief Releases a note at `offset` samples into the next block. */
    void noteOff(int key, int offset = 0) noexcept
    {
        push({ EventType::NoteOff, std::clamp(key, 0, 127), 0.0, offset });
    }

    /** @brief Sustain pedal: while down, released keys keep sounding. */
    void setSustainPedal(bool down, int offset = 0) noexcept
    {
        push({ EventType::Pedal, 0, down ? 1.0 : 0.0, offset });
    }

    /** @brief Pitch bend in semitones (-48..48), applied to every voice. */
    void setPitchBend(T semitones, int offset = 0) noexcept
    {
        if (!std::isfinite(semitones)) return;
        push({ EventType::Bend, 0, std::clamp(static_cast<double>(semitones), -48.0, 48.0), offset });
    }

    /** @brief Releases every sounding note (and lifts the pedal). */
    void allNotesOff(int offset = 0) noexcept
    {
        push({ EventType::AllOff, 0, 0.0, offset });
    }

    // -- Processing (audio thread) -------------------------------------------------

    /**
     * @brief Renders the block, overwriting `out`, and applies the queued
     *        events at their offsets. Allocation free, lock free.
     *
     * One output channel receives the mean of a zone's channels; two or more
     * receive a mono zone on both of the first two (constant-power pan,
     * unity at centre) or a multichannel zone's first two channels with the
     * pan as balance. Further output channels are left silent.
     */
    void processBlock(AudioBufferView<T> out) noexcept
    {
        const int n = out.getNumSamples();
        const int nCh = out.getNumChannels();
        for (int c = 0; c < nCh; ++c)
            std::fill_n(out.getChannel(c), n, T(0));
        if (!prepared_ || n <= 0) { eventCount_ = 0; return; }

        adoptPending();

        int cursor = 0;
        for (int e = 0; e < eventCount_; ++e)
        {
            const int at = std::clamp(events_[static_cast<size_t>(e)].offset, 0, n);
            if (at > cursor) { render(out, cursor, at); cursor = at; }
            apply(events_[static_cast<size_t>(e)]);
        }
        eventCount_ = 0;
        if (cursor < n) render(out, cursor, n);

        const double g = outputGain_.load(std::memory_order_relaxed);
        if (g != 1.0)
            for (int c = 0; c < std::min(nCh, 2); ++c)
            {
                T* d = out.getChannel(c);
                for (int i = 0; i < n; ++i) d[i] = static_cast<T>(d[i] * g);
            }

        retireFadedSet();
        int count = 0;
        for (const auto& v : voices_) count += v.active ? 1 : 0;
        activeVoices_.store(count, std::memory_order_relaxed);
    }

    /** @brief Voices sounding after the last block, stealing fades included. */
    [[nodiscard]] int getActiveVoiceCount() const noexcept
    {
        return activeVoices_.load(std::memory_order_relaxed);
    }

    /** @brief Polyphony set by prepare(). */
    [[nodiscard]] int getPolyphony() const noexcept { return polyphony_; }

private:
    // -- Interpolation kernel -------------------------------------------------------

    /// Zero crossings on each side of the kernel at unit rate: 64 taps.
    static constexpr int kHalfTaps = 32;
    /// Table points per zero crossing (linear interpolation between them).
    static constexpr int kTablePerUnit = 2048;
    /// Highest playback rate; the kernel then spans 8 x 64 taps.
    static constexpr double kMaxRate = 8.0;
    static constexpr int kMaxHalfTaps = static_cast<int>(kHalfTaps * kMaxRate);
    /// Cutoff at 0.45 of the sample rate (normalised to Nyquist): passband to
    /// 0.40, stopband from 0.50, the width a 64-tap Kaiser needs for 100 dB.
    static constexpr double kCutoff = 0.9;
    static constexpr double kBeta = 10.06;

    /** Kernel h(t) for t >= 0 in zero crossings, sampled kTablePerUnit per unit. */
    static const std::vector<double>& kernelTable()
    {
        static const std::vector<double> table = [] {
            std::vector<double> t(static_cast<size_t>(kHalfTaps * kTablePerUnit + 2), 0.0);
            auto i0 = [](double x) {
                double sum = 1.0, term = 1.0;
                for (int k = 1; k < 64; ++k)
                {
                    term *= (x / (2.0 * k)) * (x / (2.0 * k));
                    sum += term;
                    if (term < sum * 1e-17) break;
                }
                return sum;
            };
            const double norm = i0(kBeta);
            for (size_t j = 0; j + 1 < t.size(); ++j)
            {
                const double x = static_cast<double>(j) / kTablePerUnit;
                if (x >= kHalfTaps) break;
                const double arg = pi<double> * kCutoff * x;
                const double sinc = (x == 0.0) ? 1.0 : std::sin(arg) / arg;
                const double r = x / kHalfTaps;
                t[j] = kCutoff * sinc * i0(kBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) / norm;
            }
            return t;
        }();
        return table;
    }

    /// Phases of the unit-rate polyphase table (rows interpolated linearly).
    static constexpr int kPhases = 2048;
    static constexpr int kTaps = 2 * kHalfTaps;

    /**
     * The unit-rate kernel as rows: row q holds the 64 tap weights for a
     * fractional position q / kPhases, taps k = -31 .. 32 at distance k - frac.
     * A voice at or below the recording's rate blends two adjacent rows, which
     * is the same kernel as the scalar table read tap by tap, laid out so the
     * blend and the dot products run over contiguous memory.
     */
    static const std::vector<T>& polyphaseTable()
    {
        static const std::vector<T> rows = [] {
            const auto& h = kernelTable();
            std::vector<T> r(static_cast<size_t>((kPhases + 1) * kTaps), T(0));
            for (int q = 0; q <= kPhases; ++q)
                for (int t = 0; t < kTaps; ++t)
                {
                    // |k - frac| in table units, exactly: (t - 31) * P - q.
                    const long u = std::labs(static_cast<long>(t - (kHalfTaps - 1)) * kTablePerUnit - q);
                    r[static_cast<size_t>(q * kTaps + t)] =
                        (static_cast<size_t>(u) < h.size()) ? static_cast<T>(h[static_cast<size_t>(u)]) : T(0);
                }
            return r;
        }();
        return rows;
    }

    // -- Zones -----------------------------------------------------------------------

    struct ZoneData
    {
        std::vector<T> audio;        ///< Channel-major, `length` per channel.
        std::vector<T> baked;        ///< Crossfaded copy of [loopEnd - xfade, loopEnd).
        int channels = 0;
        int64_t length = 0;
        double sampleRate = 48000.0;
        int rootKey = 60;
        double tuneCents = 0.0;
        int keyLow = 0, keyHigh = 127;
        double velLow = 0.0, velHigh = 1.0;
        double gain = 1.0;
        double pan = 0.0;
        int64_t start = 0;
        LoopMode loop = LoopMode::None;
        int64_t loopStart = 0, loopEnd = 0, xfade = 0;
    };

    struct ZoneSet { std::vector<ZoneData> zones; };

    static bool buildZone(const ZoneSpec& z, ZoneData& d)
    {
        const int ch = z.audio.getNumChannels();
        const int64_t len = z.audio.getNumSamples();
        if (ch < 1 || len < 1 || !(z.sampleRate > 0.0) || !std::isfinite(z.sampleRate))
            return false;
        for (int c = 0; c < ch; ++c)
            if (z.audio.getChannel(c) == nullptr) return false;
        if (z.keyLow > z.keyHigh || z.velocityLow > z.velocityHigh
            || !std::isfinite(z.tuneCents) || !std::isfinite(z.gainDb) || !std::isfinite(z.pan))
            return false;

        d.channels = ch;
        d.length = len;
        d.sampleRate = z.sampleRate;
        d.rootKey = z.rootKey;
        d.tuneCents = z.tuneCents;
        d.keyLow = std::clamp(z.keyLow, 0, 127);
        d.keyHigh = std::clamp(z.keyHigh, 0, 127);
        d.velLow = std::clamp(z.velocityLow, 0.0, 1.0);
        d.velHigh = std::clamp(z.velocityHigh, 0.0, 1.0);
        d.gain = std::pow(10.0, std::clamp(z.gainDb, -120.0, 24.0) / 20.0);
        d.pan = std::clamp(z.pan, -1.0, 1.0);
        d.start = std::clamp<int64_t>(z.startOffset, 0, len - 1);
        d.audio.resize(static_cast<size_t>(ch) * static_cast<size_t>(len));
        for (int c = 0; c < ch; ++c)
        {
            const T* src = z.audio.getChannel(c);
            T* dst = d.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(len);
            for (int64_t i = 0; i < len; ++i)
                dst[i] = std::isfinite(src[i]) ? src[i] : T(0);
        }

        d.loop = z.loopMode;
        d.loopStart = std::clamp<int64_t>(z.loopStart, 0, len);
        d.loopEnd = std::clamp<int64_t>(z.loopEnd, 0, len);
        if (d.loop != LoopMode::None && d.loopEnd - d.loopStart < 2)
            d.loop = LoopMode::None;
        if (d.loop == LoopMode::None) return true;

        d.xfade = std::clamp<int64_t>(z.loopCrossfade, 0,
                                      std::min(d.loopStart, d.loopEnd - d.loopStart));
        if (d.xfade > 0)
        {
            d.baked.resize(static_cast<size_t>(ch) * static_cast<size_t>(d.xfade));
            for (int c = 0; c < ch; ++c)
            {
                const T* src = d.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(len);
                T* dst = d.baked.data() + static_cast<size_t>(c) * static_cast<size_t>(d.xfade);
                for (int64_t i = 0; i < d.xfade; ++i)
                {
                    // Fade the loop's tail out into the audio that precedes the
                    // loop start, so the jump lands on continuous material.
                    const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(d.xfade);
                    const double fin = z.equalPowerCrossfade ? std::sin(0.5 * pi<double> * x) : x;
                    const double fout = z.equalPowerCrossfade ? std::cos(0.5 * pi<double> * x) : 1.0 - x;
                    const double tail = static_cast<double>(src[d.loopEnd - d.xfade + i]);
                    const double lead = static_cast<double>(src[d.loopStart - d.xfade + i]);
                    dst[i] = static_cast<T>(fout * tail + fin * lead);
                }
            }
        }
        return true;
    }

    // -- Publication -------------------------------------------------------------------

    /** Frees the set the audio thread handed back. Control thread. */
    void reclaim() noexcept
    {
        delete retired_.exchange(nullptr, std::memory_order_acq_rel);
    }

    /** Offers a set; an offer the audio thread never took is freed here. */
    void publish(ZoneSet* set) noexcept
    {
        delete pending_.exchange(set, std::memory_order_acq_rel);
    }

    /** Audio thread: takes an offered set once the previous one has drained. */
    void adoptPending() noexcept
    {
        if (fading_ != nullptr) return;   // one handover at a time
        ZoneSet* next = pending_.exchange(nullptr, std::memory_order_acq_rel);
        if (next == nullptr) return;
        fading_ = active_;
        active_ = next;
    }

    /** Audio thread: hands a drained set back to the control thread. */
    void retireFadedSet() noexcept
    {
        if (fading_ == nullptr) return;
        for (const auto& v : voices_)
            if (v.active && v.set == fading_) return;
        if (retired_.load(std::memory_order_acquire) != nullptr) return;
        retired_.store(fading_, std::memory_order_release);
        fading_ = nullptr;
    }

    // -- Voices --------------------------------------------------------------------------

    struct Voice
    {
        bool active = false;
        const ZoneSet* set = nullptr;
        const ZoneData* zone = nullptr;
        int key = 0;
        double pos = 0.0;         ///< Read position in recording samples.
        double baseRatio = 1.0;   ///< Rate before pitch bend.
        bool loopEngaged = false;
        bool releaseLoop = false; ///< Sustain loop: disengage once clear of the fade.
        bool held = true;         ///< Key down.
        bool sustained = false;   ///< Released while the pedal was down.
        double gainL = 1.0, gainR = 1.0, gainMono = 1.0;
        int stealLeft = -1;       ///< Stealing fade samples left; -1 when not stealing.
        uint64_t age = 0;
        ADSREnvelope<double> env;
        // Scatter form: the next recording sample to spread, the output time
        // it falls on (relative to the next output sample), and the output
        // accumulators it is spread into.
        bool scatter = false;
        bool inputDone = false;
        int drainLeft = 0;
        int64_t jIn = 0;
        double tNext = 0.0;
        int accPos = 0;
        std::array<std::vector<T>, 2> acc;
    };

    enum class EventType { NoteOn, NoteOff, Pedal, Bend, AllOff };
    struct Event { EventType type; int key; double value; int offset; };

    void push(const Event& e) noexcept
    {
        if (!prepared_ || eventCount_ >= kMaxEvents) return;
        // Keep the queue ordered by offset; events with equal offsets keep
        // their call order.
        int i = eventCount_++;
        while (i > 0 && events_[static_cast<size_t>(i - 1)].offset > e.offset)
        {
            events_[static_cast<size_t>(i)] = events_[static_cast<size_t>(i - 1)];
            --i;
        }
        events_[static_cast<size_t>(i)] = e;
    }

    void apply(const Event& e) noexcept
    {
        switch (e.type)
        {
            case EventType::NoteOn:  startNote(e.key, e.value); break;
            case EventType::NoteOff: releaseKey(e.key); break;
            case EventType::Bend:    pitchBend_ = e.value; break;
            case EventType::Pedal:
                sustainPedal_ = e.value > 0.5;
                if (!sustainPedal_)
                    for (auto& v : voices_)
                        if (v.active && v.sustained) release(v);
                break;
            case EventType::AllOff:
                sustainPedal_ = false;
                for (auto& v : voices_)
                    if (v.active && v.stealLeft < 0) release(v);
                break;
        }
    }

    void release(Voice& v) noexcept
    {
        v.held = false;
        v.sustained = false;
        v.env.setRelease(static_cast<double>(releaseMs_.load(std::memory_order_relaxed)));
        v.env.noteOff();
        if (v.zone->loop == LoopMode::Sustain) v.releaseLoop = true;
    }

    void releaseKey(int key) noexcept
    {
        for (auto& v : voices_)
            if (v.active && v.held && v.key == key && v.stealLeft < 0)
            {
                if (sustainPedal_) { v.held = false; v.sustained = true; }
                else release(v);
            }
    }

    void startNote(int key, double velocity) noexcept
    {
        if (active_ == nullptr) return;
        // A retriggered key releases what it was playing.
        for (auto& v : voices_)
            if (v.active && v.key == key && v.stealLeft < 0 && (v.held || v.sustained))
                release(v);

        const double velGain = std::pow(10.0, velocityRangeDb_.load(std::memory_order_relaxed)
                                                  * (velocity - 1.0) / 20.0);
        for (const ZoneData& z : active_->zones)
        {
            if (key < z.keyLow || key > z.keyHigh || velocity < z.velLow || velocity > z.velHigh)
                continue;
            Voice* v = freeVoice();
            if (v == nullptr) return;
            v->active = true;
            v->set = active_;
            v->zone = &z;
            v->key = key;
            v->pos = static_cast<double>(z.start);
            v->baseRatio = z.sampleRate / sampleRate_
                         * std::pow(2.0, (key - z.rootKey + z.tuneCents / 100.0) / 12.0);
            v->scatter = v->baseRatio * std::pow(2.0, pitchBend_ / 12.0) > 1.0;
            v->inputDone = false;
            v->drainLeft = 0;
            v->jIn = z.start;
            v->tNext = 0.0;
            v->accPos = 0;
            if (v->scatter)
                for (auto& a : v->acc) std::fill(a.begin(), a.end(), T(0));
            v->loopEngaged = z.loop != LoopMode::None;
            v->releaseLoop = false;
            v->held = true;
            v->sustained = false;
            v->stealLeft = -1;
            v->age = ++clock_;
            const double g = z.gain * velGain;
            const double theta = 0.25 * pi<double> * (z.pan + 1.0);
            v->gainMono = g;
            if (z.channels == 1)
            {
                // Constant power, unity at centre.
                v->gainL = g * std::cos(theta) * sqrt2<double>;
                v->gainR = g * std::sin(theta) * sqrt2<double>;
            }
            else
            {
                v->gainL = g * std::min(1.0, 1.0 - z.pan);
                v->gainR = g * std::min(1.0, 1.0 + z.pan);
            }
            v->env.setParameters(attackMs_.load(std::memory_order_relaxed),
                                 decayMs_.load(std::memory_order_relaxed),
                                 sustain_.load(std::memory_order_relaxed),
                                 releaseMs_.load(std::memory_order_relaxed));
            v->env.reset();
            v->env.noteOn();
        }
    }

    /** A voice for a new note: a free one within the polyphony, else a
     *  stolen one handed to a spare slot to fade out. */
    Voice* freeVoice() noexcept
    {
        int sounding = 0;
        Voice* idle = nullptr;
        for (auto& v : voices_)
        {
            if (!v.active) { if (idle == nullptr) idle = &v; }
            else if (v.stealLeft < 0) ++sounding;
        }
        if (sounding < polyphony_ && idle != nullptr) return idle;

        // Steal: the oldest released voice, else the quietest, else the oldest.
        Voice* victim = nullptr;
        for (auto& v : voices_)
        {
            if (!v.active || v.stealLeft >= 0) continue;
            if (victim == nullptr) { victim = &v; continue; }
            const bool vr = !v.held && !v.sustained, cr = !victim->held && !victim->sustained;
            if (vr != cr) { if (vr) victim = &v; continue; }
            const double va = v.env.getCurrentValue(), ca = victim->env.getCurrentValue();
            if (va < ca - 1e-9 || (std::abs(va - ca) <= 1e-9 && v.age < victim->age)) victim = &v;
        }
        if (victim != nullptr) victim->stealLeft = stealFadeSamples_;
        if (idle != nullptr) return idle;
        // No spare slot left: the oldest stealing fade is cut short.
        Voice* oldest = nullptr;
        for (auto& v : voices_)
            if (v.active && v.stealLeft >= 0 && (oldest == nullptr || v.age < oldest->age)) oldest = &v;
        if (oldest != nullptr) oldest->active = false;
        return oldest;
    }

    // -- Rendering --------------------------------------------------------------------------

    /** Dot product over eight independent partial sums: a single running sum
     *  is one long dependency chain the compiler may not reorder. */
    [[nodiscard]] static double dot(const T* w, const T* x, int n) noexcept
    {
        T a[8] = { T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0) };
        int t = 0;
        for (; t + 8 <= n; t += 8)
            for (int l = 0; l < 8; ++l)
                a[l] += w[t + l] * x[t + l];
        T sum = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
        for (; t < n; ++t) sum += w[t] * x[t];
        return static_cast<double>(sum);
    }

    /** Reads sample j of channel c through the loop and the baked crossfade. */
    [[nodiscard]] static double fetch(const ZoneData& z, const T* chan, const T* baked,
                                      int64_t j, bool loopEngaged) noexcept
    {
        if (loopEngaged && j >= z.loopEnd)
            j = z.loopStart + (j - z.loopEnd) % (z.loopEnd - z.loopStart);
        if (j < 0 || j >= z.length) return 0.0;
        if (loopEngaged && baked != nullptr && j >= z.loopEnd - z.xfade && j < z.loopEnd)
            return static_cast<double>(baked[j - (z.loopEnd - z.xfade)]);
        return static_cast<double>(chan[j]);
    }

    /** Fills weights_ with the kernel at scale fc for taps k = -half+1 .. half
     *  at distance (k - frac); returns the tap count. */
    int kernelWeights(double frac, double fc, int half) noexcept
    {
        if (fc >= 1.0)
        {
            const T* rows = polyphaseTable().data();
            const double qf = frac * kPhases;
            const int q = std::min(static_cast<int>(qf), kPhases - 1);
            const T g = static_cast<T>(qf - static_cast<double>(q));
            const T* r0 = rows + static_cast<size_t>(q) * kTaps;
            const T* r1 = r0 + kTaps;
            T* w = weights_.data();
            for (int t = 0; t < kTaps; ++t)
                w[t] = r0[t] + g * (r1[t] - r0[t]);
            return kTaps;
        }
        // Stretched kernel: |k - frac| * fc in table units, one step of fc per tap.
        const auto& table = kernelTable();
        const double step = fc * kTablePerUnit;
        double x = (static_cast<double>(-half + 1) - frac) * step;
        const size_t last = table.size() - 1;
        for (int t = 0; t < 2 * half; ++t, x += step)
        {
            const double u = std::abs(x);
            const size_t iu = static_cast<size_t>(u);
            double w = 0.0;
            if (iu < last)
            {
                const double f = u - static_cast<double>(iu);
                w = table[iu] + f * (table[iu + 1] - table[iu]);
            }
            weights_[static_cast<size_t>(t)] = static_cast<T>(w * fc);
        }
        return 2 * half;
    }

    /** Envelope and stealing fade for the output sample about to be emitted;
     *  false when the voice has just ended. */
    static bool advanceGain(Voice& v, int stealFade, double& gain) noexcept
    {
        gain = v.env.getNextValue();
        if (v.stealLeft >= 0)
        {
            gain *= static_cast<double>(v.stealLeft) / static_cast<double>(stealFade);
            if (--v.stealLeft < 0) { v.active = false; return false; }
        }
        if (!v.env.isActive()) { v.active = false; return false; }
        return true;
    }

    void emit(AudioBufferView<T> out, int s, const Voice& v, double a0, double a1, double gain) noexcept
    {
        const int nCh = out.getNumChannels();
        if (nCh == 1)
            out.getChannel(0)[s] += static_cast<T>(a0 * v.gainMono * gain);
        else
        {
            out.getChannel(0)[s] += static_cast<T>(a0 * v.gainL * gain);
            out.getChannel(1)[s] += static_cast<T>((v.zone->channels == 1 ? a0 : a1) * v.gainR * gain);
        }
    }

    void render(AudioBufferView<T> out, int from, int to) noexcept
    {
        const double bendRatio = std::pow(2.0, pitchBend_ / 12.0);
        for (auto& v : voices_)
        {
            if (!v.active) continue;
            const double ratio = std::clamp(v.baseRatio * bendRatio, 1.0 / kMaxRate, kMaxRate);
            if (v.scatter) renderScatter(v, out, from, to, ratio);
            else renderGather(v, out, from, to, ratio);
        }
    }

    /**
     * Gather form, for voices that start at or below the recording's rate:
     * each output sample is a dot product of the recording with the kernel
     * centred on the read position. At a rate r above 1 (a bend upwards) the
     * kernel is stretched by r.
     */
    void renderGather(Voice& v, AudioBufferView<T> out, int from, int to, double ratio) noexcept
    {
        const ZoneData& z = *v.zone;
        const int nCh = out.getNumChannels();
        const double fc = std::min(1.0, 1.0 / ratio);
        const int half = static_cast<int>(std::ceil(kHalfTaps / fc));
        const double loopLen = static_cast<double>(z.loopEnd - z.loopStart);
        const int zc = z.channels;
        const int used = (nCh == 1) ? zc : std::min(zc, 2);

        for (int s = from; s < to; ++s)
        {
            // Sustain loop released: leave the loop once the kernel is clear
            // of the crossfade region, so no read jumps sources.
            if (v.releaseLoop && v.pos + half < static_cast<double>(z.loopEnd - z.xfade))
            {
                v.loopEngaged = false;
                v.releaseLoop = false;
            }
            if (!v.loopEngaged && v.pos >= static_cast<double>(z.length)) { v.active = false; return; }

            const double fl = std::floor(v.pos);
            const int64_t ip = static_cast<int64_t>(fl);
            const int taps = kernelWeights(v.pos - fl, fc, half);

            double gain = 0.0;
            if (!advanceGain(v, stealFadeSamples_, gain)) return;

            const int64_t first = ip - half + 1;
            const bool direct = first >= 0 && first + taps <= z.length
                             && (!v.loopEngaged || first + taps <= z.loopEnd - z.xfade);
            double acc[2] = { 0.0, 0.0 };
            for (int c = 0; c < used; ++c)
            {
                const T* chan = z.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(z.length);
                const T* baked = z.baked.empty() ? nullptr
                               : z.baked.data() + static_cast<size_t>(c) * static_cast<size_t>(z.xfade);
                double sum = 0.0;
                if (direct)
                    sum = dot(weights_.data(), chan + first, taps);
                else
                    for (int t = 0; t < taps; ++t)
                        sum += static_cast<double>(weights_[static_cast<size_t>(t)])
                             * fetch(z, chan, baked, first + t, v.loopEngaged);
                if (nCh == 1) acc[0] += sum / zc;
                else acc[c] = sum;
            }
            emit(out, s, v, acc[0], acc[1], gain);

            v.pos += ratio;
            if (v.loopEngaged && v.pos >= static_cast<double>(z.loopEnd))
                v.pos -= loopLen;
        }
    }

    /**
     * Scatter (transposed) form, for voices that start above the recording's
     * rate: each recording sample is spread over the output samples around
     * the instant it falls on, with the kernel laid out in OUTPUT time. Its
     * stopband therefore starts at the output Nyquist frequency - exactly
     * the anti-aliasing a note played up needs - and the weights come from
     * the same contiguous polyphase rows as the gather form, so the cost is
     * r times that of an untransposed voice instead of a stretched gather.
     * Loop jumps happen at whole recording samples. At a rate below 1 (a
     * bend downwards) the output-time kernel is stretched by 1 / r, which
     * keeps the stopband at the recording's Nyquist frequency.
     */
    void renderScatter(Voice& v, AudioBufferView<T> out, int from, int to, double ratio) noexcept
    {
        const ZoneData& z = *v.zone;
        const int nCh = out.getNumChannels();
        const double fc = std::min(1.0, ratio);          // output-time kernel scale
        const int half = static_cast<int>(std::ceil(kHalfTaps / fc));
        const double spacing = 1.0 / ratio;             // output samples per input sample
        const int zc = z.channels;
        const int used = (nCh == 1) ? zc : std::min(zc, 2);
        const int rings = (nCh == 1 || zc == 1) ? 1 : 2;

        for (int s = from; s < to; ++s)
        {
            // Spread every input whose support reaches this output sample.
            while (!v.inputDone && v.tNext < static_cast<double>(half))
            {
                if (v.releaseLoop && v.jIn < z.loopEnd - z.xfade)
                {
                    v.loopEngaged = false;
                    v.releaseLoop = false;
                }
                double x[2] = { 0.0, 0.0 };
                for (int c = 0; c < used; ++c)
                {
                    const T* chan = z.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(z.length);
                    const T* baked = z.baked.empty() ? nullptr
                                   : z.baked.data() + static_cast<size_t>(c) * static_cast<size_t>(z.xfade);
                    const double sample = fetch(z, chan, baked, v.jIn, v.loopEngaged);
                    if (rings == 1) x[0] += sample / used;
                    else x[c] = sample;
                }
                const double fl = std::floor(v.tNext);
                const int taps = kernelWeights(v.tNext - fl, fc, half);
                const int base = v.accPos + static_cast<int>(fl) - half + 1;
                for (int r = 0; r < rings; ++r)
                {
                    const T xs = static_cast<T>(x[r] * spacing);
                    T* acc = v.acc[static_cast<size_t>(r)].data();
                    const T* w = weights_.data();
                    for (int t = 0; t < taps; ++t)
                    {
                        const int idx = base + t;
                        if (idx >= v.accPos) acc[idx] += w[t] * xs;   // earlier outputs are gone
                    }
                }
                ++v.jIn;
                if (v.loopEngaged && v.jIn >= z.loopEnd) v.jIn = z.loopStart;
                if (!v.loopEngaged && v.jIn >= z.length) { v.inputDone = true; v.drainLeft = 2 * half; }
                v.tNext += spacing;
            }

            double gain = 0.0;
            if (!advanceGain(v, stealFadeSamples_, gain)) return;
            const double a0 = static_cast<double>(v.acc[0][static_cast<size_t>(v.accPos)]);
            const double a1 = (rings == 2) ? static_cast<double>(v.acc[1][static_cast<size_t>(v.accPos)]) : a0;
            emit(out, s, v, a0, a1, gain);

            // Advance the output: the accumulator slides once it nears its end.
            for (int r = 0; r < rings; ++r) v.acc[static_cast<size_t>(r)][static_cast<size_t>(v.accPos)] = T(0);
            ++v.accPos;
            v.tNext -= 1.0;
            if (v.accPos + 2 * kMaxHalfTaps + 2 >= kAccLength)
            {
                const int keep = kAccLength - v.accPos;
                for (int r = 0; r < rings; ++r)
                {
                    T* acc = v.acc[static_cast<size_t>(r)].data();
                    std::copy(acc + v.accPos, acc + kAccLength, acc);
                    std::fill(acc + keep, acc + kAccLength, T(0));
                }
                v.accPos = 0;
            }
            if (v.inputDone && --v.drainLeft <= 0) { v.active = false; return; }
        }
    }

    // -- Members ------------------------------------------------------------------------------

    static constexpr int kMaxEvents = 1024;
    /// Scatter accumulator per channel: room for the widest kernel plus a
    /// stretch of output before it slides back.
    static constexpr int kAccLength = 4 * kMaxHalfTaps + 256;

    double sampleRate_ = 48000.0;
    int polyphony_ = 0;
    bool prepared_ = false;
    int stealFadeSamples_ = 144;
    uint64_t clock_ = 0;

    std::vector<Voice> voices_;
    std::vector<T> weights_;
    std::array<Event, kMaxEvents> events_ {};
    int eventCount_ = 0;
    bool sustainPedal_ = false;
    double pitchBend_ = 0.0;

    ZoneSet* active_ = nullptr;                 ///< Audio thread's current set.
    ZoneSet* fading_ = nullptr;                 ///< Previous set, until its voices end.
    std::atomic<ZoneSet*> pending_ { nullptr }; ///< Offered by the control thread.
    std::atomic<ZoneSet*> retired_ { nullptr }; ///< Handed back for freeing.

    std::atomic<double> attackMs_ { 1.0 };
    std::atomic<double> decayMs_ { 100.0 };
    std::atomic<double> sustain_ { 1.0 };
    std::atomic<double> releaseMs_ { 100.0 };
    std::atomic<double> velocityRangeDb_ { 24.0 };
    std::atomic<double> outputGain_ { 1.0 };
    std::atomic<int> activeVoices_ { 0 };
};

} // namespace dspark
