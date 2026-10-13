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
 * - noteOn() / noteOnTuned() / noteOff() / setSustainPedal() / setPitchBend() / allNotesOff()
 *   and processBlock() / reset(): audio thread, the stream owner. Events are
 *   queued, allocation free, for the next processBlock().
 * - getActiveVoiceCount(): any thread, lock free.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O. The shared interpolation tables (about 1 MB) are
 * built once per process, on the first prepare().
 *
 * The zone description (SampleZone.h) and the band-limited reader
 * (detail/SampleReader.h) are shared with Synthesizer's sample source.
 *
 * Dependencies: DspMath.h, AudioBuffer.h, AudioSpec.h, EnvelopeGenerator.h,
 * SampleZone.h, detail/SampleReader.h.
 */

#include "DspMath.h"
#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "EnvelopeGenerator.h"
#include "SampleZone.h"
#include "detail/SampleReader.h"

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
    /** Maximum queued events between processBlock calls. Events beyond
     *  this capacity are ignored; hosts must split dense spans beforehand.
     *  Every event kind shares the same queue. */
    static constexpr int kMaxEvents = 1024;

    /** @brief How a zone loops (see SampleLoopMode). */
    using LoopMode = SampleLoopMode;

    /** @brief One zone, as the caller describes it (see SampleZoneSpec). The
     *         audio is copied. */
    using ZoneSpec = SampleZoneSpec<T>;

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
            v.reader.allocate();
        }
        weights_.assign(static_cast<size_t>(Reader::kWeightsLength), T(0));
        (void)Reader::polyphaseTable();   // build the shared tables here, never on the audio thread
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
            if (!detail::buildSampleZone(z, d)) return false;
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
        noteOnTuned(key, velocity, T(0), offset);
    }

    /** @brief Starts a note with individual fine tuning in cents, clamped to
     *  -100..100. Tuning adds to the zone tuning and channel pitch bend;
     *  key selection, retrigger and note-off identity remain the integer key.
     *  Non-finite velocity or tuning is ignored. Zero velocity is note-off.
     *  Audio thread; the event applies at offset in the next block. */
    void noteOnTuned(int key, T velocity, T tuningCents, int offset = 0) noexcept
    {
        if (!std::isfinite(velocity) || !std::isfinite(tuningCents)) return;
        if (velocity <= T(0)) { noteOff(key, offset); return; }
        push({ EventType::NoteOn, std::clamp(key, 0, 127),
               std::clamp(static_cast<double>(velocity), 0.0, 1.0), offset,
               std::clamp(static_cast<double>(tuningCents), -100.0, 100.0) });
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
    using Reader = detail::SampleReader<T>;
    using ZoneData = detail::SampleZoneData<T>;

    struct ZoneSet { std::vector<ZoneData> zones; };

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
        int key = 0;
        double baseRatio = 1.0;   ///< Rate before pitch bend.
        bool held = true;         ///< Key down.
        bool sustained = false;   ///< Released while the pedal was down.
        double gainL = 1.0, gainR = 1.0, gainMono = 1.0;
        int stealLeft = -1;       ///< Stealing fade samples left; -1 when not stealing.
        uint64_t age = 0;
        ADSREnvelope<double> env;
        Reader reader;            ///< Read position in the zone.
    };

    enum class EventType { NoteOn, NoteOff, Pedal, Bend, AllOff };
    struct Event { EventType type; int key; double value; int offset; double tuningCents = 0.0; };

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
            case EventType::NoteOn:  startNote(e.key, e.value, e.tuningCents); break;
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
        v.reader.release();
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

    void startNote(int key, double velocity, double tuningCents) noexcept
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
            v->key = key;
            v->baseRatio = z.sampleRate / sampleRate_
                         * std::pow(2.0, (key - z.rootKey + (z.tuneCents + tuningCents) / 100.0) / 12.0);
            v->reader.start(z, v->baseRatio * std::pow(2.0, pitchBend_ / 12.0));
            v->held = true;
            v->sustained = false;
            v->stealLeft = -1;
            v->age = ++clock_;
            // Constant power for a mono zone (unity at centre), balance otherwise.
            detail::sampleZonePanGains(z.pan, z.channels, z.gain * velGain,
                                       v->gainL, v->gainR, v->gainMono);
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
            out.getChannel(1)[s] += static_cast<T>((v.reader.zone()->channels == 1 ? a0 : a1) * v.gainR * gain);
        }
    }

    void render(AudioBufferView<T> out, int from, int to) noexcept
    {
        const double bendRatio = std::pow(2.0, pitchBend_ / 12.0);
        const int nCh = out.getNumChannels();
        for (auto& v : voices_)
        {
            if (!v.active) continue;
            const double ratio = std::clamp(v.baseRatio * bendRatio, 1.0 / Reader::kMaxRate, Reader::kMaxRate);
            for (int s = from; s < to; ++s)
            {
                double a0 = 0.0, a1 = 0.0;
                const auto frame = v.reader.next(ratio, nCh, weights_.data(), a0, a1);
                if (frame == Reader::Frame::Ended) { v.active = false; break; }
                double gain = 0.0;
                if (!advanceGain(v, stealFadeSamples_, gain)) break;
                emit(out, s, v, a0, a1, gain);
                if (frame == Reader::Frame::Last) { v.active = false; break; }
            }
        }
    }

    // -- Members ------------------------------------------------------------------------------


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
