// DSParkLab — Audio Engine
// File loading (WAV/MP3), real-time playback via miniaudio, effect processing.
//
// Threading
// ---------
// Two threads reach this object: the interface thread, which owns every public
// call, and the audio thread the playback device creates, which runs
// audioCallback() and renderToneBlock().
//
// - loadFile(), update(), play(), pause(), stop(), togglePlay(), setLooping(),
//   seekTo(), setBypass(), setTestTone(), setTestToneFreq(), setTestToneDb():
//   interface thread only.
// - setEffects(): setup only. It publishes nothing, so it must be called
//   before the first loadFile() brings a device up, which is what the
//   application does.
// - Every readout -- isLoading(), isPlaying(), isLooping(), isBypassed(),
//   hasFile(), getPosition(), getDurationSeconds(), getSampleRate(),
//   getChannels(), getFilePath(), getSpec(), getPeakL(), getPeakR(),
//   getWaveform(), computeThd(), getSpectrumDb(), getSpectrumBins(),
//   binToFrequency(), isDeviceReady(), getTestTone(), getTestToneFreq(),
//   getTestToneDb() -- runs on that same ONE interface thread, concurrently
//   with the audio thread. The spectrum getters forward a single-reader
//   contract of the analyzer they wrap, so they belong to that thread and to
//   no other.
// - audioCallback() and renderToneBlock(): audio thread only.
//
// Every word the two threads share is published, never plain:
//
// - fileSamples_ is the sentinel guarding the DECODED FILE and the position
//   walked through it. The interface thread stores it with release once the
//   file is in place; the audio thread loads it with acquire, once per
//   callback, before it touches the buffer. A callback that sees a non-zero
//   count therefore sees the whole file that goes with it, and zero means
//   "no source": it outputs silence and reads none of it.
// - The sentinel does NOT guard the test-tone path. audioCallback branches
//   into renderToneBlock BEFORE it loads the sentinel, and that path reads the
//   sample rate, the channel count, the work buffer, the analyzer and the
//   meter without consulting it at all. What makes those safe is the next
//   point, not this one.
// - Publication is not enough on its own to REPLACE that state, because the
//   previous buffers would still be under a callback already in flight, and
//   the analyzer's and the meter's prepare() are setup-only by their own
//   documented contract. finalizeLoad() therefore stops the device first: the
//   block in flight drains, the swap becomes a setup-time operation, and the
//   device that starts afterwards is a fresh thread which by construction sees
//   everything written before it existed.
// - The waveform snapshot travels the other way, audio thread to interface. It
//   is a set with an invariant -- the samples and their count come from one
//   block -- so it goes through a sequence counter that lets the reader detect
//   a copy taken across a publication and repeat it. See publishWaveform().
// - The tone capture ring is a stream, not a set: its words are published one
//   by one and a reader may span the boundary between two blocks, which is of
//   no consequence over a window of some 0.7 s.
// - The play position is owned by the audio thread while it renders, so a seek
//   is handed over as a request the callback adopts rather than as a store the
//   callback would overwrite.

#pragma once

#include "../DSPark.h"
#include "EffectSlot.h"

#define MINIAUDIO_IMPLEMENTATION
#include "vendor/miniaudio.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace dsplab {

class AudioEngine
{
public:
    /// Frames the engine renders per chunk, and the capacity of the waveform
    /// snapshot getWaveform() publishes.
    static constexpr int kBlockSize = 512;

    AudioEngine() = default;

    ~AudioEngine()
    {
        // Wait for any in-flight load before tearing down the device — the
        // worker thread still holds references to staging buffers owned by
        // this object.
        if (loadThread_.joinable()) loadThread_.join();

        if (deviceInit_)
        {
            ma_device_uninit(&device_);
            deviceInit_ = false;
        }
    }

    // --- File loading -------------------------------------------------------
    //
    // Async load architecture:
    //
    //  GUI thread → loadFile(path) → spawns worker → returns immediately.
    //  Worker thread → opens file, fully decodes into staging buffers,
    //                  then sets loadState_ to Ready or Failed.
    //  GUI thread → calls update() each frame; if state == Ready, swap
    //               staging into active buffers and re-init the audio
    //               device. While loading, fileSamples_ is held at 0 so
    //               the audio callback outputs silence.
    //
    // The motivation: MP3 decode of a 5-minute file takes seconds even
    // with the LUT-optimised decoder; doing that synchronously on the GUI
    // thread froze the entire UI during open. Async load keeps the
    // interface responsive and frees us to display a "loading" indicator.

    enum class LoadState : int
    {
        Idle    = 0,
        Loading = 1,
        Ready   = 2,
        Failed  = 3
    };

    /// Begins loading a file. Returns immediately; check isLoading() and
    /// call update() each frame to detect completion.
    /// Subsequent calls cancel-by-waiting for the prior load.
    bool loadFile(const char* path)
    {
        // Drain any prior load before we mutate staging buffers.
        if (loadThread_.joinable()) loadThread_.join();

        stop();
        // Retire the sentinel: from the moment the audio thread reads this
        // zero it stops consulting the file, its rate, its channel count and
        // the work buffer, all of which finalizeLoad() is about to replace.
        fileSamples_.store(0, std::memory_order_release);

        loadState_.store(static_cast<int>(LoadState::Loading));
        std::string pathCopy(path);
        loadThread_ = std::thread([this, pathCopy] { doLoadAsync(pathCopy); });
        return true;
    }

    /// Must be called every frame from the GUI thread. Finalises any
    /// completed background load (swaps in decoded buffers, re-initialises
    /// the audio device, prepares the effect chain).
    void update()
    {
        auto state = static_cast<LoadState>(loadState_.load());
        if (state == LoadState::Ready)
        {
            if (loadThread_.joinable()) loadThread_.join();
            finalizeLoad();
            loadState_.store(static_cast<int>(LoadState::Idle));
        }
        else if (state == LoadState::Failed)
        {
            if (loadThread_.joinable()) loadThread_.join();
            loadState_.store(static_cast<int>(LoadState::Idle));
        }
    }

    [[nodiscard]] bool isLoading() const
    {
        return loadState_.load() == static_cast<int>(LoadState::Loading);
    }

    // --- Transport ----------------------------------------------------------

    void play()    { playing_.store(true); }
    void pause()   { playing_.store(false); }
    void stop()    { playing_.store(false); requestPosition(0); }
    void togglePlay() { playing_.store(!playing_.load()); }

    void setLooping(bool v)           { looping_.store(v); }
    void seekTo(float normPos)
    {
        // One load of the sentinel decides both the scale and the clamp: read
        // twice, it could change between them and leave the clamp inert.
        const int total = fileSamples_.load(std::memory_order_acquire);
        // The fraction is bounded BEFORE it is scaled and converted, not
        // after. Converting a float that does not fit an int is undefined, and
        // a clamp on the result runs too late; the test is written so a NaN
        // takes the 0 branch instead of failing both comparisons and passing
        // through.
        const float fraction = !(normPos > 0.0f) ? 0.0f
                             : (normPos > 1.0f ? 1.0f : normPos);
        // The fraction is bounded above, but the PRODUCT still has to be:
        // (float) of a sample count near the top of the int range rounds UP,
        // past what an int can hold, and the conversion is undefined before
        // any clamp on its result can run. Scaling in double and bounding
        // there settles it for every count a file can carry.
        const double scaled = static_cast<double>(fraction) * static_cast<double>(total);
        const int converted = scaled >= static_cast<double>(total)
                            ? total : static_cast<int>(scaled);
        // Bounded into a named local rather than through std::clamp on the
        // argument: the published store below inlines, and a library frame
        // sitting on top of it is what a race oracle then reports, which costs
        // a reader of the log the one thing it needs to classify the access.
        const int pos = converted < 0 ? 0 : (converted > total ? total : converted);
        requestPosition(pos);
    }

    [[nodiscard]] bool  isPlaying()  const { return playing_.load(); }
    [[nodiscard]] bool  isLooping()  const { return looping_.load(); }
    [[nodiscard]] bool  hasFile()    const
    {
        return fileSamples_.load(std::memory_order_acquire) > 0;
    }
    [[nodiscard]] float getPosition() const
    {
        const int total = fileSamples_.load(std::memory_order_acquire);
        if (total <= 0) return 0.0f;
        return static_cast<float>(position_.load()) / static_cast<float>(total);
    }
    [[nodiscard]] float getDurationSeconds() const
    {
        if (fileSampleRate_ <= 0) return 0.0f;
        return static_cast<float>(fileSamples_.load(std::memory_order_acquire))
             / static_cast<float>(fileSampleRate_);
    }
    [[nodiscard]] const std::string& getFilePath() const { return filePath_; }
    [[nodiscard]] double getSampleRate() const { return fileSampleRate_; }
    [[nodiscard]] int    getChannels()   const { return fileChannels_; }
    [[nodiscard]] const dspark::AudioSpec& getSpec() const { return spec_; }

    // --- Effect chain -------------------------------------------------------

    void setEffects(std::vector<EffectSlot>* effects) { effects_ = effects; }

    // --- Bypass (A/B) -------------------------------------------------------

    void setBypass(bool v) { bypass_.store(v); }
    [[nodiscard]] bool isBypassed() const { return bypass_.load(); }

    // --- Test tone + live THD metering ---------------------------------------
    //
    // The tone replaces the file as the source while enabled (the audio
    // device must already be initialised, i.e. a file was loaded once). The
    // post-FX output of channel 0 is captured into a ring; computeThd()
    // estimates THD+N coherently against the generated frequency on the GUI
    // thread (Hann window + Goertzel at the exact tone frequency).

    void setTestTone(bool on)
    {
        if (on && !toneOn_.load()) thdFill_.store(0);
        toneOn_.store(on);
    }
    [[nodiscard]] bool getTestTone() const { return toneOn_.load(); }
    void setTestToneFreq(float hz) { toneFreq_.store(std::clamp(hz, 20.0f, 20000.0f)); thdFill_.store(0); }
    [[nodiscard]] float getTestToneFreq() const { return toneFreq_.load(); }
    void setTestToneDb(float db) { toneDb_.store(std::clamp(db, -60.0f, 0.0f)); thdFill_.store(0); }
    [[nodiscard]] float getTestToneDb() const { return toneDb_.load(); }
    [[nodiscard]] bool isDeviceReady() const { return deviceInit_; }

    struct ThdReading
    {
        float thdnPercent = 0.0f;
        float fundDb = -120.0f;   ///< Fundamental level (dBFS amplitude).
        float rmsDb  = -120.0f;   ///< Total output RMS (dBFS).
        bool  valid  = false;
    };

    /// THD+N of the captured post-FX output. Interface thread; the one buffer
    /// it needs is allocated once per thread rather than per call.
    [[nodiscard]] ThdReading computeThd() const
    {
        ThdReading r;
        if (!toneOn_.load() || thdFill_.load() < static_cast<long long>(kThdCap))
            return r;

        // Snapshot the ring. Each word is published on its own, so the copy is
        // a well-defined read of a stream the audio thread keeps filling; what
        // it does not promise is that every sample came from the same block.
        // It need not: the window spans ~0.7 s and the samples near the head
        // are the oldest in it.
        static thread_local std::vector<float> snap(kThdCap);
        const int head = thdHead_.load();
        for (int i = 0; i < kThdCap; ++i)
            snap[static_cast<size_t>(i)] =
                thdRing_[static_cast<size_t>((head + i) % kThdCap)]
                    .load(std::memory_order_relaxed);

        const double fs = (fileSampleRate_ > 0) ? fileSampleRate_ : 48000.0;
        const double f0 = static_cast<double>(toneFreq_.load());
        const double w  = 2.0 * 3.14159265358979323846 * f0 / fs;
        const double cw = 2.0 * std::cos(w);

        double s1 = 0, s2 = 0, winSum = 0, totSq = 0;
        for (int i = 0; i < kThdCap; ++i)
        {
            const double win = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846
                                                    * i / (kThdCap - 1));
            const double x = win * static_cast<double>(snap[static_cast<size_t>(i)]);
            const double s0 = x + cw * s1 - s2;
            s2 = s1; s1 = s0;
            winSum += win;
            totSq  += x * x;
        }
        const double re = s1 - s2 * std::cos(w);
        const double im = s2 * std::sin(w);
        const double amp = 2.0 * std::sqrt(re * re + im * im) / std::max(winSum, 1.0);
        const double fundSq = 0.5 * amp * amp;   // coherent tone power
        // Windowed total power normalised to the same (coherent) scale.
        double winSqSum = 0;
        for (int i = 0; i < kThdCap; ++i)
        {
            const double win = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846
                                                    * i / (kThdCap - 1));
            winSqSum += win * win;
        }
        const double totPow = totSq / std::max(winSqSum, 1.0);

        r.fundDb = static_cast<float>(20.0 * std::log10(std::max(amp, 1e-9)));
        r.rmsDb  = static_cast<float>(10.0 * std::log10(std::max(totPow, 1e-18)));
        r.thdnPercent = static_cast<float>(
            100.0 * std::sqrt(std::max(totPow - fundSq, 0.0) / std::max(fundSq, 1e-18)));
        r.valid = true;
        return r;
    }

    // --- Visualization data (read from GUI thread) --------------------------

    [[nodiscard]] float getPeakL() const { return peakL_.load(); }
    [[nodiscard]] float getPeakR() const { return peakR_.load(); }

    [[nodiscard]] const float* getSpectrumDb() const
    {
        return spectrum_.getMagnitudesDb();
    }
    [[nodiscard]] int getSpectrumBins() const
    {
        return spectrum_.getNumBins();
    }
    [[nodiscard]] float binToFrequency(int k) const
    {
        return spectrum_.binToFrequency(k);
    }

    /// Copies the most recent whole block of the processed waveform into the
    /// caller's arrays and returns how many frames it wrote (0 if nothing has
    /// been published yet, or if the arguments leave nowhere to write).
    ///
    /// The engine hands out a copy rather than a pointer because the audio
    /// thread overwrites this snapshot every block: any pointer into it would
    /// be read while it is being written. The copy is validated against the
    /// snapshot's sequence counter and repeated if a publication landed in the
    /// middle of it, so what comes back is always one block and never a
    /// mixture of two. Interface thread.
    [[nodiscard]] int getWaveform(float* left, float* right, int maxFrames) const noexcept
    {
        if (left == nullptr || right == nullptr || maxFrames <= 0)
            return 0;

        const int cap = std::min(maxFrames, kBlockSize);
        for (;;)
        {
            const unsigned s0 = waveSeq_.load(std::memory_order_acquire);
            if ((s0 & 1u) != 0u)
            {
                // A publication is in progress. This reader is exempt from the
                // attempt bound (it is the interface thread; the audio thread
                // is the writer here and never waits), but on a single-core
                // machine spinning against a counter only the other thread can
                // advance burns the whole quantum for nothing.
                std::this_thread::yield();
                continue;
            }

            // The published count is read into its own named local rather
            // than folded into a std::min: the fold makes a race oracle
            // attribute this atomic load to a standard-library line, which
            // costs a reader of the log the one thing it needs to classify it.
            const int published = waveCount_.load(std::memory_order_relaxed);
            const int n = published < cap ? published : cap;
            for (int i = 0; i < n; ++i)
            {
                left[i]  = waveL_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
                right[i] = waveR_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
            }
            // Orders the copy above BEFORE the second read of the counter. A
            // plain acquire load would only order what comes after it, so on a
            // weakly ordered machine the copy could sink past the check and a
            // mixture of two blocks would pass validation.
            std::atomic_thread_fence(std::memory_order_acquire);
            if (s0 == waveSeq_.load(std::memory_order_relaxed))
                return n;
        }
    }

private:

    // --- Async load worker ---------------------------------------------------

    void doLoadAsync(const std::string& path)
    {
        // Detect extension
        std::string ext;
        if (auto dot = path.rfind('.'); dot != std::string::npos)
            ext = path.substr(dot);
        for (auto& c : ext)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        bool ok = false;

        if (ext == ".wav")
        {
            dspark::WavFile wav;
            if (wav.openRead(path.c_str()))
            {
                stagingInfo_ = wav.getInfo();
                stagingBuffer_.resize(static_cast<int>(stagingInfo_.numChannels),
                                      static_cast<int>(stagingInfo_.numSamples));
                ok = wav.readSamples(stagingBuffer_.toView());
                wav.close();
            }
        }
        else if (ext == ".mp3")
        {
            dspark::Mp3File mp3;
            if (mp3.openRead(path.c_str()))
            {
                stagingInfo_ = mp3.getInfo();
                stagingBuffer_.resize(static_cast<int>(stagingInfo_.numChannels),
                                      static_cast<int>(stagingInfo_.numSamples));
                ok = mp3.readSamples(stagingBuffer_.toView());
                mp3.close();
            }
        }

        if (ok)
        {
            stagingPath_ = path;
            loadState_.store(static_cast<int>(LoadState::Ready));
        }
        else
        {
            loadState_.store(static_cast<int>(LoadState::Failed));
        }
    }

    // Swap staging buffers into active state and bring up the audio device.
    // Runs on the GUI thread (the worker has already finished).
    void finalizeLoad()
    {
        // Everything below replaces state the audio thread reads: the decoded
        // file, the work buffer, the analyzer and the meter -- and the last two
        // have setup-only prepare() methods that must not run beside a
        // processing call. The sentinel already told a future callback to stay
        // away; stopping the device retires the one that may still be inside a
        // block. initDevice() brings a fresh device, and a fresh thread, up
        // again at the end.
        if (deviceInit_) ma_device_stop(&device_);

        fileInfo_ = stagingInfo_;
        // Move staging buffer into the active fileBuffer_. AudioBuffer is
        // assumed to support move-assignment (and even if it falls back to
        // copy, the callback is silent so nothing observes the transient).
        fileBuffer_ = std::move(stagingBuffer_);
        stagingBuffer_ = {};  // reset to a clean empty buffer
        filePath_ = stagingPath_;

        requestPosition(0);
        fileSampleRate_ = fileInfo_.sampleRate;
        fileChannels_   = fileInfo_.numChannels;

        spec_ = { fileSampleRate_, kBlockSize, std::min(fileChannels_, 2) };
        workBuffer_.resize(spec_.numChannels, kBlockSize);

        spectrum_.prepare(fileSampleRate_, 4096);
        meter_.prepare(spec_);
        meter_.setAttackMs(5.0f);
        meter_.setReleaseMs(100.0f);

        if (effects_)
        {
            for (auto& e : *effects_)
            {
                e.prepare(spec_);
                e.applyAllDefaults();
            }
        }

        // Publish the sample count *after* every other piece of state is in
        // place, so that a callback which sees a non-zero count sees the file
        // that goes with it. The device is still stopped here, and the thread
        // initDevice() starts below begins after this store, so the release
        // and the thread's own creation both carry the same guarantee.
        fileSamples_.store(static_cast<int>(fileInfo_.numSamples),
                           std::memory_order_release);

        initDevice();
    }

    // --- Audio callback (runs on audio thread) ------------------------------

    static void audioCallback(ma_device* device, void* output, const void*, ma_uint32 frameCount)
    {
        auto* self = static_cast<AudioEngine*>(device->pUserData);
        auto* out  = static_cast<float*>(output);
        const int frames = static_cast<int>(frameCount);
        const int outCh  = static_cast<int>(device->playback.channels);

        // Test tone replaces the file source while enabled (transport-agnostic).
        if (self->toneOn_.load())
        {
            self->renderToneBlock(out, frames, outCh);
            return;
        }

        // One acquire load of the sentinel decides this whole callback, and it
        // is what makes the file, its channel count and the work buffer safe
        // to touch below. Re-reading it per chunk would let the block straddle
        // two different sources.
        const int fileSamples = self->fileSamples_.load(std::memory_order_acquire);

        if (!self->playing_.load() || fileSamples <= 0)
        {
            std::memset(out, 0, sizeof(float) * frames * outCh);
            return;
        }

        int pos = self->position_.load();
        const int requested = self->seekRequest_.exchange(-1, std::memory_order_acquire);
        if (requested >= 0) pos = std::clamp(requested, 0, fileSamples);
        const int nCh = std::min(self->fileChannels_, 2);
        int written = 0;

        while (written < frames)
        {
            int remaining = fileSamples - pos;
            if (remaining <= 0)
            {
                if (self->looping_.load())
                    pos = 0;
                else
                {
                    // Fill rest with silence
                    std::memset(out + written * outCh, 0,
                                sizeof(float) * (frames - written) * outCh);
                    self->playing_.store(false);
                    break;
                }
                remaining = fileSamples - pos;
            }

            const int chunk = std::min(frames - written, std::min(remaining, self->kBlockSize));

            // Copy from source to work buffer
            for (int ch = 0; ch < nCh; ++ch)
            {
                const float* src = self->fileBuffer_.getChannel(ch) + pos;
                float* dst = self->workBuffer_.getChannel(ch);
                std::copy(src, src + chunk, dst);
            }

            // Apply effects (unless bypassed)
            if (!self->bypass_.load() && self->effects_)
            {
                auto view = self->workBuffer_.toView().getSubView(0, chunk);
                for (auto& e : *self->effects_)
                    e.process(view);
            }

            // Feed analysis
            self->spectrum_.pushSamples(self->workBuffer_.getChannel(0), chunk);
            auto meterView = self->workBuffer_.toView().getSubView(0, chunk);
            self->meter_.process(meterView);

            // Interleave to output
            for (int i = 0; i < chunk; ++i)
            {
                for (int c = 0; c < outCh; ++c)
                {
                    int srcCh = std::min(c, nCh - 1);
                    out[(written + i) * outCh + c] = self->workBuffer_.getChannel(srcCh)[i];
                }
            }

            // Waveform snapshot
            if (chunk > 0)
            {
                const float* left  = self->workBuffer_.getChannel(0);
                const float* right = (nCh > 1) ? self->workBuffer_.getChannel(1) : left;
                self->publishWaveform(left, right, chunk);
            }

            pos += chunk;
            written += chunk;
        }

        self->position_.store(pos);

        // Update peak meters
        self->peakL_.store(self->meter_.getPeakLevel(0));
        if (nCh > 1)
            self->peakR_.store(self->meter_.getPeakLevel(1));
        else
            self->peakR_.store(self->peakL_.load());
    }

    /// Generates the test tone, runs it through the effect chain, feeds the
    /// analysis taps and the THD capture ring, and interleaves to the output.
    void renderToneBlock(float* out, int frames, int outCh) noexcept
    {
        const double fs = (fileSampleRate_ > 0) ? fileSampleRate_ : 48000.0;
        const double inc = 2.0 * 3.14159265358979323846
                         * static_cast<double>(toneFreq_.load()) / fs;
        const float amp = std::pow(10.0f, toneDb_.load() / 20.0f);
        const int nCh = std::max(1, std::min(fileChannels_, 2));

        int written = 0;
        while (written < frames)
        {
            const int chunk = std::min(frames - written, kBlockSize);

            for (int i = 0; i < chunk; ++i)
            {
                const float s = amp * static_cast<float>(std::sin(tonePhase_));
                tonePhase_ += inc;
                if (tonePhase_ > 6.28318530717958647692)
                    tonePhase_ -= 6.28318530717958647692;
                for (int ch = 0; ch < nCh; ++ch)
                    workBuffer_.getChannel(ch)[i] = s;
            }

            if (!bypass_.load() && effects_)
            {
                auto view = workBuffer_.toView().getSubView(0, chunk);
                for (auto& e : *effects_)
                    e.process(view);
            }

            spectrum_.pushSamples(workBuffer_.getChannel(0), chunk);
            auto meterView = workBuffer_.toView().getSubView(0, chunk);
            meter_.process(meterView);

            // THD capture (post-FX, channel 0).
            int head = thdHead_.load();
            for (int i = 0; i < chunk; ++i)
            {
                thdRing_[static_cast<size_t>(head)].store(
                    workBuffer_.getChannel(0)[i], std::memory_order_relaxed);
                head = (head + 1) % kThdCap;
            }
            thdHead_.store(head);
            // fetch_add, not load-then-store: the interface thread resets this
            // counter to zero when the tone changes, and a reset landing
            // between a load and its store would simply be overwritten. Then
            // the fill gate would still be satisfied and the next reading
            // would be computed at the NEW frequency over a ring still holding
            // the OLD tone. The counter is never clamped, so a reset always
            // wins; it counts frames since the last one, and at any supported
            // rate the 64-bit range outlasts any session.
            thdFill_.fetch_add(chunk, std::memory_order_relaxed);

            for (int i = 0; i < chunk; ++i)
                for (int c = 0; c < outCh; ++c)
                    out[(written + i) * outCh + c]
                        = workBuffer_.getChannel(std::min(c, nCh - 1))[i];

            publishWaveform(workBuffer_.getChannel(0),
                            workBuffer_.getChannel(nCh > 1 ? 1 : 0), chunk);

            written += chunk;
        }

        peakL_.store(meter_.getPeakLevel(0));
        peakR_.store(nCh > 1 ? meter_.getPeakLevel(1) : peakL_.load());
    }

    // --- Published readouts and requests ------------------------------------

    /// Audio thread. Publishes one whole block of the processed waveform for
    /// the interface to draw: the counter goes odd while the samples and their
    /// count are stored and even again once the set is complete, so a reader
    /// can tell a copy taken across a publication from a valid one. The writer
    /// never waits, which is what keeps this affordable on the audio thread.
    void publishWaveform(const float* left, const float* right, int frames) noexcept
    {
        const int n = std::min(frames, kBlockSize);

        waveSeq_.fetch_add(1, std::memory_order_acq_rel);   // odd: writing
        // Release fence: without it the relaxed sample stores below are not
        // ordered against the counter, and a reader could validate a copy it
        // took while they were still landing.
        std::atomic_thread_fence(std::memory_order_release);
        for (int i = 0; i < n; ++i)
        {
            waveL_[static_cast<size_t>(i)].store(left[i], std::memory_order_relaxed);
            waveR_[static_cast<size_t>(i)].store(right[i], std::memory_order_relaxed);
        }
        waveCount_.store(n, std::memory_order_relaxed);
        waveSeq_.fetch_add(1, std::memory_order_release);   // even: complete
    }

    /// Interface thread. Hands a play position to the audio thread. A plain
    /// store into position_ would be overwritten by up to one block: the
    /// callback stores the position its block ended at, computed from a value
    /// it read before the store. The request is adopted by the callback
    /// instead, and position_ is set as well so the interface reads the new
    /// value back immediately.
    void requestPosition(int pos) noexcept
    {
        position_.store(pos, std::memory_order_relaxed);
        seekRequest_.store(pos, std::memory_order_release);
    }

    // --- Device init --------------------------------------------------------

    void initDevice()
    {
        if (deviceInit_)
        {
            ma_device_uninit(&device_);
            deviceInit_ = false;
        }

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format   = ma_format_f32;
        config.playback.channels = static_cast<ma_uint32>(std::min(fileChannels_, 2));
        config.sampleRate        = static_cast<ma_uint32>(fileSampleRate_);
        config.dataCallback      = audioCallback;
        config.pUserData         = this;
        config.periodSizeInFrames = kBlockSize;

        if (ma_device_init(nullptr, &config, &device_) == MA_SUCCESS)
        {
            ma_device_start(&device_);
            deviceInit_ = true;
        }
    }

    // --- State --------------------------------------------------------------

    std::string              filePath_;
    dspark::AudioFileInfo    fileInfo_ {};
    dspark::AudioBuffer<float> fileBuffer_;
    dspark::AudioBuffer<float> workBuffer_;
    dspark::AudioSpec        spec_ {};

    // Written only while the device is stopped, which is what protects them:
    // the tone path reads them (renderToneBlock) WITHOUT ever loading the
    // sentinel, because the callback branches into it before the sentinel is
    // consulted. The sentinel guards the file, not these; the device stop in
    // finalizeLoad() is what makes replacing them safe.
    double fileSampleRate_ = 0;
    int    fileChannels_   = 0;

    /// The sentinel. Non-zero means "the file and everything describing it are
    /// in place"; zero means the callback must output silence and read none of
    /// it. Published with release, consumed with acquire.
    std::atomic<int>  fileSamples_ { 0 };

    std::atomic<int>  position_ { 0 };
    std::atomic<int>  seekRequest_ { -1 };   ///< -1 = none pending.
    std::atomic<bool> playing_  { false };
    std::atomic<bool> looping_  { true };
    std::atomic<bool> bypass_   { false };

    // Analysis
    dspark::SpectrumAnalyzer<float> spectrum_;
    dspark::LevelFollower<float>    meter_;
    std::atomic<float> peakL_ { 0.0f };
    std::atomic<float> peakR_ { 0.0f };

    // Test tone + live THD capture (post-FX channel 0)
    std::atomic<bool>  toneOn_   { false };
    std::atomic<float> toneFreq_ { 1000.0f };
    std::atomic<float> toneDb_   { -12.0f };
    double             tonePhase_ = 0.0;
    static constexpr int kThdCap = 32768;
    /// A stream of published words, not a set: the reader takes a window that
    /// may span two blocks, and over 0.7 s that costs it nothing.
    std::vector<std::atomic<float>> thdRing_ =
        std::vector<std::atomic<float>>(kThdCap);
    std::atomic<int>       thdHead_ { 0 };
    /// Frames captured since the last reset. Only ever advanced by the audio
    /// thread with fetch_add and only ever reset by the interface thread, so a
    /// reset cannot be swallowed by a read-modify-write it raced.
    std::atomic<long long> thdFill_ { 0 };

    // Waveform snapshot: a set published behind waveSeq_ (odd while writing).
    std::array<std::atomic<float>, kBlockSize> waveL_ {};
    std::array<std::atomic<float>, kBlockSize> waveR_ {};
    std::atomic<int>      waveCount_ { 0 };
    std::atomic<unsigned> waveSeq_   { 0 };

    // Effects
    std::vector<EffectSlot>* effects_ = nullptr;

    // miniaudio
    ma_device device_ {};
    bool      deviceInit_ = false;

    // Async load (worker thread + staging)
    std::thread                loadThread_;
    std::atomic<int>           loadState_ { 0 };  // LoadState as int (atomic-friendly)
    dspark::AudioBuffer<float> stagingBuffer_;
    dspark::AudioFileInfo      stagingInfo_ {};
    std::string                stagingPath_;
};

} // namespace dsplab
