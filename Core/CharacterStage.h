// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file CharacterStage.h
 * @brief An analog colour stage: emphasis, a chain of static saturators and
 *        the de-emphasis that undoes it, with slow drift, supply sag and a
 *        noise floor that follows the signal.
 *
 * The circuit an analog unit puts around its gain element (an input or
 * output amplifier) colours the signal without changing its level:
 *
 * ```
 *   x -> emphasis -> [ tanh(g e + b) - tanh(b) ] -> [ tanh(k s) / k ] -> / (g sech^2 b) -> de-emphasis -> + noise
 * ```
 *
 * - The emphasis is a first-order high shelf (+emphasis dB, centred on
 *   3 kHz) and the de-emphasis its exact inverse, so a small signal leaves
 *   as it came: the highs reach the saturators louder than the lows and
 *   come back to their level after them, and the stage saturates the top of
 *   the spectrum first, as tape and many amplifiers do.
 * - Two static saturators: one with a bias (the asymmetry, even
 *   harmonics), one symmetric that rounds what the first lets through. The
 *   output is divided by the small-signal gain of the chain, so at low
 *   levels the stage has unity gain and only the colour changes with drive.
 * - Drift: the drive and the bias wander slowly at random (about 0.3 Hz),
 *   each channel on its own, as the parts of two analog channels do.
 * - Sag: the drive rises with a slow envelope of the input (30 ms up,
 *   300 ms down), as a supply that droops under load lowers the point where
 *   the circuit clips.
 * - Noise: a floor proportional to a 50 ms envelope of the input, so
 *   silence stays silent.
 * - A first-order high-pass at 5 Hz removes the offset the asymmetry leaves
 *   under a signal (the coupling capacitor of the circuit).
 *
 * The emphasis and the saturators run at an internal rate of at least
 * 176.4 kHz (Oversampling::factorForRate(): 4x at 44.1 and 48 kHz, 2x at
 * 88.2 and 96 kHz, none from 176.4 kHz), through the polyphase IIR
 * half-bands, a few samples of delay (getLatency(), rounded at low
 * frequencies: 4 at 48 kHz). Measured at 48 kHz it costs 198 to 618 ns per
 * stereo frame (two saturators per channel at four times the rate). Every
 * setting is a lock-free atomic, safe from any thread
 * and before prepare(); the stage takes them at the next block, the drive in
 * a linear ramp over the block and the emphasis in a 20 ms glide.
 *
 * Threading: prepare() belongs to the setup thread (allocates); process()
 * and reset() to the audio thread. Allocation-free after prepare().
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DenormalGuard.h, DspMath.h,
 *               Oversampling.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "DenormalGuard.h"
#include "DspMath.h"
#include "Oversampling.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class CharacterStage
 * @brief Emphasis, static saturators and de-emphasis, with drift, sag and noise.
 *
 * @tparam T Sample type (float or double); the arithmetic runs in double.
 */
template <FloatType T>
class CharacterStage
{
public:
    /** @brief The settings of the stage. */
    struct Settings
    {
        T driveDb = T(6);        ///< Gain into the saturators, [0, 24] dB (the level out does not follow it).
        T emphasisDb = T(6);     ///< High shelf before the saturators, undone after them, [0, 12] dB.
        T asymmetry = T(0.3);    ///< Bias of the first saturator, [0, 1]: even harmonics.
        T drift = T(0);          ///< Slow random wander of drive (up to 1.5 dB) and bias, [0, 1].
        T sag = T(0);            ///< How much a slow envelope of the input raises the drive, [0, 1].
        T noiseDb = T(-140);     ///< Noise relative to the input's envelope, [-140, -40] dB; -140 is none.
    };

    static constexpr int kMaxChannels = 16;

    /**
     * @brief Allocates the oversampler and the scratch, and clears the state.
     *
     * An invalid spec is a no-op that keeps the previous state.
     *
     * @param spec Host rate, largest block and channel count.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        sampleRate_ = spec.sampleRate;
        maxBlock_ = spec.maxBlockSize;
        numChannels_ = std::min(spec.numChannels, kMaxChannels);
        factor_ = Oversampling<T>::factorForRate(sampleRate_);
        if (factor_ > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(factor_, Oversampling<T>::Quality::High,
                                                             Oversampling<T>::Mode::PolyphaseIir);
            oversampler_->prepare(AudioSpec { sampleRate_, maxBlock_, numChannels_ });
        }
        else
        {
            oversampler_.reset();
        }
        const double fs = sampleRate_;
        sagUp_ = 1.0 - std::exp(-1.0 / (0.030 * fs));
        sagDown_ = 1.0 - std::exp(-1.0 / (0.300 * fs));
        noiseRate_ = 1.0 - std::exp(-1.0 / (0.050 * fs));
        dcPole_ = std::exp(-2.0 * 3.141592653589793 * 5.0 / fs);
        driftRate_ = 1.0 - std::exp(-2.0 * 3.141592653589793 * kDriftHz * kDriftTick / fs);
        // A one-pole of rate r over a uniform in [-1, 1) has a deviation of
        // sqrt(r / (3 (2 - r))): scaled to about one.
        driftScale_ = 1.0 / std::sqrt(driftRate_ / (3.0 * (2.0 - driftRate_)));
        emphasisGlide_ = 1.0 - std::exp(-1.0 / (0.020 * fs));
        ch_.assign(static_cast<size_t>(numChannels_), Channel {});
        driveLin_.assign(static_cast<size_t>(numChannels_) * static_cast<size_t>(maxBlock_), 1.0);
        biasNow_.assign(driveLin_.size(), 0.0);
        reset();
    }

    /** @brief Clears every memory and restarts the random sources. Audio thread. */
    void reset() noexcept
    {
        for (size_t ch = 0; ch < ch_.size(); ++ch)
        {
            Channel& c = ch_[ch];
            c = Channel {};
            c.rng = 0x9E3779B9u * static_cast<std::uint32_t>(ch + 1) + 0x7F4A7C15u;
            c.driftTick = 0;
        }
        const Settings s = settings();
        driveNow_ = static_cast<double>(s.driveDb);
        emphasisNow_ = static_cast<double>(s.emphasisDb);
        if (sampleRate_ > 0.0) designEmphasis(emphasisNow_);
        if (oversampler_) oversampler_->reset();
    }

    /** @brief Sets every setting at once (clamped; non-finite fields are ignored). Any thread. */
    void setSettings(const Settings& s) noexcept
    {
        store(driveDb_, s.driveDb, T(0), T(24));
        store(emphasisDb_, s.emphasisDb, T(0), T(12));
        store(asymmetry_, s.asymmetry, T(0), T(1));
        store(drift_, s.drift, T(0), T(1));
        store(sag_, s.sag, T(0), T(1));
        store(noiseDb_, s.noiseDb, T(-140), T(-40));
    }

    /** @return The settings in use. */
    [[nodiscard]] Settings settings() const noexcept
    {
        return { driveDb_.load(std::memory_order_relaxed), emphasisDb_.load(std::memory_order_relaxed),
                 asymmetry_.load(std::memory_order_relaxed), drift_.load(std::memory_order_relaxed),
                 sag_.load(std::memory_order_relaxed), noiseDb_.load(std::memory_order_relaxed) };
    }

    /** @return The delay of the stage in host samples (the oversampler's, rounded at low frequencies). */
    [[nodiscard]] int getLatency() const noexcept { return oversampler_ ? oversampler_->getLatency() : 0; }

    /** @return The internal rate's factor over the host rate (1 before prepare()). */
    [[nodiscard]] int getFactor() const noexcept { return factor_; }

    /**
     * @brief Colours a block in place.
     *
     * Channels beyond those prepared (or beyond 16) are left untouched, as is
     * the whole block before prepare(). Blocks longer than the prepared size
     * run in pieces. Non-finite samples read as silence.
     *
     * @param buffer Audio to colour.
     */
    void process(AudioBufferView<T> buffer) noexcept
    {
        if (!(sampleRate_ > 0.0) || maxBlock_ <= 0) return;
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        if (nCh <= 0 || nS <= 0) return;
        DenormalGuard guard;
        T* channels[kMaxChannels] = {};
        for (int off = 0; off < nS; off += maxBlock_)
        {
            const int m = std::min(maxBlock_, nS - off);
            for (int ch = 0; ch < nCh; ++ch) channels[ch] = buffer.getChannel(ch) + off;
            processPart(AudioBufferView<T>(channels, nCh, m));
        }
    }

private:
    struct Channel
    {
        double pre1 = 0.0;          ///< Emphasis memory.
        double post1 = 0.0;         ///< De-emphasis memory.
        double sagEnv = 0.0;        ///< Slow envelope of the input (sag).
        double noiseEnv = 0.0;      ///< Envelope the noise follows.
        double dcX1 = 0.0, dcY1 = 0.0; ///< Output high-pass memory.
        double driftDb = 0.0, driftBias = 0.0;           ///< Drift now.
        double driftDbFrom = 0.0, driftBiasFrom = 0.0;   ///< Drift at the start of the tick.
        double driftDbTo = 0.0, driftBiasTo = 0.0;       ///< Drift at its end.
        double wanderDb = 0.0, wanderBias = 0.0;         ///< The smoothed random walks.
        int driftTick = 0;          ///< Samples into the tick.
        std::uint32_t rng = 1u;     ///< Noise and drift source.
    };

    static constexpr double kEmphasisHz = 3000.0;   ///< Centre of the emphasis shelf.
    static constexpr double kSecond = 1.5;          ///< Gain into the second saturator.
    static constexpr double kBias = 0.35;           ///< Bias at full asymmetry.
    static constexpr double kSagGain = 2.0;         ///< Drive rise per unit of envelope at full sag.
    static constexpr double kDriftDb = 1.5;         ///< Drive wander at full drift.
    static constexpr double kDriftBias = 0.08;      ///< Bias wander at full drift.
    static constexpr double kDriftHz = 0.3;         ///< Corner of the random walks.
    static constexpr int kDriftTick = 64;           ///< Host samples between draws.

    static void store(std::atomic<T>& a, T v, T lo, T hi) noexcept
    {
        if (std::isfinite(v)) a.store(std::clamp(v, lo, hi), std::memory_order_relaxed);
    }

    /** @brief Uniform in [-1, 1). */
    static double uniform(std::uint32_t& s) noexcept
    {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<double>(s) * (2.0 / 4294967296.0) - 1.0;
    }

    /**
     * @brief The emphasis at the internal rate: (s + wz) / (s + wp) * wp / wz,
     *        wz and wp an octave-scaled pair around kEmphasisHz with wp / wz
     *        the shelf's gain, prewarped; the de-emphasis is its inverse.
     */
    void designEmphasis(double db) noexcept
    {
        const double fsInt = sampleRate_ * factor_;
        const double gain = std::pow(10.0, db / 20.0);
        const double root = std::sqrt(gain);
        const double kk = 2.0 * fsInt;
        const double wz = kk * std::tan(3.141592653589793 * std::min(kEmphasisHz / root, 0.45 * fsInt) / fsInt);
        const double wp = kk * std::tan(3.141592653589793 * std::min(kEmphasisHz * root, 0.45 * fsInt) / fsInt);
        const double scale = wp / wz;
        preB0_ = scale * (kk + wz) / (kk + wp);
        preB1_ = scale * (wz - kk) / (kk + wp);
        preA1_ = (wp - kk) / (kk + wp);
    }

    void processPart(AudioBufferView<T> part) noexcept
    {
        const int nCh = part.getNumChannels();
        const int n = part.getNumSamples();
        const Settings s = settings();

        // The emphasis glides once per block towards its setting.
        const double emphTarget = static_cast<double>(s.emphasisDb);
        if (emphasisNow_ != emphTarget)
        {
            const double glide = 1.0 - std::pow(1.0 - emphasisGlide_, static_cast<double>(n));
            emphasisNow_ += glide * (emphTarget - emphasisNow_);
            if (std::abs(emphTarget - emphasisNow_) < 1e-4) emphasisNow_ = emphTarget;
            designEmphasis(emphasisNow_);
        }
        const double driveFrom = driveNow_, driveTo = static_cast<double>(s.driveDb);
        driveNow_ = driveTo;
        const double asym = static_cast<double>(s.asymmetry) * kBias;
        const double drift = static_cast<double>(s.drift);
        const double sag = static_cast<double>(s.sag) * kSagGain;
        const bool noisy = s.noiseDb > T(-140);
        const double noiseGain = noisy ? std::pow(10.0, static_cast<double>(s.noiseDb) / 20.0) * 1.7320508075688772 : 0.0;

        // Host rate: the envelopes and the drift, per sample, into arrays the
        // internal rate reads (held across each group of factor_ samples).
        for (int ch = 0; ch < nCh; ++ch)
        {
            Channel& c = ch_[static_cast<size_t>(ch)];
            T* x = part.getChannel(ch);
            double* drive = driveLin_.data() + static_cast<size_t>(ch) * static_cast<size_t>(maxBlock_);
            double* bias = biasNow_.data() + static_cast<size_t>(ch) * static_cast<size_t>(maxBlock_);
            for (int i = 0; i < n; ++i)
            {
                const T raw = x[i];
                if (!std::isfinite(raw)) x[i] = T(0);
                const double level = std::abs(static_cast<double>(x[i]));
                c.sagEnv += (level > c.sagEnv ? sagUp_ : sagDown_) * (level - c.sagEnv);
                c.noiseEnv += noiseRate_ * (level - c.noiseEnv);
                if (c.driftTick == 0)
                {
                    c.driftDbFrom = c.driftDbTo;
                    c.driftBiasFrom = c.driftBiasTo;
                    c.wanderDb += driftRate_ * (uniform(c.rng) - c.wanderDb);
                    c.wanderBias += driftRate_ * (uniform(c.rng) - c.wanderBias);
                    c.driftDbTo = driftScale_ * c.wanderDb;
                    c.driftBiasTo = driftScale_ * c.wanderBias;
                }
                const double t = static_cast<double>(c.driftTick + 1) / kDriftTick;
                c.driftDb = c.driftDbFrom + t * (c.driftDbTo - c.driftDbFrom);
                c.driftBias = c.driftBiasFrom + t * (c.driftBiasTo - c.driftBiasFrom);
                c.driftTick = (c.driftTick + 1) % kDriftTick;

                const double ramp = driveFrom + (driveTo - driveFrom) * (static_cast<double>(i + 1) / n);
                const double db = ramp + drift * kDriftDb * std::clamp(c.driftDb, -1.0, 1.0);
                drive[i] = std::pow(10.0, db / 20.0) * (1.0 + sag * c.sagEnv);
                bias[i] = asym + drift * kDriftBias * std::clamp(c.driftBias, -1.0, 1.0);
            }
        }

        // Internal rate: emphasis, saturators, de-emphasis.
        AudioBufferView<T> work = oversampler_ ? oversampler_->upsample(part) : part;
        const double invB0 = 1.0 / preB0_, postB1 = preA1_ * invB0, postA1 = preB1_ * invB0;
        for (int ch = 0; ch < nCh; ++ch)
        {
            Channel& c = ch_[static_cast<size_t>(ch)];
            T* u = work.getChannel(ch);
            const double* drive = driveLin_.data() + static_cast<size_t>(ch) * static_cast<size_t>(maxBlock_);
            const double* bias = biasNow_.data() + static_cast<size_t>(ch) * static_cast<size_t>(maxBlock_);
            double pre1 = c.pre1, post1 = c.post1;
            for (int i = 0, j = 0; i < n; ++i)
            {
                const double g = drive[i], b = bias[i];
                const double tb = std::tanh(b);
                const double back = 1.0 / (g * (1.0 - tb * tb));   // the chain's small-signal gain, undone
                for (int k = 0; k < factor_; ++k, ++j)
                {
                    const double in = static_cast<double>(u[j]);
                    // Emphasis, transposed direct form II.
                    const double e = preB0_ * in + pre1;
                    pre1 = preB1_ * in - preA1_ * e;
                    const double s1 = std::tanh(g * e + b) - tb;
                    const double y = std::tanh(kSecond * s1) * (back / kSecond);
                    // De-emphasis: the inverse of the emphasis.
                    const double d = invB0 * y + post1;
                    post1 = postB1 * y - postA1 * d;
                    u[j] = static_cast<T>(d);
                }
            }
            if (!std::isfinite(pre1 + post1)) { pre1 = 0.0; post1 = 0.0; }
            c.pre1 = pre1;
            c.post1 = post1;
        }
        if (oversampler_) oversampler_->downsample(part);

        // Host rate: the coupling high-pass and the noise floor.
        for (int ch = 0; ch < nCh; ++ch)
        {
            Channel& c = ch_[static_cast<size_t>(ch)];
            T* x = part.getChannel(ch);
            double x1 = c.dcX1, y1 = c.dcY1, env = c.noiseEnv;
            for (int i = 0; i < n; ++i)
            {
                const double v = static_cast<double>(x[i]);
                const double hp = v - x1 + dcPole_ * y1;
                x1 = v;
                y1 = hp;
                double out = hp;
                if (noisy) out += noiseGain * env * uniform(c.rng);
                x[i] = static_cast<T>(out);
            }
            if (!std::isfinite(x1 + y1)) { x1 = 0.0; y1 = 0.0; }
            c.dcX1 = x1;
            c.dcY1 = y1;
        }
    }

    std::atomic<T> driveDb_ { T(6) };
    std::atomic<T> emphasisDb_ { T(6) };
    std::atomic<T> asymmetry_ { T(0.3) };
    std::atomic<T> drift_ { T(0) };
    std::atomic<T> sag_ { T(0) };
    std::atomic<T> noiseDb_ { T(-140) };

    double sampleRate_ = 0.0;
    int maxBlock_ = 0;
    int numChannels_ = 0;
    int factor_ = 1;
    std::unique_ptr<Oversampling<T>> oversampler_;
    std::vector<Channel> ch_;                ///< Per channel (allocated by prepare()).
    std::vector<double> driveLin_;           ///< Drive per host sample, channel after channel.
    std::vector<double> biasNow_;            ///< Bias per host sample, channel after channel.

    double preB0_ = 1.0, preB1_ = 0.0, preA1_ = 0.0;   ///< The emphasis (its inverse is the de-emphasis).
    double driveNow_ = 6.0, emphasisNow_ = 6.0;
    double sagUp_ = 0.0, sagDown_ = 0.0, noiseRate_ = 0.0, dcPole_ = 0.0, driftRate_ = 0.0, emphasisGlide_ = 0.0;
    double driftScale_ = 1.0;
};

} // namespace dspark
