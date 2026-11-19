// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TonalBalance.h
 * @brief An adaptive equalizer driven by an auditory model: it measures the
 *        loudness of the programme in bands of the ear's scale and moves a
 *        bank of filters toward a balanced shape, raising what is missing
 *        and taming what stands out, while the total loudness stays.
 *
 * The analysis looks at the programme every hop (128 samples at 48 kHz)
 * through a transform of 2048 with an asymmetric window whose weight sits
 * on the most recent samples (low latency); the analysis scales by octaves
 * with the sample rate, so it sees the same span at any rate. The power
 * spectrum is smoothed across frequency with a width that follows frequency
 * (SpectralEnvelope), read in 72 bands uniform on the Bark scale from 20 Hz
 * to 20 kHz (AuditoryBands) and smoothed across the bands (the ear's
 * spread). In each band the covariance of the two channels is smoothed in
 * time, longer in the low end, and the band's power is its larger
 * eigenvalue: the level of the dominant component wherever it sits in the
 * stereo image. The specific loudness of each band (the power law over the
 * threshold in quiet) is smoothed in time and read in decibels of power.
 *
 * From there: a broad trend across the bands, and two deviations per band,
 * the trend against a reference shape (a tilt in dB per octave; the global
 * balance) and the band against its trend (the local contrast). Two
 * soft-knee curves (hyperbolas, with a different slope on each side) turn a
 * deviation into a correction: setBalance() moves the trend toward the
 * reference, setBoost() raises bands that sit under their trend and
 * setTame() lowers those that stand over it. The corrections are limited
 * (setRange()), faded at the edges of setFrequencyRange() over one Bark, and
 * then offset so the total loudness of the programme does not change (the
 * correction redistributes). setLevelCompensation(true) converts loudness
 * to decibels with each band's own loudness exponent (the low end, whose
 * loudness grows faster with level, moves less in decibels for the same
 * change of loudness).
 *
 * The gains drive a GraphicBank (a bell per band, a shelf at each end,
 * their gains solved by least squares so the bank's response is the one
 * asked for) whose sections glide over each hop. The programme is delayed
 * to line up with the analysis. processBlock(buffer, key) analyses an
 * external key instead, with a fade of 50 ms when the key comes and goes.
 *
 * Latency: the analysis delay, about 16 ms at 48 kHz, plus 7 samples of the
 * bank in mono (getLatency()).
 *
 * Threading: prepare() and setState() on the setup thread (allocate);
 * processBlock() and reset() on the audio thread; the setters are lock-free
 * atomic publications, safe from any thread. Non-finite values are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, AuditoryBands.h, DenormalGuard.h,
 * DspMath.h, FFT.h, GraphicBank.h, SpectralEnvelope.h, StateBlob.h,
 * WindowFunctions.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AuditoryBands.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/GraphicBank.h"
#include "../Core/SpectralEnvelope.h"
#include "../Core/StateBlob.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class TonalBalance
 * @brief Adaptive equalizer toward a balanced loudness shape.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TonalBalance
{
public:
    static constexpr int kBands = 72;      ///< Bands of the analysis and of the bank.
    static constexpr int kMaxChannels = 2; ///< Channels processed (more pass through).

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        fs_ = spec.sampleRate;
        channels_ = std::min(spec.numChannels, kMaxChannels);
        // The analysis scales by octaves with the rate (2048 at 44.1 to 48 kHz).
        const int octaves = static_cast<int>(std::lround(std::log2(std::max(fs_, 8000.0) / 48000.0)));
        fft_ = std::clamp(2048 << std::max(0, octaves), 1024, 16384);
        if (octaves < 0) fft_ = std::max(512, 2048 >> -octaves);
        hop_ = std::max(16, fft_ / 16);
        const int fall = fft_ / 8;
        transform_ = std::make_unique<FFTReal<double>>(static_cast<size_t>(fft_));
        window_.resize(static_cast<size_t>(fft_));
        WindowFunctions<double>::asymmetricHann(window_.data(), fft_, fall);
        double sw = 0.0, swi = 0.0;
        for (int i = 0; i < fft_; ++i) { sw += window_[static_cast<size_t>(i)]; swi += window_[static_cast<size_t>(i)] * i; }
        // Power scale: a full-scale sine reads 1 in its bin.
        powerScale_ = 4.0 / (sw * sw);
        // The programme waits for the analysis: from the window's centre of
        // mass to its end, plus half a hop (the gains glide over a hop).
        analysisDelay_ = std::max(1, static_cast<int>(std::lround(static_cast<double>(fft_ - 1) - swi / sw + 0.5 * hop_)));
        bins_ = fft_ / 2 + 1;
        bands_.prepare(fs_, fft_, kBands, 20.0, std::min(20000.0, 0.45 * fs_));
        bands_.setReferenceSpl(static_cast<double>(referenceSpl_.load(std::memory_order_relaxed)));
        envelope_.prepare(bins_, 1.0 / 6.0);
        std::array<double, kBands> centres {};
        for (int b = 0; b < kBands; ++b) centres[static_cast<size_t>(b)] = bands_.getCentre(b);
        bank_.prepare(fs_, centres.data(), kBands, channels_);
        ring_.assign(static_cast<size_t>(channels_), std::vector<double>(static_cast<size_t>(fft_), 0.0));
        spectra_.assign(static_cast<size_t>(channels_), std::vector<double>(static_cast<size_t>(fft_ + 2), 0.0));
        delay_.assign(static_cast<size_t>(channels_), std::vector<T>(static_cast<size_t>(analysisDelay_), T(0)));
        frame_.assign(static_cast<size_t>(fft_), 0.0);
        binPower_.assign(static_cast<size_t>(bins_), 0.0);
        binSmooth_.assign(static_cast<size_t>(bins_), 0.0);
        for (auto* v : { &ll_, &rr_, &lr_, &loud_, &corr_, &tmpA_, &tmpB_, &tmpC_ }) v->assign(kBands, 0.0);
        for (int b = 0; b < kBands; ++b)
        {
            const double tau = 0.001 * std::clamp(200.0 * std::pow(1000.0 / bands_.getCentre(b), 0.3), 20.0, 2000.0);
            timeCoeff_[static_cast<size_t>(b)] = std::exp(-static_cast<double>(hop_) / (tau * fs_));
        }
        prepared_ = true;
        reset();
    }

    /** @brief Clears the analysis, the corrections, the bank and the delay. */
    void reset() noexcept
    {
        for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
        for (auto& d : delay_) std::fill(d.begin(), d.end(), T(0));
        for (auto* v : { &ll_, &rr_, &lr_, &loud_, &corr_ }) std::fill(v->begin(), v->end(), 0.0);
        ringPos_ = delayPos_ = hopCounter_ = hops_ = 0;
        keyFade_ = 0.0;
        warm_ = false;
        bank_.reset();
        std::array<T, kBands> zero {};
        bank_.setGains(zero.data(), 0, 0);
        for (auto& g : published_) g.store(T(0), std::memory_order_relaxed);
    }

    // -- Controls ------------------------------------------------------------

    /** @brief How far the trend moves toward the reference shape, 0 to 1 (default 0.5). */
    void setBalance(T amount) noexcept { store(balance_, amount, T(0), T(1)); }
    /** @brief How much bands under their trend are raised, 0 to 1 (default 0.3). */
    void setBoost(T amount) noexcept { store(boost_, amount, T(0), T(1)); }
    /** @brief How much bands over their trend are lowered, 0 to 1 (default 0.5). */
    void setTame(T amount) noexcept { store(tame_, amount, T(0), T(1)); }
    /** @brief Tilt of the reference shape in dB per octave around 1 kHz, -6 to +6 (default 0: flat loudness per Bark). */
    void setTilt(T dbPerOctave) noexcept { store(tilt_, dbPerOctave, T(-6), T(6)); }
    /** @brief Largest correction of a band, 0 to 24 dB (default 6). */
    void setRange(T db) noexcept { store(range_, db, T(0), T(24)); }
    /** @brief Knee of the correction curves, 0.1 to 12 dB (default 3). */
    void setKnee(T db) noexcept { store(knee_, db, T(0.1), T(12)); }
    /** @brief Frequencies the corrections act on, with edges one Bark wide (default 40 Hz to 16 kHz: the ends of the band, where the threshold in quiet climbs, are left alone). */
    void setFrequencyRange(T loHz, T hiHz) noexcept
    {
        if (!std::isfinite(loHz) || !std::isfinite(hiHz)) return;
        const T lo = std::clamp(std::min(loHz, hiHz), T(10), T(24000));
        loHz_.store(lo, std::memory_order_relaxed);
        hiHz_.store(std::clamp(std::max(loHz, hiHz), lo, T(24000)), std::memory_order_relaxed);
    }
    /** @brief Loudness to decibels with each band's exponent (default off: one exponent). */
    void setLevelCompensation(bool on) noexcept { levelComp_.store(on, std::memory_order_relaxed); }
    /** @brief The SPL a full-scale sine stands for, 60 to 130 dB (default 100). */
    void setReferenceLevel(T dbSpl) noexcept { store(referenceSpl_, dbSpl, T(60), T(130)); refDirty_.store(true, std::memory_order_release); }

    [[nodiscard]] T getBalance() const noexcept { return balance_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBoost() const noexcept { return boost_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTame() const noexcept { return tame_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getTilt() const noexcept { return tilt_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRange() const noexcept { return range_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getKnee() const noexcept { return knee_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLowFrequency() const noexcept { return loHz_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHighFrequency() const noexcept { return hiHz_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getLevelCompensation() const noexcept { return levelComp_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getReferenceLevel() const noexcept { return referenceSpl_.load(std::memory_order_relaxed); }

    /** @return The latency in samples (the analysis delay plus the bank's). */
    [[nodiscard]] int getLatency() const noexcept { return analysisDelay_ + bank_.getLatency(); }
    /** @return A band's centre in hertz. */
    [[nodiscard]] double getBandCentre(int band) const noexcept { return bands_.getCentre(band); }
    /** @return The correction of a band in dB (a readout for any thread). */
    [[nodiscard]] T getCorrection(int band) const noexcept
    {
        return published_[static_cast<size_t>(std::clamp(band, 0, kBands - 1))].load(std::memory_order_relaxed);
    }
    /** @return The bank's response at a frequency in dB (audio thread or after processing). */
    [[nodiscard]] double getResponse(double hz) const noexcept { return bank_.responseDb(hz); }

    // -- Processing ----------------------------------------------------------

    /** @brief Analyses and equalizes the programme in place. */
    void processBlock(AudioBufferView<T> buffer) noexcept { run(buffer, nullptr); }

    /** @brief Equalizes the programme by the analysis of a key (same channels or one). */
    void processBlock(AudioBufferView<T> buffer, AudioBufferView<T> key) noexcept
    {
        run(buffer, key.getNumSamples() >= buffer.getNumSamples() && key.getNumChannels() > 0 ? &key : nullptr);
    }

    // -- State ---------------------------------------------------------------

    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TBAL"), 1);
        w.write("balance", static_cast<float>(getBalance()));
        w.write("boost", static_cast<float>(getBoost()));
        w.write("tame", static_cast<float>(getTame()));
        w.write("tilt", static_cast<float>(getTilt()));
        w.write("range", static_cast<float>(getRange()));
        w.write("knee", static_cast<float>(getKnee()));
        w.write("loHz", static_cast<float>(getLowFrequency()));
        w.write("hiHz", static_cast<float>(getHighFrequency()));
        w.write("levelComp", getLevelCompensation());
        w.write("refSpl", static_cast<float>(getReferenceLevel()));
        return w.blob();
    }

    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TBAL")) return false;
        setBalance(static_cast<T>(r.read("balance", 0.5f)));
        setBoost(static_cast<T>(r.read("boost", 0.3f)));
        setTame(static_cast<T>(r.read("tame", 0.5f)));
        setTilt(static_cast<T>(r.read("tilt", 0.0f)));
        setRange(static_cast<T>(r.read("range", 6.0f)));
        setKnee(static_cast<T>(r.read("knee", 3.0f)));
        setFrequencyRange(static_cast<T>(r.read("loHz", 40.0f)), static_cast<T>(r.read("hiHz", 16000.0f)));
        setLevelCompensation(r.read("levelComp", false));
        setReferenceLevel(static_cast<T>(r.read("refSpl", 100.0f)));
        return true;
    }

private:
    static void store(std::atomic<T>& a, T v, T lo, T hi) noexcept
    {
        if (std::isfinite(v)) a.store(std::clamp(v, lo, hi), std::memory_order_relaxed);
    }

    void run(AudioBufferView<T>& buffer, const AudioBufferView<T>* key) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        if (refDirty_.exchange(false, std::memory_order_acquire))
            bands_.setReferenceSpl(static_cast<double>(referenceSpl_.load(std::memory_order_relaxed)));
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        const int nS = buffer.getNumSamples();
        if (nCh <= 0) return;
        const double fadeStep = 1.0 / (0.05 * fs_);
        const double fadeTo = key != nullptr ? 1.0 : 0.0;
        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, hop_ - hopCounter_);
            // Feed the analysis (programme or key, faded) and delay the programme.
            for (int k = 0; k < chunk; ++k)
            {
                keyFade_ = fadeTo > keyFade_ ? std::min(fadeTo, keyFade_ + fadeStep) : std::max(fadeTo, keyFade_ - fadeStep);
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* data = buffer.getChannel(ch);
                    double a = static_cast<double>(data[i + k]);
                    if (keyFade_ > 0.0 && key != nullptr)
                    {
                        const int kc = std::min(ch, key->getNumChannels() - 1);
                        a += keyFade_ * (static_cast<double>(key->getChannel(kc)[i + k]) - a);
                    }
                    ring_[static_cast<size_t>(ch)][static_cast<size_t>((ringPos_ + k) & (fft_ - 1))] = std::isfinite(a) ? a : 0.0;
                    auto& d = delay_[static_cast<size_t>(ch)];
                    const T x = data[i + k];
                    data[i + k] = d[static_cast<size_t>(delayPos_)];
                    d[static_cast<size_t>(delayPos_)] = std::isfinite(x) ? x : T(0);
                }
                delayPos_ = delayPos_ + 1 == static_cast<int>(delay_[0].size()) ? 0 : delayPos_ + 1;
            }
            ringPos_ = (ringPos_ + chunk) & (fft_ - 1);
            // The bank on the delayed chunk.
            {
                T* ptr[kMaxChannels] {};
                for (int ch = 0; ch < nCh; ++ch) ptr[ch] = buffer.getChannel(ch) + i;
                bank_.processBlock(AudioBufferView<T>(ptr, nCh, chunk));
            }
            hopCounter_ += chunk;
            if (hopCounter_ >= hop_)
            {
                hopCounter_ = 0;
                analyse(nCh);
            }
            i += chunk;
        }
    }

    /** @brief One hop of the analysis, and the new gains of the bank. */
    void analyse(int nCh) noexcept
    {
        for (int ch = 0; ch < nCh; ++ch)
        {
            const auto& r = ring_[static_cast<size_t>(ch)];
            for (int k = 0; k < fft_; ++k)
                frame_[static_cast<size_t>(k)] = r[static_cast<size_t>((ringPos_ + k) & (fft_ - 1))] * window_[static_cast<size_t>(k)];
            transform_->forward(frame_.data(), spectra_[static_cast<size_t>(ch)].data());
        }
        // Band powers of the two channels and of their cross term, each from
        // a spectrum smoothed across frequency.
        const auto bandsOf = [&](auto&& binValue, std::vector<double>& out) {
            for (int k = 0; k < bins_; ++k) binPower_[static_cast<size_t>(k)] = binValue(k);
            envelope_.compute(binPower_.data(), binSmooth_.data());
            bands_.bandPowers(binSmooth_.data(), out.data());
            AuditoryBands<double>::smoothAcross(out.data(), kBands, 0.25);
        };
        const double* s0 = spectra_[0].data();
        const double* s1 = spectra_[static_cast<size_t>(nCh > 1 ? 1 : 0)].data();
        bandsOf([&](int k) { return powerScale_ * (s0[2 * k] * s0[2 * k] + s0[2 * k + 1] * s0[2 * k + 1]); }, tmpA_);
        if (nCh > 1)
        {
            bandsOf([&](int k) { return powerScale_ * (s1[2 * k] * s1[2 * k] + s1[2 * k + 1] * s1[2 * k + 1]); }, tmpB_);
            bandsOf([&](int k) { return powerScale_ * (s0[2 * k] * s1[2 * k] + s0[2 * k + 1] * s1[2 * k + 1]); }, tmpC_);
        }
        // The covariance in time, longer in the low end; the dominant power.
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double c = warm_ ? timeCoeff_[i] : 0.0;
            ll_[i] = tmpA_[i] + c * (ll_[i] - tmpA_[i]);
            if (nCh > 1)
            {
                rr_[i] = tmpB_[i] + c * (rr_[i] - tmpB_[i]);
                lr_[i] = tmpC_[i] + c * (lr_[i] - tmpC_[i]);
                tmpA_[i] = AuditoryScale::dominantPower(ll_[i], rr_[i], lr_[i]);
            }
            else
            {
                tmpA_[i] = ll_[i];
            }
        }
        bands_.specificLoudness(tmpA_.data(), tmpB_.data());
        const bool comp = levelComp_.load(std::memory_order_relaxed);
        constexpr double kMidExponent = 0.23;
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double c = warm_ ? timeCoeff_[i] : 0.0;
            loud_[i] = tmpB_[i] + c * (loud_[i] - tmpB_[i]);
        }
        warm_ = true;
        // Loudness in decibels of power, the trend, the two deviations.
        const double tilt = static_cast<double>(tilt_.load(std::memory_order_relaxed));
        const double lo = static_cast<double>(loHz_.load(std::memory_order_relaxed));
        const double hi = static_cast<double>(hiHz_.load(std::memory_order_relaxed));
        const double zLo = AuditoryScale::hzToBark(lo), zHi = AuditoryScale::hzToBark(hi);
        std::array<double, kBands> level {}, trend {}, weight {}, exponent {};
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            exponent[i] = comp ? bands_.getExponent(b) : kMidExponent;
            level[i] = 10.0 * std::log10(loud_[i] + 1e-12) / exponent[i];
            trend[i] = level[i];
            const double z = bands_.getBark(b);
            weight[i] = std::clamp(z - zLo + 0.5, 0.0, 1.0) * std::clamp(zHi - z + 0.5, 0.0, 1.0);
        }
        AuditoryBands<double>::smoothAcross(trend.data(), kBands, 0.9);
        double meanDev = 0.0, wsum = 0.0;
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double ref = tilt * std::log2(bands_.getCentre(b) / 1000.0);
            tmpC_[i] = trend[i] - ref;
            meanDev += weight[i] * tmpC_[i];
            wsum += weight[i];
        }
        meanDev = wsum > 0.0 ? meanDev / wsum : 0.0;
        const double balance = static_cast<double>(balance_.load(std::memory_order_relaxed));
        const double boost = static_cast<double>(boost_.load(std::memory_order_relaxed));
        const double tame = static_cast<double>(tame_.load(std::memory_order_relaxed));
        const double range = static_cast<double>(range_.load(std::memory_order_relaxed));
        const double knee = static_cast<double>(knee_.load(std::memory_order_relaxed));
        const auto soft = [knee](double d) { return std::copysign(std::sqrt(d * d + knee * knee) - knee, d); };
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double global = tmpC_[i] - meanDev;
            const double local = level[i] - trend[i];
            double c = -balance * soft(global) - (local > 0.0 ? tame : boost) * soft(local);
            c = std::clamp(c, -range, range) * weight[i];
            corr_[i] = c;
        }
        // The total loudness stays: offset every band by the same loudness ratio.
        double before = 0.0, after = 0.0;
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double w = (b == 0 || b == kBands - 1) ? 0.5 : 1.0;
            before += w * loud_[i];
            after += w * loud_[i] * std::pow(10.0, exponent[i] * corr_[i] / 10.0);
        }
        const double k = after > 1e-20 ? before / after : 1.0;
        std::array<T, kBands> gains {};
        for (int b = 0; b < kBands; ++b)
        {
            const auto i = static_cast<size_t>(b);
            corr_[i] = std::clamp(corr_[i] + 10.0 * std::log10(k) / exponent[i] * weight[i], -range, range);
            gains[i] = static_cast<T>(corr_[i]);
            published_[i].store(gains[i], std::memory_order_relaxed);
        }
        bank_.setGains(gains.data(), hop_, (hops_++ & 7) == 0 ? 1 : 0);
    }

    double fs_ = 48000.0, powerScale_ = 1.0, keyFade_ = 0.0;
    int channels_ = 0, fft_ = 2048, hop_ = 128, bins_ = 1025, analysisDelay_ = 0;
    int ringPos_ = 0, delayPos_ = 0, hopCounter_ = 0, hops_ = 0;
    bool prepared_ = false, warm_ = false;

    std::atomic<T> balance_ { T(0.5) }, boost_ { T(0.3) }, tame_ { T(0.5) }, tilt_ { T(0) };
    std::atomic<T> range_ { T(6) }, knee_ { T(3) }, loHz_ { T(40) }, hiHz_ { T(16000) }, referenceSpl_ { T(100) };
    std::atomic<bool> levelComp_ { false }, refDirty_ { false };
    std::array<std::atomic<T>, kBands> published_ {};

    std::unique_ptr<FFTReal<double>> transform_;
    AuditoryBands<double> bands_;
    SpectralEnvelope<double> envelope_;
    GraphicBank<T, 8> bank_;
    std::vector<double> window_, frame_, binPower_, binSmooth_;
    std::vector<std::vector<double>> ring_, spectra_;
    std::vector<std::vector<T>> delay_;
    std::vector<double> ll_, rr_, lr_, loud_, corr_, tmpA_, tmpB_, tmpC_;
    std::array<double, kBands> timeCoeff_ {};
};

} // namespace dspark
