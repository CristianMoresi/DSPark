// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file AnalogRandom.h
 * @brief Analog-style random modulation: white/pink/brown drift generators.
 *
 * Header-only C++20 generator of smooth, analog-style random values for
 * modulation. Supports white, pink, and brown noise types.
 *
 * Features:
 * - True audio-rate analog drift (continuous 1/f and 1/f^2 filtering).
 * - Sample-and-Hold LFO behavior with 1-pole smoothing.
 * - Zero allocations, cache-friendly block processing.
 * - Lock-free thread safety between GUI and audio threads.
 * - DC-free denormal mitigation.
 *
 * Dependencies: AnalogConstants.h, C++20 standard library (<algorithm>,
 * <array>, <atomic>, <chrono>, <cmath>, <cstdint>, <span>, <type_traits>).
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>

#include "AnalogConstants.h"

/**
 * @namespace dspark
 * @brief Main namespace for the DSPark framework.
 */
namespace dspark
{
    /**
     * @namespace AnalogRandom
     * @brief Main namespace for the analog random generation library.
     */
    namespace AnalogRandom
    {
        //==============================================================================
        // PRNG and Internal Details
        // (Reference constants live in Core/AnalogConstants.h, included above.)
        //==============================================================================
        namespace Detail
        {
            /**
             * @class Xoshiro256pp
             * @brief A compact xoshiro256++ PRNG implementation (non-threadsafe).
             */
            struct Xoshiro256pp
            {
                std::uint64_t s[4];

                explicit Xoshiro256pp(std::uint64_t seed = 1) noexcept
                {
                    reseed(seed);
                }

                void reseed(std::uint64_t seed) noexcept
                {
                    if (seed == 0) seed = 1;
                    s[0] = splitmix64(seed + 0x9E3779B97F4A7C15ull);
                    s[1] = splitmix64(s[0]);
                    s[2] = splitmix64(s[1]);
                    s[3] = splitmix64(s[2]);
                }

                [[nodiscard]] std::uint64_t next() noexcept
                {
                    const std::uint64_t result = rotl(s[0] + s[3], 23) + s[0];
                    const std::uint64_t t = s[1] << 17;
                    s[2] ^= s[0];
                    s[3] ^= s[1];
                    s[1] ^= s[2];
                    s[0] ^= s[3];
                    s[2] ^= t;
                    s[3] = rotl(s[3], 45);
                    return result;
                }

                [[nodiscard]] double next_double() noexcept
                {
                    const std::uint64_t x = next();
                    constexpr double inv2pow53 = 1.0 / 9007199254740992.0;
                    return static_cast<double>(x >> 11) * inv2pow53;
                }

            private:
                static std::uint64_t rotl(std::uint64_t x, int k) noexcept
                {
                    return (x << k) | (x >> (64 - k));
                }

                static std::uint64_t splitmix64(std::uint64_t x) noexcept
                {
                    x += 0x9E3779B97F4A7C15ull;
                    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
                    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
                    return x ^ (x >> 31);
                }
            };

            // NOTE: lock-free safety is checked at runtime per Generator instance
            // (checkLockFree() -> isSafeToRun_). A preprocessor #warning here would
            // fire on every translation unit unconditionally (the directive ignores
            // the runtime `if`), so it is intentionally omitted.
        }

        //==============================================================================
        // Public API
        //==============================================================================

        /**
         * @enum NoiseType
         * @brief Defines the spectral distribution of the random fluctuations.
         */
        enum class NoiseType
        {
            Pink,  ///< 1/f noise for organic drift.
            Brown, ///< 1/f^2 noise for slow "wow/flutter".
            White  ///< Uncorrelated noise for "sample-and-hold" effects.
        };

        enum class BpmDivision
        {
            One, Half, Quarter, Eighth, Sixteenth, ThirtySecond,
            HalfTriplet, QuarterTriplet, EighthTriplet, SixteenthTriplet,
            DottedHalf, DottedQuarter, DottedEighth,
        };

        enum class AnalogComponent
        {
            TapeMachine, VacuumTube, Transistor, Compressor, Equalizer
        };

        /**
         * @class Generator
         * @brief Main generator class for analog-style random modulation.
         * @tparam Real Floating point type to represent values (float or double).
         */
        template <typename Real = float>
        class Generator
        {
            static_assert(std::is_floating_point_v<Real>,
                        "AnalogRandom::Generator requires a floating-point Real type.");

        public:
            Generator() noexcept : prng_(generateUniqueSeed())
            {
                checkLockFree();
            }

            explicit Generator(std::uint64_t seed) noexcept : prng_(seed)
            {
                checkLockFree();
            }

            // std::atomic members are not move-constructible by default, but we
            // own this Generator's state and need it to be storable in standard
            // containers (e.g. std::vector). The atomic loads/stores below are
            // safe because the source object is being moved-from; by contract
            // it is not concurrently observed by any other thread.
            Generator(Generator&& other) noexcept
                : isSafeToRun_(other.isSafeToRun_),
                  prng_(other.prng_),
                  sampleRate_(other.sampleRate_),
                  phaseAccumulator_(other.phaseAccumulator_),
                  triggerNext_(other.triggerNext_.load(std::memory_order_relaxed)),
                  currentValue_(other.currentValue_),
                  targetValue_(other.targetValue_),
                  noiseType_(other.noiseType_.load(std::memory_order_relaxed)),
                  useBpmSync_(other.useBpmSync_.load(std::memory_order_relaxed)),
                  rateHz_(other.rateHz_.load(std::memory_order_relaxed)),
                  bpm_(other.bpm_.load(std::memory_order_relaxed)),
                  bpmDivision_(other.bpmDivision_.load(std::memory_order_relaxed)),
                  min_(other.min_.load(std::memory_order_relaxed)),
                  max_(other.max_.load(std::memory_order_relaxed)),
                  smoothingEnabled_(other.smoothingEnabled_.load(std::memory_order_relaxed)),
                  smoothingCoeff_(other.smoothingCoeff_.load(std::memory_order_relaxed)),
                  quantizationStep_(other.quantizationStep_.load(std::memory_order_relaxed)),
                  pendingSeed_(other.pendingSeed_.load(std::memory_order_relaxed)),
                  brownNoiseState_(other.brownNoiseState_),
                  pinkNoiseOctaves_(other.pinkNoiseOctaves_),
                  denormalFlip_(other.denormalFlip_)
            {
            }

            Generator& operator=(Generator&& other) noexcept
            {
                if (this == &other) return *this;
                isSafeToRun_       = other.isSafeToRun_;
                prng_              = other.prng_;
                sampleRate_        = other.sampleRate_;
                phaseAccumulator_  = other.phaseAccumulator_;
                triggerNext_.store(other.triggerNext_.load(std::memory_order_relaxed),       std::memory_order_relaxed);
                currentValue_      = other.currentValue_;
                targetValue_       = other.targetValue_;
                noiseType_.store  (other.noiseType_.load(std::memory_order_relaxed),         std::memory_order_relaxed);
                useBpmSync_.store (other.useBpmSync_.load(std::memory_order_relaxed),        std::memory_order_relaxed);
                rateHz_.store     (other.rateHz_.load(std::memory_order_relaxed),            std::memory_order_relaxed);
                bpm_.store        (other.bpm_.load(std::memory_order_relaxed),               std::memory_order_relaxed);
                bpmDivision_.store(other.bpmDivision_.load(std::memory_order_relaxed),       std::memory_order_relaxed);
                min_.store        (other.min_.load(std::memory_order_relaxed),               std::memory_order_relaxed);
                max_.store        (other.max_.load(std::memory_order_relaxed),               std::memory_order_relaxed);
                smoothingEnabled_.store(other.smoothingEnabled_.load(std::memory_order_relaxed), std::memory_order_relaxed);
                smoothingCoeff_.store  (other.smoothingCoeff_.load(std::memory_order_relaxed),   std::memory_order_relaxed);
                quantizationStep_.store(other.quantizationStep_.load(std::memory_order_relaxed), std::memory_order_relaxed);
                pendingSeed_.store(other.pendingSeed_.load(std::memory_order_relaxed),       std::memory_order_relaxed);
                brownNoiseState_   = other.brownNoiseState_;
                pinkNoiseOctaves_  = other.pinkNoiseOctaves_;
                denormalFlip_      = other.denormalFlip_;
                return *this;
            }

            // Copy operations stay deleted: the atomics make a meaningful copy
            // semantically ambiguous (concurrent observers of the source) and
            // we don't need them.
            Generator(const Generator&)            = delete;
            Generator& operator=(const Generator&) = delete;

            /**
             * @brief Prepare the generator with the audio sample rate.
             * @param sampleRate Sample rate in Hz (must be > 0).
             */
            void prepare(double sampleRate) noexcept
            {
                if (sampleRate > 0.0) sampleRate_ = sampleRate;
                reset();
            }

            /**
             * @brief Reset internal state (phase, smoothing, noise states).
             */
            void reset() noexcept
            {
                phaseAccumulator_ = 0.0;
                triggerNext_.store(true, std::memory_order_relaxed);
                currentValue_ = static_cast<Real>(0);
                targetValue_ = static_cast<Real>(0);
                brownNoiseState_ = static_cast<Real>(0);
                pinkNoiseOctaves_.fill(static_cast<Real>(0));
            }

            /**
             * @brief Request a lock-free reseed of the internal PRNG.
             * @param newSeed New seed value. Applied on the next audio tick.
             */
            void reseed(std::uint64_t newSeed) noexcept
            {
                if (newSeed == 0) newSeed = 1;
                pendingSeed_.store(newSeed, std::memory_order_release);
            }

            //----------------------------------------------------------------------
            // Main audio API
            //----------------------------------------------------------------------

            /**
             * @brief Generate and return the next modulation sample.
             * @return Next modulation value mapped into [min, max].
             * @note Safe for real-time audio thread. Mitigates denormals dynamically.
             */
            [[nodiscard]] Real getNextSample() noexcept
            {
                if (!isSafeToRun_) [[unlikely]] return static_cast<Real>(0);

                const auto pending = pendingSeed_.exchange(0, std::memory_order_acq_rel);
                if (pending != 0)
                {
                    prng_.reseed(pending);
                    reset();
                }

                const Real continuousNoise = tickContinuousNoise();
                updatePhase();

                if (triggerNext_.exchange(false, std::memory_order_acquire))
                {
                    generateNewTarget(continuousNoise);
                }

                if (smoothingEnabled_.load(std::memory_order_relaxed))
                {
                    const Real coeff = smoothingCoeff_.load(std::memory_order_relaxed);
                    currentValue_ += coeff * (targetValue_ - currentValue_);
                }
                else
                {
                    currentValue_ = targetValue_;
                }

                // Professional DC-free denormal mitigation
                denormalFlip_ = -denormalFlip_;
                currentValue_ += denormalFlip_;

                return currentValue_;
            }

            /**
             * @brief Generate a block of modulation samples.
             * @param outputBuffer C++20 span representing the destination buffer.
             * @note Highly optimized SIMD/Cache-friendly path.
             */
            void getNextBlock(std::span<Real> outputBuffer) noexcept
            {
                if (!isSafeToRun_ || outputBuffer.empty()) [[unlikely]] return;

                const auto pending = pendingSeed_.exchange(0, std::memory_order_acq_rel);
                if (pending != 0)
                {
                    prng_.reseed(pending);
                    reset();
                }

                // Cache atomics to local registers for the block duration
                const bool smoothing = smoothingEnabled_.load(std::memory_order_relaxed);
                const Real coeff = smoothingCoeff_.load(std::memory_order_relaxed);
                const NoiseType noiseType = noiseType_.load(std::memory_order_relaxed);
                const Real quantStep = quantizationStep_.load(std::memory_order_relaxed);
                Real currentMin = min_.load(std::memory_order_relaxed);
                Real currentMax = max_.load(std::memory_order_relaxed);

                if (currentMin > currentMax) std::swap(currentMin, currentMax);

                for (Real& sample : outputBuffer)
                {
                    const Real white = static_cast<Real>(prng_.next_double() * 2.0 - 1.0);
                    Real continuousNoise = white;

                    if (noiseType == NoiseType::Pink) continuousNoise = tickPinkNoise(white);
                    else if (noiseType == NoiseType::Brown) continuousNoise = tickBrownNoise(white);

                    updatePhase();

                    if (triggerNext_.exchange(false, std::memory_order_acquire))
                    {
                        // Clamp to [-1,1] before mapping, matching generateNewTarget()
                        // (pink noise can momentarily exceed unity).
                        const Real cn = std::clamp(continuousNoise, static_cast<Real>(-1), static_cast<Real>(1));
                        targetValue_ = currentMin + ((cn * static_cast<Real>(0.5)) + static_cast<Real>(0.5)) * (currentMax - currentMin);
                        if (quantStep > static_cast<Real>(0)) targetValue_ = std::round(targetValue_ / quantStep) * quantStep;
                    }

                    if (smoothing)
                    {
                        currentValue_ += coeff * (targetValue_ - currentValue_);
                    }
                    else
                    {
                        currentValue_ = targetValue_;
                    }

                    denormalFlip_ = -denormalFlip_;
                    currentValue_ += denormalFlip_;

                    sample = currentValue_;
                }
            }

            [[nodiscard]] Real getCurrentValue() const noexcept { return currentValue_; }
            [[nodiscard]] Real getPhase() const noexcept { return static_cast<Real>(phaseAccumulator_); }

            //----------------------------------------------------------------------
            // Configuration API
            //----------------------------------------------------------------------

            void setNoiseType(NoiseType type) noexcept
            {
                noiseType_.store(type, std::memory_order_relaxed);
            }

            void setRateHz(Real rateInHz) noexcept
            {
                useBpmSync_.store(false, std::memory_order_relaxed);
                rateHz_.store(static_cast<float>(rateInHz), std::memory_order_relaxed);
            }

            void setRateBPM(double bpm, BpmDivision division) noexcept
            {
                bpm_.store(static_cast<float>(bpm), std::memory_order_relaxed);
                bpmDivision_.store(division, std::memory_order_relaxed);
                useBpmSync_.store(true, std::memory_order_relaxed);
            }

            void updateBPM(double newBpm) noexcept
            {
                bpm_.store(static_cast<float>(newBpm), std::memory_order_relaxed);
            }

            /**
             * @brief Set the output range for values. Safe from tearing.
             */
            template <typename T>
            void setRange(T min, T max) noexcept
            {
                static_assert(std::is_floating_point_v<T>, "setRange only accepts floating-point types.");
                if (min > max) std::swap(min, max);
                min_.store(static_cast<Real>(min), std::memory_order_relaxed);
                max_.store(static_cast<Real>(max), std::memory_order_relaxed);
            }

            void setSmoothing(bool shouldBeEnabled, Real timeInMs = static_cast<Real>(50.0)) noexcept
            {
                smoothingEnabled_.store(shouldBeEnabled, std::memory_order_relaxed);
                if (sampleRate_ > 0 && timeInMs > static_cast<Real>(0))
                {
                    const double coeff = std::exp(-1.0 / (sampleRate_ * (static_cast<double>(timeInMs) / 1000.0)));
                    smoothingCoeff_.store(static_cast<Real>(1.0 - coeff), std::memory_order_relaxed);
                }
                else
                {
                    // A zero (or negative) smoothing time means instantaneous.
                    // Keeping the previous coefficient here (possibly the
                    // initial 0) would silently freeze the output short of
                    // every new target while smoothing is enabled.
                    smoothingCoeff_.store(static_cast<Real>(1), std::memory_order_relaxed);
                }
            }

            void setQuantization(Real step) noexcept
            {
                if (std::isnan(step) || step < static_cast<Real>(0)) step = static_cast<Real>(0);
                quantizationStep_.store(step, std::memory_order_relaxed);
            }

            void setAnalogDefault(AnalogComponent component) noexcept
            {
                switch (component)
                {
                    case AnalogComponent::TapeMachine:
                        setNoiseType(NoiseType::Brown);
                        setRange(-VariationPercent::TapeMachine / 100.0f, VariationPercent::TapeMachine / 100.0f);
                        setRateHz(0.5f);
                        setSmoothing(true, 100.0f);
                        break;
                    case AnalogComponent::VacuumTube:
                        setNoiseType(NoiseType::Pink);
                        setRange(-VariationPercent::VacuumTube / 100.0f, VariationPercent::VacuumTube / 100.0f);
                        setRateHz(2.0f);
                        setSmoothing(true, 50.0f);
                        break;
                    case AnalogComponent::Transistor:
                        setNoiseType(NoiseType::Pink);
                        setRange(-VariationPercent::Transistor / 100.0f, VariationPercent::Transistor / 100.0f);
                        setRateHz(1.0f);
                        setSmoothing(true, 75.0f);
                        break;
                    case AnalogComponent::Compressor:
                        setNoiseType(NoiseType::Pink);
                        setRange(-VariationPercent::Compressor / 100.0f, VariationPercent::Compressor / 100.0f);
                        setRateHz(1.5f);
                        setSmoothing(true, 80.0f);
                        break;
                    case AnalogComponent::Equalizer:
                        setNoiseType(NoiseType::Brown);
                        setRange(-VariationPercent::Equalizer / 100.0f, VariationPercent::Equalizer / 100.0f);
                        setRateHz(0.8f);
                        setSmoothing(true, 120.0f);
                        break;
                }
            }

            template <typename Int>
            [[nodiscard]] Int getNextDiscrete(Int imin, Int imax) noexcept
            {
                static_assert(std::is_integral_v<Int>, "getNextDiscrete requires an integral type.");
                if (imax <= imin) return imin;
                const Real value = getNextSample();

                Real minVal = min_.load(std::memory_order_relaxed);
                Real maxVal = max_.load(std::memory_order_relaxed);
                if (minVal > maxVal) std::swap(minVal, maxVal);
                if (maxVal == minVal) return imin;

                const double t = static_cast<double>((value - minVal) / (maxVal - minVal));
                const double mapped = t * static_cast<double>(imax - imin) + static_cast<double>(imin);
                // llround/long long, not lround/long: on LLP64 platforms
                // (Windows) long is 32-bit and would truncate 64-bit Int.
                return static_cast<Int>(std::clamp(std::llround(mapped),
                                                   static_cast<long long>(imin),
                                                   static_cast<long long>(imax)));
            }

            [[nodiscard]] int getNextDiscreteInt(int imin, int imax) noexcept
            {
                return static_cast<int>(getNextDiscrete<int>(imin, imax));
            }

        private:
            void checkLockFree() noexcept
            {
                // Guard against the atomics this class actually uses. On every
                // supported platform these are all lock-free; a hypothetical
                // platform where they are not degrades to silence instead of
                // risking a blocking atomic on the audio thread.
                isSafeToRun_ = std::atomic<Real>::is_always_lock_free
                            && std::atomic<float>::is_always_lock_free
                            && std::atomic<std::uint64_t>::is_always_lock_free
                            && std::atomic<NoiseType>::is_always_lock_free
                            && std::atomic<BpmDivision>::is_always_lock_free
                            && std::atomic<bool>::is_always_lock_free;
            }

            /**
             * @brief Ticks the continuous noise generators at audio rate to preserve mathematical 1/f structures.
             */
            [[nodiscard]] Real tickContinuousNoise() noexcept
            {
                const Real white = static_cast<Real>(prng_.next_double() * 2.0 - 1.0);
                switch (noiseType_.load(std::memory_order_relaxed))
                {
                    case NoiseType::White: return white;
                    case NoiseType::Pink:  return tickPinkNoise(white);
                    case NoiseType::Brown: return tickBrownNoise(white);
                    default: return white;
                }
            }

            /**
             * @brief Maps the sampled audio-rate noise to the user boundaries.
             */
            void generateNewTarget(Real sampledNoise) noexcept
            {
                sampledNoise = std::clamp(sampledNoise, static_cast<Real>(-1), static_cast<Real>(1));

                Real currentMin = min_.load(std::memory_order_relaxed);
                Real currentMax = max_.load(std::memory_order_relaxed);
                if (currentMin > currentMax) std::swap(currentMin, currentMax);

                targetValue_ = currentMin + ((sampledNoise * static_cast<Real>(0.5)) + static_cast<Real>(0.5)) * (currentMax - currentMin);

                const Real quantStep = quantizationStep_.load(std::memory_order_relaxed);
                if (quantStep > static_cast<Real>(0))
                {
                    targetValue_ = std::round(targetValue_ / quantStep) * quantStep;
                }
            }

            void updatePhase() noexcept
            {
                Real rate = static_cast<Real>(0);
                if (useBpmSync_.load(std::memory_order_relaxed))
                {
                    const float bpm = bpm_.load(std::memory_order_relaxed);
                    if (bpm > 0.0f)
                    {
                        const double noteLengthInBeats = 4.0 / getBpmDivisionMultiplier(bpmDivision_.load(std::memory_order_relaxed));
                        const double periodInSeconds = (60.0 / static_cast<double>(bpm)) * noteLengthInBeats;
                        if (periodInSeconds > 0.0) rate = static_cast<Real>(1.0 / periodInSeconds);
                    }
                }
                else
                {
                    rate = static_cast<Real>(rateHz_.load(std::memory_order_relaxed));
                }

                if (rate <= static_cast<Real>(0)) return;

                phaseAccumulator_ += static_cast<double>(rate) / sampleRate_;
                if (phaseAccumulator_ >= 1.0)
                {
                    // floor(), not a single -1.0: with rates at or above the
                    // sample rate the fixed decrement would let the
                    // accumulator grow without bound.
                    phaseAccumulator_ -= std::floor(phaseAccumulator_);
                    triggerNext_.store(true, std::memory_order_release);
                }
            }

            [[nodiscard]] Real tickPinkNoise(Real white) noexcept
            {
                Real b0 = pinkNoiseOctaves_[0];
                Real b1 = pinkNoiseOctaves_[1];
                Real b2 = pinkNoiseOctaves_[2];

                b0 = static_cast<Real>(0.99886) * b0 + white * static_cast<Real>(0.0555179);
                b1 = static_cast<Real>(0.99332) * b1 + white * static_cast<Real>(0.0750759);
                b2 = static_cast<Real>(0.96900) * b2 + white * static_cast<Real>(0.1538520);

                pinkNoiseOctaves_[0] = b0;
                pinkNoiseOctaves_[1] = b1;
                pinkNoiseOctaves_[2] = b2;

                return (b0 + b1 + b2 + white * static_cast<Real>(0.1848)) * static_cast<Real>(0.16666666666666666);
            }

            [[nodiscard]] Real tickBrownNoise(Real white) noexcept
            {
                brownNoiseState_ += white * static_cast<Real>(0.02);
                brownNoiseState_ *= static_cast<Real>(0.995);
                brownNoiseState_ = std::clamp(brownNoiseState_, static_cast<Real>(-1.0), static_cast<Real>(1.0));
                return brownNoiseState_;
            }

            [[nodiscard]] static constexpr double getBpmDivisionMultiplier(BpmDivision division) noexcept
            {
                switch (division)
                {
                    case BpmDivision::One:              return 4.0;
                    case BpmDivision::Half:             return 2.0;
                    case BpmDivision::Quarter:          return 1.0;
                    case BpmDivision::Eighth:           return 0.5;
                    case BpmDivision::Sixteenth:        return 0.25;
                    case BpmDivision::ThirtySecond:     return 0.125;
                    case BpmDivision::HalfTriplet:      return 4.0 / 3.0;
                    case BpmDivision::QuarterTriplet:   return 2.0 / 3.0;
                    case BpmDivision::EighthTriplet:    return 1.0 / 3.0;
                    case BpmDivision::SixteenthTriplet: return 1.0 / 6.0;
                    case BpmDivision::DottedHalf:       return 3.0;
                    case BpmDivision::DottedQuarter:    return 1.5;
                    case BpmDivision::DottedEighth:     return 0.75;
                    default:                            return 1.0;
                }
            }

            [[nodiscard]] static std::uint64_t generateUniqueSeed() noexcept
            {
                static std::atomic_uint64_t instanceCounter{ 0 };
                const auto instanceId = instanceCounter.fetch_add(1, std::memory_order_relaxed);
                const auto thisPtrAddress = reinterpret_cast<std::uintptr_t>(&instanceCounter);
                const auto now = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
                std::uint64_t seed = instanceId ^ (thisPtrAddress << 1) ^ (now + 0x9E3779B97F4A7C15ull);
                return seed == 0 ? 1 : seed;
            }

            bool isSafeToRun_{ false };
            Detail::Xoshiro256pp prng_;
            double sampleRate_{ 44100.0 };
            double phaseAccumulator_{ 0.0 };
            std::atomic<bool> triggerNext_{ true };
            Real currentValue_{ static_cast<Real>(0) };
            Real targetValue_{ static_cast<Real>(0) };
            std::atomic<NoiseType> noiseType_{ NoiseType::Pink };
            std::atomic<bool> useBpmSync_{ false };
            std::atomic<float> rateHz_{ 1.0f };
            std::atomic<float> bpm_{ 120.0f };
            std::atomic<BpmDivision> bpmDivision_{ BpmDivision::Quarter };
            // Value-domain atomics use Real: with Real = double, storing them
            // as float would silently round the user's range/quantization.
            // Time-domain ones (rateHz_, bpm_) stay float on purpose.
            std::atomic<Real> min_{ static_cast<Real>(-1) };
            std::atomic<Real> max_{ static_cast<Real>(1) };
            std::atomic<bool> smoothingEnabled_{ false };
            std::atomic<Real> smoothingCoeff_{ static_cast<Real>(0) };
            std::atomic<Real> quantizationStep_{ static_cast<Real>(0) };
            std::atomic<std::uint64_t> pendingSeed_{ 0 };
            Real brownNoiseState_{ static_cast<Real>(0) };
            std::array<Real, 3> pinkNoiseOctaves_{};
            Real denormalFlip_{ static_cast<Real>(1e-18) }; // DC-Free denormal mitigation state
        };
    } // namespace AnalogRandom
} // namespace dspark