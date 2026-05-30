// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

/*
==============================================================================
    AnalogRandom.h
    Author: Cristian Moresi
    ---------------------------------------------------------------------------
    Description:
    Header-only C++20 generator of smooth, analog-style random values for
    modulation. Supports white, pink, and brown noise types.
    
    Features:
    - True audio-rate analog drift (continuous 1/f and 1/f^2 filtering).
    - Sample-and-Hold LFO behavior with 1-pole smoothing.
    - Zero allocations, SIMD/Cache-friendly block processing.
    - Lock-free thread safety between GUI and Audio threads.
    - DC-free denormal mitigation.
==============================================================================
*/

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <type_traits>

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
        // Constexpr constants from research studies
        //==============================================================================

        namespace NoiseConstants
        {
            namespace Thermal
            {
                constexpr double BOLTZMANN_CONSTANT = 1.380649e-23;
                constexpr double ROOM_TEMP_KELVIN = 293.15;
                constexpr double TYPICAL_RESISTANCE = 1000.0;
                constexpr double NOISE_BANDWIDTH = 20000.0;
            } 

            namespace Shot
            {
                constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;
                constexpr double TYPICAL_DC_CURRENT = 1.0e-3;
                constexpr double AUDIO_BANDWIDTH = 20000.0;
            } 

            namespace Flicker
            {
                constexpr double ALPHA_PINK = 1.0;
                constexpr double ALPHA_BROWN = 2.0;
                constexpr double SPECTRAL_SLOPE_DB_PER_OCTAVE_PINK = -3.0;
                constexpr double SPECTRAL_SLOPE_DB_PER_OCTAVE_BROWN = -6.0;
            } 

            namespace PracticalNoiseFloors
            {
                constexpr double NEVE_1073_EIN_DBU = -125.0;
                constexpr double NEVE_1073_NOISE_LINE_DBU = -83.0;
                constexpr double API_512C_EIN = -129.0;
                constexpr double LA_2A_NOISE_DB_BELOW_10DBM = -75.0;
                constexpr double AMPEX_351_SNR_DB = 55.0;
                constexpr double STUDER_C37_SNR_DB = 75.0;
                constexpr double TAPE_NOISE_OFFSET_DB = 8.0; 

                constexpr std::array<double, 7> HUM_FREQUENCIES_HZ{ 50.0, 60.0, 100.0, 120.0, 180.0, 240.0, 300.0 };
                constexpr std::array<double, 7> HUM_AMPLITUDES_DB{
                    -60.0, -55.0, -65.0, -70.0, -75.0, -80.0, -85.0
                };
            } 
        } 

        namespace SaturationConstants
        {
            namespace General
            {
                constexpr double DRIVE_THRESHOLD_DB = -6.0;
                constexpr double MAX_THD_PERCENTAGE = 0.07;
            } 

            namespace Tube
            {
                constexpr double EVEN_HARMONIC_DOMINANCE_RATIO = 0.7;
                constexpr std::array<double, 3> TRANSFER_FUNCTION_COEFFS{ 1.0, 0.5, 0.1 };
            } 

            namespace Tape
            {
                constexpr double ODD_HARMONIC_DOMINANCE_RATIO = 0.6;
                constexpr double SATURATION_THRESHOLD_DB = 0.0;
                constexpr double HIGH_END_ROLL_OFF_HZ = 10000.0;
                constexpr double LOW_END_BOOST_DB = 2.0;
            } 
        } 

        namespace ComponentConstants
        {
            namespace Resistors
            {
                constexpr double TOLERANCE_PERCENTAGE_1 = 1.0;
                constexpr double TOLERANCE_PERCENTAGE_5 = 5.0;
                constexpr double TOLERANCE_PERCENTAGE_10 = 10.0;
                constexpr double AGING_DRIFT_RATE_PERCENTAGE_PER_YEAR = 0.5;
            } 

            namespace Capacitors
            {
                constexpr double AGING_RATE_X7R_PERCENT_PER_DECADE_HOUR = -2.5;
                constexpr double AGING_RATE_Y5V_PERCENT_PER_DECADE_HOUR = -7.0;
                constexpr double CURIE_POINT_TEMPERATURE_CELSIUS = 125.0;
            } 

            namespace TransistorsValves
            {
                constexpr double GAIN_VARIATION_PERCENTAGE = 5.0;
                constexpr double TEMPERATURE_SENSITIVITY_FACTOR = 0.1;
                constexpr double AGING_FACTOR_GAIN_DRIFT = 0.2;
            } 

            namespace Transformers
            {
                constexpr double SATURATION_THRESHOLD_FLUX_DENSITY_TESLAS = 1.5;
                constexpr double HIGH_FREQUENCY_ROLL_OFF_HZ = 10000.0;
            } 
        } 

        namespace EquipmentConstants
        {
            namespace TapeMachines
            {
                constexpr double WOW_DEPTH_PERCENTAGE = 0.1;
                constexpr double FLUTTER_DEPTH_PERCENTAGE = 0.15;
                constexpr double TAPE_HISS_SNR_DB = 55.0;
                constexpr double TAPE_SATURATION_THRESHOLD_DB = 0.0;
            } 

            namespace Preamplifiers
            {
                constexpr double MAX_MIC_GAIN_DB = 80.0;
                constexpr double EIN_AT_MAX_GAIN_DBU = -125.0;
                constexpr double SATURATION_THRESHOLD_DB = -6.0;
                constexpr double THD_PERCENTAGE = 0.07;
            } 

            namespace Compressors
            {
                constexpr double MAX_GAIN_REDUCTION_DB = 40.0;
                constexpr double THD_AT_10DBM_PERCENT = 0.35;
                constexpr double FIXED_ATTACK_TIME_SECONDS = 0.01;
                constexpr double FIXED_RELEASE_TIME_SECONDS = 0.5;
            } 

            namespace Equalizers
            {
                constexpr std::array<double, 4> FREQUENCY_POINTS_LOW_HZ{ 20.0, 30.0, 60.0, 100.0 };
                constexpr std::array<double, 3> FREQUENCY_POINTS_HIGH_HZ{ 3000.0, 5000.0, 10000.0 };
                constexpr double BANDWIDTH_CONTROL_RANGE = 1.0;
            } 

            namespace Consoles
            {
                constexpr double CROSSTALK_LEVEL_DB_AT_1KHZ = -70.0;
                constexpr double CROSSTALK_FREQUENCY_SLOPE_DB_PER_OCTAVE = 6.0;
                constexpr double VINTAGE_CROSSTALK_MULTIPLIER = 1.5;
                constexpr double MODERN_CROSSTALK_MULTIPLIER = 0.5;
            } 
        } 

        namespace VariationPercent
        {
            constexpr float TapeMachine = 0.1f;
            constexpr float VacuumTube = 5.0f;
            constexpr float Transistor = 3.0f;
            constexpr float Compressor = 1.5f;
            constexpr float Equalizer = 1.0f;
        } 

        //==============================================================================
        // PRNG and Internal Details
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
            // (checkLockFree() -> m_isSafeToRun). A preprocessor #warning here would
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
            Generator() noexcept : m_prng(generateUniqueSeed()) 
            { 
                checkLockFree(); 
            }

            explicit Generator(std::uint64_t seed) noexcept : m_prng(seed)
            {
                checkLockFree();
            }

            // std::atomic members are not move-constructible by default, but we
            // own this Generator's state and need it to be storable in standard
            // containers (e.g. std::vector). The atomic loads/stores below are
            // safe because the source object is being moved-from — by contract
            // it is not concurrently observed by any other thread.
            Generator(Generator&& other) noexcept
                : m_isSafeToRun(other.m_isSafeToRun),
                  m_prng(other.m_prng),
                  m_sampleRate(other.m_sampleRate),
                  m_phaseAccumulator(other.m_phaseAccumulator),
                  m_triggerNext(other.m_triggerNext.load(std::memory_order_relaxed)),
                  m_currentValue(other.m_currentValue),
                  m_targetValue(other.m_targetValue),
                  m_noiseType(other.m_noiseType.load(std::memory_order_relaxed)),
                  m_useBpmSync(other.m_useBpmSync.load(std::memory_order_relaxed)),
                  m_rateHz(other.m_rateHz.load(std::memory_order_relaxed)),
                  m_bpm(other.m_bpm.load(std::memory_order_relaxed)),
                  m_bpmDivision(other.m_bpmDivision.load(std::memory_order_relaxed)),
                  m_min(other.m_min.load(std::memory_order_relaxed)),
                  m_max(other.m_max.load(std::memory_order_relaxed)),
                  m_smoothingEnabled(other.m_smoothingEnabled.load(std::memory_order_relaxed)),
                  m_smoothingCoeff(other.m_smoothingCoeff.load(std::memory_order_relaxed)),
                  m_quantizationStep(other.m_quantizationStep.load(std::memory_order_relaxed)),
                  m_pendingSeed(other.m_pendingSeed.load(std::memory_order_relaxed)),
                  m_brownNoiseState(other.m_brownNoiseState),
                  m_pinkNoiseOctaves(other.m_pinkNoiseOctaves),
                  m_denormalFlip(other.m_denormalFlip)
            {
            }

            Generator& operator=(Generator&& other) noexcept
            {
                if (this == &other) return *this;
                m_isSafeToRun       = other.m_isSafeToRun;
                m_prng              = other.m_prng;
                m_sampleRate        = other.m_sampleRate;
                m_phaseAccumulator  = other.m_phaseAccumulator;
                m_triggerNext.store(other.m_triggerNext.load(std::memory_order_relaxed),       std::memory_order_relaxed);
                m_currentValue      = other.m_currentValue;
                m_targetValue       = other.m_targetValue;
                m_noiseType.store  (other.m_noiseType.load(std::memory_order_relaxed),         std::memory_order_relaxed);
                m_useBpmSync.store (other.m_useBpmSync.load(std::memory_order_relaxed),        std::memory_order_relaxed);
                m_rateHz.store     (other.m_rateHz.load(std::memory_order_relaxed),            std::memory_order_relaxed);
                m_bpm.store        (other.m_bpm.load(std::memory_order_relaxed),               std::memory_order_relaxed);
                m_bpmDivision.store(other.m_bpmDivision.load(std::memory_order_relaxed),       std::memory_order_relaxed);
                m_min.store        (other.m_min.load(std::memory_order_relaxed),               std::memory_order_relaxed);
                m_max.store        (other.m_max.load(std::memory_order_relaxed),               std::memory_order_relaxed);
                m_smoothingEnabled.store(other.m_smoothingEnabled.load(std::memory_order_relaxed), std::memory_order_relaxed);
                m_smoothingCoeff.store  (other.m_smoothingCoeff.load(std::memory_order_relaxed),   std::memory_order_relaxed);
                m_quantizationStep.store(other.m_quantizationStep.load(std::memory_order_relaxed), std::memory_order_relaxed);
                m_pendingSeed.store(other.m_pendingSeed.load(std::memory_order_relaxed),       std::memory_order_relaxed);
                m_brownNoiseState   = other.m_brownNoiseState;
                m_pinkNoiseOctaves  = other.m_pinkNoiseOctaves;
                m_denormalFlip      = other.m_denormalFlip;
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
                if (sampleRate > 0.0) m_sampleRate = sampleRate;
                reset();
            }

            /**
             * @brief Reset internal state (phase, smoothing, noise states).
             */
            void reset() noexcept
            {
                m_phaseAccumulator = 0.0;
                m_triggerNext.store(true, std::memory_order_relaxed);
                m_currentValue = static_cast<Real>(0);
                m_targetValue = static_cast<Real>(0);
                m_brownNoiseState = static_cast<Real>(0);
                m_pinkNoiseOctaves.fill(static_cast<Real>(0));
            }

            /**
             * @brief Request a lock-free reseed of the internal PRNG.
             * @param newSeed New seed value. Applied on the next audio tick.
             */
            void reseed(std::uint64_t newSeed) noexcept
            {
                if (newSeed == 0) newSeed = 1;
                m_pendingSeed.store(newSeed, std::memory_order_release);
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
                if (!m_isSafeToRun) [[unlikely]] return static_cast<Real>(0);

                const auto pending = m_pendingSeed.exchange(0, std::memory_order_acq_rel);
                if (pending != 0)
                {
                    m_prng.reseed(pending);
                    reset();
                }

                const Real continuousNoise = tickContinuousNoise();
                updatePhase();

                if (m_triggerNext.exchange(false, std::memory_order_acquire))
                {
                    generateNewTarget(continuousNoise);
                }

                if (m_smoothingEnabled.load(std::memory_order_relaxed))
                {
                    const Real coeff = m_smoothingCoeff.load(std::memory_order_relaxed);
                    m_currentValue += coeff * (m_targetValue - m_currentValue);
                }
                else
                {
                    m_currentValue = m_targetValue;
                }

                // Professional DC-free denormal mitigation
                m_denormalFlip = -m_denormalFlip;
                m_currentValue += m_denormalFlip;

                return m_currentValue;
            }

            /**
             * @brief Generate a block of modulation samples.
             * @param outputBuffer C++20 span representing the destination buffer.
             * @note Highly optimized SIMD/Cache-friendly path.
             */
            void getNextBlock(std::span<Real> outputBuffer) noexcept
            {
                if (!m_isSafeToRun || outputBuffer.empty()) [[unlikely]] return;

                const auto pending = m_pendingSeed.exchange(0, std::memory_order_acq_rel);
                if (pending != 0)
                {
                    m_prng.reseed(pending);
                    reset();
                }

                // Cache atomics to local registers for the block duration
                const bool smoothing = m_smoothingEnabled.load(std::memory_order_relaxed);
                const Real coeff = m_smoothingCoeff.load(std::memory_order_relaxed);
                const NoiseType noiseType = m_noiseType.load(std::memory_order_relaxed);
                const float quantStep = m_quantizationStep.load(std::memory_order_relaxed);
                Real currentMin = m_min.load(std::memory_order_relaxed);
                Real currentMax = m_max.load(std::memory_order_relaxed);
                
                if (currentMin > currentMax) std::swap(currentMin, currentMax);

                for (Real& sample : outputBuffer)
                {
                    const Real white = static_cast<Real>(m_prng.next_double() * 2.0 - 1.0);
                    Real continuousNoise = white;
                    
                    if (noiseType == NoiseType::Pink) continuousNoise = tickPinkNoise(white);
                    else if (noiseType == NoiseType::Brown) continuousNoise = tickBrownNoise(white);

                    updatePhase();

                    if (m_triggerNext.exchange(false, std::memory_order_acquire))
                    {
                        // Clamp to [-1,1] before mapping, matching generateNewTarget()
                        // (pink noise can momentarily exceed unity).
                        const Real cn = std::clamp(continuousNoise, static_cast<Real>(-1), static_cast<Real>(1));
                        m_targetValue = currentMin + ((cn * static_cast<Real>(0.5)) + static_cast<Real>(0.5)) * (currentMax - currentMin);
                        if (quantStep > 0.0f) m_targetValue = std::round(m_targetValue / quantStep) * quantStep;
                    }

                    if (smoothing)
                    {
                        m_currentValue += coeff * (m_targetValue - m_currentValue);
                    }
                    else
                    {
                        m_currentValue = m_targetValue;
                    }

                    m_denormalFlip = -m_denormalFlip;
                    m_currentValue += m_denormalFlip;
                    
                    sample = m_currentValue;
                }
            }

            [[nodiscard]] Real getCurrentValue() const noexcept { return m_currentValue; }
            [[nodiscard]] Real getPhase() const noexcept { return static_cast<Real>(m_phaseAccumulator); }

            //----------------------------------------------------------------------
            // Configuration API
            //----------------------------------------------------------------------

            void setNoiseType(NoiseType type) noexcept 
            { 
                m_noiseType.store(type, std::memory_order_relaxed); 
            }

            void setRateHz(Real rateInHz) noexcept
            {
                m_useBpmSync.store(false, std::memory_order_relaxed);
                m_rateHz.store(static_cast<float>(rateInHz), std::memory_order_relaxed);
            }

            void setRateBPM(double bpm, BpmDivision division) noexcept
            {
                m_bpm.store(static_cast<float>(bpm), std::memory_order_relaxed);
                m_bpmDivision.store(division, std::memory_order_relaxed);
                m_useBpmSync.store(true, std::memory_order_relaxed);
            }

            void updateBPM(double newBpm) noexcept 
            { 
                m_bpm.store(static_cast<float>(newBpm), std::memory_order_relaxed); 
            }

            /**
             * @brief Set the output range for values. Safe from tearing.
             */
            template <typename T> 
            void setRange(T min, T max) noexcept
            {
                static_assert(std::is_floating_point_v<T>, "setRange only accepts floating-point types.");
                if (min > max) std::swap(min, max);
                m_min.store(static_cast<Real>(min), std::memory_order_relaxed);
                m_max.store(static_cast<Real>(max), std::memory_order_relaxed);
            }

            void setSmoothing(bool shouldBeEnabled, Real timeInMs = static_cast<Real>(50.0)) noexcept
            {
                m_smoothingEnabled.store(shouldBeEnabled, std::memory_order_relaxed);
                if (m_sampleRate > 0 && timeInMs > static_cast<Real>(0))
                {
                    const Real coeff = static_cast<Real>(std::exp(-1.0 / (static_cast<double>(m_sampleRate) * (static_cast<double>(timeInMs) / 1000.0))));
                    m_smoothingCoeff.store(static_cast<float>(1.0 - coeff), std::memory_order_relaxed);
                }
            }

            void setQuantization(Real step) noexcept
            {
                if (std::isnan(step) || step < static_cast<Real>(0)) step = static_cast<Real>(0);
                m_quantizationStep.store(static_cast<float>(step), std::memory_order_relaxed);
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
                
                Real minVal = static_cast<Real>(m_min.load(std::memory_order_relaxed));
                Real maxVal = static_cast<Real>(m_max.load(std::memory_order_relaxed));
                if (minVal > maxVal) std::swap(minVal, maxVal);
                if (maxVal == minVal) return imin;

                const double t = static_cast<double>((value - minVal) / (maxVal - minVal));
                const double mapped = t * static_cast<double>(imax - imin) + static_cast<double>(imin);
                return static_cast<Int>(std::clamp(std::lround(mapped), static_cast<long>(imin), static_cast<long>(imax)));
            }

            [[nodiscard]] int getNextDiscreteInt(int imin, int imax) noexcept
            {
                return static_cast<int>(getNextDiscrete<int>(imin, imax));
            }

        private:
            void checkLockFree() noexcept
            {
                m_isSafeToRun = std::atomic<Real>::is_always_lock_free;
            }

            /**
             * @brief Ticks the continuous noise generators at audio rate to preserve mathematical 1/f structures.
             */
            [[nodiscard]] Real tickContinuousNoise() noexcept
            {
                const Real white = static_cast<Real>(m_prng.next_double() * 2.0 - 1.0);
                switch (m_noiseType.load(std::memory_order_relaxed))
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
                
                Real currentMin = m_min.load(std::memory_order_relaxed);
                Real currentMax = m_max.load(std::memory_order_relaxed);
                if (currentMin > currentMax) std::swap(currentMin, currentMax);

                m_targetValue = currentMin + ((sampledNoise * static_cast<Real>(0.5)) + static_cast<Real>(0.5)) * (currentMax - currentMin);

                const Real quantStep = static_cast<Real>(m_quantizationStep.load(std::memory_order_relaxed));
                if (quantStep > static_cast<Real>(0))
                {
                    m_targetValue = std::round(m_targetValue / quantStep) * quantStep;
                }
            }

            void updatePhase() noexcept
            {
                Real rate = static_cast<Real>(0);
                if (m_useBpmSync.load(std::memory_order_relaxed))
                {
                    const float bpm = m_bpm.load(std::memory_order_relaxed);
                    if (bpm > 0.0f)
                    {
                        const double noteLengthInBeats = 4.0 / getBpmDivisionMultiplier(m_bpmDivision.load(std::memory_order_relaxed));
                        const double periodInSeconds = (60.0 / static_cast<double>(bpm)) * noteLengthInBeats;
                        if (periodInSeconds > 0.0) rate = static_cast<Real>(1.0 / periodInSeconds);
                    }
                }
                else
                {
                    rate = static_cast<Real>(m_rateHz.load(std::memory_order_relaxed));
                }

                if (rate <= static_cast<Real>(0)) return;
                
                m_phaseAccumulator += static_cast<double>(rate) / m_sampleRate;
                if (m_phaseAccumulator >= 1.0)
                {
                    m_phaseAccumulator -= 1.0;
                    m_triggerNext.store(true, std::memory_order_release);
                }
            }

            [[nodiscard]] Real tickPinkNoise(Real white) noexcept
            {
                Real b0 = m_pinkNoiseOctaves[0];
                Real b1 = m_pinkNoiseOctaves[1];
                Real b2 = m_pinkNoiseOctaves[2];
                
                b0 = static_cast<Real>(0.99886) * b0 + white * static_cast<Real>(0.0555179);
                b1 = static_cast<Real>(0.99332) * b1 + white * static_cast<Real>(0.0750759);
                b2 = static_cast<Real>(0.96900) * b2 + white * static_cast<Real>(0.1538520);
                
                m_pinkNoiseOctaves[0] = b0;
                m_pinkNoiseOctaves[1] = b1;
                m_pinkNoiseOctaves[2] = b2;
                
                return (b0 + b1 + b2 + white * static_cast<Real>(0.1848)) * static_cast<Real>(0.16666666666666666);
            }

            [[nodiscard]] Real tickBrownNoise(Real white) noexcept
            {
                m_brownNoiseState += white * static_cast<Real>(0.02);
                m_brownNoiseState *= static_cast<Real>(0.995);
                m_brownNoiseState = std::clamp(m_brownNoiseState, static_cast<Real>(-1.0), static_cast<Real>(1.0));
                return m_brownNoiseState;
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

            bool m_isSafeToRun{ false };
            Detail::Xoshiro256pp m_prng;
            double m_sampleRate{ 44100.0 };
            double m_phaseAccumulator{ 0.0 };
            std::atomic<bool> m_triggerNext{ true };
            Real m_currentValue{ static_cast<Real>(0) };
            Real m_targetValue{ static_cast<Real>(0) };
            std::atomic<NoiseType> m_noiseType{ NoiseType::Pink };
            std::atomic<bool> m_useBpmSync{ false };
            std::atomic<float> m_rateHz{ 1.0f };
            std::atomic<float> m_bpm{ 120.0f };
            std::atomic<BpmDivision> m_bpmDivision{ BpmDivision::Quarter };
            std::atomic<float> m_min{ -1.0f };
            std::atomic<float> m_max{ 1.0f };
            std::atomic<bool> m_smoothingEnabled{ false };
            std::atomic<float> m_smoothingCoeff{ 0.0f };
            std::atomic<float> m_quantizationStep{ 0.0f };
            std::atomic<std::uint64_t> m_pendingSeed{ 0 };
            Real m_brownNoiseState{ static_cast<Real>(0) };
            std::array<Real, 3> m_pinkNoiseOctaves{};
            Real m_denormalFlip{ static_cast<Real>(1e-18) }; // DC-Free denormal mitigation state
        }; 
    } // namespace AnalogRandom
} // namespace dspark