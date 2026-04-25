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
  Usage:
  1. Declare AnalogRandom<float/double> as a member in your processor class.
  2. In prepareToPlay():
     - Call gen.prepare(sampleRate);
     - Optionally set BPM via gen.setBpm(bpmFromPlayHead);
  3. In processBlock():
     - Call gen.getNextValue() per sample for a smoothed random multiplier.
     - Or call gen.getNextBlock(numSamples) for a block of values.
  4. Apply the output as a multiplier to your target parameter.
  Example (gain modulation):
     for (int s = 0; s < buffer.getNumSamples(); ++s)
         gainSample = baseGain * gen.getNextValue();
  Notes:
  - Smooth transitions are applied internally to avoid abrupt jumps.
  - BPM affects modulation rate if using tempo-synced modes.
  - Added support for analog component default configurations via setAnalogDefault().
  - Manual configuration of noise type, range, rate, etc., remains fully supported and unchanged.
  - Extensive constexpr constants from research studies for analog variations, noise, saturation,
etc.
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
#include <type_traits>

/**
 * @namespace AnalogRandom
 * @brief Main namespace for the analog random generation library.
 */
namespace dspark {
namespace AnalogRandom
{
//==============================================================================
// Constexpr constants from research studies
//==============================================================================

// From Study 1: Intrinsic Noise Characteristics
namespace NoiseConstants
{
namespace Thermal
{
constexpr double BOLTZMANN_CONSTANT = 1.380649e-23;
constexpr double ROOM_TEMP_KELVIN = 293.15;
constexpr double TYPICAL_RESISTANCE = 1000.0;
constexpr double NOISE_BANDWIDTH = 20000.0;
} // namespace Thermal

namespace Shot
{
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;
constexpr double TYPICAL_DC_CURRENT = 1.0e-3;
constexpr double AUDIO_BANDWIDTH = 20000.0;
} // namespace Shot

namespace Flicker
{
constexpr double ALPHA_PINK = 1.0;
constexpr double ALPHA_BROWN = 2.0;
constexpr double SPECTRAL_SLOPE_DB_PER_OCTAVE_PINK = -3.0;
constexpr double SPECTRAL_SLOPE_DB_PER_OCTAVE_BROWN = -6.0;
} // namespace Flicker

namespace PracticalNoiseFloors
{
// Example values from equipment
constexpr double NEVE_1073_EIN_DBU = -125.0;
constexpr double NEVE_1073_NOISE_LINE_DBU = -83.0;
constexpr double API_512C_EIN = -129.0;
constexpr double LA_2A_NOISE_DB_BELOW_10DBM = -75.0;
constexpr double AMPEX_351_SNR_DB = 55.0;
constexpr double STUDER_C37_SNR_DB = 75.0;
constexpr double TAPE_NOISE_OFFSET_DB = 8.0; // Typical 8-10 dB higher than equipment noise

constexpr std::array<double, 7> HUM_FREQUENCIES_HZ{ 50.0, 60.0, 100.0, 120.0, 180.0, 240.0, 300.0 };
// Relative amplitudes (example values, can be adjusted)
constexpr std::array<double, 7> HUM_AMPLITUDES_DB{
    -60.0, -55.0, -65.0, -70.0, -75.0, -80.0, -85.0
};
} // namespace PracticalNoiseFloors
} // namespace NoiseConstants

// From Study 1: Non-Linearities
namespace SaturationConstants
{
namespace General
{
constexpr double DRIVE_THRESHOLD_DB = -6.0; // Example threshold where saturation begins
constexpr double MAX_THD_PERCENTAGE = 0.07; // From Neve 1073
} // namespace General

namespace Tube
{
constexpr double EVEN_HARMONIC_DOMINANCE_RATIO = 0.7; // Relative strength of even harmonics
// Example coefficients for tanh-based transfer function
constexpr std::array<double, 3> TRANSFER_FUNCTION_COEFFS{ 1.0, 0.5, 0.1 };
} // namespace Tube

namespace Tape
{
constexpr double ODD_HARMONIC_DOMINANCE_RATIO = 0.6;
constexpr double SATURATION_THRESHOLD_DB = 0.0;
// Example for frequency shift: high-end roll-off start Hz, low-end boost dB
constexpr double HIGH_END_ROLL_OFF_HZ = 10000.0;
constexpr double LOW_END_BOOST_DB = 2.0;
} // namespace Tape
} // namespace SaturationConstants

// From Study 1: Component Tolerances and Aging
namespace ComponentConstants
{
namespace Resistors
{
constexpr double TOLERANCE_PERCENTAGE_1 = 1.0;
constexpr double TOLERANCE_PERCENTAGE_5 = 5.0;
constexpr double TOLERANCE_PERCENTAGE_10 = 10.0;
constexpr double AGING_DRIFT_RATE_PERCENTAGE_PER_YEAR = 0.5; // Example
} // namespace Resistors

namespace Capacitors
{
constexpr double AGING_RATE_X7R_PERCENT_PER_DECADE_HOUR = -2.5;
constexpr double AGING_RATE_Y5V_PERCENT_PER_DECADE_HOUR = -7.0;
constexpr double CURIE_POINT_TEMPERATURE_CELSIUS = 125.0;
} // namespace Capacitors

namespace TransistorsValves
{
constexpr double GAIN_VARIATION_PERCENTAGE = 5.0;
constexpr double TEMPERATURE_SENSITIVITY_FACTOR = 0.1; // Example
constexpr double AGING_FACTOR_GAIN_DRIFT = 0.2;        // Example per year
} // namespace TransistorsValves

namespace Transformers
{
constexpr double SATURATION_THRESHOLD_FLUX_DENSITY_TESLAS = 1.5; // Example
constexpr double HIGH_FREQUENCY_ROLL_OFF_HZ = 10000.0;
} // namespace Transformers
} // namespace ComponentConstants

// From Study 1: Equipment-Specific Variations
namespace EquipmentConstants
{
namespace TapeMachines
{
constexpr double WOW_DEPTH_PERCENTAGE = 0.1;
constexpr double FLUTTER_DEPTH_PERCENTAGE = 0.15;
constexpr double TAPE_HISS_SNR_DB = 55.0;
constexpr double TAPE_SATURATION_THRESHOLD_DB = 0.0;
} // namespace TapeMachines

namespace Preamplifiers
{
constexpr double MAX_MIC_GAIN_DB = 80.0; // Neve 1073
constexpr double EIN_AT_MAX_GAIN_DBU = -125.0;
constexpr double SATURATION_THRESHOLD_DB = -6.0;
constexpr double THD_PERCENTAGE = 0.07;
} // namespace Preamplifiers

namespace Compressors
{
constexpr double MAX_GAIN_REDUCTION_DB = 40.0; // LA-2A
constexpr double THD_AT_10DBM_PERCENT = 0.35;
constexpr double FIXED_ATTACK_TIME_SECONDS = 0.01; // Example
constexpr double FIXED_RELEASE_TIME_SECONDS = 0.5;
} // namespace Compressors

namespace Equalizers
{
constexpr std::array<double, 4> FREQUENCY_POINTS_LOW_HZ{ 20.0, 30.0, 60.0,
                                                         100.0 }; // Pultec example
constexpr std::array<double, 3> FREQUENCY_POINTS_HIGH_HZ{ 3000.0, 5000.0, 10000.0 };
constexpr double BANDWIDTH_CONTROL_RANGE = 1.0; // Q factor range
} // namespace Equalizers

namespace Consoles
{
constexpr double CROSSTALK_LEVEL_DB_AT_1KHZ = -70.0;
constexpr double CROSSTALK_FREQUENCY_SLOPE_DB_PER_OCTAVE = 6.0;
constexpr double VINTAGE_CROSSTALK_MULTIPLIER = 1.5;
constexpr double MODERN_CROSSTALK_MULTIPLIER = 0.5;
} // namespace Consoles
} // namespace EquipmentConstants

// From Study 2: Variation Percentages
namespace VariationPercent
{
constexpr float TapeMachine = 0.1f; // 0.1% for wow/flutter
constexpr float VacuumTube = 5.0f;  // 5.0% for gain variation
constexpr float Transistor = 3.0f;  // 3.0% for gain variation
constexpr float Compressor = 1.5f;  // 1.5% for gain reduction
constexpr float Equalizer = 1.0f;   // 1.0% for frequency drift
} // namespace VariationPercent

//==============================================================================
// PRNG and Internal Details
//==============================================================================
namespace Detail
{
/**
 * @class Xoshiro256pp
 * @brief A compact xoshiro256++ PRNG implementation (non-threadsafe).
 *
 * This type is intentionally lightweight and expects external callers to
 * ensure thread-safety when required (see Generator).
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
        // Avoid seed==0 for splitmix
        if (seed == 0)
            seed = 1;
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
    /**
     * @brief Return a double in [0, 1).
     *
     * Produces 53-bit precision by shifting the 64-bit output and scaling.
     */
    [[nodiscard]] double next_double() noexcept
    {
        const std::uint64_t x = next();
        // Keep top 53 bits -> shift right 11, divide by 2^53
        constexpr double inv2pow53 = 1.0 / 9007199254740992.0; // 1/2^53
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
// Compile-time warning for platforms that do not guarantee lock-free atomics.
struct AtomicCheck
{
    AtomicCheck()
    {
        if (!std::atomic<float>::is_always_lock_free)
        {
#if defined(_MSC_VER)
#pragma message(                                                                                   \
    "WARNING: std::atomic<float> is not guaranteed to be lock-free on this platform. AnalogRandom will fall back to runtime checks, but performance may be affected.")
#else
#warning                                                                                           \
    "std::atomic<float> is not guaranteed to be lock-free on this platform. AnalogRandom will fall back to runtime checks, but performance may be affected."
#endif
        }
    }
};
// This inline instance triggers the compile-time check once per TU.
inline AtomicCheck s_atomicCheck;
} // namespace Detail

//==============================================================================
// Public API
//==============================================================================
/**
 * @enum NoiseType
 * @brief Defines the "color" or character of the random fluctuations.
 */
enum class NoiseType
{
    Pink,  ///< 1/f noise for organic drift.
    Brown, ///< 1/f^2 noise for slow "wow/flutter".
    White  ///< Uncorrelated noise for "sample-and-hold" effects.
};
/**
 * @enum BpmDivision
 * @brief Defines musical note divisions for BPM-synced modulation.
 */
enum class BpmDivision
{
    One,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    ThirtySecond,
    HalfTriplet,
    QuarterTriplet,
    EighthTriplet,
    SixteenthTriplet,
    DottedHalf,
    DottedQuarter,
    DottedEighth,
};
/**
 * @enum AnalogComponent
 * @brief Defines analog component types for default configuration.
 */
enum class AnalogComponent
{
    TapeMachine,
    VacuumTube,
    Transistor,
    Compressor,
    Equalizer
};
/**
 * @class Generator
 * @brief Main generator class for analog-style random modulation.
 *
 * @tparam Real Floating point type to represent values (float or double).
 *
 * @note The class is optimized for audio use: the audio thread should call
 * `getNextSample()` per-sample. Many setters are atomic and thread-safe from
 * other threads (GUI), while `reseed()` uses a lock-free pending-seed
 * mechanism: the actual reseed is applied on the audio thread on the next
 * sample invocation to avoid races.
 */
template <typename Real = float> class Generator
{
    static_assert(std::is_floating_point_v<Real>,
                  "AnalogRandom::Generator requires a floating-point Real type.");

  public:
    //----------------------------------------------------------------------
    // Constructors / lifecycle
    //----------------------------------------------------------------------
    /**
     * @brief Default constructor. A unique seed is generated automatically.
     */
    Generator() noexcept : m_prng(generateUniqueSeed())
    {
        checkLockFree();
    }
    /**
     * @brief Construct with explicit PRNG seed.
     * @param seed PRNG seed (0 is mapped to 1 internally).
     */
    explicit Generator(std::uint64_t seed) noexcept : m_prng(seed)
    {
        checkLockFree();
    }
    /**
     * @brief Prepare the generator with the audio sample rate.
     * @param sampleRate Sample rate in Hz (must be > 0).
     * @note This calls reset().
     */
    void prepare(double sampleRate) noexcept
    {
        if (sampleRate > 0.0)
            m_sampleRate = sampleRate;
        reset();
    }
    /**
     * @brief Reset internal state (phase, smoothing, noise states).
     * @note Safe to call from any thread; when called from non-audio threads it
     * will be applied on the audio thread at the next sample due to the
     * pending-seed mechanism.
     */
    void reset() noexcept
    {
        m_phaseAccumulator = 0.0;
        m_triggerNext.store(true, std::memory_order_relaxed);
        m_currentValue = static_cast<Real>(0);
        m_targetValue = static_cast<Real>(0);
        m_brownNoiseState = static_cast<Real>(0);
        m_pinkNoiseOctaves.fill(static_cast<Real>(0));
        // keep sample rate unchanged if not prepared
    }
    /**
     * @brief Request a reseed of the internal PRNG.
     * @param newSeed New seed value; 0 is accepted but internally remapped.
     *
     * This function is lock-free and can be called from GUI threads. The new
     * seed will be applied by the audio thread on the next call to
     * getNextSample().
     */
    void reseed(std::uint64_t newSeed) noexcept
    {
        if (newSeed == 0)
            newSeed = 1; // 0 reserved sentinel
        m_pendingSeed.store(newSeed, std::memory_order_release);
    }
    //----------------------------------------------------------------------
    // Main audio API
    //----------------------------------------------------------------------
    /**
     * @brief Generate and return the next modulation sample.
     * @return Next modulation value mapped into [min, max].
     *
     * Threading: This function is intended to be called on the audio thread.
     * It will apply any pending reseed (set from other threads) before
     * generating samples, avoiding locks.
     */
    [[nodiscard]] Real getNextSample() noexcept
    {
        // If the platform doesn't guarantee safe atomics for Real, return zero.
        if (!m_isSafeToRun) [[unlikely]]
        {
            return static_cast<Real>(0);
        }
        // Apply pending reseed safely on the audio thread
        const auto pending = m_pendingSeed.exchange(0, std::memory_order_acq_rel);
        if (pending != 0)
        {
            m_prng.reseed(pending);
            reset();
        }
        updatePhase();
        if (m_triggerNext.exchange(false, std::memory_order_acquire))
        {
            generateNewTarget();
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
        // Denormal kill: small additive epsilon to avoid denormals.
        // Apply to the internal state as well as the returned sample so
        // recursive feedback (smoother) cannot accumulate a denormal tail.
        constexpr Real denormEps = static_cast<Real>(1e-25);
        m_currentValue += denormEps;
        return m_currentValue;
    }
    /**
     * @brief Get the current (last returned) value.
     */
    [[nodiscard]] Real getCurrentValue() const noexcept
    {
        return m_currentValue;
    }
    /**
     * @brief Get the internal phase accumulator in [0,1).
     */
    [[nodiscard]] Real getPhase() const noexcept
    {
        return static_cast<Real>(m_phaseAccumulator);
    }
    //----------------------------------------------------------------------
    // Configuration API (thread-safe where noted)
    //----------------------------------------------------------------------
    /**
     * @brief Set the noise color/type.
     * @param type NoiseType enum.
     * @note This setter is atomic and safe to call from GUI threads.
     */
    void setNoiseType(NoiseType type) noexcept
    {
        m_noiseType.store(type, std::memory_order_relaxed);
    }
    /**
     * @brief Set the rate in Hertz (samples-per-second of new target generation).
     * @param rateInHz Rate in Hz. Caller must call prepare(sampleRate) beforehand.
     * @note This setter is atomic and safe for GUI threads.
     */
    void setRateHz(Real rateInHz) noexcept
    {
        m_useBpmSync.store(false, std::memory_order_relaxed);
        m_rateHz.store(static_cast<float>(rateInHz), std::memory_order_relaxed);
    }
    /**
     * @brief Set rate using BPM and a musical division (BPM-synced mode).
     * @param bpm Beats per minute.
     * @param division Musical division enum value.
     * @note This setter is atomic and safe for GUI threads.
     */
    void setRateBPM(double bpm, BpmDivision division) noexcept
    {
        m_bpm.store(static_cast<float>(bpm), std::memory_order_relaxed);
        m_bpmDivision.store(division, std::memory_order_relaxed);
        m_useBpmSync.store(true, std::memory_order_relaxed);
    }
    /**
     * @brief Update an already-set BPM value (when using BPM sync).
     * @param newBpm New BPM value.
     */
    void updateBPM(double newBpm) noexcept
    {
        m_bpm.store(static_cast<float>(newBpm), std::memory_order_relaxed);
    }
    /**
     * @brief Set the output range for values.
     * @tparam T Floating-point type accepted by the call (must be floating point).
     * @param min Minimum output value.
     * @param max Maximum output value.
     *
     * This setter is atomic and safe to call from GUI threads.
     */
    template <typename T> void setRange(T min, T max) noexcept
    {
        static_assert(std::is_floating_point_v<T>, "setRange only accepts floating-point types.");
        m_min.store(static_cast<Real>(min), std::memory_order_relaxed);
        m_max.store(static_cast<Real>(max), std::memory_order_relaxed);
    }
    /**
     * @brief Enable or disable smoothing and set smoothing time.
     * @param shouldBeEnabled true to enable smoothing.
     * @param timeInMs Time constant in milliseconds for the one-pole smoothing filter.
     *
     * This setter is safe for GUI threads. Computation of the coefficient is
     * done using the configured sample rate and applied atomically.
     */
    void setSmoothing(bool shouldBeEnabled, Real timeInMs = static_cast<Real>(50.0)) noexcept
    {
        m_smoothingEnabled.store(shouldBeEnabled, std::memory_order_relaxed);
        if (m_sampleRate > 0 && timeInMs > static_cast<Real>(0))
        {
            const Real coeff =
                static_cast<Real>(std::exp(-1.0 / (static_cast<double>(m_sampleRate) *
                                                   (static_cast<double>(timeInMs) / 1000.0))));
            m_smoothingCoeff.store(static_cast<float>(1.0 - coeff), std::memory_order_relaxed);
        }
    }
    /**
     * @brief Set quantization step. Zero disables quantization.
     * @param step Step value (>= 0).
     */
    void setQuantization(Real step) noexcept
    {
        if (std::isnan(step) || step < static_cast<Real>(0))
        {
            step = static_cast<Real>(0);
        }
        m_quantizationStep.store(static_cast<float>(step), std::memory_order_relaxed);
    }
    /**
     * @brief Configure the generator with defaults for an analog component type.
     * @param component The analog component to emulate.
     * @note This is an additional convenience function; manual configuration is still fully
     * supported.
     */
    void setAnalogDefault(AnalogComponent component) noexcept
    {
        switch (component)
        {
        case AnalogComponent::TapeMachine:
            setNoiseType(NoiseType::Brown);
            setRange(-VariationPercent::TapeMachine / 100.0f,
                     VariationPercent::TapeMachine / 100.0f);
            setRateHz(0.5f); // Slow drift characteristic
            setSmoothing(true, 100.0f);
            break;
        case AnalogComponent::VacuumTube:
            setNoiseType(NoiseType::Pink);
            setRange(-VariationPercent::VacuumTube / 100.0f, VariationPercent::VacuumTube / 100.0f);
            setRateHz(2.0f); // Organic, medium-speed variation
            setSmoothing(true, 50.0f);
            break;
        case AnalogComponent::Transistor:
            setNoiseType(NoiseType::Pink);
            setRange(-VariationPercent::Transistor / 100.0f, VariationPercent::Transistor / 100.0f);
            setRateHz(1.0f); // Reasonable default for solid-state
            setSmoothing(true, 75.0f);
            break;
        case AnalogComponent::Compressor:
            setNoiseType(NoiseType::Pink);
            setRange(-VariationPercent::Compressor / 100.0f, VariationPercent::Compressor / 100.0f);
            setRateHz(1.5f); // Dynamic compression variation
            setSmoothing(true, 80.0f);
            break;
        case AnalogComponent::Equalizer:
            setNoiseType(NoiseType::Brown);
            setRange(-VariationPercent::Equalizer / 100.0f, VariationPercent::Equalizer / 100.0f);
            setRateHz(0.8f); // Slow frequency drift
            setSmoothing(true, 120.0f);
            break;
        }
    }
    //----------------------------------------------------------------------
    // Discrete/Integral helpers
    //----------------------------------------------------------------------
    /**
     * @brief Get a discrete integral value by mapping the generator output to [imin, imax].
     *
     * This helper is intended to be used when you need an integer parameter
     * modulated (for example, a selector from 1..10). It uses the exact same
     * internal pipeline as getNextSample(): pending reseed is applied,
     * phase/smoothing/quantization are honored, and the resulting continuous
     * value is mapped to the integer interval, rounded and clamped.
     *
     * @tparam Int Integral type to return (int, long, ...).
     * @param imin Minimum integer value (inclusive).
     * @param imax Maximum integer value (inclusive).
     * @return Mapped integer in [imin, imax].
     *
     * Threading: Calls getNextSample(), so this helper should be called on the
     * audio thread (or wherever calling getNextSample() is safe).
     */
    template <typename Int> [[nodiscard]] Int getNextDiscrete(Int imin, Int imax) noexcept
    {
        static_assert(std::is_integral_v<Int>, "getNextDiscrete requires an integral type.");
        if (imax <= imin)
            return imin; // trivial or malformed interval
        // Acquire the continuous sample (applies reseed, smoothing, quantization...)
        const Real value = getNextSample(); // already in [m_min, m_max] per design
        const Real minVal = static_cast<Real>(m_min.load(std::memory_order_relaxed));
        const Real maxVal = static_cast<Real>(m_max.load(std::memory_order_relaxed));
        if (maxVal == minVal)
            return imin;
        // Map [minVal, maxVal] -> [0, 1]
        const double t = static_cast<double>((value - minVal) / (maxVal - minVal));
        // Map to [imin, imax] in double domain for precision, then round and clamp
        const double mapped = t * static_cast<double>(imax - imin) + static_cast<double>(imin);
        const long rounded = std::lround(mapped);
        const long clamped = std::clamp(rounded, static_cast<long>(imin), static_cast<long>(imax));
        return static_cast<Int>(clamped);
    }
    /**
     * @brief Convenience overload returning an int directly.
     * @param imin Minimum integer value (inclusive).
     * @param imax Maximum integer value (inclusive).
     * @return int Mapped integer in [imin, imax].
     *
     * This calls the template helper and casts to int.
     */
    [[nodiscard]] int getNextDiscreteInt(int imin, int imax) noexcept
    {
        return static_cast<int>(getNextDiscrete<int>(imin, imax));
    }

  private:
    //----------------------------------------------------------------------
    // Internal utilities
    //----------------------------------------------------------------------
    void checkLockFree() noexcept
    {
        // Uses compile-time guarantee when available for the Real atomic type.
        m_isSafeToRun = std::atomic<Real>::is_always_lock_free;
    }
    void generateNewTarget() noexcept
    {
        Real rawNoise = static_cast<Real>(0);
        switch (m_noiseType.load(std::memory_order_relaxed))
        {
        case NoiseType::White:
        {
            const Real white = static_cast<Real>(m_prng.next_double() * 2.0 - 1.0);
            rawNoise = white;
        }
        break;
        case NoiseType::Pink:
            rawNoise = generatePinkNoise();
            break;
        case NoiseType::Brown:
            rawNoise = generateBrownNoise();
            break;
        }
        rawNoise = std::clamp(rawNoise, static_cast<Real>(-1), static_cast<Real>(1));
        const Real minVal = static_cast<Real>(m_min.load(std::memory_order_relaxed));
        const Real maxVal = static_cast<Real>(m_max.load(std::memory_order_relaxed));
        // Map [-1,1] -> [min,max]
        m_targetValue = minVal + ((rawNoise * static_cast<Real>(0.5)) + static_cast<Real>(0.5)) *
                                     (maxVal - minVal);
        const Real quantStep =
            static_cast<Real>(m_quantizationStep.load(std::memory_order_relaxed));
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
                const double noteLengthInBeats =
                    4.0 / getBpmDivisionMultiplier(m_bpmDivision.load(std::memory_order_relaxed));
                const double periodInSeconds =
                    (60.0 / static_cast<double>(bpm)) * noteLengthInBeats;
                if (periodInSeconds > 0.0)
                    rate = static_cast<Real>(1.0 / periodInSeconds);
            }
        }
        else
        {
            rate = static_cast<Real>(m_rateHz.load(std::memory_order_relaxed));
        }
        if (rate <= static_cast<Real>(0))
            return;
        m_phaseAccumulator += static_cast<double>(rate) / m_sampleRate;
        if (m_phaseAccumulator >= 1.0)
        {
            m_phaseAccumulator -= 1.0;
            m_triggerNext.store(true, std::memory_order_release);
        }
    }
    // Pink (Voss-McCartney like) implementation using three octave filters
    [[nodiscard]] Real generatePinkNoise() noexcept
    {
        const Real white = static_cast<Real>(m_prng.next_double() * 2.0 - 1.0);
        Real b0 = m_pinkNoiseOctaves[0];
        Real b1 = m_pinkNoiseOctaves[1];
        Real b2 = m_pinkNoiseOctaves[2];
        b0 = static_cast<Real>(0.99886) * b0 + white * static_cast<Real>(0.0555179);
        b1 = static_cast<Real>(0.99332) * b1 + white * static_cast<Real>(0.0750759);
        b2 = static_cast<Real>(0.96900) * b2 + white * static_cast<Real>(0.1538520);
        Real pink = b0 + b1 + b2 + white * static_cast<Real>(0.1848);
        m_pinkNoiseOctaves[0] = b0;
        m_pinkNoiseOctaves[1] = b1;
        m_pinkNoiseOctaves[2] = b2;
        // Scale to keep roughly in [-1,1]. The constant chosen empirically.
        return pink * static_cast<Real>(0.16666666666666666);
    }
    // Simple Brownian integrator with gentle leaky factor and normalization.
    [[nodiscard]] Real generateBrownNoise() noexcept
    {
        const Real white = static_cast<Real>(m_prng.next_double() * 2.0 - 1.0);
        // Integrate and leak a bit for stability
        m_brownNoiseState += white * static_cast<Real>(0.02); // smaller step to avoid fast drift
        m_brownNoiseState *= static_cast<Real>(0.995);
        // Normalize to [-1,1] safely
        if (m_brownNoiseState > static_cast<Real>(1.0))
            m_brownNoiseState = static_cast<Real>(1.0);
        if (m_brownNoiseState < static_cast<Real>(-1.0))
            m_brownNoiseState = static_cast<Real>(-1.0);
        return m_brownNoiseState;
    }
    [[nodiscard]] static constexpr double getBpmDivisionMultiplier(BpmDivision division) noexcept
    {
        switch (division)
        {
        case BpmDivision::One:
            return 4.0;
        case BpmDivision::Half:
            return 2.0;
        case BpmDivision::Quarter:
            return 1.0;
        case BpmDivision::Eighth:
            return 0.5;
        case BpmDivision::Sixteenth:
            return 0.25;
        case BpmDivision::ThirtySecond:
            return 0.125;
        case BpmDivision::HalfTriplet:
            return 4.0 / 3.0; // 1.333...
        case BpmDivision::QuarterTriplet:
            return 2.0 / 3.0; // 0.666...
        case BpmDivision::EighthTriplet:
            return 1.0 / 3.0;
        case BpmDivision::SixteenthTriplet:
            return 1.0 / 6.0;
        case BpmDivision::DottedHalf:
            return 3.0; // 2 * 1.5
        case BpmDivision::DottedQuarter:
            return 1.5;
        case BpmDivision::DottedEighth:
            return 0.75;
        default:
            return 1.0;
        }
    }
    [[nodiscard]] static std::uint64_t generateUniqueSeed() noexcept
    {
        static std::atomic_uint64_t instanceCounter{ 0 };
        const auto instanceId = instanceCounter.fetch_add(1, std::memory_order_relaxed);
        const auto thisPtrAddress = reinterpret_cast<std::uintptr_t>(&instanceCounter);
        // Mix with time for extra entropy
        const auto now =
            static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uint64_t seed = instanceId ^ (thisPtrAddress << 1) ^ (now + 0x9E3779B97F4A7C15ull);
        if (seed == 0)
            seed = 1;
        return seed;
    }
    //----------------------------------------------------------------------
    // Member variables
    //----------------------------------------------------------------------
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
    // Lock-free pending reseed applied on audio thread
    std::atomic<std::uint64_t> m_pendingSeed{ 0 };
    // Noise internal states
    Real m_brownNoiseState{ static_cast<Real>(0) };
    std::array<Real, 3> m_pinkNoiseOctaves{};
}; // class Generator
} // namespace AnalogRandom
} // namespace dspark
