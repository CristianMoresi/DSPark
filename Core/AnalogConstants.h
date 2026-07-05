// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file AnalogConstants.h
 * @brief Reference constants from analog-hardware research and measurement.
 *
 * A curated collection of physical constants, measured noise floors, component
 * tolerances, and classic-equipment behaviour figures. They document the
 * real-world values that the framework's analog-modelling defaults are derived
 * from, and are available for users building their own component models.
 *
 * The constants deliberately live inside the dspark::AnalogRandom namespace:
 * that is the analog noise/drift subsystem defined in Core/AnalogRandom.h
 * (Generator, NoiseType, AnalogComponent), and this header supplies the
 * measured data that subsystem's defaults are calibrated against. Including
 * either header alone is fine; they reopen the same namespace.
 *
 * None of these constants is required by the DSP hot paths: this header is
 * pure reference data with zero runtime cost.
 *
 * Dependencies: C++20 standard library only (<array>).
 */

#include <array>

namespace dspark {
namespace AnalogRandom {

//==============================================================================
// Constants from research studies and hardware measurements
//==============================================================================

namespace NoiseConstants
{
    namespace Thermal
    {
        constexpr double BOLTZMANN_CONSTANT = 1.380649e-23; // J/K (exact, SI 2019)
        constexpr double ROOM_TEMP_KELVIN = 293.15;         // 20 degrees C
        constexpr double TYPICAL_RESISTANCE = 1000.0;       // ohms
        constexpr double NOISE_BANDWIDTH = 20000.0;         // Hz
    }

    namespace Shot
    {
        constexpr double ELEMENTARY_CHARGE = 1.602176634e-19; // coulombs (exact, SI 2019)
        constexpr double TYPICAL_DC_CURRENT = 1.0e-3;         // amperes
        constexpr double AUDIO_BANDWIDTH = 20000.0;           // Hz
    }

    namespace Flicker
    {
        constexpr double ALPHA_PINK = 1.0;  // 1/f^alpha exponent
        constexpr double ALPHA_BROWN = 2.0; // 1/f^alpha exponent
        constexpr double SPECTRAL_SLOPE_DB_PER_OCTAVE_PINK = -3.0;
        constexpr double SPECTRAL_SLOPE_DB_PER_OCTAVE_BROWN = -6.0;
    }

    namespace PracticalNoiseFloors
    {
        constexpr double NEVE_1073_EIN_DBU = -125.0;         // EIN, 200 ohm source, max gain
        constexpr double NEVE_1073_NOISE_LINE_DBU = -83.0;   // line-amp noise
        constexpr double API_512C_EIN = -129.0;              // dBu, EIN at max gain
        constexpr double LA_2A_NOISE_DB_BELOW_10DBM = -75.0; // spec: noise 75 dB below +10 dBm
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

} // namespace AnalogRandom
} // namespace dspark
