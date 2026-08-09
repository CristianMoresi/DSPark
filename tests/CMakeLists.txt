# DSPark test suite
#
#   cmake -S . -B build -DDSPARK_BUILD_TESTS=ON
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Four targets:
#   dspark_tests            the suite proper (630+ cases, zero dependencies)
#   dspark_alloc_failure    allocation-failure injection, deliberately apart
#   dspark_standalone       compile gate: umbrella header alone, float + double
#   dspark_smoke            every effect at defaults, checked for NaN/Inf/runaway

# Assertions must survive an optimised build. A number of cases exercise the
# debug-time contracts of the framework (bounds, ranges, preconditions), so
# NDEBUG is stripped from the optimised configurations here: the suite is built
# with optimisations for speed, not to switch off its own checks. These are
# directory-scoped copies of the cache variables, so nothing outside tests/ is
# affected.
foreach(config RELEASE RELWITHDEBINFO MINSIZEREL)
    string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
endforeach()

add_executable(dspark_tests
    main.cpp
    TestAnalysis.cpp
    TestCoreFilters.cpp
    TestCoreFoundation.cpp
    TestCoreGenerators.cpp
    TestCoreSpectral.cpp
    TestCoreUtilities.cpp
    TestEffectsAdvanced.cpp
    TestEffectsDynamics.cpp
    TestEffectsModulation.cpp
    TestEffectsStereo.cpp
    TestEffectsTone.cpp
    TestIO.cpp
    TestIntegration.cpp
    TestMusic.cpp
    TestBeat.cpp
    TestKey.cpp
    TestLoudnessNormalizer.cpp
    TestOnset.cpp
    TestQualityAudit.cpp
    TestState.cpp)

add_executable(dspark_standalone standalone_compile_test.cpp)
add_executable(dspark_smoke      functional_smoke_test.cpp)

# Allocation-failure injection is a separate executable, not one more source
# file of dspark_tests. Its translation unit defines replacement operator
# new/delete, and those are program-wide: every allocation in whatever binary
# they are linked into goes through malloc/free, which leaves AddressSanitizer
# with matched malloc/free pairs and nothing to check, so its
# alloc-dealloc-mismatch / new-delete-type-mismatch class stops firing for every
# case in that binary. Keeping it in its own target confines the replacement to
# the one case that needs it and leaves the suite with full sanitizer coverage.
# It reuses main.cpp: the harness registry is per-binary.
add_executable(dspark_alloc_failure main.cpp TestAllocFailure.cpp)

set(dspark_test_targets dspark_tests dspark_alloc_failure
                        dspark_standalone dspark_smoke)

foreach(target IN LISTS dspark_test_targets)
    target_link_libraries(${target} PRIVATE dspark::dspark)

    if(MSVC)
        # /wd4324: benign "structure was padded due to alignment specifier" on
        # the deliberately 32-byte-aligned coefficient structs.
        # /bigobj:  the heavier translation units exceed the default section
        #           limit once the templates are instantiated at -O2.
        target_compile_options(${target} PRIVATE /W4 /wd4324 /bigobj)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
endforeach()

# TestIO writes and deletes small WAV/MP3 files in the working directory.
add_test(NAME suite         COMMAND dspark_tests
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME alloc_failure COMMAND dspark_alloc_failure)
add_test(NAME smoke         COMMAND dspark_smoke)
add_test(NAME standalone    COMMAND dspark_standalone)
