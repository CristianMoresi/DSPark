# DSPark — header-only audio DSP framework (C++20, zero dependencies)
#
# Consumption:
#   add_subdirectory(DSPark)            # or FetchContent
#   target_link_libraries(app PRIVATE dspark::dspark)
#
# Options:
#   DSPARK_BUILD_CONFORMANCE  Build the public conformance suite (default: ON
#                             when DSPark is the top-level project).
#   DSPARK_BUILD_TESTS        Build the full test suite (default: ON when
#                             DSPark is the top-level project).

cmake_minimum_required(VERSION 3.21)
project(DSPark VERSION 1.7.0 LANGUAGES CXX)

add_library(dspark INTERFACE)
add_library(dspark::dspark ALIAS dspark)

target_compile_features(dspark INTERFACE cxx_std_20)
target_include_directories(dspark INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/dspark>)

# ------------------------------------------------------------------------------
# Public conformance suite
# ------------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL)
    option(DSPARK_BUILD_CONFORMANCE "Build the DSPark public conformance suite" ON)
    option(DSPARK_BUILD_TESTS       "Build the DSPark test suite"               ON)
    option(DSPARK_VERIFY_FLOAT_CAST_OVERFLOW_SANITIZER
           "Build and run the float-cast-overflow sanitizer negative control" OFF)
else()
    option(DSPARK_BUILD_CONFORMANCE "Build the DSPark public conformance suite" OFF)
    option(DSPARK_BUILD_TESTS       "Build the DSPark test suite"               OFF)
    option(DSPARK_VERIFY_FLOAT_CAST_OVERFLOW_SANITIZER
           "Build and run the float-cast-overflow sanitizer negative control" OFF)
endif()

if(DSPARK_BUILD_CONFORMANCE OR DSPARK_BUILD_TESTS)
    enable_testing()
endif()

if(DSPARK_BUILD_CONFORMANCE)
    add_executable(dspark_conformance conformance/dspark_conformance.cpp)
    target_link_libraries(dspark_conformance PRIVATE dspark::dspark)

    if(MSVC)
        target_compile_options(dspark_conformance PRIVATE /W4 /WX /wd4324)
    else()
        target_compile_options(dspark_conformance PRIVATE
                               -Wall -Wextra -Wpedantic -Werror)
    endif()

    add_test(NAME conformance COMMAND dspark_conformance)
endif()

# ------------------------------------------------------------------------------
# Test suite
# ------------------------------------------------------------------------------
if(DSPARK_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(DSPARK_VERIFY_FLOAT_CAST_OVERFLOW_SANITIZER)
    if(MSVC)
        message(FATAL_ERROR
            "MSVC does not provide the required float-cast-overflow sanitizer")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(dspark_float_cast_overflow_control
                   tests/float_cast_overflow_subject.cpp)
    target_compile_options(dspark_float_cast_overflow_control PRIVATE
        -Wall -Wextra -Wpedantic -Werror
        -fsanitize=float-cast-overflow
        -fno-sanitize-recover=float-cast-overflow)
    target_link_options(dspark_float_cast_overflow_control PRIVATE
        -fsanitize=float-cast-overflow
        -fno-sanitize-recover=float-cast-overflow)
    add_test(NAME float_cast_overflow_sanitizer_control
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_float_cast_overflow_sanitizer.py
                --subject $<TARGET_FILE:dspark_float_cast_overflow_control>)
endif()

# ------------------------------------------------------------------------------
# Install (header-only)
# ------------------------------------------------------------------------------
include(GNUInstallDirs)
install(DIRECTORY Core Effects Analysis IO Music
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dspark
        FILES_MATCHING PATTERN "*.h")
install(FILES DSPark.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dspark)
install(TARGETS dspark EXPORT dsparkTargets)
install(EXPORT dsparkTargets
        NAMESPACE dspark::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dspark)

# Package config, so find_package(dspark CONFIG) works from any install or
# package manager (vcpkg/Conan). ARCH_INDEPENDENT: header-only.
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/dsparkConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dsparkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dspark)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dsparkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dsparkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dsparkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dspark)
